use super::*;

fn rate_limit_version(timestamp: u64) -> RateLimitVersion {
    RateLimitVersion::new(timestamp, ReplicaId::new())
}

fn stored(epoch: u64, count: i64, timestamp: u64) -> Vec<u8> {
    state_from_insert_value(&encode(epoch, count), rate_limit_version(timestamp))
        .expect("raw epoch/count insert normalizes")
        .encode_live()
        .expect("live state has encoded bytes")
}

#[test]
fn raw_epoch_count_payload_round_trip() {
    for (epoch, count) in [
        (0_u64, 0_i64),
        (1, 1),
        (5, 30),
        (u64::MAX, i64::MAX),
        (u64::MAX, i64::MIN),
        (42, -1),
    ] {
        let buf = encode(epoch, count);
        assert_eq!(buf.len(), EPOCH_MAX_WINS_ENCODED_LEN);
        let decoded = decode_raw_epoch_count(&buf).expect("encoded buffer is 16 bytes");
        assert_eq!(decoded, EpochCount { epoch, count });
    }
}

#[test]
fn public_decode_accepts_raw_epoch_count_payload() {
    assert_eq!(
        decode(&encode(1, 2)),
        Some(EpochCount { epoch: 1, count: 2 })
    );
}

#[test]
fn raw_epoch_count_decode_rejects_wrong_lengths() {
    assert_eq!(decode_raw_epoch_count(&[]), None);
    assert_eq!(decode_raw_epoch_count(&[0u8; 15]), None);
    assert_eq!(decode_raw_epoch_count(&[0u8; 17]), None);
    assert!(decode_raw_epoch_count(&[0u8; 16]).is_some());
}

#[test]
fn normalized_shard_decodes_to_epoch_count() {
    let encoded = state_from_insert_value(&encode(7, 42), rate_limit_version(10))
        .expect("raw epoch/count insert normalizes to shard state")
        .encode_live()
        .expect("live shard encodes");
    assert_ne!(encoded.len(), EPOCH_MAX_WINS_ENCODED_LEN);
    assert_eq!(
        decode(&encoded),
        Some(EpochCount {
            epoch: 7,
            count: 42
        })
    );
}

#[test]
fn same_epoch_max_count_wins() {
    let local = stored(5, 30, 1);
    let remote = stored(5, 42, 2);
    let merged = merge(&local, &remote);
    assert_eq!(
        decode(&merged).unwrap(),
        EpochCount {
            epoch: 5,
            count: 42
        }
    );
    assert_eq!(merge(&remote, &local), merged);
}

#[test]
fn higher_epoch_wins_even_with_lower_count() {
    let merged = merge(&stored(5, 30, 1), &stored(6, 0, 2));
    assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 6, count: 0 });
}

#[test]
fn lower_epoch_loses_to_local_newer_window() {
    let merged = merge(&stored(6, 10, 1), &stored(5, 100, 2));
    assert_eq!(
        decode(&merged).unwrap(),
        EpochCount {
            epoch: 6,
            count: 10
        }
    );
}

#[test]
fn near_simultaneous_reset_both_at_zero() {
    let merged = merge(&stored(5, 0, 1), &stored(5, 0, 2));
    assert_eq!(decode(&merged).unwrap(), EpochCount { epoch: 5, count: 0 });
}

#[test]
fn malformed_remote_keeps_local() {
    let local = stored(5, 30, 1);
    let merged = merge(&local, &[0xFFu8; 15]);
    assert_eq!(merged, local);
}

#[test]
fn malformed_local_is_replaced_by_remote() {
    let remote = stored(5, 30, 1);
    let merged = merge(&[], &remote);
    assert_eq!(merged, remote);
}

#[test]
fn both_malformed_keeps_local_no_panic() {
    let corrupt_local = vec![1u8, 2, 3];
    let merged = merge(&corrupt_local, &[0xFFu8; 17]);
    assert_eq!(merged, corrupt_local);
}

#[test]
fn signed_count_preserves_sign() {
    let merged = merge(&stored(5, -10, 1), &stored(5, -5, 2));
    assert_eq!(
        decode(&merged).unwrap(),
        EpochCount {
            epoch: 5,
            count: -5
        }
    );
}

#[test]
fn merge_is_idempotent() {
    let value = stored(42, 7, 1);
    assert_eq!(merge(&value, &value), value);
}

#[test]
fn merge_is_associative_on_three_values() {
    let a = stored(5, 10, 1);
    let b = stored(6, 3, 2);
    let c = stored(6, 9, 3);
    let ab_then_c = merge(&merge(&a, &b), &c);
    let a_then_bc = merge(&a, &merge(&b, &c));
    assert_eq!(ab_then_c, a_then_bc);
    assert_eq!(
        decode(&ab_then_c).unwrap(),
        EpochCount { epoch: 6, count: 9 }
    );
}

#[test]
fn compacted_live_state_remembers_tombstone_boundary() {
    let key = "rl:global:node-a".to_string();
    let ops = [
        Operation::insert(key.clone(), encode(9, 99).to_vec(), 10, ReplicaId::new()),
        Operation::remove(key.clone(), 20, ReplicaId::new()),
        Operation::insert(key.clone(), encode(1, 1).to_vec(), 30, ReplicaId::new()),
    ];

    let compacted =
        compact_operations(ops.iter()).expect("post-tombstone live insert remains live");
    assert!(matches!(compacted, Operation::Insert { .. }));

    let delayed = Operation::insert(key.clone(), encode(9, 99).to_vec(), 10, ReplicaId::new());
    let compacted_again =
        compact_operations([compacted, delayed].iter()).expect("compacted live shard remains live");
    let Operation::Insert { value, .. } = compacted_again else {
        panic!("expected live compacted shard");
    };
    assert_eq!(
        decode(&value),
        Some(EpochCount { epoch: 1, count: 1 }),
        "pre-tombstone high-epoch insert must stay suppressed after compaction",
    );
}

#[test]
fn compacted_live_state_uses_newest_live_version() {
    let key = "rl:global:node-a".to_string();
    let ops = [
        Operation::remove(key.clone(), 50, ReplicaId::new()),
        Operation::insert(key.clone(), encode(7, 100).to_vec(), 60, ReplicaId::new()),
        Operation::insert(key.clone(), encode(6, 1).to_vec(), 70, ReplicaId::new()),
    ];

    let compacted = compact_operations(ops.iter()).expect("live state wins");
    let Operation::Insert {
        value, timestamp, ..
    } = compacted
    else {
        panic!("expected live compacted shard");
    };
    assert_eq!(timestamp, 70);
    assert_eq!(
        decode(&value),
        Some(EpochCount {
            epoch: 7,
            count: 100
        })
    );
}

#[test]
fn compact_operations_skips_malformed_inserts() {
    let key = "rl:global:node-a".to_string();
    let malformed = Operation::insert(key.clone(), vec![1, 2, 3], 100, ReplicaId::new());
    let valid = Operation::insert(key.clone(), encode(5, 42).to_vec(), 10, ReplicaId::new());
    let compacted =
        compact_operations([malformed.clone(), valid].iter()).expect("valid insert survives");

    let Operation::Insert { value, .. } = compacted else {
        panic!("valid insert should remain after skipping malformed insert");
    };
    assert_eq!(
        decode(&value),
        Some(EpochCount {
            epoch: 5,
            count: 42
        })
    );

    let tombstone = Operation::remove(key.clone(), 110, ReplicaId::new());
    let compacted = compact_operations([malformed, tombstone].iter()).expect("tombstone survives");
    let Operation::Remove { timestamp, .. } = compacted else {
        panic!("tombstone should remain after skipping malformed insert");
    };
    assert_eq!(timestamp, 110);
}
