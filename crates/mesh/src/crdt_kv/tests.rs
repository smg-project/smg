use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, Once,
    },
    thread,
    time::Duration,
};

use tracing::info;
use tracing_subscriber::{
    filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

use super::{
    crdt::CrdtOrMap,
    epoch_max_wins::{self, decode, encode, EpochCount},
    merge_strategy::MergeStrategy,
    operation::{Operation, OperationLog},
    replica::ReplicaId,
};
static INIT: Once = Once::new();

/// Initialize test logging infrastructure
fn init_test_logging() {
    INIT.call_once(|| {
        let _ = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            )
            .try_init();
    });
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_basic_insert_and_get() {
    init_test_logging();
    let map = CrdtOrMap::new();

    // Insert data
    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());

    // Verify retrieval
    assert_eq!(map.get("key1"), Some(b"value1".to_vec()));
    assert_eq!(map.get("key2"), Some(b"value2".to_vec()));
    assert_eq!(map.get("key3"), None);
}

#[test]
fn test_basic_remove() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    assert!(map.contains_key("key1"));

    map.remove("key1");
    assert!(!map.contains_key("key1"));
    assert_eq!(map.get("key1"), None);
}

#[test]
fn test_update_value() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    assert_eq!(map.get("key1"), Some(b"value1".to_vec()));

    map.insert("key1".to_string(), b"value2".to_vec());
    assert_eq!(map.get("key1"), Some(b"value2".to_vec()));
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[test]
fn test_concurrent_inserts() {
    init_test_logging();
    let map = CrdtOrMap::new();
    let mut handles = vec![];

    // 10 threads inserting concurrently
    for i in 0..10 {
        let map_clone = map.clone();
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let key = format!("key_{i}_{j}");
                let value = format!("value_{i}_{j}").into_bytes();
                map_clone.insert(key, value);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all data was inserted successfully
    for i in 0..10 {
        for j in 0..100 {
            let key = format!("key_{i}_{j}");
            assert!(map.contains_key(&key));
        }
    }
}

// ============================================================================
// CRDT Merge Tests
// ============================================================================

#[test]
fn test_merge_two_replicas() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Replica 1 inserts data
    replica1.insert("key1".to_string(), b"value1_from_r1".to_vec());
    replica1.insert("key2".to_string(), b"value2_from_r1".to_vec());

    // Replica 2 inserts data
    replica2.insert("key3".to_string(), b"value3_from_r2".to_vec());
    replica2.insert("key4".to_string(), b"value4_from_r2".to_vec());
    replica2.remove("key3");

    // Get replica 2's operation log and merge into replica 1
    let log2 = replica2.get_operation_log();

    info!(
        "Replica 1 merging Replica 2's log with \n====\n{:?}\n====",
        log2
    );
    replica1.merge(&log2);

    // Verify merged data
    assert_eq!(replica1.get("key1"), Some(b"value1_from_r1".to_vec()));
    assert_eq!(replica1.get("key2"), Some(b"value2_from_r1".to_vec()));
    assert_eq!(replica1.get("key3"), None);
    assert_eq!(replica1.get("key4"), Some(b"value4_from_r2".to_vec()));
}

#[test]
fn test_merge_emits_change_for_new_value() {
    init_test_logging();
    let map = CrdtOrMap::new();
    let mut log = OperationLog::new();
    log.append(Operation::insert(
        "worker:a".to_string(),
        b"v".to_vec(),
        5,
        ReplicaId::new(),
    ));

    let changes = map.merge(&log);
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].key, "worker:a");
    assert_eq!(changes[0].value, Some(b"v".to_vec()));
}

#[test]
fn test_merge_no_change_on_byte_identical_newer_insert() {
    // A newer-version insert that rewrites byte-identical bytes is accepted by
    // LWW (advances the winner's version) but does not change the observable
    // `get` value, so `merge` must report no CrdtChange.
    init_test_logging();
    let map = CrdtOrMap::new();

    let mut first = OperationLog::new();
    first.append(Operation::insert(
        "worker:a".to_string(),
        b"v".to_vec(),
        5,
        ReplicaId::new(),
    ));
    assert_eq!(map.merge(&first).len(), 1, "first insert is a change");

    let mut identical_newer = OperationLog::new();
    identical_newer.append(Operation::insert(
        "worker:a".to_string(),
        b"v".to_vec(),
        10,
        ReplicaId::new(),
    ));
    assert!(
        map.merge(&identical_newer).is_empty(),
        "byte-identical newer-version rewrite must fire no CrdtChange"
    );
}

#[test]
fn test_merge_no_change_on_tombstone_for_never_seen_key() {
    // A remove for a key that was never live leaves `get` at None before and
    // after, so it must fire no CrdtChange (both LWW and EpochMaxWins).
    init_test_logging();
    let map = CrdtOrMap::new();
    map.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let mut log = OperationLog::new();
    log.append(Operation::remove(
        "rl:global:node-a".to_string(),
        50,
        ReplicaId::new(),
    ));
    assert!(
        map.merge(&log).is_empty(),
        "tombstone for a never-seen key must fire no CrdtChange"
    );

    let mut worker_tombstone = OperationLog::new();
    worker_tombstone.append(Operation::remove(
        "worker:ghost".to_string(),
        50,
        ReplicaId::new(),
    ));
    assert!(
        map.merge(&worker_tombstone).is_empty(),
        "LWW tombstone for a never-seen key must fire no CrdtChange"
    );
}

#[test]
fn test_merge_emits_none_when_tombstone_kills_live_key() {
    init_test_logging();
    let map = CrdtOrMap::new();
    let replica = ReplicaId::new();

    let mut insert = OperationLog::new();
    insert.append(Operation::insert(
        "worker:a".to_string(),
        b"v".to_vec(),
        5,
        replica,
    ));
    map.merge(&insert);

    let mut tombstone = OperationLog::new();
    tombstone.append(Operation::remove(
        "worker:a".to_string(),
        10,
        ReplicaId::new(),
    ));
    let changes = map.merge(&tombstone);
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].key, "worker:a");
    assert_eq!(
        changes[0].value, None,
        "killing a live key emits value None"
    );
}

#[test]
fn test_concurrent_insert_same_key() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Two replicas insert the same key concurrently
    replica1.insert("key1".to_string(), b"value_from_r1".to_vec());
    replica2.insert("key1".to_string(), b"value_from_r2".to_vec());

    // Get replica 2's log and merge
    let log2 = replica2.get_operation_log();
    info!(
        "Replica 1 merging Replica 2's log with \n====\n{:?}\n====",
        log2
    );
    replica1.merge(&log2);

    // LWW semantic: conflicts resolve by (timestamp, replica_id), so one value wins.
    // The winner displayed here is deterministic under that ordering.
    info!("{:?}", String::from_utf8(replica1.get("key1").unwrap()));
    assert!(replica1.contains_key("key1"));
}

#[test]
fn test_remove_after_insert() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Replica 1 inserts
    replica1.insert("key1".to_string(), b"value1".to_vec());

    // Replica 2 also inserts the same key
    replica2.insert("key1".to_string(), b"value1".to_vec());

    // Replica 1 removes
    replica1.remove("key1");

    // Get replica 2's log and merge into replica 1
    let log2 = replica2.get_operation_log();
    replica1.merge(&log2);

    // Remove operation should win (because remove has newer timestamp)
    assert!(!replica1.contains_key("key1"));
}

#[test]
fn test_older_insert_applied_later_does_not_overwrite_winner() {
    init_test_logging();
    let source = CrdtOrMap::new();

    source.insert("key1".to_string(), b"older_value".to_vec());
    source.insert("key1".to_string(), b"newer_value".to_vec());

    let full_log = source.get_operation_log();
    let stale_insert = full_log
        .operations()
        .iter()
        .find_map(|op| match op {
            Operation::Insert { value, .. } if value.as_slice() == b"older_value" => {
                Some(op.clone())
            }
            _ => None,
        })
        .unwrap();

    let replica = CrdtOrMap::new();
    replica.merge(&full_log);
    assert_eq!(replica.get("key1"), Some(b"newer_value".to_vec()));

    let mut stale_log = OperationLog::new();
    stale_log.append(stale_insert);
    replica.merge(&stale_log);

    assert_eq!(replica.get("key1"), Some(b"newer_value".to_vec()));
}

#[test]
fn test_epoch_max_wins_compaction_uses_value_epoch() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    let older_reset =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 1, ReplicaId::new());
    let newer_stale_count = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        2,
        ReplicaId::new(),
    );

    let mut log = OperationLog::new();
    log.append(newer_stale_count);
    log.append(older_reset);

    replica.merge(&log);

    let value = replica.get(key).expect("rate-limit shard should exist");
    assert_eq!(decode(&value), Some(EpochCount { epoch: 6, count: 0 }));
}

#[test]
fn test_epoch_max_wins_preserves_newer_tombstone() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:dead-node";
    let stale_insert =
        Operation::insert(key.to_string(), encode(6, 50).to_vec(), 1, ReplicaId::new());
    let tombstone = Operation::remove(key.to_string(), 2, ReplicaId::new());

    let mut log = OperationLog::new();
    log.append(stale_insert);
    log.append(tombstone);

    replica.merge(&log);

    assert_eq!(replica.get(key), None);
}

#[test]
fn test_epoch_max_wins_local_write_cannot_rewind_epoch() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    replica.insert(key.to_string(), encode(6, 0).to_vec());
    replica.insert(key.to_string(), encode(5, 100).to_vec());

    let value = replica.get(key).expect("rate-limit shard should exist");
    assert_eq!(decode(&value), Some(EpochCount { epoch: 6, count: 0 }));
}

#[test]
fn test_epoch_max_wins_tombstone_filters_pre_tombstone_inserts_per_point() {
    // Spec §2.5: a tombstone partitions history by (timestamp, replica_id).
    // Inserts before the newest tombstone are stale and dropped; inserts
    // after it survive. Verified per-point so the live store matches what
    // `compact_operations` would produce.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    let post_tombstone_low_epoch = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        100,
        ReplicaId::new(),
    );
    let pre_tombstone_high_epoch =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 90, ReplicaId::new());
    let tombstone_between_them = Operation::remove(key.to_string(), 95, ReplicaId::new());

    let mut log1 = OperationLog::new();
    log1.append(post_tombstone_low_epoch);
    replica.merge(&log1);

    let mut log2 = OperationLog::new();
    log2.append(pre_tombstone_high_epoch);
    replica.merge(&log2);
    assert_eq!(
        decode(&replica.get(key).expect("high-epoch insert merges in")),
        Some(EpochCount { epoch: 6, count: 0 }),
    );

    let mut log3 = OperationLog::new();
    log3.append(tombstone_between_them);
    replica.merge(&log3);

    assert_eq!(
        decode(
            &replica
                .get(key)
                .expect("post-tombstone live point still exists"),
        ),
        Some(EpochCount {
            epoch: 5,
            count: 100,
        }),
        "pre-tombstone (ts=90) high-epoch point must be filtered out; \
         post-tombstone (ts=100) low-epoch point survives",
    );
}

#[test]
fn test_epoch_max_wins_live_store_matches_compacted_log_after_tombstone() {
    // After sequential merges, the live store and the canonical compacted
    // operation log must decode to the same EpochCount. Without per-point
    // tombstone filtering in the live path, they diverge.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    for op in [
        Operation::insert(
            key.to_string(),
            encode(5, 100).to_vec(),
            100,
            ReplicaId::new(),
        ),
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 90, ReplicaId::new()),
        Operation::remove(key.to_string(), 95, ReplicaId::new()),
    ] {
        let mut log = OperationLog::new();
        log.append(op);
        replica.merge(&log);
    }

    let live_value = decode(&replica.get(key).expect("store has a survivor"));
    let log = replica.get_operation_log();
    let compacted_value = log
        .operations()
        .iter()
        .find_map(|op| match op {
            Operation::Insert { value, .. } => Some(decode(value)),
            Operation::Remove { .. } => None,
        })
        .expect("compacted log retains the surviving insert");
    assert_eq!(live_value, compacted_value);
}

#[test]
fn test_epoch_max_wins_tombstone_kills_all_points_removes_key() {
    // When the tombstone is newer than every live point, the key is removed
    // from the store but the tombstone metadata persists so a delayed
    // pre-tombstone insert cannot resurrect it.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";
    let mut log = OperationLog::new();
    log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        50,
        ReplicaId::new(),
    ));
    log.append(Operation::remove(key.to_string(), 100, ReplicaId::new()));
    replica.merge(&log);

    assert!(replica.get(key).is_none(), "all live points killed");

    let mut late_log = OperationLog::new();
    late_log.append(Operation::insert(
        key.to_string(),
        encode(8, 88).to_vec(),
        40,
        ReplicaId::new(),
    ));
    replica.merge(&late_log);
    assert!(
        replica.get(key).is_none(),
        "pre-tombstone insert must not resurrect a fully-killed key",
    );
}

#[test]
fn test_epoch_max_wins_snapshot_only_propagation_preserves_tombstone_boundary() {
    // Snapshot-only path: the source replica compacts its log so a
    // peer receives just one Insert per key (with the shard's
    // `tombstone_version` embedded), never the original Remove op.
    // A late peer that still holds the pre-tombstone high-epoch
    // insert must not be able to resurrect it.
    init_test_logging();
    let key = "rl:global:node-a";

    // Source: pre-tombstone high-epoch insert, then tombstone, then
    // post-tombstone lower-epoch insert. After merge+compact, the
    // log holds a single shard insert with tombstone_version=65.
    let source = CrdtOrMap::new();
    source.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    let mut source_log = OperationLog::new();
    source_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    source_log.append(Operation::remove(key.to_string(), 65, ReplicaId::new()));
    source_log.append(Operation::insert(
        key.to_string(),
        encode(6, 1).to_vec(),
        70,
        ReplicaId::new(),
    ));
    source.merge(&source_log);

    let snapshot_log = source.get_operation_log();
    assert_eq!(
        snapshot_log.operations().len(),
        1,
        "compaction must reduce to a single shard insert",
    );

    // Receiver applies the snapshot — gets the shard with
    // tombstone_version embedded but no Remove op in its log.
    let receiver = CrdtOrMap::new();
    receiver.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    receiver.merge(&snapshot_log);
    assert_eq!(
        decode(&receiver.get(key).expect("post-tombstone insert applied")),
        Some(EpochCount { epoch: 6, count: 1 }),
    );

    // Late peer that never saw the Remove gossips the original
    // pre-tombstone high-epoch insert. The receiver must reject it
    // — the shard's embedded tombstone_version (65) > the late
    // insert's version (60), so it gets filtered.
    let mut late_log = OperationLog::new();
    late_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    receiver.merge(&late_log);

    assert_eq!(
        decode(
            &receiver
                .get(key)
                .expect("post-tombstone state must survive late pre-tombstone insert")
        ),
        Some(EpochCount { epoch: 6, count: 1 }),
        "pre-tombstone insert must not resurrect when only the snapshot \
         (no Remove op) has reached the receiver",
    );
}

#[test]
fn test_epoch_max_wins_compacted_snapshot_applies_when_op_id_already_seen() {
    // Receiver sees the post-tombstone raw insert first (op-id (70, c)),
    // then the source's compacted snapshot that reuses that same op-id but
    // carries the embedded tombstone_version. Without strategy-aware merge
    // the op-id collision drops the richer payload, and a delayed
    // pre-tombstone insert can resurrect the deleted shard.
    init_test_logging();
    let key = "rl:global:node-a";

    let pre_tombstone_replica = ReplicaId::new();
    let tombstone_replica = ReplicaId::new();
    let post_tombstone_replica = ReplicaId::new();

    // Source applies the full history and compacts into one Insert that
    // embeds tombstone_version=65 and reuses the post-tombstone op-id.
    let source = CrdtOrMap::new();
    source.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    let mut source_log = OperationLog::new();
    source_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        pre_tombstone_replica,
    ));
    source_log.append(Operation::remove(key.to_string(), 65, tombstone_replica));
    source_log.append(Operation::insert(
        key.to_string(),
        encode(6, 1).to_vec(),
        70,
        post_tombstone_replica,
    ));
    source.merge(&source_log);
    let compacted_log = source.get_operation_log();
    assert_eq!(compacted_log.operations().len(), 1);

    // Receiver already has the raw post-tombstone insert (same op-id as the
    // compacted snapshot will use), but no tombstone information yet.
    let receiver = CrdtOrMap::new();
    receiver.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    let mut raw_log = OperationLog::new();
    raw_log.append(Operation::insert(
        key.to_string(),
        encode(6, 1).to_vec(),
        70,
        post_tombstone_replica,
    ));
    receiver.merge(&raw_log);

    // Compacted snapshot arrives. Op-id collides with the raw insert, but
    // the payload now embeds tombstone_version=65 - it must be applied.
    receiver.merge(&compacted_log);
    assert_eq!(
        decode(&receiver.get(key).expect("post-tombstone shard remains")),
        Some(EpochCount { epoch: 6, count: 1 }),
    );

    // Delayed pre-tombstone insert from yet another replica. Without the
    // embedded tombstone_version reaching the receiver, this would
    // resurrect epoch=7 (the high-epoch deleted value).
    let mut delayed = OperationLog::new();
    delayed.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    receiver.merge(&delayed);

    assert_eq!(
        decode(
            &receiver
                .get(key)
                .expect("post-tombstone live point still survives")
        ),
        Some(EpochCount { epoch: 6, count: 1 }),
        "compacted snapshot must override the same-op-id raw payload so \
         delayed pre-tombstone inserts cannot resurrect",
    );
}

#[test]
fn test_epoch_max_wins_same_op_id_folds_within_single_batch() {
    // The raw post-tombstone insert and the compacted snapshot that reuses its
    // op-id (and embeds tombstone_version=65) arrive in the SAME batch.
    // apply_remote_ops appends both and lets per-key compaction fold them, so
    // the snapshot's tombstone_version must survive without a separate op-id
    // index - otherwise a later delayed pre-tombstone insert resurrects the
    // deleted high-epoch shard.
    init_test_logging();
    let key = "rl:global:node-a";

    let pre_tombstone_replica = ReplicaId::new();
    let tombstone_replica = ReplicaId::new();
    let post_tombstone_replica = ReplicaId::new();

    // Produce the compacted snapshot op: a single Insert at op-id
    // (70, post_tombstone_replica) embedding tombstone_version=65.
    let source = CrdtOrMap::new();
    source.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    let mut source_log = OperationLog::new();
    source_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        pre_tombstone_replica,
    ));
    source_log.append(Operation::remove(key.to_string(), 65, tombstone_replica));
    source_log.append(Operation::insert(
        key.to_string(),
        encode(6, 1).to_vec(),
        70,
        post_tombstone_replica,
    ));
    source.merge(&source_log);
    let compacted = source.get_operation_log();
    assert_eq!(compacted.operations().len(), 1);
    let snapshot_op = compacted.operations()[0].clone();

    // Receiver gets BOTH the raw post-tombstone insert and the compacted
    // snapshot (same op-id) in one batch.
    let receiver = CrdtOrMap::new();
    receiver.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
    let mut batch = OperationLog::new();
    batch.append(Operation::insert(
        key.to_string(),
        encode(6, 1).to_vec(),
        70,
        post_tombstone_replica,
    ));
    batch.append(snapshot_op);
    receiver.merge(&batch);
    // The two same-op-id ops must append-then-compact to a single exported op,
    // not survive as two entries masked by the state merge.
    assert_eq!(
        receiver.get_operation_log().operations().len(),
        1,
        "same-op-id batch must fold to a single exported op",
    );
    assert_eq!(
        decode(&receiver.get(key).expect("post-tombstone shard remains")),
        Some(EpochCount { epoch: 6, count: 1 }),
    );

    // Delayed pre-tombstone insert must be suppressed by the folded-in tombstone.
    let mut delayed = OperationLog::new();
    delayed.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    receiver.merge(&delayed);
    assert_eq!(
        decode(
            &receiver
                .get(key)
                .expect("post-tombstone live point still survives")
        ),
        Some(EpochCount { epoch: 6, count: 1 }),
        "same-op-id snapshot folded within one batch must preserve the tombstone",
    );
}

#[test]
fn test_epoch_max_wins_remove_for_never_seen_key_blocks_delayed_pre_tombstone_insert() {
    // Replica receives a tombstone for a key it has never seen, then later
    // receives a delayed pre-tombstone insert for the same key. The tombstone
    // must record ordering metadata so the delayed insert is suppressed -
    // otherwise the live store resurrects state the compacted operation log
    // says is dead (spec §2.5 / §5.8).
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";

    let mut remove_log = OperationLog::new();
    remove_log.append(Operation::remove(key.to_string(), 100, ReplicaId::new()));
    replica.merge(&remove_log);
    assert!(
        replica.get(key).is_none(),
        "no live value after remove-only merge"
    );

    let mut insert_log = OperationLog::new();
    insert_log.append(Operation::insert(
        key.to_string(),
        encode(7, 99).to_vec(),
        60,
        ReplicaId::new(),
    ));
    replica.merge(&insert_log);

    assert!(
        replica.get(key).is_none(),
        "pre-tombstone insert (ts=60) must not resurrect a never-seen-key tombstone (ts=100)",
    );
}

#[test]
fn test_epoch_max_wins_older_delayed_remove_preserves_tombstone_age() {
    // After the dominant tombstone is recorded, an older delayed Remove
    // from a lagging peer must not refresh the existing tombstone's
    // `created_at`. Otherwise stale gossip from a slow node pins the GC
    // clock and the tombstone never gets collected, leaking metadata and
    // key-lock entries indefinitely.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";

    // Live insert at ts=50, then dominant Remove at ts=100. After merge, the
    // key is tombstoned and metadata holds the (100, _) tombstone.
    let mut initial = OperationLog::new();
    initial.append(Operation::insert(
        key.to_string(),
        encode(1, 1).to_vec(),
        50,
        ReplicaId::new(),
    ));
    initial.append(Operation::remove(key.to_string(), 100, ReplicaId::new()));
    replica.merge(&initial);
    assert!(replica.get(key).is_none(), "key fully tombstoned");

    // Let the tombstone age past the GC grace window we will use below.
    thread::sleep(Duration::from_millis(60));

    // Older Remove from a different replica arrives (stale gossip from a
    // lagging peer). It is dominated by the existing (100, _) tombstone, so
    // the on-disk shape doesn't change - but if the metadata entry is
    // re-created with `Instant::now()`, the GC clock resets.
    let mut delayed = OperationLog::new();
    delayed.append(Operation::remove(key.to_string(), 50, ReplicaId::new()));
    replica.merge(&delayed);

    // Grace is shorter than the time since the original tombstone but longer
    // than the time since the delayed Remove. With the fix, GC sees the
    // tombstone as old enough and collects it.
    let removed = replica.gc_tombstones_with_grace(Duration::from_millis(50));
    assert_eq!(
        removed, 1,
        "older delayed Remove must not refresh the tombstone GC clock",
    );
}

#[test]
fn test_epoch_max_wins_newer_tombstone_refreshes_tombstone_age() {
    // Symmetric to the older-delayed-remove case: when a *newer* winning
    // tombstone supersedes the existing one, the GC clock must reset. If the
    // clock kept the original tombstone's `created_at`, GC could collect the
    // entry immediately after the newer tombstone arrived, deleting state
    // that a later delayed insert (older than the winning tombstone but
    // newer than any remaining local frontier) would then resurrect.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a";

    // Live insert at ts=10, then Remove at ts=50. The key is tombstoned with
    // the (50, _) tombstone as the winning version.
    let mut initial = OperationLog::new();
    initial.append(Operation::insert(
        key.to_string(),
        encode(1, 1).to_vec(),
        10,
        ReplicaId::new(),
    ));
    initial.append(Operation::remove(key.to_string(), 50, ReplicaId::new()));
    replica.merge(&initial);
    assert!(replica.get(key).is_none(), "key fully tombstoned");

    // Let the first tombstone age past the grace window we will use below.
    // The grace value below also bounds how much wall-clock can elapse
    // between the merge of `newer` and the GC check before the assertion
    // becomes flaky, so keep this sleep / grace gap wide.
    thread::sleep(Duration::from_millis(250));

    // Newer winning Remove arrives at ts=200. The merged tombstone version
    // advances; the GC clock must restart so the new winning tombstone gets
    // its full grace period.
    let mut newer = OperationLog::new();
    newer.append(Operation::remove(key.to_string(), 200, ReplicaId::new()));
    replica.merge(&newer);

    // Grace is longer than the time since the newer tombstone but shorter
    // than the time since the original tombstone. Without the refresh, GC
    // would collect immediately; with it, GC must keep the entry. 150ms
    // gives ample headroom against CI scheduler stalls between the merge
    // and the GC call (which has to land inside grace for the assertion
    // to be meaningful).
    let removed = replica.gc_tombstones_with_grace(Duration::from_millis(150));
    assert_eq!(
        removed, 0,
        "newer winning tombstone must refresh the GC clock",
    );
}

#[test]
fn test_epoch_max_wins_put_returns_none_for_per_point_update() {
    // A second insert on the same `rl:` key merges into the existing shard's
    // live-points frontier; there is no clean "previously displaced" value.
    // The trait contract for that case is `None`.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a".to_string();
    assert_eq!(
        replica.insert(key.clone(), encode(1, 5).to_vec()),
        None,
        "first insert on a vacant key has no previous",
    );
    assert_eq!(
        replica.insert(key.clone(), encode(1, 7).to_vec()),
        None,
        "per-point update on existing live shard has no well-defined previous",
    );
}

#[test]
fn test_epoch_max_wins_put_malformed_returns_current_live_bytes() {
    // Malformed payload is rejected; the trait says return current live bytes
    // so the caller can see what is actually live without a second `get`.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a".to_string();
    replica.insert(key.clone(), encode(1, 5).to_vec());
    let live = replica.get(&key).expect("key is live after insert");

    let result = replica.insert(key.clone(), vec![0xFF, 0xFF, 0xFF]);
    assert_eq!(
        result,
        Some(live),
        "malformed put returns current live bytes",
    );
}

#[test]
fn test_epoch_max_wins_delete_returns_prior_live_when_killing_key() {
    // When a delete kills the last live points, the trait says return the
    // prior live bytes. EpochMaxWins kills the live frontier when the
    // tombstone version dominates every live point's version.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    let key = "rl:global:node-a".to_string();
    replica.insert(key.clone(), encode(1, 5).to_vec());
    let live = replica.get(&key).expect("key is live after insert");

    // Local `remove` ticks the local clock past the insert's version, so the
    // tombstone dominates the live frontier and the entry transitions Live
    // -> Tombstone.
    let removed = replica.remove(&key);
    assert_eq!(
        removed,
        Some(live),
        "delete that kills the last live points returns prior live bytes",
    );
    assert!(replica.get(&key).is_none(), "key is no longer live");
}

#[test]
fn test_epoch_max_wins_delete_returns_none_for_never_seen_key() {
    // A delete that arrives at a vacant key did not "remove an existing
    // value"; the trait returns `None`.
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    assert_eq!(replica.remove("rl:global:node-a"), None);
}

#[test]
fn test_lww_remove_for_never_seen_key_blocks_delayed_insert() {
    // Same gap exists for LWW: a tombstone for a never-seen key must record
    // metadata so a delayed older insert cannot win by LWW comparison.
    init_test_logging();
    let replica = CrdtOrMap::new();

    let key = "worker:1";

    let mut remove_log = OperationLog::new();
    remove_log.append(Operation::remove(key.to_string(), 100, ReplicaId::new()));
    replica.merge(&remove_log);
    assert!(replica.get(key).is_none());

    let mut insert_log = OperationLog::new();
    insert_log.append(Operation::insert(
        key.to_string(),
        b"stale".to_vec(),
        50,
        ReplicaId::new(),
    ));
    replica.merge(&insert_log);

    assert!(
        replica.get(key).is_none(),
        "older insert (ts=50) must lose to never-seen-key tombstone (ts=100)",
    );
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_operation_log_json_serialization() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());
    map.remove("key1");

    let log = map.get_operation_log();

    // Serialize to bytes
    let bytes = log.to_bytes().unwrap();

    // Deserialize
    let deserialized_log = OperationLog::from_bytes(&bytes).unwrap();
    assert_eq!(log.len(), deserialized_log.len());
}

#[test]
fn test_operation_log_binary_serialization() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());
    map.remove("key1");

    let log = map.get_operation_log();

    // Serialize to binary
    let bytes = log.to_bytes().unwrap();
    assert!(!bytes.is_empty());

    // Deserialize
    let deserialized_log = OperationLog::from_bytes(&bytes).unwrap();
    assert_eq!(log.len(), deserialized_log.len());
}

#[test]
fn test_lww_apply_remote_ops_is_idempotent() {
    // LWW dedups by op-id on remote apply: a log already absorbed once must
    // not grow the local log when re-applied. Previously tested via
    // `OperationLog::merge_with_strategy`; now that merge policy lives in
    // `LwwEngine::apply_remote_ops`, exercise it through `CrdtOrMap`.
    init_test_logging();
    let source = CrdtOrMap::new();
    source.insert("key1".to_string(), b"value1".to_vec());
    source.insert("key2".to_string(), b"value2".to_vec());
    source.remove("key1");

    let log = source.get_operation_log();
    let receiver = CrdtOrMap::new();

    receiver.merge(&log);
    let after_first = receiver.get_operation_log().len();

    receiver.merge(&log);
    let after_second = receiver.get_operation_log().len();

    assert_eq!(
        after_first, after_second,
        "re-applying the same log must be a no-op for log length",
    );
}

#[test]
fn test_lww_apply_remote_ops_keeps_within_batch_duplicate_op_ids() {
    // A single batch may carry duplicate op-ids (e.g. a concatenated gossip
    // log). The batch-sized dedup filters only against the local log (via
    // `contains`), so it keeps every batch op - duplicates included - exactly
    // as the prior log-sized-set filter did. Compaction folds the duplicates
    // to one winner, and the clock advances once per applied op (matching the
    // non-idempotent `LamportClock::update`).
    init_test_logging();
    let receiver = CrdtOrMap::new();
    let replica = ReplicaId::new();

    let mut batch = OperationLog::new();
    batch.append(Operation::insert(
        "k".to_string(),
        b"v".to_vec(),
        5,
        replica,
    ));
    batch.append(Operation::insert(
        "k".to_string(),
        b"v".to_vec(),
        5,
        replica,
    ));
    receiver.merge(&batch);

    assert_eq!(receiver.get("k"), Some(b"v".to_vec()));
    assert_eq!(
        receiver.get_operation_log().len(),
        1,
        "duplicate op-ids fold to a single log entry",
    );

    // The store value and the folded log are identical whether the duplicate
    // is kept or dropped, so they cannot distinguish `contains` from `remove`.
    // The observable effect of keeping the duplicate is that the apply loop
    // calls the non-idempotent `LamportClock::update` once per copy. Compare a
    // two-copy merge against a one-copy merge: the extra `update` must make the
    // next local write land at a strictly higher timestamp. A regression back
    // to within-batch dedup collapses both paths and fails this assertion.
    let single = CrdtOrMap::new();
    let mut single_batch = OperationLog::new();
    single_batch.append(Operation::insert(
        "k".to_string(),
        b"v".to_vec(),
        5,
        replica,
    ));
    single.merge(&single_batch);

    receiver.insert("after_dup".to_string(), b"x".to_vec());
    single.insert("after_single".to_string(), b"x".to_vec());

    let local_write_ts = |map: &CrdtOrMap, key: &str| {
        map.get_operation_log()
            .operations()
            .iter()
            .find_map(|op| match op {
                Operation::Insert {
                    key: k, timestamp, ..
                } if k == key => Some(*timestamp),
                _ => None,
            })
            .expect("local write should be in the log")
    };
    assert!(
        local_write_ts(&receiver, "after_dup") > local_write_ts(&single, "after_single"),
        "two same-op-id remote copies must advance the Lamport clock one step \
         more than a single copy",
    );
}

#[test]
fn test_operation_log_snapshot_uses_merge_strategy() {
    let key = "rl:global:node-a";
    let stale_newer_timestamp = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        2,
        ReplicaId::new(),
    );
    let epoch_winner_older_timestamp =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 1, ReplicaId::new());

    let mut log = OperationLog::new();
    log.append(stale_newer_timestamp);
    log.append(epoch_winner_older_timestamp);

    log.compact_by_key(|ops| epoch_max_wins::compact_operations(ops.iter()));
    let winner = log
        .operations()
        .iter()
        .find(|op| op.key() == key)
        .expect("compaction keeps rl shard");

    let Operation::Insert { value, .. } = winner else {
        panic!("compaction should keep an insert");
    };
    assert_eq!(decode(value), Some(EpochCount { epoch: 6, count: 0 }));
    assert_eq!(log.len(), 1, "compaction folds duplicates per key");
}

#[test]
fn test_operation_log_epoch_max_wins_tombstone_selection_is_order_independent() {
    let key = "rl:global:node-a";
    let stale_lower_epoch = Operation::insert(
        key.to_string(),
        encode(5, 100).to_vec(),
        80,
        ReplicaId::new(),
    );
    let epoch_winner_older_timestamp =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 90, ReplicaId::new());
    let tombstone_after_epoch_winner = Operation::remove(key.to_string(), 95, ReplicaId::new());
    let orders = [
        [
            stale_lower_epoch.clone(),
            epoch_winner_older_timestamp.clone(),
            tombstone_after_epoch_winner.clone(),
        ],
        [
            stale_lower_epoch.clone(),
            tombstone_after_epoch_winner.clone(),
            epoch_winner_older_timestamp.clone(),
        ],
        [
            epoch_winner_older_timestamp.clone(),
            stale_lower_epoch.clone(),
            tombstone_after_epoch_winner.clone(),
        ],
        [
            epoch_winner_older_timestamp.clone(),
            tombstone_after_epoch_winner.clone(),
            stale_lower_epoch.clone(),
        ],
        [
            tombstone_after_epoch_winner.clone(),
            stale_lower_epoch.clone(),
            epoch_winner_older_timestamp.clone(),
        ],
        [
            tombstone_after_epoch_winner.clone(),
            epoch_winner_older_timestamp.clone(),
            stale_lower_epoch.clone(),
        ],
    ];

    for order in orders {
        let mut log = OperationLog::new();
        for operation in order {
            log.append(operation);
        }

        log.compact_by_key(|ops| epoch_max_wins::compact_operations(ops.iter()));
        let winner = log.operations().iter().find(|op| op.key() == key);
        let Some(Operation::Remove { timestamp, .. }) = winner else {
            panic!("tombstone should win consistently; got {winner:?}");
        };
        assert_eq!(*timestamp, 95);
    }
}

#[test]
fn test_operation_log_epoch_max_wins_post_tombstone_insert_revives_key() {
    let key = "rl:global:node-a";
    let pre_tombstone_higher_epoch =
        Operation::insert(key.to_string(), encode(7, 0).to_vec(), 90, ReplicaId::new());
    let tombstone = Operation::remove(key.to_string(), 95, ReplicaId::new());
    let post_tombstone_lower_epoch = Operation::insert(
        key.to_string(),
        encode(6, 0).to_vec(),
        100,
        ReplicaId::new(),
    );
    let orders = [
        [
            pre_tombstone_higher_epoch.clone(),
            tombstone.clone(),
            post_tombstone_lower_epoch.clone(),
        ],
        [
            pre_tombstone_higher_epoch.clone(),
            post_tombstone_lower_epoch.clone(),
            tombstone.clone(),
        ],
        [
            tombstone.clone(),
            pre_tombstone_higher_epoch.clone(),
            post_tombstone_lower_epoch.clone(),
        ],
        [
            tombstone.clone(),
            post_tombstone_lower_epoch.clone(),
            pre_tombstone_higher_epoch.clone(),
        ],
        [
            post_tombstone_lower_epoch.clone(),
            pre_tombstone_higher_epoch.clone(),
            tombstone.clone(),
        ],
        [
            post_tombstone_lower_epoch.clone(),
            tombstone.clone(),
            pre_tombstone_higher_epoch.clone(),
        ],
    ];

    for order in orders {
        let mut log = OperationLog::new();
        for operation in order {
            log.append(operation);
        }

        log.compact_by_key(|ops| epoch_max_wins::compact_operations(ops.iter()));
        let winner = log.operations().iter().find(|op| op.key() == key);
        let Some(Operation::Insert {
            value, timestamp, ..
        }) = winner
        else {
            panic!("post-tombstone insert should revive key; got {winner:?}");
        };
        assert_eq!(*timestamp, 100);
        assert_eq!(decode(value), Some(EpochCount { epoch: 6, count: 0 }));
    }
}

#[test]
fn test_operation_log_epoch_max_wins_post_tombstone_insert_wins_over_pre_tombstone_equal_epoch() {
    let key = "rl:global:node-a";
    let newer_insert = Operation::insert(
        key.to_string(),
        encode(6, 0).to_vec(),
        100,
        ReplicaId::new(),
    );
    let older_equal_insert =
        Operation::insert(key.to_string(), encode(6, 0).to_vec(), 10, ReplicaId::new());
    let tombstone_between = Operation::remove(key.to_string(), 50, ReplicaId::new());

    let mut log = OperationLog::new();
    log.append(older_equal_insert);
    log.append(tombstone_between);
    log.append(newer_insert);

    log.compact_by_key(|ops| epoch_max_wins::compact_operations(ops.iter()));
    let winner = log.operations().iter().find(|op| op.key() == key);
    let Some(Operation::Insert {
        value, timestamp, ..
    }) = winner
    else {
        panic!("newer equal-value insert should win over intermediate tombstone");
    };
    assert_eq!(*timestamp, 100);
    assert_eq!(decode(value), Some(EpochCount { epoch: 6, count: 0 }));
}

#[test]
fn test_apply_operation_log() {
    init_test_logging();
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();

    // Replica 1 executes operations
    replica1.insert("key1".to_string(), b"value1".to_vec());
    replica1.insert("key2".to_string(), b"value2".to_vec());
    replica1.remove("key1");

    // Get operation log
    let log = replica1.get_operation_log();

    // Replica 2 merges operation log
    replica2.merge(&log);

    // Verify replica 2's state matches replica 1
    assert!(!replica2.contains_key("key1"));
    assert_eq!(replica2.get("key2"), Some(b"value2".to_vec()));
}

// ============================================================================
// Complex Scenario Tests
// ============================================================================

#[test]
fn test_distributed_scenario() {
    init_test_logging();
    // Simulate distributed scenario: 3 replicas operate independently then merge
    let replica1 = CrdtOrMap::new();
    let replica2 = CrdtOrMap::new();
    let replica3 = CrdtOrMap::new();

    // Replica 1 operations
    replica1.insert("user:1".to_string(), b"Alice".to_vec());
    replica1.insert("user:2".to_string(), b"Bob".to_vec());

    // Replica 2 operations
    replica2.insert("user:3".to_string(), b"Charlie".to_vec());
    replica2.insert("user:1".to_string(), b"Alice_Updated".to_vec());

    // Replica 3 operations
    replica3.insert("user:4".to_string(), b"David".to_vec());
    // OR-Map remove only applies to observed keys, so replica3 first observes replica1 state.
    let log1 = replica1.get_operation_log();
    replica3.merge(&log1);
    replica3.remove("user:2");

    // Merge all replicas into replica 1
    let log2 = replica2.get_operation_log();
    let log3 = replica3.get_operation_log();

    // Idempotent + unordered merge
    replica1.merge(&log3);
    replica1.merge(&log2);
    replica1.merge(&log3);

    // Verify final state
    assert!(replica1.contains_key("user:1")); // Exists (updated)
    assert!(!replica1.contains_key("user:2")); // Removed
    assert!(replica1.contains_key("user:3")); // Exists
    assert!(replica1.contains_key("user:4")); // Exists

    assert_eq!(replica1.get("user:1"), Some(b"Alice_Updated".to_vec()));
    assert_eq!(replica1.get("user:3"), Some(b"Charlie".to_vec()));
    assert_eq!(replica1.get("user:4"), Some(b"David".to_vec()));
}

// ============================================================================
// Tombstone GC Grace Period Tests
// ============================================================================

#[test]
fn test_gc_tombstones_respects_grace_period() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.remove("key1");

    // GC with a long grace period — tombstone is too young, should NOT be collected.
    let removed = map.gc_tombstones_with_grace(Duration::from_secs(3600));
    assert_eq!(removed, 0, "Young tombstone should not be GC'd");

    // GC with zero grace period — tombstone should be collected immediately.
    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 1, "Expired tombstone should be GC'd");
}

#[test]
fn test_gc_tombstones_does_not_remove_live_keys() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"value1".to_vec());
    map.insert("key2".to_string(), b"value2".to_vec());

    // GC should not remove live (non-tombstoned) keys.
    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 0);
    assert_eq!(map.get("key1"), Some(b"value1".to_vec()));
    assert_eq!(map.get("key2"), Some(b"value2".to_vec()));
}

#[test]
fn test_gc_purges_collected_keys_ops_from_log() {
    // GC must reclaim the log too: leaving the collected keys' Remove ops
    // behind would let dead tombstones gossip forever and let the compacted
    // log sit permanently above the size-tracking compaction trigger.
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"v1".to_vec());
    map.insert("key2".to_string(), b"v2".to_vec());
    map.remove("key1");

    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 1);
    let log = map.get_operation_log();
    assert!(
        log.operations().iter().all(|op| op.key() != "key1"),
        "GC'd key's ops must leave the log"
    );
    assert!(
        log.operations().iter().any(|op| op.key() == "key2"),
        "live keys' ops stay"
    );
}

#[test]
fn test_epoch_max_wins_gc_purges_collected_keys_ops_from_log() {
    init_test_logging();
    let replica = CrdtOrMap::new();
    replica.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);

    replica.insert("rl:global:node-a".to_string(), encode(1, 1).to_vec());
    replica.insert("rl:global:node-b".to_string(), encode(1, 2).to_vec());
    replica.remove("rl:global:node-a");

    let removed = replica.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 1);
    let log = replica.get_operation_log();
    assert!(
        log.operations()
            .iter()
            .all(|op| op.key() != "rl:global:node-a"),
        "GC'd key's ops must leave the log"
    );
    assert!(
        log.operations()
            .iter()
            .any(|op| op.key() == "rl:global:node-b"),
        "live keys' ops stay"
    );
}

#[test]
fn test_gc_tombstones_multiple_keys() {
    init_test_logging();
    let map = CrdtOrMap::new();

    map.insert("key1".to_string(), b"v1".to_vec());
    map.insert("key2".to_string(), b"v2".to_vec());
    map.insert("key3".to_string(), b"v3".to_vec());

    map.remove("key1");
    map.remove("key3");
    // key2 stays alive.

    let removed = map.gc_tombstones_with_grace(Duration::ZERO);
    assert_eq!(removed, 2, "Two tombstoned keys should be GC'd");
    assert_eq!(map.get("key2"), Some(b"v2".to_vec()));
}

// ============================================================================
// The op-log keeper both engines share
// ============================================================================

/// Counts the WARN events whose message carries `needle` and collects the
/// `engine` field they name; a scoped subscriber for one test.
struct WarnCounter {
    needle: &'static str,
    count: Arc<AtomicUsize>,
    engines: Arc<Mutex<BTreeSet<String>>>,
}

impl<S: tracing::Subscriber> Layer<S> for WarnCounter {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        #[derive(Default)]
        struct Fields {
            message: String,
            engine: Option<String>,
        }
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.message = format!("{value:?}");
                }
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "engine" {
                    self.engine = Some(value.to_string());
                }
            }
        }
        let mut fields = Fields::default();
        event.record(&mut fields);
        if fields.message.contains(self.needle) {
            self.count.fetch_add(1, Ordering::Relaxed);
            if let Some(engine) = fields.engine {
                self.engines.lock().unwrap().insert(engine);
            }
        }
    }
}

/// Both engines keep their log through the one keeper: the compaction
/// trigger (more than twice the live keys, min 64) and the memory backstop
/// over 10,000 retained ops are the same for the LWW and the EpochMaxWins
/// namespace, and the backstop warns once per overflow.
#[test]
fn both_engines_compact_at_one_trigger_and_warn_once_past_the_backstop() {
    let warnings = Arc::new(AtomicUsize::new(0));
    let engines = Arc::new(Mutex::new(BTreeSet::new()));
    let subscriber = tracing_subscriber::registry().with(WarnCounter {
        needle: "over threshold",
        count: Arc::clone(&warnings),
        engines: Arc::clone(&engines),
    });
    tracing::subscriber::with_default(subscriber, || {
        let map = CrdtOrMap::new();
        map.register_merge_strategy("rl:".to_string(), MergeStrategy::EpochMaxWins);
        let ops_under = |prefix: &str| {
            map.get_operation_log()
                .operations()
                .iter()
                .filter(|op| op.key().starts_with(prefix))
                .count()
        };

        // 40 keys per namespace: the trigger is 80 ops. Two versions per key
        // fill each log to the trigger exactly; the 81st op compacts it to
        // one op per key.
        for round in 0..2u64 {
            for k in 0..40 {
                map.insert(format!("cfg:key{k}"), vec![round as u8]);
                map.insert(format!("rl:key{k}"), encode(round + 1, 0).to_vec());
            }
        }
        assert_eq!((ops_under("cfg:"), ops_under("rl:")), (80, 80));
        map.insert("cfg:key0".to_string(), vec![9]);
        map.insert("rl:key0".to_string(), encode(9, 0).to_vec());
        assert_eq!(
            (ops_under("cfg:"), ops_under("rl:")),
            (40, 40),
            "the op past the trigger compacts each namespace to one op per key"
        );
        assert_eq!(warnings.load(Ordering::Relaxed), 0);

        // Past the backstop: 10,001 more keys with two versions each leave
        // each log at 20,042 ops under a trigger of 20,082; the 41st of a
        // hundred further writes crosses it, the compaction leaves 10,041
        // ops, over the 10,000 backstop, so the oldest go down to 7,500 with
        // one warning per engine, and the 59 writes after it append again.
        for round in 1..=2u64 {
            for k in 0..10_001 {
                map.insert(format!("cfg:bulk{k}"), vec![round as u8]);
                map.insert(format!("rl:bulk{k}"), encode(round, 1).to_vec());
            }
        }
        assert_eq!((ops_under("cfg:"), ops_under("rl:")), (20_042, 20_042));
        for k in 0..100 {
            map.insert(format!("cfg:bulk{k}"), vec![3]);
            map.insert(format!("rl:bulk{k}"), encode(3, 1).to_vec());
        }
        assert_eq!((ops_under("cfg:"), ops_under("rl:")), (7_559, 7_559));
        assert_eq!(
            warnings.load(Ordering::Relaxed),
            2,
            "one overflow warning per engine"
        );
        assert_eq!(
            *engines.lock().unwrap(),
            BTreeSet::from(["LwwEngine".to_string(), "RateLimitEngine".to_string()]),
            "each engine names itself in its one warning"
        );
    });
}
