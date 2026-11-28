use super::*;

/// Deterministic replica id `00…00NN` so tie-break tests are reproducible
/// (higher `n` => higher `ReplicaId`).
fn replica(n: u8) -> ReplicaId {
    ReplicaId::from_string(&format!("00000000-0000-0000-0000-0000000000{n:02x}")).unwrap()
}

fn op(key: &str, ts: u64) -> Operation {
    op_r(key, ts, 1)
}

fn op_r(key: &str, ts: u64, n: u8) -> Operation {
    Operation::insert(key.to_owned(), vec![1], ts, replica(n))
}

#[test]
fn unknown_key_is_unset_and_allowed() {
    let wm = CrdtWatermark::new();
    assert_eq!(wm.get("worker:a"), None);
    assert!(wm.allows(&op("worker:a", 1)));
}

#[test]
fn allows_only_strictly_newer_versions() {
    let wm = CrdtWatermark::from_ops(&[op("k", 5)]);
    assert!(!wm.allows(&op("k", 4)), "older version already acked");
    assert!(!wm.allows(&op("k", 5)), "equal op-id already acked");
    assert!(wm.allows(&op("k", 6)), "newer version must be sent");
    assert!(
        wm.allows(&op("other", 1)),
        "other key tracked independently"
    );
}

#[test]
fn same_timestamp_higher_replica_is_a_newer_winner() {
    // The replica tie-break: at equal timestamps LWW orders by replica_id,
    // so a same-timestamp, higher-replica write is a newer winner and must
    // be sent — a timestamp-only watermark would wrongly drop it.
    let wm = CrdtWatermark::from_ops(&[op_r("k", 5, 1)]);
    assert!(!wm.allows(&op_r("k", 5, 1)), "same op-id already acked");
    assert!(
        wm.allows(&op_r("k", 5, 2)),
        "same timestamp, higher replica is a newer winner"
    );
    assert!(
        !wm.allows(&op_r("k", 5, 0)),
        "same timestamp, lower replica is older"
    );
}

#[test]
fn from_ops_keeps_max_op_id_per_key() {
    let wm = CrdtWatermark::from_ops(&[op("k", 3), op("k", 9), op("k", 7), op("j", 2)]);
    assert_eq!(wm.get("k"), Some((9, replica(1))));
    assert_eq!(wm.get("j"), Some((2, replica(1))));
}

#[test]
fn merge_max_takes_per_key_maximum() {
    let mut a = CrdtWatermark::from_ops(&[op("k", 5), op("j", 8)]);
    let b = CrdtWatermark::from_ops(&[op("k", 3), op("j", 12), op("new", 1)]);
    a.merge_max(&b);
    assert_eq!(
        a.get("k"),
        Some((5, replica(1))),
        "higher op-id not lowered"
    );
    assert_eq!(a.get("j"), Some((12, replica(1))), "op-id raised");
    assert_eq!(a.get("new"), Some((1, replica(1))), "new key added");
}

#[test]
fn merge_max_is_idempotent() {
    let mut a = CrdtWatermark::from_ops(&[op("k", 5)]);
    let b = a.clone();
    a.merge_max(&b);
    a.merge_max(&b);
    assert_eq!(a.get("k"), Some((5, replica(1))));
}

#[test]
fn merge_max_owned_takes_per_key_maximum() {
    let mut a = CrdtWatermark::from_ops(&[op("k", 5)]);
    a.merge_max_owned([
        ("k".to_owned(), (3, replica(1))),
        ("j".to_owned(), (8, replica(2))),
    ]);
    assert_eq!(a.get("k"), Some((5, replica(1))), "lower op-id ignored");
    assert_eq!(a.get("j"), Some((8, replica(2))), "new key added");
}

#[test]
fn iter_yields_all_key_versions() {
    let wm = CrdtWatermark::from_ops(&[op("k", 5), op("j", 8)]);
    let mut pairs: Vec<(String, CrdtVersion)> =
        wm.iter().map(|(k, id)| (k.to_owned(), id)).collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("j".to_owned(), (8, replica(1))),
            ("k".to_owned(), (5, replica(1))),
        ]
    );
}

#[test]
fn is_empty_reflects_contents() {
    assert!(CrdtWatermark::new().is_empty());
    assert!(!CrdtWatermark::from_ops(&[op("k", 1)]).is_empty());
}

#[test]
fn from_iter_takes_max_per_key() {
    let wm: CrdtWatermark = [
        ("k".to_owned(), (3, replica(1))),
        ("k".to_owned(), (9, replica(1))),
        ("j".to_owned(), (2, replica(1))),
    ]
    .into_iter()
    .collect();
    assert_eq!(wm.get("k"), Some((9, replica(1))));
    assert_eq!(wm.get("j"), Some((2, replica(1))));
}
