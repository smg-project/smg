use bytes::Bytes;

use super::*;

fn round_batch_with(drain: Vec<(&str, &[u8])>, targeted: Vec<(&str, &str, &[u8])>) -> RoundBatch {
    RoundBatch {
        drain_entries: drain
            .into_iter()
            .map(|(k, v)| (k.to_string(), Bytes::copy_from_slice(v)))
            .collect(),
        targeted_entries: targeted
            .into_iter()
            .map(|(t, k, v)| (t.to_string(), k.to_string(), Bytes::copy_from_slice(v)))
            .collect(),
        crdt_ops: Default::default(),
    }
}

#[test]
fn empty_round_batch_emits_nothing() {
    let rb = round_batch_with(vec![], vec![]);
    assert!(build_peer_stream_batches(&rb, "peer1").is_empty());
}

#[test]
fn drain_only_is_emitted_to_every_peer() {
    let rb = round_batch_with(vec![("td:abc", b"hello")], vec![]);
    let a = build_peer_stream_batches(&rb, "peer1");
    let b = build_peer_stream_batches(&rb, "peer2");
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(a[0].entries.len(), 1);
    assert_eq!(a[0].entries[0].key, "td:abc");
}

#[test]
fn targeted_entries_filter_by_peer() {
    let rb = round_batch_with(
        vec![],
        vec![
            ("peer1", "tree:req:a", b"req-a".as_slice()),
            ("peer2", "tree:req:b", b"req-b".as_slice()),
        ],
    );
    let a = build_peer_stream_batches(&rb, "peer1");
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].entries.len(), 1);
    assert_eq!(a[0].entries[0].key, "tree:req:a");

    let none = build_peer_stream_batches(&rb, "");
    assert!(none.is_empty());
}

#[test]
fn drain_emitted_even_when_peer_unknown() {
    let rb = round_batch_with(
        vec![("td:abc", b"hello".as_slice())],
        vec![("peer1", "tree:req:a", b"req-a".as_slice())],
    );
    // Empty peer_id -> targeted entries excluded, drain still emitted.
    let batches = build_peer_stream_batches(&rb, "");
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].entries.len(), 1);
    assert_eq!(batches[0].entries[0].key, "td:abc");
}

#[test]
fn empty_peer_id_skips_target_with_empty_string() {
    // Defensive: a targeted entry whose `target` is the empty
    // string MUST NOT match the "peer unknown" sentinel
    // (`peer_id = ""`). Without the explicit `is_empty` guard the
    // `target == peer_id` comparison would let it through.
    let rb = round_batch_with(vec![], vec![("", "tree:req:bad", b"oops".as_slice())]);
    let batches = build_peer_stream_batches(&rb, "");
    assert!(batches.is_empty(), "empty peer_id must skip all targeted");
}

#[test]
fn wrap_stream_batch_envelope_shape() {
    let batch = StreamBatch::default();
    let msg = wrap_stream_batch(batch, 42, "node-1");
    assert_eq!(msg.message_type, StreamMessageType::StreamBatch as i32);
    assert_eq!(msg.sequence, 42);
    assert_eq!(msg.peer_id, "node-1");
    assert!(matches!(msg.payload, Some(StreamPayload::StreamBatch(_))));
}

#[test]
fn build_heartbeat_envelope_shape() {
    let msg = build_heartbeat(7, "node-2");
    assert_eq!(msg.message_type, StreamMessageType::Heartbeat as i32);
    assert_eq!(msg.sequence, 7);
    assert_eq!(msg.peer_id, "node-2");
    assert!(msg.payload.is_none());
}
