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
