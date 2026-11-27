use smg_grpc_client::common_proto::{kv_cache_event, KvCacheCleared, KvCacheEvent};

use super::*;

fn batch(seq: u64, events: usize) -> KvEventBatch {
    KvEventBatch {
        sequence_number: seq,
        timestamp: 1.0,
        events: (0..events)
            .map(|id| KvCacheEvent {
                event_id: id as u64,
                data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
            })
            .collect(),
        dp_rank: None,
        snapshot: None,
        load: None,
    }
}

fn seqs(batches: &[EncodedBatch]) -> Vec<u64> {
    batches
        .iter()
        .map(|b| b.decode().expect("decodes").sequence_number)
        .collect()
}

#[test]
fn serves_cursors_inside_the_window_and_names_why_not_otherwise() {
    let mut history = History::new(10, usize::MAX);
    assert_eq!(history.after(0), Err(Window::Empty));
    assert!(!history.complete_from_start());
    for seq in 3..=7 {
        history.push(seq, EncodedBatch::new(&batch(seq, 1)));
    }
    assert_eq!((history.oldest(), history.newest()), (Some(3), Some(7)));
    assert_eq!(seqs(&history.after(2).unwrap()), vec![3, 4, 5, 6, 7]);
    assert_eq!(seqs(&history.after(5).unwrap()), vec![6, 7]);
    assert_eq!(
        history.after(6).unwrap()[0].decode(),
        Some(batch(7, 1)),
        "a replayed batch is the one pushed"
    );
    assert!(
        history.after(7).unwrap().is_empty(),
        "nothing after the newest"
    );
    assert_eq!(history.after(1), Err(Window::Behind { oldest: 3 }));
    assert_eq!(history.after(8), Err(Window::Ahead { newest: 7 }));
    assert!(
        !history.complete_from_start(),
        "the publisher's first batches were never seen"
    );
}

#[test]
fn evicts_by_count_and_by_bytes_keeping_the_window_contiguous() {
    let mut history = History::new(3, usize::MAX);
    for seq in 0..=5 {
        history.push(seq, EncodedBatch::new(&batch(seq, 1)));
    }
    assert_eq!(
        (history.oldest(), history.newest(), history.len()),
        (Some(3), Some(5), 3)
    );
    // Cursor 2 wants 3, the oldest kept; cursor 1 wants 2, gone.
    assert_eq!(seqs(&history.after(2).unwrap()), vec![3, 4, 5]);
    assert_eq!(history.after(1), Err(Window::Behind { oldest: 3 }));

    // Sequences 1..=5 encode to the same size (sequence 0 is elided).
    let per_batch = batch(1, 4).encoded_len() + ENTRY_OVERHEAD;
    let mut bounded = History::new(1_000, per_batch * 2 + 1);
    for seq in 1..=5 {
        bounded.push(seq, EncodedBatch::new(&batch(seq, 4)));
    }
    assert_eq!(bounded.len(), 2, "two batches fit the byte budget");
    assert_eq!(bounded.oldest(), Some(4));
    assert!(bounded.bytes() <= per_batch * 2 + 1);
}

#[test]
fn holes_keep_their_slot_and_are_skipped_on_resume() {
    let mut history = History::new(10, usize::MAX);
    history.push(0, EncodedBatch::new(&batch(0, 1)));
    history.push_lost(1);
    history.push_lost(2);
    history.push(3, EncodedBatch::new(&batch(3, 1)));
    assert_eq!(history.holes(), 2);
    assert_eq!(history.newest(), Some(3));
    assert_eq!(seqs(&history.after(0).unwrap()), vec![3]);
    assert_eq!(seqs(&history.after(1).unwrap()), vec![3]);
    assert!(
        !history.complete_from_start(),
        "a hole means a batch is missing"
    );
    // Evicting the holes restores completeness-from-start? No: the start moved.
    let mut small = History::new(2, usize::MAX);
    small.push(0, EncodedBatch::new(&batch(0, 1)));
    small.push_lost(1);
    small.push(2, EncodedBatch::new(&batch(2, 1)));
    assert_eq!(small.oldest(), Some(1));
    assert_eq!(small.holes(), 1);
    small.push(3, EncodedBatch::new(&batch(3, 1)));
    assert_eq!((small.oldest(), small.holes()), (Some(2), 0));
}

#[test]
fn a_complete_window_from_zero_is_the_publishers_whole_state() {
    let mut history = History::new(10, usize::MAX);
    history.push(0, EncodedBatch::new(&batch(0, 1)));
    history.push(1, EncodedBatch::new(&batch(1, 1)));
    assert!(history.complete_from_start());
    assert_eq!(seqs(&history.all()), vec![0, 1]);
    history.clear();
    assert!(history.is_empty());
    assert_eq!(history.after(0), Err(Window::Empty));
    history.push(0, EncodedBatch::new(&batch(0, 1)));
    assert!(
        history.complete_from_start(),
        "a new incarnation starts over"
    );
}

#[test]
fn a_run_of_lost_sequences_keeps_only_what_the_window_would() {
    let mut history = History::new(4, usize::MAX);
    history.push_lost_range(0, 9);
    assert_eq!((history.oldest(), history.newest()), (Some(6), Some(9)));
    assert_eq!((history.len(), history.holes()), (4, 4));
    history.push(10, EncodedBatch::new(&batch(10, 1)));
    // The batch evicts the oldest hole: the window is 7..=10.
    assert_eq!(history.after(5), Err(Window::Behind { oldest: 7 }));
    assert_eq!(seqs(&history.after(6).unwrap()), vec![10]);
    assert!(!history.complete_from_start());
    // A short run after batches stays contiguous with them.
    let mut short = History::new(10, usize::MAX);
    short.push(0, EncodedBatch::new(&batch(0, 1)));
    short.push_lost_range(1, 3);
    short.push(4, EncodedBatch::new(&batch(4, 1)));
    assert_eq!((history.len() + short.len(), short.holes()), (9, 3));
    assert_eq!(seqs(&short.after(0).unwrap()), vec![4]);
    short.push_lost_range(5, 4);
    assert_eq!(short.newest(), Some(4), "an empty run is nothing");
}

#[test]
fn a_non_contiguous_append_is_ignored() {
    let mut history = History::new(10, usize::MAX);
    history.push(0, EncodedBatch::new(&batch(0, 1)));
    history.push(5, EncodedBatch::new(&batch(5, 1)));
    assert_eq!(history.newest(), Some(0));
}
