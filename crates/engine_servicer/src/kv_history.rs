//! A bounded, contiguous history of relayed KV-event batches: what a
//! subscriber's `start_sequence_number` is served from, and what a lagging
//! subscriber catches up from.
//!
//! The history covers one unbroken range of the publisher's sequence numbers.
//! A sequence the relay could not obtain (the publisher dropped it and its
//! replay did not have it) occupies its slot as a hole, so the window stays
//! contiguous and a resume past it skips it the way the live stream did.
//! Two caps bound it: a batch count (the engines keep `buffer_steps`, 10,000,
//! for their own replay) and a byte budget over the encoded batches.

use std::{collections::VecDeque, sync::Arc};

use prost::Message;
use smg_grpc_client::common_proto::KvEventBatch;

/// What a slot of the window holds.
enum Entry {
    Batch(Arc<KvEventBatch>),
    /// The publisher's sequence passed with no batch to show for it.
    Lost,
}

/// The per-entry bookkeeping charged against the byte budget on top of the
/// encoded batch: the queue slot, the `Arc`, the proto's own allocations.
const ENTRY_OVERHEAD: usize = 96;

/// Why a cursor cannot be served from the history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Window {
    /// Nothing relayed yet.
    Empty,
    /// The batch after the cursor is older than the oldest one kept.
    Behind { oldest: u64 },
    /// The cursor is past the newest sequence relayed: it belongs to another
    /// publisher incarnation.
    Ahead { newest: u64 },
}

pub struct History {
    /// The sequence number of `entries[0]`.
    first: u64,
    entries: VecDeque<Entry>,
    bytes: usize,
    holes: usize,
    max_batches: usize,
    max_bytes: usize,
}

impl History {
    pub fn new(max_batches: usize, max_bytes: usize) -> Self {
        Self {
            first: 0,
            entries: VecDeque::new(),
            bytes: 0,
            holes: 0,
            max_batches: max_batches.max(1),
            max_bytes,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Slots occupied by holes.
    pub(crate) fn holes(&self) -> usize {
        self.holes
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn oldest(&self) -> Option<u64> {
        (!self.is_empty()).then_some(self.first)
    }

    pub(crate) fn newest(&self) -> Option<u64> {
        self.oldest()
            .map(|first| first + self.entries.len() as u64 - 1)
    }

    /// The window holds every batch the publisher ever numbered: it starts at
    /// 0 and has no holes. A subscriber without a cursor can take it as the
    /// publisher's whole state.
    pub(crate) fn complete_from_start(&self) -> bool {
        !self.entries.is_empty() && self.first == 0 && self.holes == 0
    }

    /// Append the batch for `seq`, which must be the next sequence (any
    /// sequence when the history is empty). A non-contiguous append is a
    /// caller bug and is ignored.
    pub fn push(&mut self, seq: u64, batch: Arc<KvEventBatch>) {
        let bytes = batch.encoded_len() + ENTRY_OVERHEAD;
        self.append(seq, Entry::Batch(batch), bytes);
    }

    /// Record that `seq` passed without a batch.
    pub(crate) fn push_lost(&mut self, seq: u64) {
        self.append(seq, Entry::Lost, ENTRY_OVERHEAD);
    }

    /// Record that `from..=to` passed without batches (`from` the next
    /// sequence, any when the history is empty). Only the newest
    /// `max_batches` of them could stay in the window, so a longer run
    /// starts the window over at the slots that would have survived.
    pub(crate) fn push_lost_range(&mut self, from: u64, to: u64) {
        if to < from {
            return;
        }
        let mut from = from;
        if to - from + 1 > self.max_batches as u64 {
            self.clear();
            from = to + 1 - self.max_batches as u64;
        }
        for seq in from..=to {
            self.push_lost(seq);
        }
    }

    fn append(&mut self, seq: u64, entry: Entry, bytes: usize) {
        if let Some(newest) = self.newest() {
            if seq != newest + 1 {
                return;
            }
        } else {
            self.first = seq;
        }
        if matches!(entry, Entry::Lost) {
            self.holes += 1;
        }
        self.entries.push_back(entry);
        self.bytes += bytes;
        self.evict();
    }

    fn evict(&mut self) {
        while self.entries.len() > self.max_batches
            || (self.bytes > self.max_bytes && self.entries.len() > 1)
        {
            let Some(entry) = self.entries.pop_front() else {
                break;
            };
            self.first += 1;
            match entry {
                Entry::Batch(batch) => {
                    self.bytes = self
                        .bytes
                        .saturating_sub(batch.encoded_len() + ENTRY_OVERHEAD);
                }
                Entry::Lost => {
                    self.holes -= 1;
                    self.bytes = self.bytes.saturating_sub(ENTRY_OVERHEAD);
                }
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.holes = 0;
        self.first = 0;
    }

    /// The batches after `cursor`, in order, holes skipped; or why the window
    /// cannot serve that cursor. A cursor equal to the newest sequence yields
    /// nothing and is fine.
    pub(crate) fn after(&self, cursor: u64) -> Result<Vec<Arc<KvEventBatch>>, Window> {
        let (Some(oldest), Some(newest)) = (self.oldest(), self.newest()) else {
            return Err(Window::Empty);
        };
        if cursor > newest {
            return Err(Window::Ahead { newest });
        }
        let want = cursor + 1;
        if want < oldest {
            return Err(Window::Behind { oldest });
        }
        let skip = (want - oldest) as usize;
        Ok(self.batches(skip))
    }

    /// Every batch in the window, holes skipped.
    pub(crate) fn all(&self) -> Vec<Arc<KvEventBatch>> {
        self.batches(0)
    }

    fn batches(&self, skip: usize) -> Vec<Arc<KvEventBatch>> {
        self.entries
            .iter()
            .skip(skip)
            .filter_map(|entry| match entry {
                Entry::Batch(batch) => Some(Arc::clone(batch)),
                Entry::Lost => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::common_proto::{kv_cache_event, KvCacheCleared, KvCacheEvent};

    use super::*;

    fn batch(seq: u64, events: usize) -> Arc<KvEventBatch> {
        Arc::new(KvEventBatch {
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
        })
    }

    fn seqs(batches: &[Arc<KvEventBatch>]) -> Vec<u64> {
        batches.iter().map(|b| b.sequence_number).collect()
    }

    #[test]
    fn serves_cursors_inside_the_window_and_names_why_not_otherwise() {
        let mut history = History::new(10, usize::MAX);
        assert_eq!(history.after(0), Err(Window::Empty));
        assert!(!history.complete_from_start());
        for seq in 3..=7 {
            history.push(seq, batch(seq, 1));
        }
        assert_eq!((history.oldest(), history.newest()), (Some(3), Some(7)));
        assert_eq!(seqs(&history.after(2).unwrap()), vec![3, 4, 5, 6, 7]);
        assert_eq!(seqs(&history.after(5).unwrap()), vec![6, 7]);
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
            history.push(seq, batch(seq, 1));
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
            bounded.push(seq, batch(seq, 4));
        }
        assert_eq!(bounded.len(), 2, "two batches fit the byte budget");
        assert_eq!(bounded.oldest(), Some(4));
        assert!(bounded.bytes() <= per_batch * 2 + 1);
    }

    #[test]
    fn holes_keep_their_slot_and_are_skipped_on_resume() {
        let mut history = History::new(10, usize::MAX);
        history.push(0, batch(0, 1));
        history.push_lost(1);
        history.push_lost(2);
        history.push(3, batch(3, 1));
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
        small.push(0, batch(0, 1));
        small.push_lost(1);
        small.push(2, batch(2, 1));
        assert_eq!(small.oldest(), Some(1));
        assert_eq!(small.holes(), 1);
        small.push(3, batch(3, 1));
        assert_eq!((small.oldest(), small.holes()), (Some(2), 0));
    }

    #[test]
    fn a_complete_window_from_zero_is_the_publishers_whole_state() {
        let mut history = History::new(10, usize::MAX);
        history.push(0, batch(0, 1));
        history.push(1, batch(1, 1));
        assert!(history.complete_from_start());
        assert_eq!(seqs(&history.all()), vec![0, 1]);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.after(0), Err(Window::Empty));
        history.push(0, batch(0, 1));
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
        history.push(10, batch(10, 1));
        // The batch evicts the oldest hole: the window is 7..=10.
        assert_eq!(history.after(5), Err(Window::Behind { oldest: 7 }));
        assert_eq!(seqs(&history.after(6).unwrap()), vec![10]);
        assert!(!history.complete_from_start());
        // A short run after batches stays contiguous with them.
        let mut short = History::new(10, usize::MAX);
        short.push(0, batch(0, 1));
        short.push_lost_range(1, 3);
        short.push(4, batch(4, 1));
        assert_eq!((history.len() + short.len(), short.holes()), (9, 3));
        assert_eq!(seqs(&short.after(0).unwrap()), vec![4]);
        short.push_lost_range(5, 4);
        assert_eq!(short.newest(), Some(4), "an empty run is nothing");
    }

    #[test]
    fn a_non_contiguous_append_is_ignored() {
        let mut history = History::new(10, usize::MAX);
        history.push(0, batch(0, 1));
        history.push(5, batch(5, 1));
        assert_eq!(history.newest(), Some(0));
    }
}
