//! A bounded, contiguous history of relayed KV-event batches: what a
//! subscriber's `start_sequence_number` is served from, and what a lagging
//! subscriber catches up from.
//!
//! The history covers one unbroken range of the publisher's sequence numbers.
//! A sequence the relay could not obtain (the publisher dropped it and its
//! replay did not have it) occupies its slot as a hole, so the window stays
//! contiguous and a resume past it skips it the way the live stream did.
//! Two caps bound it: a batch count (the engines keep `buffer_steps`, 10,000,
//! for their own replay) and a byte budget over the encoded batches. The
//! window keeps each batch as those bytes (one encode at the push, a decode
//! by each stream that replays it), so the relayed batch itself is shared
//! with nobody and the live channel can hand it to its subscriber as it is.

use std::collections::VecDeque;

use bytes::Bytes;
use prost::Message;
use smg_grpc_client::common_proto::KvEventBatch;
use tracing::error;

/// What a slot of the window holds.
enum Entry {
    Batch(EncodedBatch),
    /// The publisher's sequence passed with no batch to show for it.
    Lost,
}

/// The per-entry bookkeeping charged against the byte budget on top of the
/// encoded batch: the queue slot and the heap block that holds the bytes.
const ENTRY_OVERHEAD: usize = 96;

/// A relayed batch as the window keeps it: encoded, and decoded by each
/// stream that replays it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedBatch(Bytes);

impl EncodedBatch {
    /// `batch` encoded once, off any lock the caller holds.
    pub fn new(batch: &KvEventBatch) -> Self {
        Self(Bytes::from(batch.encode_to_vec()))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    /// The batch. `None` for bytes that do not decode, which this module
    /// encoded itself: a defect, logged, and the stream passes the slot over
    /// the way it passes a hole.
    pub(crate) fn decode(&self) -> Option<KvEventBatch> {
        match KvEventBatch::decode(self.0.as_ref()) {
            Ok(batch) => Some(batch),
            Err(error) => {
                error!(
                    %error,
                    bytes = self.0.len(),
                    "a KV-event batch kept by the history does not decode"
                );
                None
            }
        }
    }
}

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
    /// sequence when the history is empty), encoded. A non-contiguous append
    /// is a caller bug and is ignored.
    pub fn push(&mut self, seq: u64, encoded: EncodedBatch) {
        let bytes = encoded.len() + ENTRY_OVERHEAD;
        self.append(seq, Entry::Batch(encoded), bytes);
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
                Entry::Batch(encoded) => {
                    self.bytes = self.bytes.saturating_sub(encoded.len() + ENTRY_OVERHEAD);
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
    pub(crate) fn after(&self, cursor: u64) -> Result<Vec<EncodedBatch>, Window> {
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
    pub(crate) fn all(&self) -> Vec<EncodedBatch> {
        self.batches(0)
    }

    fn batches(&self, skip: usize) -> Vec<EncodedBatch> {
        self.entries
            .iter()
            .skip(skip)
            .filter_map(|entry| match entry {
                Entry::Batch(encoded) => Some(encoded.clone()),
                Entry::Lost => None,
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../tests/unit/kv_history/tests.rs"]
mod tests;
