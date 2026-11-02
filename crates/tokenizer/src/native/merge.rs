//! The merges of a byte-level BPE vocabulary as a pair-rank table, and the
//! merge of one piece over it: what the native encode path runs on a
//! piece-cache miss in place of the model's own `tokenize`.
//!
//! The model merges a piece through a `Word` of `Symbol`s and a heap of
//! candidate merges, and hands back one `Token` with an owned `String` per
//! id: three allocations per token on every miss, and a miss is every piece
//! the thread has not seen (code identifiers, numbers, a script without
//! word spaces, the first requests of every thread). This table holds the
//! same merges, read from the model's serialized form (its merge map is
//! private; its serialization writes the merges in rank order, and only that
//! field is taken, see `serialized`): a dense grid
//! for the pairs of small ids, which carry most of a piece's first round of
//! candidates, an open-addressed table for the rest, and the id each merge
//! produces by rank. A piece of up to [`SHORT_PIECE_SYMBOLS`] symbols merges
//! on stack arrays without an allocation; a longer one on a linked list with
//! a heap of candidates, as the model does, minus the strings.
//!
//! The ids are the model's own. The model's loop pops its heap by rank, then
//! by position, and merges the popped pair unless its entry has expired:
//! the left symbol was merged away, is last, or the pair now at that
//! position produces another token. Every adjacent pair that is a merge has
//! one live entry (pushed when it formed: at the start, or by the merge that
//! formed it; a position never holds one pair twice, as the text a position
//! stands for only grows), and the token test is a currency test: the pair
//! at a live position only ever changes to one whose concatenation is
//! longer, hence another token. So the model merges the current adjacent
//! pair of least rank, the leftmost among repeats of one pair (ranks are
//! one per pair, so equal ranks are one pair at two positions), until no
//! adjacent pair is a merge: the rule both loops below apply directly. A
//! model whose symbols are not the byte tokens (a prefix or suffix on
//! subwords, a byte without a token) or whose merges are random (dropout)
//! declines at load and keeps the model's merge for its misses.

use std::{borrow::Cow, cmp::Reverse, collections::BinaryHeap};

use serde::Deserialize;
use tokenizers::{models::ModelWrapper, Model};

use crate::encode_path::Declined;

mod serialized;

/// Pairs whose two ids are both below this bound are read from the dense
/// grid (1 MiB of ranks): the byte tokens and the earliest merges, which a
/// piece's first round of candidates is made of.
const GRID_SIDE: usize = 512;
/// Pieces of up to this many symbols (bytes) merge on stack arrays.
const SHORT_PIECE_SYMBOLS: usize = 16;
/// The rank of a pair that is no merge.
const NO_RANK: u32 = u32::MAX;
/// An empty slot of the open-addressed table; no pair of ids has this key.
const EMPTY_KEY: u64 = u64::MAX;
/// No neighbour, in the linked list of a long piece.
const NONE: u32 = u32::MAX;

/// The merges of one byte-level BPE model; see the module documentation.
pub(super) struct MergeTable {
    /// The id of each byte's token.
    byte_ids: [u32; 256],
    /// `grid[a * GRID_SIDE + b]`: the rank of the pair `(a, b)`, or
    /// [`NO_RANK`], for ids below [`GRID_SIDE`].
    grid: Box<[u32]>,
    /// The other pairs, open-addressed with linear probing: `keys[slot]` is
    /// `a << 32 | b` or [`EMPTY_KEY`], `ranks[slot]` the pair's rank.
    keys: Box<[u64]>,
    ranks: Box<[u32]>,
    /// Bits a key's hash is shifted right by to give its first slot; the
    /// table holds `1 << (64 - shift)` slots.
    shift: u32,
    /// The id the merge of each rank produces.
    new_ids: Box<[u32]>,
    /// The model's `ignore_merges`: a piece that is a token is that token,
    /// looked up before any merge (by the caller, in the model's vocabulary).
    pub(super) ignore_merges: bool,
}

/// The model's `merges` field read back: `[[a, b], ...]` in rank order.
#[derive(Deserialize)]
#[serde(transparent)]
struct SerializedMerges<'a>(#[serde(borrow)] Vec<(Cow<'a, str>, Cow<'a, str>)>);

impl MergeTable {
    /// The table of `model`'s merges, or why the model's own merge stays in
    /// place for the path's misses.
    pub(super) fn from_model(model: &ModelWrapper) -> Result<Self, Declined> {
        let ModelWrapper::BPE(bpe) = model else {
            return Err(Declined::from("the model is not BPE"));
        };
        if bpe.dropout.is_some_and(|dropout| dropout > 0.0) {
            return Err(Declined::from("the model merges with dropout"));
        }
        // An empty prefix or suffix leaves every symbol as it is, as none does.
        let marks = |mark: &Option<String>| mark.as_deref().is_some_and(|mark| !mark.is_empty());
        if marks(&bpe.continuing_subword_prefix) || marks(&bpe.end_of_word_suffix) {
            return Err(Declined::from(
                "the model marks subwords with a prefix or a suffix",
            ));
        }
        let mut byte_ids = [0u32; 256];
        let mut utf8 = [0u8; 4];
        for (byte, alphabet) in super::bytes_char().into_iter().enumerate() {
            byte_ids[byte] = model
                .token_to_id(alphabet.encode_utf8(&mut utf8))
                .ok_or_else(|| Declined::from(format!("byte {byte:#04x} has no token")))?;
        }
        // The merges alone, through the model's `Serialize` (see `serialized`):
        // its vocabulary is never serialized, so a vocabulary with gaps in its
        // ids does not have the library warn about them at every load.
        let serialized = serialized::merges_json(bpe)
            .map_err(|e| Declined::from(format!("the model does not serialize: {e}")))?;
        let SerializedMerges(merges) = serde_json::from_str(&serialized)
            .map_err(|e| Declined::from(format!("the model's merges do not read back: {e}")))?;
        let mut table = Self::with_capacity(merges.len(), bpe.ignore_merges);
        table.byte_ids = byte_ids;
        let mut token = String::new();
        for (rank, (left, right)) in merges.iter().enumerate() {
            let rank = u32::try_from(rank)
                .ok()
                .filter(|&rank| rank != NO_RANK)
                .ok_or_else(|| Declined::from("more merges than the table ranks"))?;
            token.clear();
            token.push_str(left);
            token.push_str(right);
            let (Some(a), Some(b), Some(new_id)) = (
                model.token_to_id(left),
                model.token_to_id(right),
                model.token_to_id(&token),
            ) else {
                return Err(Declined::from(format!(
                    "merge {left:?} + {right:?} names a token outside the vocabulary"
                )));
            };
            table.new_ids[rank as usize] = new_id;
            table.insert(a, b, rank);
        }
        Ok(table)
    }

    /// An empty table sized for `merges` pairs.
    fn with_capacity(merges: usize, ignore_merges: bool) -> Self {
        // The open-addressed part at a load of 0.6 at most, two slots at least
        // (so that a probe always meets an empty slot).
        let slots = (merges * 5 / 3 + 1).next_power_of_two().max(2);
        let shift = u64::BITS - slots.trailing_zeros();
        Self {
            byte_ids: [0; 256],
            grid: vec![NO_RANK; GRID_SIDE * GRID_SIDE].into_boxed_slice(),
            keys: vec![EMPTY_KEY; slots].into_boxed_slice(),
            ranks: vec![NO_RANK; slots].into_boxed_slice(),
            shift,
            new_ids: vec![0; merges].into_boxed_slice(),
            ignore_merges,
        }
    }

    /// Bytes the table holds, for the load log.
    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + size_of_val(&*self.grid)
            + size_of_val(&*self.keys)
            + size_of_val(&*self.ranks)
            + size_of_val(&*self.new_ids)
    }

    /// Merges the table holds.
    pub(super) fn merges(&self) -> usize {
        self.new_ids.len()
    }

    #[inline]
    fn first_slot(&self, key: u64) -> usize {
        // Fibonacci hashing: the pair's two ids spread over the high bits.
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// Records `(a, b)` as the merge of `rank`; a pair recorded twice keeps
    /// the later rank, as the model's merge map does.
    fn insert(&mut self, a: u32, b: u32, rank: u32) {
        if (a as usize) < GRID_SIDE && (b as usize) < GRID_SIDE {
            self.grid[a as usize * GRID_SIDE + b as usize] = rank;
            return;
        }
        let key = u64::from(a) << 32 | u64::from(b);
        let mask = self.keys.len() - 1;
        let mut slot = self.first_slot(key);
        while self.keys[slot] != EMPTY_KEY && self.keys[slot] != key {
            slot = (slot + 1) & mask;
        }
        self.keys[slot] = key;
        self.ranks[slot] = rank;
    }

    /// The rank of the pair `(a, b)`, [`NO_RANK`] when it is no merge.
    #[inline]
    fn rank_of(&self, a: u32, b: u32) -> u32 {
        if (a as usize) < GRID_SIDE && (b as usize) < GRID_SIDE {
            return self.grid[a as usize * GRID_SIDE + b as usize];
        }
        let key = u64::from(a) << 32 | u64::from(b);
        let mask = self.keys.len() - 1;
        let mut slot = self.first_slot(key);
        loop {
            let found = self.keys[slot];
            if found == key {
                return self.ranks[slot];
            }
            if found == EMPTY_KEY {
                return NO_RANK;
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Appends the ids `piece` merges to under the table: the model's
    /// `tokenize` of the piece's byte-level alphabet form, minus its strings
    /// (and minus the whole-piece lookup of `ignore_merges`, the caller's).
    pub(super) fn merge(&self, piece: &[u8], ids: &mut Vec<u32>) {
        if piece.len() <= SHORT_PIECE_SYMBOLS {
            self.merge_short(piece, ids);
        } else {
            self.merge_long(piece, ids);
        }
    }

    /// The merge of a short piece: the symbols and the ranks of the adjacent
    /// pairs on stack arrays, each round a scan for the least rank (the
    /// leftmost of equal ranks, which can only be the same pair twice), the
    /// two pairs the merged symbol forms recomputed.
    fn merge_short(&self, piece: &[u8], ids: &mut Vec<u32>) {
        let mut symbols = [0u32; SHORT_PIECE_SYMBOLS];
        let mut ranks = [NO_RANK; SHORT_PIECE_SYMBOLS - 1];
        let mut len = piece.len();
        for (symbol, &byte) in symbols.iter_mut().zip(piece) {
            *symbol = self.byte_ids[usize::from(byte)];
        }
        for (rank, pair) in ranks.iter_mut().zip(symbols[..len].windows(2)) {
            *rank = self.rank_of(pair[0], pair[1]);
        }
        loop {
            let (mut best, mut at) = (NO_RANK, 0);
            for (i, &rank) in ranks[..len.saturating_sub(1)].iter().enumerate() {
                if rank < best {
                    (best, at) = (rank, i);
                }
            }
            if best == NO_RANK {
                break;
            }
            symbols[at] = self.new_ids[best as usize];
            if at + 2 < len {
                // The symbols past the merged pair, and the pairs among them.
                symbols.copy_within(at + 2..len, at + 1);
                ranks.copy_within(at + 2..len - 1, at + 1);
            }
            len -= 1;
            if at > 0 {
                ranks[at - 1] = self.rank_of(symbols[at - 1], symbols[at]);
            }
            if at + 1 < len {
                ranks[at] = self.rank_of(symbols[at], symbols[at + 1]);
            }
        }
        ids.extend_from_slice(&symbols[..len]);
    }

    /// The merge of a long piece, the model's own loop: a linked list of
    /// symbols and a min-heap of `(rank, position)` candidates, an entry
    /// taken only when the pair at its position is still the pair it was
    /// pushed for (the rank names the pair; the model compares the token
    /// the pair produces, which comes to the same, see the module docs).
    fn merge_long(&self, piece: &[u8], ids: &mut Vec<u32>) {
        let last = piece.len() - 1;
        let mut symbols: Vec<Symbol> = piece
            .iter()
            .enumerate()
            .map(|(i, &byte)| Symbol {
                id: self.byte_ids[usize::from(byte)],
                prev: if i == 0 { NONE } else { i as u32 - 1 },
                next: if i == last { NONE } else { i as u32 + 1 },
            })
            .collect();
        let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = (0..last)
            .filter_map(|i| {
                let rank = self.rank_of(symbols[i].id, symbols[i + 1].id);
                (rank != NO_RANK).then_some(Reverse((rank, i as u32)))
            })
            .collect();
        while let Some(Reverse((rank, pos))) = candidates.pop() {
            let at = pos as usize;
            let Symbol { id, prev, next } = symbols[at];
            if id == NONE || next == NONE {
                continue;
            }
            let right = symbols[next as usize];
            if self.rank_of(id, right.id) != rank {
                continue;
            }
            symbols[at].id = self.new_ids[rank as usize];
            symbols[at].next = right.next;
            symbols[next as usize].id = NONE;
            if right.next != NONE {
                symbols[right.next as usize].prev = pos;
                let ahead = self.rank_of(symbols[at].id, symbols[right.next as usize].id);
                if ahead != NO_RANK {
                    candidates.push(Reverse((ahead, pos)));
                }
            }
            if prev != NONE {
                let behind = self.rank_of(symbols[prev as usize].id, symbols[at].id);
                if behind != NO_RANK {
                    candidates.push(Reverse((behind, prev)));
                }
            }
        }
        ids.extend(
            symbols
                .iter()
                .map(|symbol| symbol.id)
                .filter(|&id| id != NONE),
        );
    }
}

/// One symbol of a long piece's merge: its token id ([`NONE`] once merged
/// into its left neighbour: no vocabulary reaches that id) and its live
/// neighbours.
#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: u32,
    next: u32,
}

#[cfg(test)]
mod tests;
