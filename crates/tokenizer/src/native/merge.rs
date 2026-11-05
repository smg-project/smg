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
//!
//! A tiktoken vocabulary lists its tokens by rank and no merges: its encoder
//! merges, at every step, the adjacent pair whose concatenation is the token
//! of least rank, so every split of a token into two tokens is a merge of
//! the token's rank, and the rank is the id. [`MergeTable::from_ranks`]
//! reads those merges off the ranks; the loops below then apply the same
//! rule (a pair's rank is its concatenation's rank, the least rank first,
//! the leftmost of equal ranks: equal ranks are one token, formed at two
//! positions by the same or by different splits).

use std::{borrow::Cow, cmp::Reverse, collections::BinaryHeap};

use rustc_hash::FxHashMap;
use serde::Deserialize;
use tokenizers::{models::ModelWrapper, Model};

use crate::encode_path::Declined;

mod serialized;

/// Pairs whose two ids are both below this bound are read from the dense
/// grid (1 MiB of ranks): the byte tokens and the earliest merges, which a
/// piece's first round of candidates is made of.
const GRID_SIDE: usize = 512;
/// Pieces of up to this many symbols merge on stack arrays, a [`ShortMerge`]:
/// a run of forty CJK ideographs, a word of a script without word spaces, a
/// long identifier. A piece of up to [`SHORT_PIECE_SMALL`] symbols takes
/// arrays of that size, so a word's set-up is a few stores.
const SHORT_PIECE_SYMBOLS: usize = 128;
/// The smaller arrays of a short merge.
const SHORT_PIECE_SMALL: usize = 32;
/// Bits of a position in a short merge's key, `rank << SHORT_KEY_BITS |
/// position`: an unsigned minimum over the keys is the least rank and,
/// among its repeats, the leftmost position.
const SHORT_KEY_BITS: u32 = SHORT_PIECE_SYMBOLS.trailing_zeros();
/// The position bits of a short merge's key.
const SHORT_KEY_MASK: u32 = (1 << SHORT_KEY_BITS) - 1;
/// The key of a live symbol whose pair is no merge, past every packed key.
const NO_KEY: u32 = u32::MAX;
/// Ranks a short merge's key carries; a model with more merges declines
/// the table (a vocabulary has some hundred thousand).
const MAX_RANKS: usize = 1 << (u32::BITS - SHORT_KEY_BITS);
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
        if merges.len() >= MAX_RANKS {
            return Err(Declined::from(
                "more merges than a short merge's keys carry",
            ));
        }
        let mut table = Self::with_capacity(merges.len(), merges.len(), bpe.ignore_merges);
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

    /// The table of a tiktoken vocabulary's merges (see the module
    /// documentation), or why the file's own encoder stays in place: a byte
    /// without a token, a token that is no two tokens' concatenation, or a
    /// token its own bytes do not merge back into (the file's encoder looks
    /// a whole piece up before it merges; the table does not, so every token
    /// must be the merge of its bytes for the two to agree on every piece).
    pub(super) fn from_ranks(ranks: &FxHashMap<Vec<u8>, u32>) -> Result<Self, Declined> {
        let mut byte_ids = [0u32; 256];
        for (byte, id) in byte_ids.iter_mut().enumerate() {
            *id = ranks
                .get([byte as u8].as_slice())
                .copied()
                .ok_or_else(|| Declined::from(format!("byte {byte:#04x} has no token")))?;
        }
        let ranks_end = ranks.values().max().map_or(0, |&rank| rank as usize + 1);
        if ranks_end >= MAX_RANKS {
            return Err(Declined::from("more ranks than a short merge's keys carry"));
        }
        // Every split of every token into two tokens, at the token's rank.
        let mut splits = Vec::with_capacity(ranks.len() * 2);
        for (token, &rank) in ranks {
            if token.len() < 2 {
                continue;
            }
            let before = splits.len();
            for cut in 1..token.len() {
                if let (Some(&left), Some(&right)) =
                    (ranks.get(&token[..cut]), ranks.get(&token[cut..]))
                {
                    splits.push((left, right, rank));
                }
            }
            if splits.len() == before {
                return Err(Declined::from(format!(
                    "token {rank} is no two tokens' concatenation"
                )));
            }
        }
        let mut table = Self::with_capacity(splits.len(), ranks_end, false);
        table.byte_ids = byte_ids;
        for &(left, right, rank) in &splits {
            table.new_ids[rank as usize] = rank;
            table.insert(left, right, rank);
        }
        let mut ids = Vec::new();
        for (token, &rank) in ranks {
            if token.len() < 2 {
                continue;
            }
            ids.clear();
            table.merge(token, &mut ids);
            if ids != [rank] {
                return Err(Declined::from(format!(
                    "token {rank} is not the merge of its bytes"
                )));
            }
        }
        Ok(table)
    }

    /// An empty table sized for `pairs` merges over `ranks` ranks.
    fn with_capacity(pairs: usize, ranks: usize, ignore_merges: bool) -> Self {
        // The open-addressed part at a load of 0.6 at most, two slots at least
        // (so that a probe always meets an empty slot).
        let slots = (pairs * 5 / 3 + 1).next_power_of_two().max(2);
        let shift = u64::BITS - slots.trailing_zeros();
        Self {
            byte_ids: [0; 256],
            grid: vec![NO_RANK; GRID_SIDE * GRID_SIDE].into_boxed_slice(),
            keys: vec![EMPTY_KEY; slots].into_boxed_slice(),
            ranks: vec![NO_RANK; slots].into_boxed_slice(),
            shift,
            new_ids: vec![0; ranks].into_boxed_slice(),
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

    /// Ranks the table holds (one per merge of a model; one per token of a
    /// tiktoken vocabulary, whose splits share their token's rank).
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
    ///
    /// Never inlined: with its one call site the merge was inlined into the
    /// piece lookup, whose frame then carried the arrays of the short loop
    /// and the heap of the long one on every piece, hits included (4 to 6
    /// instructions more per token on warm text, at identical work). As a
    /// call, a hit pays nothing for the merge and a miss pays the call.
    #[inline(never)]
    pub(super) fn merge(&self, piece: &[u8], ids: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let symbols = piece.iter().map(|&byte| self.byte_ids[usize::from(byte)]);
        if piece.len() <= SHORT_PIECE_SYMBOLS {
            self.merge_short(symbols, piece.len(), ids);
        } else {
            self.merge_long(symbols, ids);
        }
    }

    /// The merge of a short piece, `n` initial symbols (`1..=SHORT_PIECE_SYMBOLS`),
    /// as a [`ShortMerge`] on stack arrays, the smaller ones for a piece that
    /// fits them.
    fn merge_short(&self, symbols: impl Iterator<Item = u32>, n: usize, ids: &mut Vec<u32>) {
        if n <= SHORT_PIECE_SMALL {
            ShortMerge::<SHORT_PIECE_SMALL>::of(symbols, n).run(self, ids);
        } else {
            ShortMerge::<SHORT_PIECE_SYMBOLS>::of(symbols, n).run(self, ids);
        }
    }

    /// The merge of a long piece, the model's own loop: a linked list of
    /// symbols and a min-heap of `(rank, position)` candidates, an entry
    /// taken only when the pair at its position is still the pair it was
    /// pushed for (the rank names the pair; the model compares the token
    /// the pair produces, which comes to the same, see the module docs).
    fn merge_long(&self, initial: impl ExactSizeIterator<Item = u32>, ids: &mut Vec<u32>) {
        let last = initial.len() - 1;
        let mut symbols: Vec<Symbol> = initial
            .enumerate()
            .map(|(i, id)| Symbol {
                id,
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

/// The merge of a short piece: the live symbols as a linked list over arrays
/// of `N` (a multiple of eight, at most [`SHORT_PIECE_SYMBOLS`]), and for
/// each live symbol the key of the pair it starts with its right neighbour,
/// its rank over its position ([`NO_KEY`] for no merge), so that the least
/// key names the next merge: the least rank and, among its repeats, the
/// leftmost position (the rule of the module documentation). A merge unlinks
/// the right symbol, writes the product, looks up the two pairs the product
/// now forms, and takes the minimum over the other keys while those lookups
/// are in flight (their latency is the merge's critical path otherwise); the
/// minimum runs eight lanes wide, vector instructions without a branch per
/// key. This replaces a scan over the ranks of arrays of sixteen that moved
/// the symbols past each merge down by one: a run of six CJK ideographs or
/// a word of a script without word spaces took the heap path, with its two
/// allocations.
struct ShortMerge<const N: usize> {
    symbols: [u32; N],
    /// The id the pair each live symbol starts produces, when it is a merge.
    products: [u32; N],
    keys: [u32; N],
    /// The live neighbours by position: `next` of the last symbol is `n`,
    /// `prev` of the first is 255, both past every position.
    next: [u8; N],
    prev: [u8; N],
    /// Initial symbols.
    n: usize,
}

impl<const N: usize> ShortMerge<N> {
    /// The list of the `n` initial `symbols` (`1..=N`), no pair looked up.
    #[inline(always)]
    fn of(symbols: impl Iterator<Item = u32>, n: usize) -> Self {
        const { assert!(N.is_multiple_of(8) && N <= SHORT_PIECE_SYMBOLS) };
        debug_assert!(0 < n && n <= N);
        let mut merge = Self {
            symbols: [0; N],
            products: [0; N],
            keys: [NO_KEY; N],
            next: [0; N],
            prev: [0; N],
            n,
        };
        for (at, id) in symbols.take(N).enumerate() {
            merge.symbols[at] = id;
            merge.next[at] = (at + 1) as u8;
            merge.prev[at] = (at as u8).wrapping_sub(1);
        }
        merge
    }

    /// Merges until no live pair is a merge, and appends the ids to `ids`.
    #[inline(always)]
    fn run(mut self, table: &MergeTable, ids: &mut Vec<u32>) {
        let pairs = self.n.saturating_sub(1);
        for at in 0..pairs {
            let rank = table.rank_of(self.symbols[at], self.symbols[at + 1]);
            self.set_pair(table, at, rank);
        }
        let mut best = self.least(pairs);
        while best != NO_KEY {
            best = self.merge_at(table, (best & SHORT_KEY_MASK) as usize);
        }
        let mut at = 0;
        while at < self.n {
            ids.push(self.symbols[at]);
            at = self.next[at] as usize;
        }
    }

    /// Merges the pair the symbol at `at` starts, and answers the least key
    /// after it.
    #[inline(always)]
    fn merge_at(&mut self, table: &MergeTable, at: usize) -> u32 {
        self.symbols[at] = self.products[at];
        let dead = self.next[at] as usize;
        let right = self.next[dead] as usize;
        self.next[at] = right as u8;
        let left = self.prev[at] as usize;
        self.keys[dead] = NO_KEY;
        self.keys[at] = NO_KEY;
        // The two pairs the product forms, looked up before the minimum
        // over the other keys, so that the loads overlap it.
        let ahead = if right < self.n {
            self.prev[right] = at as u8;
            table.rank_of(self.symbols[at], self.symbols[right])
        } else {
            NO_RANK
        };
        let behind = if left < self.n {
            self.keys[left] = NO_KEY;
            table.rank_of(self.symbols[left], self.symbols[at])
        } else {
            NO_RANK
        };
        let others = self.least(self.n - 1);
        let key_at = self.set_pair(table, at, ahead);
        let key_left = if left < self.n {
            self.set_pair(table, left, behind)
        } else {
            NO_KEY
        };
        others.min(key_at).min(key_left)
    }

    /// Records the pair the symbol at `at` starts as a merge of `rank` (its
    /// key and its product; no key for [`NO_RANK`]) and answers the key.
    #[inline(always)]
    fn set_pair(&mut self, table: &MergeTable, at: usize, rank: u32) -> u32 {
        let key = if rank == NO_RANK {
            NO_KEY
        } else {
            self.products[at] = table.new_ids[rank as usize];
            rank << SHORT_KEY_BITS | at as u32
        };
        self.keys[at] = key;
        key
    }

    /// The least of the first `len` keys, eight lanes at a time (the keys
    /// past the live pairs are [`NO_KEY`], so the lanes may run on to the
    /// next eight).
    #[inline(always)]
    fn least(&self, len: usize) -> u32 {
        let lim = (len + 7) & !7;
        let mut lanes = [NO_KEY; 8];
        for chunk in self.keys[..lim].as_chunks::<8>().0 {
            for (lane, &key) in lanes.iter_mut().zip(chunk) {
                *lane = (*lane).min(key);
            }
        }
        lanes.into_iter().min().unwrap_or(NO_KEY)
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
