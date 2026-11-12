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
//! field is taken, see `serialized`): a grid of the merges among the lowest
//! ids of the vocabulary, the byte tokens and the first merges' products,
//! which the pairs of any text are mostly made of, read by a direct index
//! (see [`MergeTable::hot`]), and an open-addressed table of every other
//! pair in buckets of one cache line, each entry a pair with its rank and
//! the id the merge produces, so that the lookup of a pair reads one line
//! and answers both (see [`Entry`]). A piece of up to [`SHORT_PIECE_SYMBOLS`]
//! symbols merges on stack arrays without an allocation; a longer one on a
//! linked list with a heap of candidates, as the model does, minus the
//! strings.
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
//! subwords) or whose merges are random (dropout) declines at load and
//! keeps the model's merge for its misses; one without a token for a byte
//! keeps its table, and the pieces that hold the byte stay the model's.
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
use tokenizers::{
    models::{bpe::BPE, ModelWrapper},
    Model,
};

use crate::{
    encode_path::Declined,
    tiktoken::MAX_BYTE_LEVEL_TABLE_IDS,
    word_cache::{entries_of, ReadBpe},
};

mod preformed;
mod serialized;

use preformed::{char_len, code_point, is_continuation, Preformed, NO_CHAR};

/// Entries of one bucket of the pair table: sixteen bytes each, so a bucket
/// is one 64-byte cache line, read whole.
const BUCKET: usize = 4;
/// Pieces of up to this many symbols merge on stack arrays, a [`ShortMerge`]:
/// a run of forty CJK ideographs, a word of a script without word spaces, a
/// long identifier. A piece of up to [`SHORT_PIECE_TINY`] symbols, a word,
/// takes arrays of that size, one of up to [`SHORT_PIECE_SMALL`] the next,
/// so a word's set-up is a few stores.
const SHORT_PIECE_SYMBOLS: usize = 128;
/// The middle arrays of a short merge.
const SHORT_PIECE_SMALL: usize = 32;
/// The smallest arrays of a short merge: a word.
const SHORT_PIECE_TINY: usize = 16;
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
/// The id of a byte the vocabulary has no token for. On the byte-level path
/// a piece that holds the byte is the model's (see
/// [`MergeTable::lacks_a_token_for`]); on the character symbols' table such
/// a byte is never reached (`CharSymbols::from_model` declines one the
/// fallback could reach, and `CharSymbols::merge` asserts it in debug
/// builds).
const NO_BYTE_TOKEN: u32 = u32::MAX;
/// Filter bits per merge of the pair table (see `MergeTable::pair_filter`),
/// at least: two set per pair in one word, so at eight bits per merge about
/// one pair in twenty that is no merge passes the filter (5.3 % measured on
/// random pairs; the rounding of the word count up to a power of two gives
/// up to sixteen bits per merge and a rate down to 2 %), in 64-512 KiB for
/// the vocabularies in use (50k to 280k merges; a vocabulary built from token
/// ranks is sized by its two-token splits and can take more).
const PAIR_FILTER_BITS_PER_MERGE: usize = 8;
/// The least words of the pair filter: 8 KiB for a small vocabulary.
const PAIR_FILTER_MIN_WORDS: usize = 1024;
/// The pair filter's hash multiplier: odd, and not the buckets' own, so that
/// two pairs sharing a bucket do not share a filter word.
const PAIR_FILTER_MULTIPLIER: u64 = 0xD6E8_FEB8_6659_FD93;
/// Ids of the hot grid's window, from its base (see [`MergeTable::hot`]),
/// as a power of two: a pair is in the window when both of its ids, less
/// the base, are below this. 512 x 512 merges of eight bytes: 2 MiB per
/// loaded vocabulary.
const HOT_SIDE_BITS: u32 = 9;
const HOT_SIDE: u32 = 1 << HOT_SIDE_BITS;
/// The window base of a table without the grid: past every 32-bit id, so
/// that no pair reaches the window (the base is 64 bits for it).
const NO_HOT_BASE: u64 = 1 << 32;
/// An empty entry of the pair table; no pair of ids has this key.
const EMPTY_KEY: u64 = u64::MAX;
/// No neighbour, in the linked list of a long piece.
const NONE: u32 = u32::MAX;

/// One merge as the table stores it, in one word: the rank in the high half
/// and the id the merge produces in the low half, so that a pair's lookup
/// answers both with the one load. [`Merge::NONE`] stands for a pair that is
/// no merge; its rank reads [`NO_RANK`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Merge(u64);

impl Merge {
    const NONE: Self = Self(u64::MAX);

    fn new(rank: u32, product: u32) -> Self {
        Self(u64::from(rank) << 32 | u64::from(product))
    }

    #[inline(always)]
    fn rank(self) -> u32 {
        (self.0 >> 32) as u32
    }

    #[inline(always)]
    fn product(self) -> u32 {
        self.0 as u32
    }
}

/// One entry of the pair table: the pair's key, `a << 32 | b` or
/// [`EMPTY_KEY`], and its merge; sixteen bytes, [`BUCKET`] to a [`Bucket`].
/// Before this layout the table kept the keys, the ranks and the products in
/// three arrays and a dense grid of the small-id pairs in front of them: a
/// lookup on a cold piece read the key, then the rank, then the product by
/// rank, three lines each a memory latency apart, and the grid's branch
/// mispredicted on every round that mixed byte pairs with the pairs of
/// products (measured: the merge was a third of a cold encode, most of it
/// those reads).
#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u64,
    merge: Merge,
}

impl Entry {
    const EMPTY: Self = Self {
        key: EMPTY_KEY,
        merge: Merge::NONE,
    };
}

/// One bucket of the pair table: [`BUCKET`] entries on one cache line, the
/// alignment the type's own. A boxed slice of entries is aligned to an entry,
/// so whether a bucket shared a line was the allocator's to decide, and one
/// that hands large blocks back sixteen bytes past a line (the system
/// allocator, under the tests and the crate's benches) had every bucket span
/// two.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Bucket([Entry; BUCKET]);

const _: () = assert!(size_of::<Entry>() == 16);
const _: () = assert!(size_of::<Bucket>() == 64);

impl Bucket {
    const EMPTY: Self = Self([Entry::EMPTY; BUCKET]);
}

/// Which tokens stand for the bytes a byte-level piece starts from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ByteTokens {
    /// GPT-2's alphabet: a character per byte.
    Alphabet,
    /// Sentencepiece's `<0xHH>` tokens (a piece of such a vocabulary starts
    /// from character symbols instead, see [`MergeTable::merge_symbols`];
    /// the table carries no hot grid).
    Hex,
}

/// The merges of one byte-level BPE model; see the module documentation.
pub(super) struct MergeTable {
    /// The id of each byte's token ([`NO_BYTE_TOKEN`] for a byte without one).
    byte_ids: [u32; 256],
    /// The bytes the vocabulary has no token for, when it lacks any: on the
    /// byte-level path a piece that holds one is the model's to merge (see
    /// [`Self::lacks_a_token_for`]); the character symbols' table never
    /// reaches one (see `CharSymbols::from_model`). `None` over a complete
    /// alphabet, the usual case.
    missing: Option<Box<[bool; 256]>>,
    /// `hot[(a - hot_base) << HOT_SIDE_BITS | (b - hot_base)]`: the merge of
    /// the pair `(a, b)` when both ids are within [`HOT_SIDE`] of the window
    /// base, the least byte token's id ([`Merge::NONE`] for a pair that is
    /// no merge), the only place such a pair is recorded. The vocabularies
    /// in use number their byte tokens together (the first 256 ids, or 3 to
    /// 258, or 1000 to 1255) and the merges' products right after them in
    /// rank order, so the window holds the pairs of the most frequent
    /// operands of any text, the bytes and the products of the first merges,
    /// and a lookup there is one indexed load without the filter word, the
    /// hash and the bucket line. Counted on first-sight text, a quarter to
    /// two fifths of a merge's pair lookups fall in the window (more on
    /// short texts and on vocabularies whose first merges cover more of the
    /// text); the cold pass runs 0.8-2.6 % fewer instructions per token by
    /// cell (1.6-2.3 % by size-class median) from 256 B to 32 KB over four
    /// vocabularies for it, its cache fills unchanged, the warm pass
    /// untouched (it merges nothing). Before this grid a 256 x 256 grid by
    /// byte answered the first round's byte pairs alone, the same pairs
    /// recorded in the buckets too. Empty for a table whose pieces start
    /// from character symbols ([`ByteTokens::Hex`]): their first round reads
    /// the bucket table ([`Self::merge_symbols`]), and the grid's 2 MiB per
    /// loaded vocabulary stay unspent.
    hot: Box<[Merge]>,
    /// The first id of the hot grid's window, the least byte token's id;
    /// [`NO_HOT_BASE`] for a table without the grid.
    hot_base: u64,
    /// The pairs outside the hot grid's window, open-addressed by bucket with
    /// linear probing over whole buckets: a pair's key hashes to a bucket of
    /// [`BUCKET`] entries on one cache line, and a lookup walks it to the key
    /// or the first empty entry, going on to the next bucket only when every
    /// entry holds another pair (see [`Self::merge_of`]).
    buckets: Box<[Bucket]>,
    /// Bits a key's hash is shifted right by to give its first bucket; the
    /// table holds `1 << (64 - shift)` buckets.
    shift: u32,
    /// The pairs' filter: one 64-bit word per pair key and two bits of it,
    /// set by [`Self::insert`], [`PAIR_FILTER_BITS_PER_MERGE`] bits per
    /// merge at least (64-512 KiB for the vocabularies in use, a table the
    /// second-level cache keeps). A pair with either bit clear was never
    /// inserted, and [`Self::merge_of`] answers it from that one line; a
    /// pair with both set goes on to its bucket, which answers as before, so
    /// a false positive (about one pair in twenty at the tightest rounding)
    /// costs what every lookup cost. Measured on first-sight text: half of a
    /// merge's pair lookups are pairs that are no merge, and each read a
    /// bucket line of a 2-16 MiB table from the last-level cache or memory;
    /// with the filter the cold cells of the encode bar read 3 to 18 %
    /// faster from 2 KB texts on (1 to 10 % by class median) and the warm
    /// cells unchanged (a warm text merges nothing).
    pair_filter: Box<[u64]>,
    /// Bits a key's filter hash is shifted right by to give its filter word;
    /// the filter holds `1 << (64 - pair_filter_shift)` words.
    pair_filter_shift: u32,
    /// Merges the table holds.
    merges: usize,
    /// The characters of a multi-byte script a piece may start from as one
    /// symbol each instead of from their bytes, with the cuts between them
    /// (see `preformed`); empty for a table whose vocabulary proves none.
    /// Behind a `Box`: the table's own struct keeps the layout of its hot
    /// fields (the byte ids, the grid, the buckets, the filter).
    preformed: Box<Preformed>,
    /// The model's `ignore_merges`: a piece that is a token is that token,
    /// looked up before any merge (by the caller, in the model's vocabulary).
    pub(super) ignore_merges: bool,
}

/// The model's `merges` field read back: `[[a, b], ...]` in rank order.
#[derive(Deserialize)]
#[serde(transparent)]
struct SerializedMerges<'a>(#[serde(borrow)] Vec<(Cow<'a, str>, Cow<'a, str>)>);

impl MergeTable {
    /// The table of `model`'s merges, with `tokens` as the spelling of its
    /// byte tokens, or why the model's own merge stays in place for the
    /// path's misses.
    ///
    /// `read` is what the file's one-pass read saw of the model (its merges
    /// as ids, its largest id), so that neither is read back from the built
    /// model; `None` for a model built in memory.
    pub(super) fn from_model(
        model: &ModelWrapper,
        tokens: ByteTokens,
        read: Option<&ReadBpe>,
    ) -> Result<Self, Declined> {
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
        let mut missing: Option<Box<[bool; 256]>> = None;
        let mut utf8 = [0u8; 4];
        for (byte, alphabet) in super::bytes_char().into_iter().enumerate() {
            let spelling = match tokens {
                ByteTokens::Alphabet => Cow::Borrowed(&*alphabet.encode_utf8(&mut utf8)),
                ByteTokens::Hex => Cow::Owned(format!("<0x{byte:02X}>")),
            };
            // A vocabulary without a token for a byte (some lack one for a
            // byte no text of theirs holds): on the byte-level path the
            // pieces that hold it stay the model's, every other piece merges
            // here; the character symbols' table spells a byte only for a
            // character that is no token, and declines a missing byte that
            // could be reached (`CharSymbols::from_model`).
            byte_ids[byte] = match model.token_to_id(&spelling) {
                Some(id) => id,
                None => {
                    missing.get_or_insert_with(|| Box::new([false; 256]))[byte] = true;
                    NO_BYTE_TOKEN
                }
            };
        }
        // The merges as ids: what the file's one-pass read resolved, else
        // read back from the model (a model built in memory, a vocabulary and
        // merges pair, a merge the read could not resolve).
        let from_model;
        let merges: &[[u32; 3]] = match read.and_then(|read| read.pairs.as_deref()) {
            Some(pairs) => pairs,
            None => {
                from_model = Self::merges_of(model, bpe)?;
                &from_model
            }
        };
        if merges.len() >= MAX_RANKS {
            return Err(Declined::from(
                "more merges than a short merge's keys carry",
            ));
        }
        let mut table = Self::with_capacity(
            merges.len(),
            bpe.ignore_merges,
            tokens == ByteTokens::Alphabet,
        );
        table.byte_ids = byte_ids;
        table.missing = missing;
        if tokens == ByteTokens::Alphabet {
            table.hot_base = u64::from(
                byte_ids
                    .iter()
                    .copied()
                    .filter(|&id| id != NO_BYTE_TOKEN)
                    .min()
                    .unwrap_or(0),
            );
        }
        let mut pairs = Vec::with_capacity(merges.len());
        for (rank, &[a, b, new_id]) in merges.iter().enumerate() {
            let rank = u32::try_from(rank)
                .ok()
                .filter(|&rank| rank != NO_RANK)
                .ok_or_else(|| Declined::from("more merges than the table ranks"))?;
            table.insert(a, b, Merge::new(rank, new_id));
            pairs.push((rank, a, b));
        }
        if tokens == ByteTokens::Alphabet {
            // The least rank of a merge that takes each id as an operand.
            let ids_end = pairs
                .iter()
                .map(|&(_, a, b)| a.max(b) as usize + 1)
                .max()
                .unwrap_or(0);
            let mut min_operand = vec![NO_RANK; ids_end];
            for &(rank, a, b) in &pairs {
                min_operand[a as usize] = min_operand[a as usize].min(rank);
                min_operand[b as usize] = min_operand[b as usize].min(rank);
            }
            // The byte each alphabet character stands for, by character.
            let alphabet = super::bytes_char();
            let mut byte_of =
                vec![None; alphabet.iter().map(|&c| c as usize).max().unwrap_or(0) + 1];
            for (byte, &c) in alphabet.iter().enumerate() {
                byte_of[c as usize] = Some(byte as u8);
            }
            let raw_of = |token: &str| -> Option<Vec<u8>> {
                token
                    .chars()
                    .map(|c| byte_of.get(c as usize).copied().flatten())
                    .collect()
            };
            // The entries borrowed from the read's listing when it has one,
            // by id where the read says the ids are the range, else through
            // a clone of the vocabulary map.
            let raws = entries_of(bpe, read)
                .filter_map(|(id, token)| raw_of(token.as_ref()).map(|raw| (id, raw)))
                .filter(|(_, raw)| !table.lacks_a_token_for(raw))
                .collect();
            table.preformed = Box::new(Preformed::analyse(&table, raws, &min_operand));
        }
        Ok(table)
    }

    /// The model's merges as ids (left, right, product; the rank is the
    /// index), read back through the model's `Serialize` (see `serialized`:
    /// the merges alone, so a vocabulary with gaps in its ids does not have
    /// the library warn about them at every load) and resolved in its
    /// vocabulary; or why they cannot be.
    fn merges_of(model: &ModelWrapper, bpe: &BPE) -> Result<Vec<[u32; 3]>, Declined> {
        let serialized = serialized::merges_json(bpe)
            .map_err(|e| Declined::from(format!("the model does not serialize: {e}")))?;
        let SerializedMerges(merges) = serde_json::from_str(&serialized)
            .map_err(|e| Declined::from(format!("the model's merges do not read back: {e}")))?;
        let mut token = String::new();
        let mut out = Vec::with_capacity(merges.len());
        for (left, right) in &merges {
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
            out.push([a, b, new_id]);
        }
        Ok(out)
    }

    /// The table of a tiktoken vocabulary's merges (see the module
    /// documentation), or why the file's own encoder stays in place: a byte
    /// without a token, a token that is no two tokens' concatenation, or a
    /// token its own bytes do not merge back into (the file's encoder looks
    /// a whole piece up before it merges; the table does not, so every token
    /// must be the merge of its bytes for the two to agree on every piece).
    /// The table pre-forms characters as a model's does (see `preformed`):
    /// the tokens are raw bytes already, and an id's least operand rank is
    /// the least rank of a split that takes it as either half.
    pub(super) fn from_ranks(ranks: &FxHashMap<Vec<u8>, u32>) -> Result<Self, Declined> {
        let mut byte_ids = [0u32; 256];
        for (byte, id) in byte_ids.iter_mut().enumerate() {
            *id = ranks
                .get([byte as u8].as_slice())
                .copied()
                .ok_or_else(|| Declined::from(format!("byte {byte:#04x} has no token")))?;
        }
        let ranks_end = ranks.values().max().map_or(0, |&rank| rank as usize + 1);
        // A rank is a merge's rank and its product at once; it is bounded as
        // the decode table of such a file is, and that bound stays within
        // what a short merge's key carries of a rank.
        const _: () = assert!(MAX_BYTE_LEVEL_TABLE_IDS <= MAX_RANKS);
        if ranks_end > MAX_BYTE_LEVEL_TABLE_IDS {
            return Err(Declined::from(format!(
                "ranks up to {ranks_end} on a table bounded at {MAX_BYTE_LEVEL_TABLE_IDS}"
            )));
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
        let mut table = Self::with_capacity(splits.len(), false, true);
        table.byte_ids = byte_ids;
        table.hot_base = u64::from(byte_ids.iter().copied().min().unwrap_or(0));
        for &(left, right, rank) in &splits {
            table.insert(left, right, Merge::new(rank, rank));
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
        // The least rank of a merge that takes each id as an operand: every
        // split the table holds is such a merge, at the token's rank.
        let mut min_operand = vec![NO_RANK; ranks_end];
        for &(left, right, rank) in &splits {
            min_operand[left as usize] = min_operand[left as usize].min(rank);
            min_operand[right as usize] = min_operand[right as usize].min(rank);
        }
        let raws = ranks
            .iter()
            .map(|(token, &rank)| (rank, token.clone()))
            .collect();
        table.preformed = Box::new(Preformed::analyse(&table, raws, &min_operand));
        Ok(table)
    }

    /// An empty table sized for `merges` pairs, with the hot grid when `hot`
    /// (a table whose merges start from bytes); the caller sets the grid's
    /// window base once it knows the byte ids.
    fn with_capacity(merges: usize, ignore_merges: bool, hot: bool) -> Self {
        // Buckets for a load of one half at most, two at least: a probe
        // always meets an empty entry, and the shift that picks a bucket
        // from a key's high bits stays below the key's width.
        let buckets = merges.div_ceil(BUCKET / 2).next_power_of_two().max(2);
        let shift = u64::BITS - buckets.trailing_zeros();
        let hot = if hot {
            vec![Merge::NONE; (HOT_SIDE * HOT_SIDE) as usize].into_boxed_slice()
        } else {
            Box::default()
        };
        let filter_words = (merges * PAIR_FILTER_BITS_PER_MERGE)
            .div_ceil(u64::BITS as usize)
            .next_power_of_two()
            .max(PAIR_FILTER_MIN_WORDS);
        Self {
            byte_ids: [0; 256],
            missing: None,
            hot,
            hot_base: NO_HOT_BASE,
            buckets: vec![Bucket::EMPTY; buckets].into_boxed_slice(),
            shift,
            pair_filter: vec![0; filter_words].into_boxed_slice(),
            pair_filter_shift: u64::BITS - filter_words.trailing_zeros(),
            merges,
            preformed: Box::new(Preformed::empty()),
            ignore_merges,
        }
    }

    /// Bytes the table holds, for the load log.
    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + size_of_val(&*self.hot)
            + size_of_val(&*self.buckets)
            + self.pair_filter_bytes()
            + self.preformed.bytes()
            + self.missing.as_deref().map_or(0, size_of_val)
    }

    /// Characters of the vocabulary a piece may start from as one symbol.
    pub(super) fn preformed_characters(&self) -> usize {
        self.preformed.characters()
    }

    /// Bytes of the pair filter.
    fn pair_filter_bytes(&self) -> usize {
        size_of_val(&*self.pair_filter)
    }

    /// The id of `byte`'s token ([`NO_BYTE_TOKEN`] for a byte without one).
    #[inline(always)]
    pub(super) fn byte_id(&self, byte: u8) -> u32 {
        self.byte_ids[usize::from(byte)]
    }

    /// Whether the vocabulary has no token for `byte`.
    pub(super) fn lacks_a_token_for_byte(&self, byte: u8) -> bool {
        self.byte_ids[usize::from(byte)] == NO_BYTE_TOKEN
    }

    /// Whether `piece` holds a byte the vocabulary has no token for: such a
    /// piece is the model's to merge (the table has no id for that byte).
    /// A table over a complete alphabet answers without a look at the piece.
    #[inline]
    pub(super) fn lacks_a_token_for(&self, piece: &[u8]) -> bool {
        self.missing
            .as_ref()
            .is_some_and(|missing| piece.iter().any(|&byte| missing[usize::from(byte)]))
    }

    /// Bytes the vocabulary has no token for (0 over a complete alphabet),
    /// for the load log.
    pub(super) fn missing_bytes(&self) -> usize {
        self.missing
            .as_ref()
            .map_or(0, |missing| missing.iter().filter(|&&gone| gone).count())
    }

    /// Pairs the table was built from (one per merge of a model; one per
    /// split of a token into two tokens of a tiktoken vocabulary, the splits
    /// of one token sharing its rank).
    pub(super) fn merges(&self) -> usize {
        self.merges
    }

    #[inline(always)]
    fn first_bucket(&self, key: u64) -> usize {
        // Fibonacci hashing: the pair's two ids spread over the high bits.
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// The filter word of a pair key and its two bits: the hash's high bits
    /// pick the word, its low twelve the two bit positions.
    #[inline(always)]
    fn pair_filter_slot(&self, key: u64) -> (usize, u64) {
        let hash = key.wrapping_mul(PAIR_FILTER_MULTIPLIER);
        let word = (hash >> self.pair_filter_shift) as usize;
        let bits = 1u64 << (hash & 63) | 1u64 << (hash >> 6 & 63);
        (word, bits)
    }

    /// Whether the pair `(a, b)` passes the filter: every inserted pair does,
    /// a pair never inserted rarely.
    #[cfg(test)]
    fn pair_filter_admits(&self, a: u32, b: u32) -> bool {
        let (word, bits) = self.pair_filter_slot(Self::pair_key(a, b));
        self.pair_filter[word] & bits == bits
    }

    /// The key of the pair `(a, b)` in the bucket table.
    #[inline(always)]
    fn pair_key(a: u32, b: u32) -> u64 {
        u64::from(a) << 32 | u64::from(b)
    }

    /// The index of the pair `(a, b)` in the hot grid when both ids are in
    /// its window (one compare: the side is a power of two, and the base of
    /// a table without the grid is past every id).
    #[inline(always)]
    fn hot_index(&self, a: u32, b: u32) -> Option<usize> {
        let (x, y) = (
            u64::from(a).wrapping_sub(self.hot_base),
            u64::from(b).wrapping_sub(self.hot_base),
        );
        ((x | y) < u64::from(HOT_SIDE)).then_some((x as usize) << HOT_SIDE_BITS | y as usize)
    }

    /// Records `(a, b)` as `merge`; a pair recorded twice keeps the later
    /// merge, as the model's merge map does. The only writer of the grid and
    /// the buckets: it fills a bucket from its first entry and nothing removes
    /// an entry, which the lookup's walk ([`Self::merge_of`]) rests on.
    fn insert(&mut self, a: u32, b: u32, merge: Merge) {
        if let Some(at) = self.hot_index(a, b) {
            self.hot[at] = merge;
            return;
        }
        let key = Self::pair_key(a, b);
        let (word, bits) = self.pair_filter_slot(key);
        self.pair_filter[word] |= bits;
        let mask = self.buckets.len() - 1;
        let mut bucket = self.first_bucket(key);
        loop {
            for entry in &mut self.buckets[bucket].0 {
                if entry.key == key || entry.key == EMPTY_KEY {
                    *entry = Entry { key, merge };
                    return;
                }
            }
            bucket = (bucket + 1) & mask;
        }
    }

    /// Reads one word of every line of the table, the buckets first, then
    /// the grid and the filter (the small, hot tables last, so they stay in
    /// the nearer caches), and keeps nothing: a thread's first encode with
    /// the encoder calls it so that the table's lines come in as one stream
    /// then rather than one dependent probe at a time during the first-sight
    /// pieces of the texts after it. In a fresh process the table was built
    /// at load and the rest of the load pushed it out; measured on the
    /// encode bar, the probes of a 256-byte cold text were 1.5 fills from
    /// memory per token against 0.16 from the last-level cache, and this read
    /// takes 10-20 % off those cells (16 % by median) and 8 % by median off
    /// the 2 KB ones, nothing measurable from 32 KB up, against a parent that
    /// has the pair filter, for 0.3-0.5 ms once per thread per encoder on a
    /// 128-151k-merge table.
    #[inline(never)]
    pub(super) fn read_through(&self) {
        const GRID_STEP: usize = 64 / size_of::<Merge>();
        let mut sum = 0u64;
        for bucket in &self.buckets {
            sum = sum.wrapping_add(bucket.0[0].key);
        }
        for merge in self.hot.iter().step_by(GRID_STEP) {
            sum = sum.wrapping_add(u64::from(*merge != Merge::NONE));
        }
        for word in self.pair_filter.iter().step_by(8) {
            sum = sum.wrapping_add(*word);
        }
        // The loads stay: the sum is observed.
        std::hint::black_box(sum);
    }

    /// The merge of the pair `(a, b)`, [`Merge::NONE`] when it is none: the
    /// hot grid by index when both ids are in its window, else the pair's
    /// filter word (a pair never inserted is answered there), then the
    /// bucket the key hashes to, entry by entry up to the key or the first
    /// empty entry, and the buckets after it while a bucket is full of other
    /// pairs. The walk rests on [`Self::insert`] filling a bucket from its
    /// first entry and on no entry ever being removed: no bucket holds an
    /// empty entry before a live one.
    #[inline(always)]
    fn merge_of(&self, a: u32, b: u32) -> Merge {
        match self.hot_index(a, b) {
            Some(at) => self.hot[at],
            None => self.merge_of_bucketed(a, b),
        }
    }

    /// [`Self::merge_of`] for a pair outside the hot grid's window.
    #[inline]
    fn merge_of_bucketed(&self, a: u32, b: u32) -> Merge {
        let key = Self::pair_key(a, b);
        let (word, bits) = self.pair_filter_slot(key);
        if self.pair_filter[word] & bits != bits {
            return Merge::NONE;
        }
        let buckets = &self.buckets;
        let mask = buckets.len() - 1;
        let mut bucket = self.first_bucket(key);
        loop {
            // Most lookups end after one or two compares of the one line
            // (a bucket holds two or fewer at the table's load).
            for entry in &buckets[bucket].0 {
                if entry.key == key {
                    return entry.merge;
                }
                if entry.key == EMPTY_KEY {
                    return Merge::NONE;
                }
            }
            bucket = (bucket + 1) & mask;
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
        debug_assert!(
            !self.hot.is_empty(),
            "a byte-level merge on a table built for character symbols"
        );
        debug_assert!(
            !self.lacks_a_token_for(piece),
            "a piece holding a byte the table has no token for is the model's"
        );
        if piece.is_empty() {
            return;
        }
        self.merge_bytes(piece, ids);
    }

    /// The merge of a piece from its bytes' tokens: on stack arrays up to
    /// [`SHORT_PIECE_SYMBOLS`], on the heap beyond.
    #[inline(always)]
    fn merge_bytes(&self, piece: &[u8], ids: &mut Vec<u32>) {
        let symbols = piece.iter().map(|&byte| self.byte_ids[usize::from(byte)]);
        if piece.len() <= SHORT_PIECE_SYMBOLS {
            self.merge_short(symbols, piece.len(), ids);
        } else {
            self.merge_long(symbols, ids);
        }
    }

    /// The merge of a piece from its characters (see `preformed`): a symbol
    /// per character that pre-forms and a byte's token for every other
    /// byte, cut into segments between two pre-formed characters no token
    /// holds adjacent, each segment merged on its own (a segment with more
    /// symbols than the stack arrays hold merges from its bytes: it merges
    /// the same on its own as inside the piece). `false`, and nothing
    /// appended, when no character of the piece pre-forms, when the piece
    /// does not pay off (its first multi-byte character's block holds too
    /// few pre-formable characters for the piece's length) or when nothing
    /// pre-forms under the vocabulary: the caller merges from the bytes
    /// then. The caller sends a cold piece of a unit that holds a non-ASCII
    /// byte here first (a piece of an ASCII unit never comes); an ASCII
    /// piece of a mixed unit leaves at the pays-off test.
    #[inline(never)]
    pub(super) fn merge_preformed(&self, piece: &[u8], ids: &mut Vec<u32>) -> bool {
        debug_assert!(
            !self.lacks_a_token_for(piece),
            "a piece holding a byte the table has no token for is the model's"
        );
        let pre = &self.preformed;
        if pre.is_empty() || !pre.pays_off(piece) {
            return false;
        }
        let mut segment = [0u32; SHORT_PIECE_SYMBOLS];
        let mut n = 0usize;
        let mut overflow = false;
        let mut start = 0usize;
        let mut prev = NO_CHAR;
        let mut any = false;
        let mut i = 0usize;
        while i < piece.len() {
            let lead = piece[i];
            let len = char_len(lead);
            let whole = lead < 0x80
                || (len > 1
                    && i + len <= piece.len()
                    && piece[i + 1..i + len].iter().all(|&b| is_continuation(b)));
            if !whole {
                // A byte that starts no character of the text (the piece is
                // text, so none is met; a bare byte string merges as bytes).
                push_symbol(
                    &mut segment,
                    &mut n,
                    &mut overflow,
                    self.byte_ids[usize::from(lead)],
                );
                prev = NO_CHAR;
                i += 1;
                continue;
            }
            if lead < 0x80 {
                push_symbol(
                    &mut segment,
                    &mut n,
                    &mut overflow,
                    self.byte_ids[usize::from(lead)],
                );
                prev = NO_CHAR;
                i += 1;
                continue;
            }
            let cp = code_point(&piece[i..], len);
            match pre.token_at(piece, i, len, cp) {
                Some(id) => {
                    if prev != NO_CHAR && !pre.joinable(prev, cp) {
                        self.merge_segment(&segment, n, overflow, &piece[start..i], ids);
                        n = 0;
                        overflow = false;
                        start = i;
                    }
                    push_symbol(&mut segment, &mut n, &mut overflow, id);
                    any = true;
                    prev = cp;
                }
                None => {
                    for &byte in &piece[i..i + len] {
                        push_symbol(
                            &mut segment,
                            &mut n,
                            &mut overflow,
                            self.byte_ids[usize::from(byte)],
                        );
                    }
                    prev = NO_CHAR;
                }
            }
            i += len;
        }
        if !any {
            return false;
        }
        self.merge_segment(&segment, n, overflow, &piece[start..], ids);
        true
    }

    /// One segment of [`Self::merge_preformed`]: its `n` symbols, or its
    /// `bytes` when it outgrew the arrays.
    #[inline(always)]
    fn merge_segment(
        &self,
        segment: &[u32; SHORT_PIECE_SYMBOLS],
        n: usize,
        overflow: bool,
        bytes: &[u8],
        ids: &mut Vec<u32>,
    ) {
        if overflow {
            self.merge_bytes(bytes, ids);
        } else if n == 1 {
            ids.push(segment[0]);
        } else if n > 1 {
            self.merge_symbols(&segment[..n], ids);
        }
    }

    /// Appends the ids `symbols` merge to under the table: initial symbols
    /// of any kind, a character's token or a byte's, as the model's
    /// `merge_word` makes them for a vocabulary with byte fallback (see
    /// `sentencepiece`); the merge is the one of a piece.
    #[inline(never)]
    pub(super) fn merge_symbols(&self, symbols: &[u32], ids: &mut Vec<u32>) {
        if symbols.is_empty() {
            return;
        }
        if symbols.len() <= SHORT_PIECE_SYMBOLS {
            self.merge_short(symbols.iter().copied(), symbols.len(), ids);
        } else {
            self.merge_long(symbols.iter().copied(), ids);
        }
    }

    /// The merge of `n` (`1..=SHORT_PIECE_SYMBOLS`) initial `symbols` as a
    /// [`ShortMerge`] on stack arrays, the smaller ones for a piece that fits
    /// them. `n` is the iterator's length, passed by the caller: taken from
    /// the iterator it cost one to two instructions per token on cold text.
    fn merge_short(
        &self,
        symbols: impl ExactSizeIterator<Item = u32>,
        n: usize,
        ids: &mut Vec<u32>,
    ) {
        debug_assert_eq!(symbols.len(), n);
        if n <= SHORT_PIECE_TINY {
            ShortMerge::<SHORT_PIECE_TINY>::of(symbols, n).run(self, ids);
        } else if n <= SHORT_PIECE_SMALL {
            ShortMerge::<SHORT_PIECE_SMALL>::of(symbols, n).run(self, ids);
        } else {
            ShortMerge::<SHORT_PIECE_SYMBOLS>::of(symbols, n).run(self, ids);
        }
    }

    /// The merge of more initial symbols than a short merge's arrays hold,
    /// the model's own loop: a linked list of symbols and a min-heap of
    /// `(rank, position)` candidates, an entry taken only when the pair at
    /// its position is still the pair it was pushed for (the rank names the
    /// pair; the model compares the token the pair produces, which comes to
    /// the same, see the module docs).
    fn merge_long(&self, initial: impl Iterator<Item = u32>, ids: &mut Vec<u32>) {
        let mut symbols: Vec<Symbol> = initial
            .enumerate()
            .map(|(i, id)| Symbol {
                id,
                prev: if i == 0 { NONE } else { i as u32 - 1 },
                next: i as u32 + 1,
            })
            .collect();
        let Some(last) = symbols.last_mut() else {
            return;
        };
        last.next = NONE;
        let candidates: BinaryHeap<Reverse<(u32, u32)>> = (0..symbols.len() - 1)
            .filter_map(|i| {
                let merge = self.merge_of(symbols[i].id, symbols[i + 1].id);
                (merge != Merge::NONE).then_some(Reverse((merge.rank(), i as u32)))
            })
            .collect();
        self.merge_long_from(symbols, candidates, ids);
    }

    /// Merges `symbols` from the first round's `candidates` until none is
    /// left, and appends the ids to `ids`.
    fn merge_long_from(
        &self,
        mut symbols: Vec<Symbol>,
        mut candidates: BinaryHeap<Reverse<(u32, u32)>>,
        ids: &mut Vec<u32>,
    ) {
        while let Some(Reverse((rank, pos))) = candidates.pop() {
            let at = pos as usize;
            let Symbol { id, prev, next } = symbols[at];
            if id == NONE || next == NONE {
                continue;
            }
            let right = symbols[next as usize];
            let merge = self.merge_of(id, right.id);
            if merge.rank() != rank {
                continue;
            }
            symbols[at].id = merge.product();
            symbols[at].next = right.next;
            symbols[next as usize].id = NONE;
            if right.next != NONE {
                symbols[right.next as usize].prev = pos;
                let ahead = self.merge_of(symbols[at].id, symbols[right.next as usize].id);
                if ahead != Merge::NONE {
                    candidates.push(Reverse((ahead.rank(), pos)));
                }
            }
            if prev != NONE {
                let behind = self.merge_of(symbols[prev as usize].id, symbols[at].id);
                if behind != Merge::NONE {
                    candidates.push(Reverse((behind.rank(), prev)));
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

/// Appends `symbol` to a segment of [`MergeTable::merge_preformed`], or
/// marks the segment as outgrown.
#[inline(always)]
fn push_symbol(
    segment: &mut [u32; SHORT_PIECE_SYMBOLS],
    n: &mut usize,
    overflow: &mut bool,
    symbol: u32,
) {
    if *n < SHORT_PIECE_SYMBOLS {
        segment[*n] = symbol;
        *n += 1;
    } else {
        *overflow = true;
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
            let merge = table.merge_of(self.symbols[at], self.symbols[at + 1]);
            self.set_pair(at, merge);
        }
        self.settle(table, pairs, ids);
    }

    /// Merges from the `pairs` of the first round until no live pair is a
    /// merge, and appends the ids to `ids`.
    #[inline(always)]
    fn settle(mut self, table: &MergeTable, pairs: usize, ids: &mut Vec<u32>) {
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
            table.merge_of(self.symbols[at], self.symbols[right])
        } else {
            Merge::NONE
        };
        let behind = if left < self.n {
            self.keys[left] = NO_KEY;
            table.merge_of(self.symbols[left], self.symbols[at])
        } else {
            Merge::NONE
        };
        let others = self.least(self.n - 1);
        let key_at = self.set_pair(at, ahead);
        let key_left = if left < self.n {
            self.set_pair(left, behind)
        } else {
            NO_KEY
        };
        others.min(key_at).min(key_left)
    }

    /// Records the pair the symbol at `at` starts as `merge` (its key and
    /// its product; no key for [`Merge::NONE`]) and answers the key.
    #[inline(always)]
    fn set_pair(&mut self, at: usize, merge: Merge) -> u32 {
        let key = if merge == Merge::NONE {
            NO_KEY
        } else {
            self.products[at] = merge.product();
            merge.rank() << SHORT_KEY_BITS | at as u32
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
