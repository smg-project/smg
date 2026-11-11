//! A native encode path for byte-level BPE tokenizers.
//!
//! `tokenizers` encodes a prompt through a `NormalizedString` that tracks a
//! byte alignment per character, a pre-tokenizer whose regex runs in
//! Oniguruma (allocating through the C allocator on every search), a
//! `PreTokenizedString` of owned splits, and an `Encoding` that carries a
//! string, offsets and several per-token vectors. The gateway only ever
//! reads the ids. For the common tokenizer shape, Qwen, GPT-2, GLM, DeepSeek,
//! MiniMax and the like, this module produces the same ids directly:
//!
//! 1. added tokens are cut out of the text the way `AddedVocabulary` does it
//!    (leftmost-longest, the non-normalized set on the raw text first, the
//!    normalized set after normalization);
//! 2. normalization is NFC or nothing, read from the tables `tokenizers`
//!    normalizes with (the crate `unicode-normalization-alignments`, every
//!    NFC read of this module: a newer crate's tables know marks and
//!    compositions the library's do not and would compose other text); a
//!    text already in NFC (its ASCII stretches are not read at all, the
//!    quick check answers for most of the rest and normalizing settles what
//!    it cannot, so marks-heavy scripts cost no copy) is used as it is, any
//!    other text is normalized gap by gap between its raw added tokens, as
//!    `tokenizers` normalizes it, and encoded from those buffers;
//! 3. each `Split` pre-tokenizer runs as a [`pattern::Pattern`] over `&str`,
//!    with the same leftmost-first semantics and Unicode tables as the regex
//!    engine `tokenizers` uses, producing byte ranges instead of strings (a
//!    chain of three stages of one known shape runs as one scan instead,
//!    see [`chain`]);
//! 4. each piece is tokenized behind two tables from piece bytes to ids
//!    (`short_pieces`): a small per-thread cache of the pieces the thread
//!    has met, filled on a first sight from the vocabulary's own pieces,
//!    built at load and shared by every thread; a piece neither table
//!    holds is merged over the model's own merges by [`merge::MergeTable`]
//!    (the model's `tokenize` without its strings; `ignore_merges` looks
//!    the whole piece up in the model's vocabulary first), or by the model
//!    itself for a vocabulary the table declines (see `merge`);
//! 5. a request for special tokens gets the frame the post-processor puts
//!    around one sequence (none, or a `TemplateProcessing`'s single template:
//!    the ids of its special tokens before and after the `$A` sequence, read
//!    once at load) written around the ids.
//!
//! A vocabulary with byte fallback (sentencepiece-style: Gemma, Llama 2 and
//! their kin) has no `ByteLevel` stage and no stages here at all: the text
//! between two added tokens is one word of the model, which the module
//! [`sentencepiece`] cuts into pieces no merge crosses, with the word marker
//! for the text's spaces and, by the `Metaspace` scheme, before it; the
//! pieces take the same cache, seed and merge table, from character symbols.
//!
//! The shape is checked once at load time; a tokenizer that does not fit
//! (another normalizer, a pre-tokenizer or regex construct outside the
//! supported subset, a post-processor other than `ByteLevel` or a template
//! of that shape, truncation or padding, added tokens with
//! `single_word`/`lstrip`/`rstrip`) simply has no native path.
//!
//! A tiktoken vocabulary is a byte-level BPE of this shape with no
//! normalizer, one pattern, special tokens cut out of the raw text and no
//! frame, its merges implied by its ranks: [`NativeEncoder::from_ranks`]
//! reads one into the same encoder (the merge table from the ranks, see
//! [`merge`]), and a file whose ranks or pattern do not fit keeps its own
//! encoder.

use std::{
    borrow::Cow,
    cell::{RefCell, RefMut},
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use aho_corasick::{
    packed, AhoCorasick, AhoCorasickBuilder, Anchored, Input, MatchKind, Span, StartKind,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::Value;
use thread_local::ThreadLocal;
use tokenizers::{models::ModelWrapper, Model};
use unicode_normalization_alignments::UnicodeNormalization;

mod cased;
mod chain;
mod long_entries;
mod merge;
mod nfc;
mod pattern;
mod scan;
mod sentencepiece;
mod short_pieces;
mod unicode;

use long_entries::LongEntryFilter;
use merge::{ByteTokens, MergeTable};
use nfc::NfcQuick;
use pattern::Pattern;
use sentencepiece::{CharSymbols, Marker, WORD_MARKER};
use short_pieces::{ShortPieceKey, ShortPieceTable, Vacancy, SHORT_PIECE_MAX_BYTES};

use crate::{
    encode_path::{Declined, Fallback},
    word_cache::{Listing, ModelView, TokenizerOf},
};

/// The split the `ByteLevel` pre-tokenizer applies itself when `use_regex`
/// is on (GPT-2 and its descendants).
const BYTE_LEVEL_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Pieces up to this many bytes are cached per thread by their count (see
/// `PIECE_CACHE_CAPACITY`): those of up to `SHORT_PIECE_MAX_BYTES` in a
/// packed table, the longer ones in a map; longer pieces still, up to
/// `LONG_PIECE_MAX_BYTES`, by the bytes they take (see
/// `LONG_PIECE_BYTES_CAPACITY`).
const CACHED_PIECE_MAX_BYTES: usize = 64;
/// Longest piece cached at all, 16 KiB. The byte-pair cut leaves long pieces
/// in a script without word spaces (a clause of some 85 such characters is
/// 256 bytes, a paragraph a few KiB) and in code or prose whose word markers the
/// vocabulary bridges; a fifth of such a text can sit in pieces past 256
/// bytes, and with the cap there every encode of it merged them again. The
/// tier is bounded by bytes, so an entry costs what it takes; a longer piece
/// is tokenized every time.
const LONG_PIECE_MAX_BYTES: usize = 16 << 10;

/// Bytes of an input from which the single-stage scanners walk a letter or
/// white-space run eight bytes at a time (`Roles::run_end_as`): the scanner
/// loop is compiled with the word scans and without, and the choice is made
/// once per encode call on the input's length; a shorter input walks every
/// run byte by byte through the loop as it was before the scans, and no run
/// loop compares anything. The word scan removes the run loop's mispredicted
/// exits, which the predictor cannot learn over a long text and holds for a
/// short text it has just seen (counted per token on one core, single-stage
/// vocabularies: 32 KB-500 KB texts -13..-16 % cycles for +3..+5 %
/// instructions with the scans; 256 B and 2 KB texts +2..+7 % instructions
/// with the scans compiled into their loop, whether or not a run takes
/// them). The threshold is the pool path's unit size, so a long input keeps
/// the scans on every unit it is cut into.
const WORD_SCAN_MIN_BYTES: usize = PARALLEL_CHUNK_BYTES;
/// Entries a thread's piece cache holds for one encoder before it is cleared,
/// at most; see `PIECE_CACHE_TOTAL_CAPACITY`. This is also the cliff of a
/// repeated document: one with fewer distinct pieces is encoded again at the
/// cost of its lookups, one with more at first-sight cost on every pass,
/// since the clear takes the pieces that recur with the rest (the crate bench
/// `encode_volume` shows both, and that the cost of text no pass has seen
/// does not depend on how much a thread encoded before); a first sight of a
/// vocabulary word costs a read of the encoder's seed, of another piece the
/// model's merges. The bound holds the distinct pieces of a document of
/// several hundred thousand tokens, so a long document, and a batch of many
/// prompts, warm on a second pass as a short prompt does (at 16,384 a text
/// of 32k tokens cleared the cache on every pass, and documents of 128k and
/// 500k tokens encoded as slowly warm as cold: measured). Memory: a thread's
/// table grows with use (see `short_pieces`), so a thread that encodes
/// little holds a few KB; at the full bound it has 524,288 slots of 33
/// bytes, 16.5 MiB (the table keys pieces of up to `SHORT_PIECE_MAX_BYTES`
/// bytes with up to fifteen ids each, four in the slot and the rest in a
/// spill: 15 MiB more at most, emptied with the table, its capacity kept).
/// The pieces of 16 to 64 bytes count against the same bound in a map
/// (`PieceCache::short`): its slot with the table's slack, the piece and
/// the ids as two boxes, about 140 bytes for a typical entry (a 20-byte
/// piece with four ids) and about 430 at most (a 64-byte piece whose ids
/// are its bytes), so a thread whose entries were all such pieces holds
/// about 35 MiB, 108 MiB in that worst case. A clear at the bound keeps
/// both tiers' arrays (the packed table's slots and its spill's capacity,
/// and the map's buckets, unless the share fell below what they were grown
/// for) and drops their entries, so a thread that filled its packed table
/// on one kind of text and then meets such pieces holds both, about 51 MiB
/// with typical map entries and 140 MiB at the worst case the count bound
/// allows.
/// Measured on one thread after a corpus of 7,015 prompts: the peak RSS
/// grew by 8 MB on a 128k-token vocabulary and did not move on three
/// larger ones, whose peak is the file load.
const PIECE_CACHE_CAPACITY: usize = 262_144;
/// Entries all threads together hold for one encoder: a thread's own cap is
/// this divided by the number of threads that have encoded with the encoder
/// (and never more than `PIECE_CACHE_CAPACITY`). A thread's packed table starts
/// at 64 slots (2 KiB) and grows as the thread meets pieces (by four up to
/// 4,096 slots, by two above), up to the slots for its cap, and is cut back to
/// the slots of a smaller cap when it is cleared (made anew only past its
/// reservation), so a thread that encodes little holds little and the packed
/// tables of one tokenizer stay under 352 MiB up to some 680 threads (sixteen
/// threads at the full bound hold 264 MiB; more threads share the total, each
/// table rounded up to a power of two slots, at most eight thirds of the total
/// in slots) and under twice that however many more encode with it (a table cut
/// back inside its reservation keeps the pages it touched, 528 KiB at most,
/// while the later threads hold their share's slots: about 440 MiB in all at
/// 1,024 threads). The maps of the pieces of 16 to 64 bytes share the count at
/// about 140 bytes a typical entry (430 at most, see `PIECE_CACHE_CAPACITY`),
/// and a clear keeps the packed slots and the map's buckets (the map made anew,
/// the packed table cut back, for a smaller share), so the count-bounded tiers
/// of one tokenizer hold at most the sum of the two: about 820 MiB at sixteen
/// threads with typical map entries (about 2.2 GiB, sixteen times the 140 MiB
/// of a thread, were every map entry the worst case), toward 352 + 560 MiB with
/// the rounding of more threads (the packed part toward twice its 352 MiB past
/// some 680 threads, as above); the long pieces add `LONG_PIECE_BYTES_TOTAL` at
/// most. A thread reads its share at the start of every encode call and keeps
/// it for that call's misses (a call that was running when threads joined keeps
/// its earlier share until it ends; the long tier reads the live share at every
/// miss), so after threads join it is within the smaller share from its next
/// call on, entries, packed slots and map buckets alike (the packed table's
/// pages already touched inside its reservation stay resident,
/// `THREAD_BACKING_SLOTS` slots at most, 33 bytes each, about 528 KiB per
/// thread: more than the share's slots only once those fall under the
/// reservation, a share under 6,144 entries, some 680 threads on this total),
/// and holds at most its earlier share until then, or until it exits if it
/// never encodes again: right after a join the total is the earlier share times
/// the threads that had it plus the new share times the rest, and settles as
/// they encode. Before this bound the same ceilings were a sixteenth of these.
/// All of it is freed with the encoder.
const PIECE_CACHE_TOTAL_CAPACITY: usize = 16 * PIECE_CACHE_CAPACITY;
/// Bytes of long pieces (each piece, its ids and their slot) a thread holds
/// for one encoder before its long entries are cleared, at most; see
/// `LONG_PIECE_BYTES_TOTAL`.
const LONG_PIECE_BYTES_CAPACITY: usize = 2 << 20;
/// Bytes of long pieces all threads together hold for one encoder, shared
/// the way `PIECE_CACHE_TOTAL_CAPACITY` is. The long pieces of a prompt (its
/// clauses, in a script without word spaces) are few in number and large,
/// and each one that misses goes through the model's merges, so they are
/// bounded by what they take rather than counted.
const LONG_PIECE_BYTES_TOTAL: usize = 32 << 20;
/// What an entry of the long tier takes beyond its piece and ids.
const LONG_PIECE_ENTRY_BYTES: usize = size_of::<(Box<[u8]>, Box<[u32]>)>();

/// Inputs at least this long are encoded unit by unit on the thread pool (a
/// text between added tokens, or a chunk of one cut at a stable boundary)
/// when the pool has more than one thread (see `worth_the_pool`); shorter
/// inputs would spend more on the hand-offs than they save.
const PARALLEL_MIN_BYTES: usize = 64 << 10;
/// The size a long text unit is cut into for the pool, at the first stable
/// boundary past each multiple of it.
const PARALLEL_CHUNK_BYTES: usize = 32 << 10;
/// Threads of the pool the vocabulary seed is read on at load. The pool is
/// built for that read and joined before the load returns: on rayon's
/// global pool the first load would build that pool at the machine's
/// parallelism (a CPU quota of sixty-four, sixty-four threads) and keep its
/// threads, idle, for the life of the process. Four read a 151k-entry
/// vocabulary in about 70 ms against about 250 on one thread (see
/// `vocabulary_seed`); more threads gain tens of milliseconds on a load
/// that costs over half a second in the vocabulary file and the model. A
/// load that runs on a worker of a rayon pool already (the Python binding
/// loads on its own, sized by its host) reads on that pool instead.
const SEED_THREADS: usize = 4;
/// Text on each side of a candidate cut that every stage's pattern is run
/// over to confirm a piece starts there (widened to the spaces around it).
const CUT_CHECK_WINDOW: usize = 512;
/// Candidates one cut search may reject before it gives up: the byte scan for
/// a candidate is cheap, the window check on each one is not, so a text whose
/// candidates keep failing the check costs at most this many checks per cut.
const MAX_REJECTED_CUT_CANDIDATES: usize = 64;

/// Piece bytes to the ids they tokenize to.
type PieceIds = FxHashMap<Box<[u8]>, Box<[u32]>>;

/// The ids of one vocabulary entry while the seed is built, in a fixed slot
/// (a byte-level piece of `n` bytes has at most `n` ids), so the pass over
/// the vocabulary allocates one list rather than one per entry.
#[derive(Clone, Copy, Default)]
struct SeededIds {
    count: u8,
    ids: [u32; SHORT_PIECE_MAX_BYTES],
}

impl SeededIds {
    /// Takes one more id; `None` when the slot is full.
    fn push(&mut self, id: u32) -> Option<()> {
        *self.ids.get_mut(usize::from(self.count))? = id;
        self.count += 1;
        Some(())
    }

    fn as_slice(&self) -> &[u32] {
        &self.ids[..usize::from(self.count)]
    }
}

/// A seed job's scratch: the ids an entry merges to and, on the
/// sentencepiece path, its initial symbols; one per rayon job (a split of
/// 2048 entries or more), reused across the entries it reads, so the pass
/// allocates nothing per entry.
#[derive(Default)]
struct SeedScratch {
    ids: Vec<u32>,
    symbols: Vec<u32>,
}

/// One vocabulary entry as the pass over the vocabulary reads it: a short
/// piece with its ids, for the seed, or the bytes of a long one, for the
/// filter the whole-piece lookup consults (see `long_entries`).
enum Seeded {
    Short(ShortPieceKey, SeededIds),
    Long(Box<[u8]>),
}

/// One thread's cache of the pieces it met with one encoder, each with the
/// ids the encoder's seed or the model gave it.
struct PieceCache {
    /// Pieces of up to `SHORT_PIECE_MAX_BYTES`, packed: a table made with
    /// the fewest slots when the cache is, grown with the pieces the thread
    /// meets, up to its share, and held plain (not in an `Option`): the hit
    /// path reads it at an offset of the cache, with no test of its
    /// presence per piece. Bounded by count together with `short`.
    packed: ShortPieceTable,
    /// Pieces of more than `SHORT_PIECE_MAX_BYTES`, up to
    /// `CACHED_PIECE_MAX_BYTES`, bounded by their count together with
    /// `packed`.
    short: PieceIds,
    /// Longer pieces, up to `LONG_PIECE_MAX_BYTES`, bounded by their bytes.
    long: PieceIds,
    /// What `long` holds: each entry's piece, ids and slot.
    long_bytes: usize,
    /// The initial symbols of a piece being merged on the sentencepiece
    /// path: a scratch list of the thread.
    symbols: Vec<u32>,
    /// Whether the unit being encoded is ASCII throughout, read once per
    /// unit (see `Native::encode_unit`): the cold pieces of such a unit take
    /// the byte merge without a test of their own.
    unit_ascii: bool,
    /// The thread's share of the count-bounded tiers, read at the start of
    /// every encode call (`piece_cache_for_call`); the misses of the call
    /// read it here.
    share: usize,
}

impl PieceCache {
    /// The empty cache of a thread whose share of the count-bounded tiers
    /// is `share`: the packed table with the fewest slots, the maps empty.
    fn with_share(share: usize) -> Self {
        Self {
            packed: ShortPieceTable::new(),
            short: PieceIds::default(),
            long: PieceIds::default(),
            long_bytes: 0,
            symbols: Vec::new(),
            unit_ascii: false,
            share,
        }
    }

    /// Entries of the tiers bounded by count, `packed` and `short`.
    fn short_entries(&self) -> usize {
        self.packed.len() + self.short.len()
    }

    /// Slots of the packed tier.
    fn packed_slots(&self) -> usize {
        self.packed.slots()
    }

    /// Entries the map tier has room for without growing: its buckets, a
    /// power of two, at seven eighths.
    fn map_capacity(&self) -> usize {
        self.short.capacity()
    }

    /// Forgets the tiers bounded by count. The packed table keeps its slots
    /// and the map its buckets, unless `share` fell below what they were
    /// grown for: the table is cut back to the slots the share takes when it
    /// has more (inside its reservation the room stays, with the pages the
    /// thread already touched, `THREAD_BACKING_SLOTS` slots at most; a table
    /// that had moved past the reservation is made anew), the map is made
    /// anew when its capacity is above twice the share (a map that grew under
    /// the share stays under that, so the check never fires at a steady share
    /// and a clear at the bound keeps the buckets).
    fn clear_short(&mut self, share: usize) {
        self.packed.clear(share);
        if self.short.capacity() > 2 * share {
            self.short = PieceIds::default();
        } else {
            self.short.clear();
        }
    }

    /// The ids cached for a piece longer than the packed tier keys, if any.
    fn get(&self, piece: &[u8]) -> Option<&[u32]> {
        let tier = if piece.len() <= CACHED_PIECE_MAX_BYTES {
            &self.short
        } else {
            &self.long
        };
        tier.get(piece).map(|ids| &**ids)
    }
}

/// A literal split string as the regex that matches it, the way `Split`
/// compiles a `String` pattern: every non-alphanumeric character escaped.
fn escape_literal(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len() * 2);
    for c in literal.chars() {
        if !c.is_alphanumeric() {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// The offset of the first byte of `bytes` outside ASCII, read eight bytes
/// at a time.
fn first_non_ascii(bytes: &[u8]) -> Option<usize> {
    const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
    let (words, _) = bytes.as_chunks::<8>();
    let at = 8 * words
        .iter()
        .position(|word| u64::from_ne_bytes(*word) & HIGH_BITS != 0)
        .unwrap_or(words.len());
    bytes[at..]
        .iter()
        .position(|byte| !byte.is_ascii())
        .map(|i| at + i)
}

/// The length of the run of CJK ideographs at the start of `bytes`: the
/// characters U+4000..=U+9FFF, each a lead byte E4..=E9 and two continuation
/// bytes in UTF-8 (E4 80 80 is U+4000, E9 BF BF is U+9FFF; the kana under E3
/// and the Hangul syllables under EA..=ED are not in it). Every one of them
/// has combining class 0, no canonical decomposition and no part in any
/// canonical composition (`the_ideograph_range_is_settled_by_the_crates_own_tables`
/// walks the range against the crate's tables), so a run of them is in NFC
/// and either side of it is a cut, like an ASCII stretch.
fn ideograph_run(bytes: &[u8]) -> usize {
    let mut at = 0;
    while at + 3 <= bytes.len() && matches!(bytes[at], 0xE4..=0xE9) {
        at += 3;
    }
    at
}

/// The length of the run of non-ASCII bytes at the start of `bytes`, short
/// of its first ideograph (a continuation byte never reads as a lead byte,
/// so the first E4..=E9 byte starts one): one pass that ends at the first
/// ASCII or ideograph lead byte, so a line of ideographs and full-width
/// punctuation is read once however many pieces it splits into.
fn checked_run(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .position(|byte| byte.is_ascii() || matches!(byte, 0xE4..=0xE9))
        .unwrap_or(bytes.len())
}

/// The stretches of `text` that normalization reads on their own, each with
/// its byte offset and whether it is settled: ASCII, or CJK ideographs. An
/// ASCII stretch is in NFC in every version of Unicode and no normalization
/// changes it; so is a run of ideographs (see [`ideograph_run`]), and
/// nothing composes with an ideograph on either side. A run of other
/// non-ASCII characters goes with the ASCII character before it, the letter
/// an accent at its start composes with; nothing else crosses a cut: no
/// composition has an ASCII character or an ideograph as its second
/// element, no decomposition reaches across one, and both are starters, so
/// the canonical order of marks never spans one.
fn stretches(text: &str) -> impl Iterator<Item = (usize, &str, bool)> {
    let bytes = text.as_bytes();
    let mut at = 0;
    std::iter::from_fn(move || {
        let start = at;
        let rest = &bytes[start..];
        let (len, settled) = match rest.first() {
            None => return None,
            Some(first) if first.is_ascii() => match first_non_ascii(rest) {
                None => (rest.len(), true),
                // The whole ASCII stretch: the run after it starts with
                // ideographs, which compose with nothing before them.
                Some(run) if ideograph_run(&rest[run..]) > 0 => (run, true),
                // The ASCII stretch, short of the character before the run.
                Some(run) if run > 1 => (run - 1, true),
                // That character and the run after it, up to its first
                // ideograph.
                Some(run) => (run + checked_run(&rest[run..]), false),
            },
            // A run at the start of the text: ideographs on their own, the
            // other characters up to the first ideograph.
            Some(_) => match ideograph_run(rest) {
                0 => (checked_run(rest), false),
                run => (run, true),
            },
        };
        at = start + len;
        Some((start, &text[start..at], settled))
    })
}

/// Whether `text` is in NFC by the tables `tokenizers` normalizes with: the
/// crate's check on each stretch that is not settled (the quick check
/// answers for most text; where it cannot, marks that might compose or
/// reorder, the stretch is normalized and compared), none on the ASCII and
/// ideograph ones.
fn is_nfc(text: &str) -> bool {
    stretches(text)
        .all(|(_, stretch, settled)| settled || unicode_normalization_alignments::is_nfc(stretch))
}

/// `text` in NFC by the tables `tokenizers` normalizes with, borrowed when
/// it is in NFC already: only a stretch that is not settled and not in NFC
/// goes through the crate's normalizer (its iterator reports a byte-offset
/// shift with each character, which this path has no use for), the other
/// stretches are copied. A stretch is read by the quick check where the
/// caller has one (its tables place most characters; the library reads the
/// windows they cannot), by the library's check otherwise.
fn nfc<'a>(text: &'a str, quick: Option<&NfcQuick>) -> Cow<'a, str> {
    let in_nfc = |stretch: &str| match quick {
        Some(quick) => quick.is_nfc(stretch),
        None => unicode_normalization_alignments::is_nfc(stretch),
    };
    let mut owned: Option<String> = None;
    for (start, stretch, settled) in stretches(text) {
        if settled || in_nfc(stretch) {
            if let Some(out) = &mut owned {
                out.push_str(stretch);
            }
        } else {
            let out = owned.get_or_insert_with(|| {
                let mut out = String::with_capacity(text.len() + 16);
                out.push_str(&text[..start]);
                out
            });
            out.extend(stretch.nfc().map(|(ch, _)| ch));
        }
    }
    owned.map_or(Cow::Borrowed(text), Cow::Owned)
}

/// GPT-2's `bytes_to_unicode`: the alphabet character for each byte.
fn bytes_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next_escape = 0x100u32;
    for (byte, slot) in table.iter_mut().enumerate() {
        let byte = byte as u8;
        let printable = (b'!'..=b'~').contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte);
        *slot = if printable {
            char::from(byte)
        } else {
            next_escape += 1;
            char::from_u32(next_escape - 1).unwrap_or('\0')
        };
    }
    table
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Normalizer {
    None,
    Nfc,
}

/// One `Split` of the pre-tokenizer chain.
struct Stage {
    pattern: Pattern,
    /// `Isolated`: the text between matches forms pieces too. Otherwise
    /// (`Removed` with `invert`) only the matches do.
    keep_gaps: bool,
}

/// Added tokens by content and id, the list a matcher is built from.
type AddedTokenList = Vec<(String, u32)>;

/// One set of added tokens, matched leftmost-longest as `AddedVocabulary`
/// matches them. The automaton's own search scans a text for the tokens'
/// first bytes when there are at most three of them, all ASCII; a set the
/// scan cannot take (a token starting beyond ASCII, or more than three
/// first bytes) it searches by the tokens' rarest bytes instead, which for a
/// vocabulary of a thousand `<` tokens and one fenced in fullwidth bars were
/// the bars' continuation bytes, frequent in CJK text: that search was a
/// third of a 2 KB encode (measured). Such a set takes its candidates from
/// the library's packed searcher over the tokens' distinct prefixes
/// ([`Prefixes`]), one pass with the prefix verified, and asks the automaton
/// anchored at each: nothing on text without the prefixes, the tokens
/// themselves on text with them.
pub(crate) struct AddedTokens {
    automaton: AhoCorasick,
    ids: Vec<u32>,
    prefixes: Option<Prefixes>,
}

/// The distinct prefixes of a set's tokens as a packed searcher: the longest
/// length up to [`Self::MAX_LEN`] bytes at which there are at most
/// [`Self::MAX_COUNT`] of them (a token shorter than the length stands
/// whole), so a hit is as close to a token as the set allows. A set whose
/// prefixes would be shorter than [`Self::MIN_LEN`] bytes keeps the
/// automaton's own search. When the library declines the prefixes that fit
/// (its SIMD searcher takes a one-byte pattern among at most sixteen, and a
/// set with newline or tab runs among its tokens has such prefixes), the
/// tokens' distinct first bytes stand in when there are at most
/// [`Self::MAX_FIRST_BYTES`]: a hit then is a byte that starts some token,
/// and the automaton anchored there decides, where the unanchored automaton
/// walked every byte of the text (measured at about thirty instructions per
/// byte on a 6,415-token set).
struct Prefixes {
    searcher: packed::Searcher,
}

impl Prefixes {
    const MAX_LEN: usize = 16;
    const MIN_LEN: usize = 3;
    const MAX_COUNT: usize = 64;
    const MAX_FIRST_BYTES: usize = 16;

    /// Whether the automaton's start-byte scan declines `tokens`: a first
    /// byte beyond ASCII, or more than three distinct first bytes.
    fn wanted(tokens: &[(String, u32)]) -> bool {
        let mut firsts: Vec<u8> = tokens
            .iter()
            .filter_map(|(content, _)| content.bytes().next())
            .collect();
        firsts.sort_unstable();
        firsts.dedup();
        firsts.len() > 3 || firsts.iter().any(|first| !first.is_ascii())
    }

    /// The searcher over the prefixes of `tokens`; none when the prefixes
    /// that fit are shorter than [`Self::MIN_LEN`] bytes for a token longer
    /// than them (a set of added words: the pass would stop at nearly every
    /// byte of ordinary text, where the automaton's own walk is cheaper),
    /// when even the one-byte prefixes exceed [`Self::MAX_COUNT`], or when
    /// the library declines the set.
    fn of(tokens: &[(String, u32)]) -> Option<Self> {
        let distinct = |len: usize| {
            let mut prefixes: Vec<&[u8]> = tokens
                .iter()
                .map(|(content, _)| &content.as_bytes()[..content.len().min(len)])
                .collect();
            prefixes.sort_unstable();
            prefixes.dedup();
            prefixes
        };
        let mut len = Self::MAX_LEN;
        let mut prefixes = distinct(len);
        while prefixes.len() > Self::MAX_COUNT && len > 1 {
            len -= 1;
            prefixes = distinct(len);
        }
        let cut_short = tokens.iter().any(|(content, _)| content.len() > len);
        if prefixes.len() > Self::MAX_COUNT || (cut_short && len < Self::MIN_LEN) {
            return None;
        }
        let searcher = Self::searcher(&prefixes).or_else(|| {
            let firsts = distinct(1);
            (firsts.len() <= Self::MAX_FIRST_BYTES)
                .then(|| Self::searcher(&firsts))
                .flatten()
        })?;
        Some(Self { searcher })
    }

    /// The library's packed searcher over `patterns`, none when it declines
    /// them.
    fn searcher(patterns: &[&[u8]]) -> Option<packed::Searcher> {
        packed::Config::new()
            .match_kind(packed::MatchKind::LeftmostFirst)
            .builder()
            .extend(patterns)
            .build()
    }

    /// The first position at or after `from` where a prefix stands.
    fn next(&self, text: &str, from: usize) -> Option<usize> {
        self.searcher
            .find_in(text, Span::from(from..text.len()))
            .map(|found| found.start())
    }
}

impl AddedTokens {
    /// The matcher over `tokens`, none for no tokens.
    pub(crate) fn build(tokens: &[(String, u32)]) -> Result<Option<Self>, Declined> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let prefixes = Prefixes::wanted(tokens)
            .then(|| Prefixes::of(tokens))
            .flatten();
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .start_kind(if prefixes.is_some() {
                StartKind::Anchored
            } else {
                StartKind::Unanchored
            })
            .build(tokens.iter().map(|(content, _)| content.as_bytes()))
            .map_err(|e| Declined::from(format!("the added tokens form no automaton: {e}")))?;
        Ok(Some(Self {
            automaton,
            ids: tokens.iter().map(|(_, id)| *id).collect(),
            prefixes,
        }))
    }

    /// The added tokens in `text`, in order, each as its byte range and its
    /// index in the list: the leftmost-longest matches, each continuing after
    /// the last. With prefixes, a token starts only where a prefix stands, so
    /// the first prefix at or after the last match that begins a token begins
    /// the next one, and the longest token there is the match.
    fn matches<'a>(&'a self, text: &'a str) -> impl Iterator<Item = (Range<usize>, usize)> + 'a {
        let mut at = 0;
        std::iter::from_fn(move || loop {
            let Some(prefixes) = &self.prefixes else {
                let input = Input::new(text).range(at..);
                let found = self.automaton.find(input)?;
                at = found.end();
                return Some((found.range(), found.pattern().as_usize()));
            };
            let start = prefixes.next(text, at)?;
            let input = Input::new(text).range(start..).anchored(Anchored::Yes);
            if let Some(found) = self.automaton.find(input) {
                at = found.end();
                return Some((found.range(), found.pattern().as_usize()));
            }
            at = start + 1;
        })
    }

    /// The byte ranges of the added tokens in `text`, in order.
    pub(crate) fn ranges<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Range<usize>> + 'a {
        self.matches(text).map(|(range, _)| range)
    }

    /// Calls `f` with each segment in order: an added token's id, or a text
    /// span between added tokens.
    fn split<'t>(&self, text: &'t str, mut f: impl FnMut(Segment<'t>)) {
        let mut cursor = 0;
        for (range, index) in self.matches(text) {
            if cursor < range.start {
                f(Segment::Text(&text[cursor..range.start]));
            }
            f(Segment::Added(self.ids[index]));
            cursor = range.end;
        }
        if cursor < text.len() {
            f(Segment::Text(&text[cursor..]));
        }
    }
}

#[derive(Clone, Copy)]
enum Segment<'t> {
    Text(&'t str),
    Added(u32),
}

/// Units an encode call keeps inline: a prompt without added tokens is one
/// unit, a chat turn a handful; a long text cut for the thread pool, or a
/// render with many added tokens, spills to the heap.
const INLINE_UNITS: usize = 8;

/// The units of one encode: up to [`INLINE_UNITS`] in place, the rest in a
/// Vec made at the first spill. The short-text path makes no allocation for
/// its units this way (measured: the units list was one of the two
/// allocations of a 256-byte or 2 KB encode, the other the ids returned).
struct Units<'t> {
    inline: [Segment<'t>; INLINE_UNITS],
    len: usize,
    spill: Vec<Segment<'t>>,
}

impl<'t> Units<'t> {
    fn new() -> Self {
        Self {
            inline: [Segment::Text(""); INLINE_UNITS],
            len: 0,
            spill: Vec::new(),
        }
    }

    fn push(&mut self, unit: Segment<'t>) {
        if !self.spill.is_empty() {
            self.spill.push(unit);
        } else if self.len < INLINE_UNITS {
            self.inline[self.len] = unit;
            self.len += 1;
        } else {
            self.spill.reserve(2 * INLINE_UNITS);
            self.spill.extend_from_slice(&self.inline);
            self.spill.push(unit);
        }
    }

    fn as_slice(&self) -> &[Segment<'t>] {
        if self.spill.is_empty() {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }
}

/// A raw added token, or a text between raw added tokens in NFC: borrowed
/// where it was in NFC already, owned where normalizing changed it (see
/// [`NativeEncoder::nfc_gaps`]).
enum Gap<'t> {
    Text(Cow<'t, str>),
    Added(u32),
}

/// The special-token ids the post-processor puts around the one sequence
/// of an encode when special tokens are requested: a `TemplateProcessing`'s
/// single template, read at load (see [`NativeEncoder::frame_from`]); empty
/// for a post-processor that adds none.
#[derive(Debug, Default, PartialEq, Eq)]
struct Frame {
    before: Vec<u32>,
    after: Vec<u32>,
}

/// A `TemplateProcessing` read for its frame: the ids, and whether its
/// single template has special-token pieces at all (a piece whose id list
/// is empty adds no id, but the library still makes an encoding of it, see
/// [`NativeEncoder::frame_from`]).
struct Template {
    frame: Frame,
    special_pieces: bool,
}

/// The native encode path of one tokenizer; see the module documentation.
pub(crate) struct NativeEncoder {
    normalizer: Normalizer,
    /// The NFC quick check's tables, read when the normalizer is NFC: the
    /// check every encode under it starts with (see [`Self::in_nfc`]).
    nfc_quick: Option<NfcQuick>,
    /// Added tokens matched on the raw text (`normalized: false`).
    raw_added: Option<AddedTokens>,
    /// Added tokens matched after normalization (`normalized: true`).
    normalized_added: Option<AddedTokens>,
    stages: Vec<Stage>,
    /// The one-pass matcher of the stages, when they are a chain it covers
    /// (the stages stay for the cut checks and as the matcher of record).
    chain: Option<chain::Scanner>,
    /// What a request for special tokens adds around the ids.
    frame: Frame,
    alphabet: [char; 256],
    /// The vocabulary's own pieces with their ids, read once the model's word
    /// cache is bounded and consulted by a thread for a piece its own cache
    /// has not met, before the piece is merged (see
    /// [`Self::seed_vocabulary`]).
    seed: ShortPieceTable,
    /// Whether the seed holds every vocabulary entry of up to
    /// `SHORT_PIECE_MAX_BYTES` raw bytes: the ids `0..vocab_size` were all
    /// entries when it was read. Then a short piece the seed misses is no
    /// entry, and the whole-piece lookup `ignore_merges` asks for is spared
    /// (see [`Self::tokenize_piece`]). A vocabulary with gaps in its ids has
    /// entries past that range the seed did not read, and keeps the lookup.
    seed_covers_short_entries: bool,
    /// The vocabulary's entries longer than the seed's keys as a filter,
    /// read with the seed when the pass read every entry (`None` on the
    /// paths without the pass, on the sentencepiece path, whose pass reads
    /// no long entry, and for a vocabulary with gaps, whose entries past the
    /// range went unread): the whole-piece lookup `ignore_merges` asks for
    /// on a long piece is spared when the filter refuses it (see
    /// `long_entries` and [`Self::tokenize_piece`]).
    long_entries: Option<LongEntryFilter>,
    /// The model's merges as a table a piece neither the thread's cache nor
    /// the seed holds is merged over; `None` for a model the table cannot
    /// stand in for, whose such pieces the model itself tokenizes.
    merges: Option<MergeTable>,
    /// The tables of the sentencepiece path, for a vocabulary with byte
    /// fallback (see [`sentencepiece`]): the stages are empty and `merges`
    /// is `None` then, the pieces come from character symbols. Boxed, so
    /// the byte-level encoder keeps the layout it had.
    chars: Option<Box<CharSymbols>>,
    /// Each thread's cache from piece bytes to ids. The encoder owns them, so
    /// they go when it does: tokenizers are added and removed at runtime, and
    /// a thread never has to notice that an encoder it encoded with is gone.
    piece_cache: ThreadLocal<RefCell<PieceCache>>,
    /// Threads that hold a piece cache of this encoder.
    threads: AtomicUsize,
}

impl std::fmt::Debug for NativeEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEncoder")
            .field("normalizer", &self.normalizer)
            .field("nfc_quick", &self.nfc_quick.is_some())
            .field("stages", &self.stages.len())
            .field("seeded_pieces", &self.seed.len())
            .field("native_merge", &self.merges.is_some())
            .field("sentencepiece", &self.chars.is_some())
            .field("chain", &self.chain.is_some())
            .field("frame", &self.frame)
            .finish()
    }
}

/// A component of the tokenizer (`what` names it in a decline) in its
/// serialized form, `Null` when there is none: the shape the checks read.
fn serialized<T: Serialize>(what: &str, component: T) -> Result<Value, Declined> {
    serde_json::to_value(component)
        .map_err(|e| Declined::from(format!("the {what} does not serialize: {e}")))
}

/// The `type` a serialized component (`what` names it) declares.
fn type_of<'v>(what: &str, component: &'v Value) -> Result<&'v str, Declined> {
    component
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| Declined::from(format!("a {what} without a type")))
}

impl NativeEncoder {
    /// This path's encoder for `tokenizer`, or why it declines: a shape
    /// outside what the module documentation lists. Its seed is empty until
    /// [`Self::seed_vocabulary`] reads the vocabulary.
    pub(crate) fn from_tokenizer<M: Model + ModelView>(
        tokenizer: &TokenizerOf<M>,
    ) -> Result<Self, Declined> {
        if tokenizer.get_truncation().is_some() {
            return Err(Declined::from("truncation is configured"));
        }
        if tokenizer.get_padding().is_some() {
            return Err(Declined::from("padding is configured"));
        }
        let ModelWrapper::BPE(bpe) = tokenizer.get_model().wrapper() else {
            return Err(Declined::from("the model is not BPE"));
        };
        let pre_tokenizer = serialized("pre-tokenizer", tokenizer.get_pre_tokenizer())?;
        // A vocabulary with byte fallback and no `ByteLevel` stage is a
        // sentencepiece-style one; a byte-level vocabulary whose file sets
        // the flag keeps the byte-level path.
        if bpe.byte_fallback && !Self::ends_in_byte_level(&pre_tokenizer) {
            return Self::sentencepiece_from(tokenizer, &pre_tokenizer);
        }
        let normalizer =
            Self::normalizer_from(&serialized("normalizer", tokenizer.get_normalizer())?)?;
        let stages = Self::stages_from(&pre_tokenizer)?;
        let frame = Self::frame_from(&serialized(
            "post-processor",
            tokenizer.get_post_processor(),
        )?)?;
        let (raw, normalized) = Self::added_tokens_of(tokenizer, normalizer)?;
        let merges = match MergeTable::from_model(
            tokenizer.get_model().wrapper(),
            ByteTokens::Alphabet,
            tokenizer.get_model().read(),
        ) {
            Ok(table) => {
                tracing::debug!(
                    merges = table.merges(),
                    bytes = table.bytes(),
                    preformed_characters = table.preformed_characters(),
                    "the native encode path merges its piece-cache misses itself"
                );
                Some(table)
            }
            Err(declined) => {
                tracing::info!(
                    %declined,
                    "the native encode path hands its piece-cache misses to the model's own merge"
                );
                None
            }
        };
        let chain = chain::Scanner::recognize(&stages);
        Ok(Self {
            nfc_quick: (normalizer == Normalizer::Nfc).then(NfcQuick::read),
            normalizer,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            stages,
            chain,
            frame,
            alphabet: bytes_char(),
            seed: ShortPieceTable::for_entries(0),
            seed_covers_short_entries: false,
            long_entries: None,
            merges,
            chars: None,
            piece_cache: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
        })
    }

    /// This path's encoder for a vocabulary with byte fallback (see
    /// [`sentencepiece`]), or why it declines: the spaces of the text become
    /// the word marker by a `Replace` normalizer alone (a `Split` on the
    /// space behind it cuts nothing then, and is accepted) or by the
    /// `Metaspace` pre-tokenizer of the Llama class with no normalizer, whose
    /// prepend scheme places the marker; anything else declines, as does an
    /// added token matched after the `Replace` with a space or a marker in
    /// it (matched on the raw text here, its spelling would not be the one
    /// `tokenizers` matches after the normalizer).
    fn sentencepiece_from<M: Model + ModelView>(
        tokenizer: &TokenizerOf<M>,
        pre_tokenizer: &Value,
    ) -> Result<Self, Declined> {
        let normalizer = serialized("normalizer", tokenizer.get_normalizer())?;
        let (marker, replaces) = Self::marker_from(&normalizer, pre_tokenizer)?;
        let frame = Self::frame_from(&serialized(
            "post-processor",
            tokenizer.get_post_processor(),
        )?)?;
        let (raw, normalized) = Self::added_tokens_of(tokenizer, Normalizer::None)?;
        if replaces
            && normalized
                .iter()
                .any(|(content, _)| content.contains([' ', WORD_MARKER]))
        {
            return Err(Declined::from(
                "a byte-fallback vocabulary with an added token matched after the Replace \
                 normalizer that holds a space or the word marker",
            ));
        }
        let chars = CharSymbols::from_model(
            tokenizer.get_model().wrapper(),
            marker,
            tokenizer.get_model().read(),
        )?;
        tracing::debug!(
            merges = chars.merges(),
            bytes = chars.bytes(),
            ?marker,
            "the native encode path takes the byte-fallback vocabulary from character symbols"
        );
        Ok(Self {
            normalizer: Normalizer::None,
            nfc_quick: None,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            stages: Vec::new(),
            chain: None,
            frame,
            alphabet: bytes_char(),
            seed: ShortPieceTable::for_entries(0),
            seed_covers_short_entries: false,
            long_entries: None,
            merges: None,
            chars: Some(Box::new(chars)),
            piece_cache: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
        })
    }

    /// Whether a pre-tokenizer (serialized) ends in a `ByteLevel` stage, the
    /// shape of the byte-level path ([`Self::stages_from`]).
    fn ends_in_byte_level(pre_tokenizer: &Value) -> bool {
        let last = match pre_tokenizer.get("pretokenizers").and_then(Value::as_array) {
            Some(items) => items.last(),
            None => Some(pre_tokenizer),
        };
        last.and_then(|item| item.get("type"))
            .and_then(Value::as_str)
            == Some("ByteLevel")
    }

    /// Where the word marker comes from on a byte-fallback vocabulary, and
    /// whether a `Replace` normalizer puts it for the spaces (true in the
    /// answer; the `Metaspace` pre-tokenizer does so itself): `Never` under
    /// the normalizer, the scheme of the pre-tokenizer without one; or why
    /// the pipeline declines.
    fn marker_from(normalizer: &Value, pre_tokenizer: &Value) -> Result<(Marker, bool), Declined> {
        let replace = serde_json::json!(
            {"type": "Replace", "pattern": {"String": " "}, "content": WORD_MARKER.to_string()}
        );
        let replaces = *normalizer == replace;
        if !(normalizer.is_null() || replaces) {
            return Err(Declined::from(
                "a byte-fallback vocabulary with a normalizer other than Replace(space, marker) \
                 or none",
            ));
        }
        if pre_tokenizer.is_null() {
            return if replaces {
                Ok((Marker::Never, true))
            } else {
                Err(Declined::from(
                    "a byte-fallback vocabulary with neither a Replace normalizer nor a Metaspace \
                     pre-tokenizer",
                ))
            };
        }
        let field = |name: &str| pre_tokenizer.get(name);
        match type_of("pre-tokenizer", pre_tokenizer)? {
            // No space is left after the normalizer: the split cuts nothing.
            "Split"
                if replaces
                    && field("pattern") == Some(&serde_json::json!({"String": " "}))
                    && field("behavior").and_then(Value::as_str) == Some("MergedWithPrevious")
                    && field("invert").and_then(Value::as_bool) == Some(false) =>
            {
                Ok((Marker::Never, true))
            }
            "Metaspace"
                if !replaces
                    && field("replacement").and_then(Value::as_str) == Some("\u{2581}")
                    && field("split").and_then(Value::as_bool) == Some(false) =>
            {
                match field("prepend_scheme").and_then(Value::as_str) {
                    Some("never") => Ok((Marker::Never, false)),
                    Some("first") => Ok((Marker::First, false)),
                    Some("always") => Ok((Marker::Always, false)),
                    other => Err(Declined::from(format!(
                        "a Metaspace pre-tokenizer with the prepend scheme {other:?}"
                    ))),
                }
            }
            other => Err(Declined::from(format!(
                "a byte-fallback vocabulary with the pre-tokenizer {other} (only none or a Split \
                 on the space after the Replace normalizer, or Metaspace without split)"
            ))),
        }
    }

    /// The normalizer this path applies: none, or NFC alone.
    fn normalizer_from(normalizer: &Value) -> Result<Normalizer, Declined> {
        if normalizer.is_null() {
            return Ok(Normalizer::None);
        }
        match type_of("normalizer", normalizer)? {
            "NFC" => Ok(Normalizer::Nfc),
            "Sequence" => {
                let steps = normalizer
                    .get("normalizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Declined::from("a normalizer sequence without a list"))?;
                if steps.is_empty() {
                    Ok(Normalizer::None)
                } else {
                    Err(Declined::from(
                        "a normalizer sequence with steps (only NFC alone or none)",
                    ))
                }
            }
            other => Err(Declined::from(format!(
                "normalizer {other} (only NFC or none)"
            ))),
        }
    }

    /// What the post-processor adds around the ids on request: nothing for
    /// none or `ByteLevel` (which reshapes offsets), a `TemplateProcessing`'s
    /// single template ([`Self::template_frame`]), and for a `Sequence` of
    /// those the one template among its members with special-token pieces:
    /// `ByteLevel` members add nothing in any position, a template of the
    /// `$A` sequence alone passes the one encoding through, and any template
    /// after one with pieces declines, since `Sequence::process_encodings`
    /// hands that one's pieces on unmerged (a piece with an empty id list
    /// included): two of them take the next template's pair form, more hit
    /// a `todo!()` in the library. A sequence inside a sequence and any
    /// other kind decline.
    fn frame_from(post_processor: &Value) -> Result<Frame, Declined> {
        if post_processor.is_null() {
            return Ok(Frame::default());
        }
        match type_of("post-processor", post_processor)? {
            "ByteLevel" => Ok(Frame::default()),
            "TemplateProcessing" => {
                Self::template_frame(post_processor).map(|template| template.frame)
            }
            "Sequence" => {
                let items = post_processor
                    .get("processors")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Declined::from("a post-processor sequence without a list"))?;
                let mut frame = Frame::default();
                let mut special_pieces = false;
                for item in items {
                    match type_of("post-processor", item)? {
                        "ByteLevel" => {}
                        "TemplateProcessing" if special_pieces => {
                            return Err(Declined::from(
                                "a post-processor sequence with a template after one with \
                                 special-token pieces",
                            ));
                        }
                        "TemplateProcessing" => {
                            let template = Self::template_frame(item)?;
                            frame = template.frame;
                            special_pieces = template.special_pieces;
                        }
                        "Sequence" => {
                            return Err(Declined::from(
                                "a post-processor sequence inside a sequence",
                            ));
                        }
                        other => {
                            return Err(Declined::from(format!(
                                "post-processor {other} in a sequence (only ByteLevel or \
                                 TemplateProcessing)"
                            )));
                        }
                    }
                }
                Ok(frame)
            }
            other => Err(Declined::from(format!(
                "post-processor {other} (only ByteLevel, TemplateProcessing or none)"
            ))),
        }
    }

    /// The frame of a `TemplateProcessing`: the special-token pieces of its
    /// single template before the `$A` sequence and after it, each the ids
    /// its entry of `special_tokens` lists (the pair template is never
    /// applied to one sequence), and whether it has such pieces at all. A
    /// template the library's builder would refuse but its file reader
    /// takes as it is (no `$A`, two of them, a `$B`, a token missing from
    /// the map) declines, whatever the library makes of it.
    fn template_frame(post_processor: &Value) -> Result<Template, Declined> {
        let single = post_processor
            .get("single")
            .and_then(Value::as_array)
            .ok_or_else(|| Declined::from("a TemplateProcessing without a single template"))?;
        let tokens = post_processor.get("special_tokens");
        let mut frame = Frame::default();
        let mut special_pieces = false;
        let mut sequences = 0;
        for piece in single {
            if let Some(sequence) = piece.get("Sequence") {
                if sequence.get("id").and_then(Value::as_str) != Some("A") {
                    return Err(Declined::from(
                        "a TemplateProcessing single template with a $B sequence",
                    ));
                }
                sequences += 1;
                continue;
            }
            let name = piece
                .get("SpecialToken")
                .and_then(|token| token.get("id"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    Declined::from(
                        "a TemplateProcessing piece neither a sequence nor a special token",
                    )
                })?;
            special_pieces = true;
            let ids = tokens
                .and_then(|tokens| tokens.get(name))
                .and_then(|token| token.get("ids"))
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    Declined::from(format!(
                        "a TemplateProcessing special token {name:?} missing from its map"
                    ))
                })?;
            let side = if sequences == 0 {
                &mut frame.before
            } else {
                &mut frame.after
            };
            for id in ids {
                side.push(
                    id.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| {
                            Declined::from(format!(
                                "a TemplateProcessing special token {name:?} with an id that is not a u32"
                            ))
                        })?,
                );
            }
        }
        if sequences != 1 {
            return Err(Declined::from(
                "a TemplateProcessing single template without one $A sequence",
            ));
        }
        Ok(Template {
            frame,
            special_pieces,
        })
    }

    /// The added tokens the path cuts out itself, those matched on the raw
    /// text and those matched after normalization (normalized here, once),
    /// each in a stable order so leftmost-longest ties stay deterministic.
    fn added_tokens_of<M: Model>(
        tokenizer: &TokenizerOf<M>,
        normalizer: Normalizer,
    ) -> Result<(AddedTokenList, AddedTokenList), Declined> {
        let added = tokenizer.get_added_vocabulary();
        if added.get_encode_special_tokens() {
            return Err(Declined::from("encode_special_tokens is set"));
        }
        let mut raw = Vec::new();
        let mut normalized = Vec::new();
        for (&id, token) in added.get_added_tokens_decoder() {
            if token.single_word || token.lstrip || token.rstrip || token.content.is_empty() {
                return Err(Declined::from(format!(
                    "added token {:?} with single_word, lstrip or rstrip, or without content",
                    token.content
                )));
            }
            if token.normalized {
                let content = match normalizer {
                    Normalizer::None => token.content.clone(),
                    Normalizer::Nfc => nfc(&token.content, None).into_owned(),
                };
                normalized.push((content, id));
            } else {
                raw.push((token.content.clone(), id));
            }
        }
        raw.sort();
        normalized.sort();
        Ok((raw, normalized))
    }

    /// The vocabulary's own pieces with their ids, the table every thread
    /// reads before its own cache: each entry of the model's vocabulary whose
    /// spelling maps back from the byte-level alphabet to 1..=15 raw bytes,
    /// with the ids a cache miss on the piece computes: its own id under
    /// `ignore_merges` (the whole-piece lookup finds an entry before any
    /// merge), else the merge of its symbols over the table the misses merge
    /// over (the model's ids by the table's contract, several for an entry
    /// the merges do not reach); through the model itself on a path without
    /// a table of its own. An entry with a character outside the alphabet is
    /// no byte-level piece; one the model declines, or whose ids the table
    /// cannot hold, is left to the miss path. A model with merge dropout
    /// tokenizes at random and gets no seed. The entries are read by id, on
    /// the caller's rayon pool when the load runs on one, else on a pool of
    /// `SEED_THREADS` threads that is gone when the read is; without a copy
    /// of the vocabulary either way.
    ///
    /// Called once the model's word cache is bounded (see `word_cache`): the
    /// words a seed read through the model merges would otherwise fill that
    /// cache on every thread of the pool, and the crate never clears a
    /// thread's cache of a dropped model.
    pub(crate) fn seed_vocabulary(&mut self, model: &ModelWrapper, listing: Option<&Listing>) {
        (self.seed, self.seed_covers_short_entries, self.long_entries) = Self::vocabulary_seed(
            model,
            listing,
            &self.alphabet,
            self.merges.as_ref(),
            self.chars.as_deref(),
        );
    }

    /// The table [`Self::seed_vocabulary`] reads, and whether every id of
    /// `0..vocab_size` was an entry (so the table holds every short entry).
    /// On the sentencepiece path (`chars`) an entry's spelling is its own
    /// text: its key is its raw form (`CharSymbols::entry_key`). `merges` is
    /// the byte-level path's own table, when it has one.
    fn vocabulary_seed(
        model: &ModelWrapper,
        listing: Option<&Listing>,
        alphabet: &[char; 256],
        merges: Option<&MergeTable>,
        chars: Option<&CharSymbols>,
    ) -> (ShortPieceTable, bool, Option<LongEntryFilter>) {
        let ModelWrapper::BPE(bpe) = model else {
            return (ShortPieceTable::for_entries(0), false, None);
        };
        if bpe.dropout.is_some_and(|dropout| dropout > 0.0) {
            return (ShortPieceTable::for_entries(0), false, None);
        }
        let started = Instant::now();
        // The byte each alphabet character stands for, by character.
        let mut byte_of = vec![None; alphabet.iter().map(|&c| c as usize).max().unwrap_or(0) + 1];
        for (byte, &c) in alphabet.iter().enumerate() {
            byte_of[c as usize] = Some(byte as u8);
        }
        // The ids of one entry (`piece` its raw bytes on the byte-level
        // path): what a miss on it computes, see above. The model is asked
        // only where the path has no table: it allocates a word and a token
        // list per call.
        let ids_of = |id: u32, entry: &str, piece: &[u8], scratch: &mut SeedScratch| {
            let mut ids = SeededIds::default();
            if bpe.ignore_merges {
                ids.push(id)?;
                return Some(ids);
            }
            scratch.ids.clear();
            match (chars, merges) {
                (Some(chars), _) => chars.merge(entry, &mut scratch.symbols, &mut scratch.ids),
                (None, Some(table)) => table.merge(piece, &mut scratch.ids),
                (None, None) => {
                    for token in &model.tokenize(entry).ok()? {
                        ids.push(token.id)?;
                    }
                    return Some(ids);
                }
            }
            for &id in &scratch.ids {
                ids.push(id)?;
            }
            Some(ids)
        };
        // One id's read: `None` for an id that is no entry, else the seeded
        // piece, the bytes of a long byte-level piece, or `None` inside for
        // an entry that is no byte-level piece.
        let read = |id: u32, scratch: &mut SeedScratch| -> Option<Option<Seeded>> {
            let owned;
            let entry: &str = match listing {
                Some(listing) => listing.get(id)?,
                None => {
                    owned = bpe.id_to_token(id)?;
                    &owned
                }
            };
            let seeded = (|| {
                if chars.is_some() {
                    let key = CharSymbols::entry_key(entry)?;
                    return Some(Seeded::Short(key, ids_of(id, entry, &[], scratch)?));
                }
                let mut piece = [0u8; SHORT_PIECE_MAX_BYTES];
                let mut len = 0;
                // The bytes of an entry past the key's length, with the
                // key's bytes in front: a long entry for the filter.
                let mut long = Vec::new();
                for c in entry.chars() {
                    let byte = byte_of.get(c as usize).copied().flatten()?;
                    if len < SHORT_PIECE_MAX_BYTES {
                        piece[len] = byte;
                        len += 1;
                    } else {
                        if long.is_empty() {
                            long.extend_from_slice(&piece);
                        }
                        long.push(byte);
                    }
                }
                if !long.is_empty() {
                    return Some(Seeded::Long(long.into_boxed_slice()));
                }
                let key = ShortPieceKey::of(&piece[..len])?;
                Some(Seeded::Short(
                    key,
                    ids_of(id, entry, &piece[..len], scratch)?,
                ))
            })();
            Some(seeded)
        };
        let ids = 0..bpe.get_vocab_size() as u32;
        // One slot per id, written in place (the read keeps its index, so the
        // collect is one list and no concatenation); one scratch per rayon job
        // (2048 entries or more).
        let read_all = || {
            ids.clone()
                .into_par_iter()
                .with_min_len(2048)
                .map_init(SeedScratch::default, |scratch, id| read(id, scratch))
                .collect::<Vec<Option<Option<Seeded>>>>()
        };
        // A caller that is a worker of a rayon pool already (the Python
        // binding loads on its own, sized by its host) reads on that pool,
        // which never builds the global one. Any other caller reads on this
        // load's own pool: its threads start here and are joined before the
        // entries are returned, so the load leaves no pool behind; a process
        // that cannot spawn them reads on the calling thread and says so.
        // (The name fits the fifteen bytes a thread's name keeps.)
        let (entries, threads) = if rayon::current_thread_index().is_some() {
            (read_all(), rayon::current_num_threads())
        } else {
            let pooled = rayon::ThreadPoolBuilder::new()
                .num_threads(SEED_THREADS)
                .thread_name(|index| format!("vocab-seed-{index}"))
                .build_scoped(|thread| thread.run(), |pool| pool.install(read_all));
            match pooled {
                Ok(entries) => (entries, SEED_THREADS),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "could not build the vocabulary seed's pool; reading on the calling thread"
                    );
                    let mut scratch = SeedScratch::default();
                    (ids.map(|id| read(id, &mut scratch)).collect::<Vec<_>>(), 1)
                }
            }
        };
        // An id that was an entry has a slot that is `Some`; the seeded pieces
        // are the inner `Some`s.
        let covers = entries.iter().filter(|read| read.is_some()).count() == bpe.get_vocab_size();
        let short = entries
            .iter()
            .flatten()
            .flatten()
            .filter(|seeded| matches!(seeded, Seeded::Short(..)))
            .count();
        let long = entries.iter().flatten().flatten().count() - short;
        let mut table = ShortPieceTable::for_entries(short);
        let mut filter = LongEntryFilter::for_entries(long);
        for seeded in entries.iter().flatten().flatten() {
            match seeded {
                Seeded::Short(key, ids) => {
                    table.insert(*key, ids.as_slice(), short);
                }
                Seeded::Long(bytes) => filter.insert(bytes),
            }
        }
        let filter = (covers && chars.is_none()).then_some(filter);
        tracing::debug!(
            entries = table.len(),
            slots = table.slots(),
            long_entries = long,
            long_entry_bytes = filter.as_ref().map_or(0, LongEntryFilter::bytes),
            covers_short_entries = covers,
            threads,
            ms = started.elapsed().as_millis() as u64,
            "the native encoder's vocabulary pieces"
        );
        (table, covers, filter)
    }

    /// This path's encoder for a tiktoken vocabulary, `ranks` as the file
    /// lists them, `pattern` the file's regex and `specials` its special
    /// tokens, or why it declines: a pattern outside the engine's subset, a
    /// rank set that gives no complete merge table (see
    /// [`MergeTable::from_ranks`]), a special token that is a prefix of
    /// another. The file's encoder cuts the specials out
    /// of the raw text, runs the pattern over the text between them and
    /// encodes each match: no normalizer, one stage whose matches alone are
    /// pieces, no frame. The seed is ready at once: every token of a
    /// complete table is its own id.
    pub(crate) fn from_ranks(
        ranks: &FxHashMap<Vec<u8>, u32>,
        pattern: &str,
        specials: &[(String, u32)],
    ) -> Result<Self, Declined> {
        let pattern = Pattern::parse(pattern).ok_or_else(|| {
            Declined::from(format!(
                "a split pattern outside the supported regex subset: {pattern}"
            ))
        })?;
        if let Some((content, _)) = specials.iter().find(|(content, _)| content.is_empty()) {
            return Err(Declined::from(format!(
                "special token {content:?} without content"
            )));
        }
        // The file's own encoder matches its special tokens through an
        // alternation in no fixed order, leftmost-first; this path matches
        // leftmost-longest. The two agree unless one special is a prefix of
        // another (in sorted order, right after it).
        let mut contents: Vec<&str> = specials
            .iter()
            .map(|(content, _)| content.as_str())
            .collect();
        contents.sort_unstable();
        if let Some(pair) = contents
            .windows(2)
            .find(|pair| pair[1].starts_with(pair[0]))
        {
            return Err(Declined::from(format!(
                "special token {:?} is a prefix of {:?}",
                pair[0], pair[1]
            )));
        }
        let merges = MergeTable::from_ranks(ranks)?;
        tracing::debug!(
            merges = merges.merges(),
            bytes = merges.bytes(),
            preformed_characters = merges.preformed_characters(),
            "the native encode path merges a tiktoken vocabulary's piece-cache misses itself"
        );
        let stages = vec![Stage {
            pattern,
            keep_gaps: false,
        }];
        let chain = chain::Scanner::recognize(&stages);
        Ok(Self {
            normalizer: Normalizer::None,
            raw_added: AddedTokens::build(specials)?,
            normalized_added: None,
            nfc_quick: None,
            stages,
            chain,
            frame: Frame::default(),
            alphabet: bytes_char(),
            seed: Self::ranks_seed(ranks),
            seed_covers_short_entries: false,
            long_entries: None,
            merges: Some(merges),
            chars: None,
            piece_cache: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
        })
    }

    /// The seed of a tiktoken vocabulary: every token the key holds, with
    /// its own id (what a miss on it merges to, the table being complete).
    fn ranks_seed(ranks: &FxHashMap<Vec<u8>, u32>) -> ShortPieceTable {
        let mut table = ShortPieceTable::for_entries(ranks.len());
        for (token, &rank) in ranks {
            if let Some(key) = ShortPieceKey::of(token) {
                table.insert(key, &[rank], ranks.len());
            }
        }
        table
    }

    /// The stages of a pre-tokenizer that is zero or more `Split`s (or
    /// `Digits`, a split on numeric characters) followed by one `ByteLevel`
    /// without prefix space.
    fn stages_from(pre_tokenizer: &Value) -> Result<Vec<Stage>, Declined> {
        if pre_tokenizer.is_null() {
            return Err(Declined::from(
                "no pre-tokenizer (the path needs ByteLevel, after any Split stages)",
            ));
        }
        let items: Vec<&Value> = match type_of("pre-tokenizer", pre_tokenizer)? {
            "Sequence" => pre_tokenizer
                .get("pretokenizers")
                .and_then(Value::as_array)
                .ok_or_else(|| Declined::from("a pre-tokenizer sequence without a list"))?
                .iter()
                .collect(),
            _ => vec![pre_tokenizer],
        };
        let (byte_level, splits) = items
            .split_last()
            .ok_or_else(|| Declined::from("an empty pre-tokenizer sequence"))?;
        if byte_level.get("type").and_then(Value::as_str) != Some("ByteLevel")
            || byte_level.get("add_prefix_space").and_then(Value::as_bool) != Some(false)
        {
            return Err(Declined::from(
                "the pre-tokenizer does not end in ByteLevel without a prefix space",
            ));
        }
        let mut stages = Vec::with_capacity(splits.len() + 1);
        for split in splits {
            stages.push(Self::split_stage(split)?);
        }
        if byte_level.get("use_regex").and_then(Value::as_bool) == Some(true) {
            stages.push(Stage {
                pattern: Pattern::parse(BYTE_LEVEL_PATTERN)
                    .ok_or_else(|| Declined::from("the ByteLevel split pattern"))?,
                keep_gaps: true,
            });
        }
        Ok(stages)
    }

    /// One `Split` before the `ByteLevel` as a stage: a regex or a literal
    /// the pattern engine takes, `Isolated`, or `Removed` inverted. A
    /// `Digits` is the `Isolated` split on numeric characters, each one a
    /// piece under `individual_digits`, else each run (the reference splits
    /// on `char::is_numeric`, the `\p{N}` category).
    fn split_stage(split: &Value) -> Result<Stage, Declined> {
        match type_of("pre-tokenizer stage", split)? {
            "Split" => {}
            "Digits" => {
                let each = split
                    .get("individual_digits")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let pattern = if each { r"\p{N}" } else { r"\p{N}+" };
                return Pattern::parse(pattern)
                    .map(|pattern| Stage {
                        pattern,
                        keep_gaps: true,
                    })
                    .ok_or_else(|| Declined::from("the Digits split pattern"));
            }
            other => {
                return Err(Declined::from(format!(
                    "pre-tokenizer {other} before the ByteLevel (only Split)"
                )))
            }
        }
        let pattern = split
            .get("pattern")
            .ok_or_else(|| Declined::from("a Split without a pattern"))?;
        let regex = match (pattern.get("Regex"), pattern.get("String")) {
            (Some(regex), None) => regex
                .as_str()
                .ok_or_else(|| Declined::from("a Split whose Regex is not a string"))?
                .to_owned(),
            (None, Some(literal)) => escape_literal(
                literal
                    .as_str()
                    .ok_or_else(|| Declined::from("a Split whose String is not a string"))?,
            ),
            _ => return Err(Declined::from("a Split pattern neither Regex nor String")),
        };
        let keep_gaps = match (
            split.get("behavior").and_then(Value::as_str),
            split.get("invert").and_then(Value::as_bool),
        ) {
            (Some("Isolated"), Some(false)) => true,
            (Some("Removed"), Some(true)) => false,
            (behavior, invert) => {
                return Err(Declined::from(format!(
                    "Split behavior {behavior:?} with invert {invert:?} (only Isolated, or \
                     Removed inverted)"
                )))
            }
        };
        let pattern = Pattern::parse(&regex).ok_or_else(|| {
            Declined::from(format!(
                "a split pattern outside the supported regex subset: {regex}"
            ))
        })?;
        Ok(Stage { pattern, keep_gaps })
    }

    /// The ids of `text`, framed by the post-processor's special tokens when
    /// `add_special_tokens` is set, or why this text goes through
    /// `tokenizers` instead, for the caller to count: a piece the model
    /// declines, found after the pieces before it were tokenized, so that
    /// the error surfaces from the `tokenizers` encode. A long input is
    /// encoded on the thread pool, unit by unit (see [`Self::units`]), with
    /// the same code per unit, when the pool has more than one thread (see
    /// [`Self::worth_the_pool`]). `model` is the model behind the vocabulary,
    /// read for the whole-piece lookup of `ignore_merges` and for the
    /// pieces of a model whose merge table declined; `None` for a
    /// vocabulary read from ranks ([`Self::from_ranks`]), whose table is
    /// complete.
    pub(crate) fn encode(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, Fallback> {
        self.encode_with(model, text, Self::worth_the_pool(text), add_special_tokens)
    }

    /// Whether `text` is encoded through the thread pool: long enough
    /// (`PARALLEL_MIN_BYTES`) and a pool with more than one thread to cut
    /// it for. A pool of one thread (a process pinned to one core, a
    /// container with one CPU) encodes the units one after the other and
    /// pays the cut, the Vec per unit and the join for nothing: 5 % of the
    /// cycles and 140 allocations per encode of 2 MB on one core, measured.
    fn worth_the_pool(text: &str) -> bool {
        text.len() >= PARALLEL_MIN_BYTES && rayon::current_num_threads() > 1
    }

    /// The ids of `text` with no added token cut out of it and no frame:
    /// every spelling in it, a special token's included, goes through the
    /// stages and the model as text. The plain segments of a prompt
    /// rendered by segments are encoded this way (see `encoders::marks`),
    /// `at_start` for the one that starts the prompt (a segment after a
    /// control is not, which the sentencepiece path's `First` marker scheme
    /// reads); otherwise as [`Self::encode`].
    pub(crate) fn encode_plain(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        at_start: bool,
    ) -> Result<Vec<u32>, Fallback> {
        self.encode_units(
            model,
            text,
            Self::worth_the_pool(text),
            false,
            false,
            at_start,
        )
    }

    /// [`Self::encode`] with the thread pool used or not; the ids are the
    /// same either way (the units are encoded by the same code in the same
    /// order).
    pub(crate) fn encode_with(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        parallel: bool,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, Fallback> {
        self.encode_units(model, text, parallel, true, add_special_tokens, true)
    }

    /// The encode behind [`Self::encode_with`] and [`Self::encode_plain`]:
    /// `added` says whether the added tokens are cut out of the text first,
    /// `framed` whether the post-processor's frame goes around the ids,
    /// `at_start` whether the text starts the input (see
    /// [`Self::marker_before`]).
    fn encode_units(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        parallel: bool,
        added: bool,
        framed: bool,
        at_start: bool,
    ) -> Result<Vec<u32>, Fallback> {
        // The scanners' word scans, chosen once per call on the input's
        // length (`WORD_SCAN_MIN_BYTES`): every unit of a long input takes
        // them, a short input walks its runs byte by byte.
        let words = text.len() >= WORD_SCAN_MIN_BYTES;
        // Text not in NFC under the NFC normalizer (by the library's tables,
        // as every NFC read here) is normalized gap by gap first; the units
        // then borrow from those gaps.
        let gaps = (self.normalizer == Normalizer::Nfc && !self.in_nfc(text))
            .then(|| self.nfc_gaps(text, added));
        let units = match &gaps {
            Some(gaps) => self.units_of(gaps, parallel, added),
            None => self.units(text, parallel, added),
        };
        let units = units.as_slice();
        let (before, after): (&[u32], &[u32]) = if framed {
            (&self.frame.before, &self.frame.after)
        } else {
            (&[], &[])
        };
        if parallel && units.len() >= 2 {
            // One Vec per unit, reserved for its text's estimate up front
            // (see `ids_capacity`: a denser unit grows past it by doubling)
            // and filled in place, then joined; the pool's collection makes
            // no list of its own (measured by the crate's encode_cost bench:
            // a 32k-token encode made 60-125 allocations this way before,
            // 6-49 after; what remains is per unit, the mapping buffer below
            // growing to its first piece and the ids growing past their
            // estimate on denser text). The join copies each id once, as it
            // did.
            let mut parts: Vec<Vec<u32>> = units
                .iter()
                .map(|unit| Vec::with_capacity(Self::ids_capacity(unit)))
                .collect();
            units
                .par_iter()
                .zip(parts.par_iter_mut())
                .enumerate()
                .try_for_each(|(index, (unit, ids))| {
                    let marker = self.marker_before(units, index, at_start);
                    let mut mapped = String::new();
                    let mut cache = self.piece_cache_for_call();
                    self.encode_unit(model, unit, marker, words, &mut mapped, ids, &mut cache)
                })?;
            let body: usize = parts.iter().map(Vec::len).sum();
            let mut ids = Vec::with_capacity(before.len() + body + after.len());
            ids.extend_from_slice(before);
            for part in &parts {
                ids.extend_from_slice(part);
            }
            ids.extend_from_slice(after);
            return Ok(ids);
        }
        let mut ids = Vec::with_capacity(before.len() + text.len() / 3 + 8 + after.len());
        ids.extend_from_slice(before);
        let mut mapped = String::new();
        let mut cache = self.piece_cache_for_call();
        for (index, unit) in units.iter().enumerate() {
            let marker = self.marker_before(units, index, at_start);
            self.encode_unit(
                model,
                unit,
                marker,
                words,
                &mut mapped,
                &mut ids,
                &mut cache,
            )?;
        }
        ids.extend_from_slice(after);
        Ok(ids)
    }

    /// Room for the ids of one unit, from its bytes: about a token per three
    /// bytes of prose, with a floor for a short unit; a unit of a denser
    /// script grows its Vec past the estimate (by doubling, so a unit at one
    /// id per byte grows twice).
    fn ids_capacity(unit: &Segment<'_>) -> usize {
        match unit {
            Segment::Added(_) => 1,
            Segment::Text(text) => text.len() / 3 + 8,
        }
    }

    /// Whether `text` is in NFC by the library's tables: the quick check's
    /// own tables first, the library's stretch-wise check where they cannot
    /// tell (an encoder without the NFC normalizer never asks).
    fn in_nfc(&self, text: &str) -> bool {
        match &self.nfc_quick {
            Some(quick) => quick.is_nfc(text),
            None => is_nfc(text),
        }
    }

    /// Whether the sentencepiece path's word marker goes before the unit at
    /// `index` of `units`: a text that starts a text between added tokens
    /// (the first unit, or one after an added token; the further chunks of a
    /// text cut for the thread pool do not), and under the `First` scheme
    /// only the one that starts the input (`at_start`). Never on the
    /// byte-level path.
    fn marker_before(&self, units: &[Segment<'_>], index: usize, at_start: bool) -> bool {
        self.chars.as_ref().is_some_and(|chars| {
            let starts_text = index == 0 || matches!(units[index - 1], Segment::Added(_));
            chars.prepends(starts_text, index == 0 && at_start)
        })
    }

    /// The input as the units its encode is made of: the added tokens
    /// matched on the raw text, then the ones matched after normalization on
    /// the texts between them (the text is in its normal form here, see
    /// [`Self::encode_units`]), and the texts between all of them; for the
    /// thread pool a long text is cut further at stable boundaries
    /// ([`Self::stable_cut`]). Both kinds of added token come out before any
    /// cut: a cut inside one would turn it into pieces. With `added` off no
    /// token is cut out: the whole text is text.
    fn units<'t>(&self, text: &'t str, chunked: bool, added: bool) -> Units<'t> {
        let mut units = Units::new();
        if !added {
            self.push_text(text, chunked, &mut units);
            return units;
        }
        match &self.raw_added {
            Some(raw) => raw.split(text, |segment| match segment {
                Segment::Added(id) => units.push(Segment::Added(id)),
                Segment::Text(text) => self.push_normalized(text, chunked, &mut units),
            }),
            None => self.push_normalized(text, chunked, &mut units),
        }
        units
    }

    /// [`Self::units`] over the gaps of a text that was not in NFC
    /// ([`Self::nfc_gaps`]): the raw cut is done and every text is in NFC,
    /// what remains is the cut after normalization and the chunking.
    fn units_of<'g>(&self, gaps: &'g [Gap<'_>], chunked: bool, added: bool) -> Units<'g> {
        let mut units = Units::new();
        for gap in gaps {
            match gap {
                Gap::Added(id) => units.push(Segment::Added(*id)),
                Gap::Text(text) if added => self.push_normalized(text, chunked, &mut units),
                Gap::Text(text) => self.push_text(text, chunked, &mut units),
            }
        }
        units
    }

    /// The text cut at its raw added tokens (not at all with `added` off),
    /// each text between them in NFC: as it is where it was, normalized
    /// into an owned buffer where it was not. `tokenizers` normalizes each
    /// such text on its own, after the raw cut and before the cut after
    /// normalization: a mark right after a raw token has nothing to compose
    /// with, and a token's spelling is never matched across what
    /// normalization changed.
    fn nfc_gaps<'t>(&self, text: &'t str, added: bool) -> Vec<Gap<'t>> {
        let mut gaps = Vec::new();
        let quick = self.nfc_quick.as_ref();
        let push = |text: &'t str, gaps: &mut Vec<Gap<'t>>| gaps.push(Gap::Text(nfc(text, quick)));
        match self.raw_added.as_ref().filter(|_| added) {
            Some(raw) => raw.split(text, |segment| match segment {
                Segment::Added(id) => gaps.push(Gap::Added(id)),
                Segment::Text(text) => push(text, &mut gaps),
            }),
            None => push(text, &mut gaps),
        }
        gaps
    }

    /// Appends the units of a text between raw added tokens: the added
    /// tokens matched after normalization and the texts between them.
    fn push_normalized<'t>(&self, text: &'t str, chunked: bool, units: &mut Units<'t>) {
        match &self.normalized_added {
            Some(added) => added.split(text, |segment| match segment {
                Segment::Added(id) => units.push(Segment::Added(id)),
                Segment::Text(text) => self.push_text(text, chunked, units),
            }),
            None => self.push_text(text, chunked, units),
        }
    }

    /// Appends a text as one unit, or for the thread pool (`chunked`) as
    /// several, cut at stable boundaries ([`Self::stable_cut`]).
    fn push_text<'t>(&self, text: &'t str, chunked: bool, units: &mut Units<'t>) {
        let mut rest = text;
        while chunked && rest.len() > 2 * PARALLEL_CHUNK_BYTES {
            let Some(cut) = self.stable_cut(rest, PARALLEL_CHUNK_BYTES) else {
                break;
            };
            units.push(Segment::Text(&rest[..cut]));
            rest = &rest[cut..];
        }
        units.push(Segment::Text(rest));
    }

    /// A byte offset at or past `from` where the stages' pieces can be cut
    /// without changing them: a space followed by a letter after a character
    /// that is not whitespace, where the patterns, run over the text around
    /// it, cut cleanly ([`Self::piece_starts_at`]). No piece spans such a
    /// point (letters, digits and punctuation stop at the space, a whitespace
    /// run cannot start before it) and the patterns have no lookbehind, so
    /// the pieces of the two halves are the pieces of the whole. `None` when
    /// the text has no such point past `from`, or when
    /// [`MAX_REJECTED_CUT_CANDIDATES`] candidates past `from` failed the
    /// check: a stretch without candidates (a blob, a table of numbers) costs
    /// only the byte scan and the search goes on past it, a text whose
    /// candidates keep failing stays one unit after that many checks.
    fn stable_cut(&self, text: &str, from: usize) -> Option<usize> {
        // The sentencepiece path: any point between two characters that no
        // vocabulary token joins, read on the raw text.
        if let Some(chars) = &self.chars {
            let bytes = text.as_bytes();
            return (from.max(1)..text.len()).find(|&p| chars.cuts_at(bytes, p));
        }
        if self.stages.is_empty() {
            return None;
        }
        let bytes = text.as_bytes();
        let mut rejected = 0;
        let mut p = from;
        while p + 1 < text.len() {
            if bytes[p] != b' ' {
                p += 1;
                continue;
            }
            if text[p + 1..]
                .chars()
                .next()
                .is_some_and(char::is_alphabetic)
                && text[..p]
                    .chars()
                    .next_back()
                    .is_some_and(|c| !c.is_whitespace())
            {
                if self.piece_starts_at(text, p) {
                    return Some(p);
                }
                rejected += 1;
                if rejected >= MAX_REJECTED_CUT_CANDIDATES {
                    return None;
                }
            }
            p += 1;
        }
        None
    }

    /// Whether the stages, run over a window of text around `p` that starts
    /// and ends at whitespace, cut cleanly at `p`: at each stage in turn, no
    /// match may span `p`; a match that starts at `p` settles it (the deeper
    /// stages see that match from `p` on either way), a gap around `p` hands
    /// the question to the next stage, and a gap at the last stage would be
    /// split, so that is no cut.
    fn piece_starts_at(&self, text: &str, p: usize) -> bool {
        let mut start = p.saturating_sub(CUT_CHECK_WINDOW);
        while start > 0 && !(text.is_char_boundary(start) && text.as_bytes()[start] == b' ') {
            start -= 1;
        }
        let mut end = (p + CUT_CHECK_WINDOW).min(text.len());
        while end < text.len() && !(text.is_char_boundary(end) && text.as_bytes()[end] == b' ') {
            end += 1;
        }
        let window = &text[start..end];
        let at = p - start;
        for stage in &self.stages {
            for (piece_start, piece_end) in stage.pattern.find_iter(window) {
                if piece_start == at {
                    return true;
                }
                if piece_start < at && at < piece_end {
                    return false;
                }
                if piece_start > at {
                    break;
                }
            }
        }
        false
    }

    /// Appends the ids of one unit: an added token's id, or a text's pieces
    /// through the stages and the model (the text has no added token left in
    /// it, see [`Self::units`]); the reason when the model declines a piece
    /// (the pieces after it are tokenized all the same, as in the whole-text
    /// encode). `marker` says the sentencepiece path's word marker goes
    /// before the text ([`Self::marker_before`]). `cache` is the encoding
    /// thread's piece cache ([`Self::thread_piece_cache`]), borrowed by the
    /// caller for the unit.
    #[expect(clippy::too_many_arguments)]
    fn encode_unit(
        &self,
        model: Option<&ModelWrapper>,
        unit: &Segment<'_>,
        marker: bool,
        words: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let mut fallback = None;
        let text = match *unit {
            Segment::Added(id) => {
                ids.push(id);
                return Ok(());
            }
            Segment::Text(text) => text,
        };
        if let Some(chars) = &self.chars {
            // The sentencepiece path cuts the raw text itself, out of line.
            // An empty text gets no marker: `tokenizers` drops an empty
            // split before the pre-tokenizer runs.
            if !text.is_empty() {
                self.encode_sentencepiece(chars, text, marker, mapped, ids, cache);
            }
            return Ok(());
        }
        // One pass over the unit (word at a time in the standard library) on
        // the byte-level path, the one that reads it: the cold pieces of an
        // ASCII unit skip the pre-formed path and its per-piece test (see
        // `tokenize_piece_slow`).
        cache.unit_ascii = text.is_ascii();
        // One walk per unit. A single-stage encoder hands the unit to
        // `encode_single_stage`, out of line, where the scanner's own loop
        // probes each piece in place (no recursive walk); the chain hands
        // its pieces, as offsets into the unit, to the same probe through
        // its static callback; any other shape walks.
        match (&self.chain, self.stages.as_slice()) {
            (Some(chain), _) => {
                chain.pieces(text, &mut |start, end| {
                    if let Err(declined) =
                        self.tokenize_piece_at(model, text, start..end, mapped, ids, cache)
                    {
                        fallback = Some(declined);
                    }
                });
            }
            (None, [stage]) => {
                if let Err(declined) =
                    self.encode_single_stage(model, stage, text, words, mapped, ids, cache)
                {
                    fallback = Some(declined);
                }
            }
            (None, _) => {
                let mut per_piece = |piece: &str| {
                    if let Err(declined) = self.tokenize_piece(model, piece, mapped, ids, cache) {
                        fallback = Some(declined);
                    }
                };
                self.pieces(0, text, &mut per_piece);
            }
        }
        fallback.map_or(Ok(()), Err)
    }

    /// Applies the stages from `depth` on to `text`, calling `f` with each
    /// final piece. `f` is taken by static dispatch so that the per-piece
    /// work compiles into the walk.
    fn pieces<F: FnMut(&str)>(&self, depth: usize, text: &str, f: &mut F) {
        if text.is_empty() {
            return;
        }
        let Some(stage) = self.stages.get(depth) else {
            f(text);
            return;
        };
        let mut cursor = 0;
        for (start, end) in stage.pattern.find_iter(text) {
            if stage.keep_gaps && cursor < start {
                self.pieces(depth + 1, &text[cursor..start], f);
            }
            self.pieces(depth + 1, &text[start..end], f);
            cursor = end;
        }
        if stage.keep_gaps && cursor < text.len() {
            self.pieces(depth + 1, &text[cursor..], f);
        }
    }

    /// Appends the ids of one piece: from the thread's cache, else from the
    /// encoder's seed, else merged over the model's merges (the whole piece
    /// looked up first under `ignore_merges`), else, for a model the merge
    /// table declined, from the model itself over the piece's byte-level
    /// alphabet form (a raw piece of the sentencepiece path never comes here:
    /// it has [`Self::tokenize_raw_piece_slow`]);
    /// remembered in the thread's cache whichever answered;
    /// [`Fallback::PieceDeclined`] when the model declines the piece (or
    /// there is none to ask: a vocabulary read from ranks has a complete
    /// table and never gets here). The
    /// thread's cache comes first because it is small and hot: the pieces of
    /// the prompts a thread encodes, a few hundred KB, where the seed is
    /// several MB of lines a warm encode would read at random (measured:
    /// the seed first cost a warm prose encode a tenth more). `cache` is the
    /// thread's own ([`Self::thread_piece_cache`]), looked up and borrowed by
    /// the encode once rather than here once per piece (measured: 4 % of the
    /// instructions per token of a warm encode on a 151k byte-level
    /// vocabulary).
    #[inline(always)]
    fn tokenize_piece(
        &self,
        model: Option<&ModelWrapper>,
        piece: &str,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        // The hot path of a warm encode, and nothing else in this function:
        // the key of a short piece and the packed table's probe.
        let vacancy = match ShortPieceKey::of(piece.as_bytes()) {
            Some(key) => match cache.packed.lookup(key, ids) {
                None => return Ok(()),
                vacancy => vacancy,
            },
            None => None,
        };
        self.tokenize_piece_slow(model, piece, vacancy, mapped, ids, cache)
    }

    /// A unit of a single-stage encoder: the stage's matches through the
    /// scanner's own loop with the probe compiled into it; a gap between
    /// matches is a piece of its own through the same probe. Out of line
    /// so that [`Self::encode_unit`] keeps the frame the other shapes pay
    /// per unit: the sentencepiece path calls it once per word, and the
    /// loop inlined there cost that path 3 % of its instructions per token
    /// (measured); one call per unit here is nothing against a unit's
    /// pieces.
    #[inline(never)]
    #[expect(clippy::too_many_arguments)]
    fn encode_single_stage(
        &self,
        model: Option<&ModelWrapper>,
        stage: &Stage,
        text: &str,
        words: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let keep_gaps = stage.keep_gaps;
        let mut cursor = 0;
        let mut fallback = None;
        // The probe compiles into the scanner's loop. As a closure of its
        // own it was a call per piece: a six-register frame, a reload of
        // every captured reference and the spills the slow path needs, 135
        // instructions for a hit of which the probe is about 50 (counted on
        // the long warm cells). The gap piece stays out of line: a scanner's
        // alternatives tile the text, so a gap is the rare case there.
        stage.pattern.for_each_match(
            text,
            words,
            #[inline(always)]
            |start, end| {
                if keep_gaps && cursor < start {
                    if let Err(declined) =
                        self.tokenize_gap(model, text, cursor..start, mapped, ids, cache)
                    {
                        fallback = Some(declined);
                    }
                }
                if let Err(declined) =
                    self.tokenize_piece_at(model, text, start..end, mapped, ids, cache)
                {
                    fallback = Some(declined);
                }
                cursor = end;
            },
        );
        if keep_gaps && cursor < text.len() {
            if let Err(declined) =
                self.tokenize_piece_at(model, text, cursor..text.len(), mapped, ids, cache)
            {
                fallback = Some(declined);
            }
        }
        fallback.map_or(Ok(()), Err)
    }

    /// [`Self::tokenize_piece`] for the piece `text[piece]` of a unit: the
    /// key is read from the text around the piece (`ShortPieceKey::at`), and
    /// the piece is cut out of the text only off the hot path.
    #[inline(always)]
    fn tokenize_piece_at(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        piece: Range<usize>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let vacancy = match ShortPieceKey::at(text.as_bytes(), piece.start, piece.end) {
            Some(key) => match cache.packed.lookup(key, ids) {
                None => return Ok(()),
                vacancy => vacancy,
            },
            None => None,
        };
        self.tokenize_piece_slow(model, &text[piece], vacancy, mapped, ids, cache)
    }

    /// [`Self::tokenize_piece_at`] for a gap between two matches of a
    /// single-stage encoder, out of line: the scanners' alternatives tile
    /// the text, so the gap is the rare piece and its code stays out of the
    /// per-piece loop.
    #[cold]
    #[inline(never)]
    fn tokenize_gap(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        piece: Range<usize>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        self.tokenize_piece_at(model, text, piece, mapped, ids, cache)
    }

    /// The rest of [`Self::tokenize_piece`], out of line: the longer pieces'
    /// cache, the seed, the merge or the model, and the cache insert.
    #[inline(never)]
    fn tokenize_piece_slow(
        &self,
        model: Option<&ModelWrapper>,
        piece: &str,
        vacancy: Option<Vacancy>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let bytes = piece.as_bytes();
        let cacheable = bytes.len() <= LONG_PIECE_MAX_BYTES;
        if vacancy.is_none() && cacheable {
            if let Some(found) = cache.get(bytes) {
                ids.extend_from_slice(found);
                return Ok(());
            }
        }
        let start = ids.len();
        // A piece this thread has not met: the encoder's seed has it when it
        // is a vocabulary word, else it is merged, by the table or by the
        // model for a table that declined. A raw piece of the sentencepiece
        // path never comes here: it has its own slow path.
        debug_assert!(
            self.chars.is_none(),
            "a sentencepiece piece takes tokenize_raw_piece_slow"
        );
        if !vacancy.is_some_and(|vacancy| self.seed.ids_into(vacancy.key(), ids)) {
            if let Some(table) = &self.merges {
                // Under `ignore_merges` a piece that is an entry is that entry:
                // the model's map is asked when the piece can be one (the
                // lookup is the piece mapped to the alphabet as a string, its
                // hash, a probe of the map and a compare: a memory latency and
                // more on a cold piece, measured).
                let whole = if table.ignore_merges
                    && self.may_be_entry(vacancy.map(|vacancy| vacancy.key()), bytes)
                {
                    model.and_then(|model| model.token_to_id(self.mapped(bytes, mapped)))
                } else {
                    None
                };
                match whole {
                    Some(id) => ids.push(id),
                    // A cold piece of a unit with a non-ASCII byte merges
                    // from its pre-formed characters when it has them (the
                    // pre-formed path reads the piece's bytes itself); a
                    // piece of an ASCII unit never reads that analysis.
                    None => {
                        if cache.unit_ascii || !table.merge_preformed(bytes, ids) {
                            table.merge(bytes, ids);
                        }
                    }
                }
            } else {
                match model.map(|model| model.tokenize(self.mapped(bytes, mapped))) {
                    Some(Ok(tokens)) => ids.extend(tokens.iter().map(|token| token.id)),
                    Some(Err(_)) | None => return Err(Fallback::PieceDeclined),
                }
            }
        }
        self.remember_piece(bytes, vacancy, start, ids, cache);
        Ok(())
    }

    /// Remembers `ids[start..]` as the encoding of `piece` in the thread's
    /// cache: the packed table for a piece with a key, the map for one of up
    /// to `CACHED_PIECE_MAX_BYTES`, the long tier, by bytes, for one of up
    /// to `LONG_PIECE_MAX_BYTES`; each within the thread's share. Inlined
    /// into both slow paths: out of line it is a call frame per cache miss
    /// on every vocabulary (four to eight instructions per token on a cold
    /// byte-level text, counted).
    #[inline(always)]
    fn remember_piece(
        &self,
        piece: &[u8],
        vacancy: Option<Vacancy>,
        start: usize,
        ids: &[u32],
        cache: &mut PieceCache,
    ) {
        if let Some(vacancy) = vacancy {
            let share = cache.share;
            if cache.short_entries() >= share {
                // The vacancy named a slot of the table as it was: the cleared
                // table is probed again.
                cache.clear_short(share);
                cache.packed.insert(vacancy.key(), &ids[start..], share);
            } else {
                cache.packed.insert_at(vacancy, &ids[start..], share);
            }
        } else if piece.len() <= CACHED_PIECE_MAX_BYTES {
            let share = cache.share;
            if cache.short_entries() >= share {
                cache.clear_short(share);
            }
            cache.short.insert(piece.into(), ids[start..].into());
        } else if piece.len() <= LONG_PIECE_MAX_BYTES {
            let entry = LONG_PIECE_ENTRY_BYTES + piece.len() + size_of_val(&ids[start..]);
            let share = self.long_piece_bytes_share();
            if entry <= share {
                if cache.long_bytes + entry > share {
                    cache.long.clear();
                    cache.long_bytes = 0;
                }
                cache.long_bytes += entry;
                cache.long.insert(piece.into(), ids[start..].into());
            }
        }
    }

    /// A unit of the sentencepiece path (see [`sentencepiece`]): the raw
    /// text cut into pieces by the character tables, each piece probed in
    /// the thread's cache by its raw bytes (the key read from the text
    /// around it, as [`Self::tokenize_piece_at`] reads one), its spelling in
    /// the vocabulary's form made for a merge alone. The word marker before
    /// the text (`marker`) joins the first piece as a space, or stands alone
    /// when no token bridges it to the first byte. Out of line, as
    /// [`Self::encode_single_stage`]. Nothing on this path declines a piece.
    #[inline(never)]
    fn encode_sentencepiece(
        &self,
        chars: &CharSymbols,
        text: &str,
        marker: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) {
        let bytes = text.as_bytes();
        // A text that starts with a space or a marker has its marker already.
        let mut leading = marker && !(text.starts_with(' ') || text.starts_with(WORD_MARKER));
        if leading && chars.cuts_at(&[b' ', bytes[0]], 1) {
            self.tokenize_leading_piece(chars, "", mapped, ids, cache);
            leading = false;
        }
        chars.pieces(
            bytes,
            #[inline(always)]
            |start, end| {
                if leading {
                    leading = false;
                    self.tokenize_leading_piece(chars, &text[start..end], mapped, ids, cache);
                } else {
                    self.tokenize_raw_piece(chars, text, start..end, mapped, ids, cache);
                }
            },
        );
    }

    /// The piece `text[piece]` of a sentencepiece unit: the key read from
    /// the text around it and the packed table's probe; the rest out of line.
    #[inline(always)]
    fn tokenize_raw_piece(
        &self,
        chars: &CharSymbols,
        text: &str,
        piece: Range<usize>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) {
        let vacancy = match ShortPieceKey::at(text.as_bytes(), piece.start, piece.end) {
            Some(key) => match cache.packed.lookup(key, ids) {
                None => return,
                vacancy => vacancy,
            },
            None => None,
        };
        self.tokenize_raw_piece_slow(chars, &text[piece], vacancy, mapped, ids, cache);
    }

    /// The first piece of a unit the word marker goes before: the marker,
    /// a space in the raw form, followed by the piece. Once per unit.
    #[inline(never)]
    fn tokenize_leading_piece(
        &self,
        chars: &CharSymbols,
        piece: &str,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) {
        let mut spaced = String::with_capacity(piece.len() + 1);
        spaced.push(' ');
        spaced.push_str(piece);
        let vacancy = match ShortPieceKey::of(spaced.as_bytes()) {
            Some(key) => match cache.packed.lookup(key, ids) {
                None => return,
                vacancy => vacancy,
            },
            None => None,
        };
        self.tokenize_raw_piece_slow(chars, &spaced, vacancy, mapped, ids, cache);
    }

    /// The rest of [`Self::tokenize_raw_piece`], out of line: the longer
    /// pieces' cache (by the raw bytes), the seed, the merge from the
    /// piece's character symbols in the vocabulary's form, and the cache
    /// insert.
    #[inline(never)]
    fn tokenize_raw_piece_slow(
        &self,
        chars: &CharSymbols,
        raw: &str,
        vacancy: Option<Vacancy>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) {
        let bytes = raw.as_bytes();
        if vacancy.is_none() && bytes.len() <= LONG_PIECE_MAX_BYTES {
            if let Some(found) = cache.get(bytes) {
                ids.extend_from_slice(found);
                return;
            }
        }
        let start = ids.len();
        if !vacancy.is_some_and(|vacancy| self.seed.ids_into(vacancy.key(), ids)) {
            chars.merge(
                CharSymbols::normalized(raw, mapped),
                &mut cache.symbols,
                ids,
            );
        }
        self.remember_piece(bytes, vacancy, start, ids, cache);
    }

    /// The calling thread's piece cache for this encoder, made at the
    /// thread's first encode with it (from then on the thread counts toward
    /// every thread's share, see [`Self::piece_cache_share`]); that first
    /// encode also reads the merge table through once
    /// (`MergeTable::read_through`), so the thread's first-sight pieces find
    /// its lines in the caches.
    fn thread_piece_cache(&self) -> &RefCell<PieceCache> {
        self.piece_cache.get_or(|| {
            self.threads.fetch_add(1, Ordering::Relaxed);
            if let Some(merges) = &self.merges {
                merges.read_through();
            }
            RefCell::new(PieceCache::with_share(self.piece_cache_share()))
        })
    }

    /// The calling thread's piece cache, borrowed for one encode call and
    /// within the thread's share at the start of it: a thread whose share
    /// fell since it last read it (threads joined) clears its count-bounded
    /// tiers here, by their entries, by the slots of its packed table or by
    /// the capacity of its map (arrays grown for the earlier share keep
    /// their size through a clear at the bound; here the map is made anew for
    /// the share and the packed table is cut back to the share's slots inside
    /// its reservation, see `ShortPieceTable::clear`), so every thread that
    /// still encodes is within the share
    /// from its next call on (the share is read here once and kept in the
    /// cache for the call's misses, so a call that was running when threads
    /// joined keeps the share it started with until it ends; a thread that
    /// never encodes again holds what it had until it exits). One relaxed
    /// load, one store and three compares per call.
    fn piece_cache_for_call(&self) -> RefMut<'_, PieceCache> {
        let mut cache = self.thread_piece_cache().borrow_mut();
        let share = self.piece_cache_share();
        cache.share = share;
        let slots = ShortPieceTable::slots_for(share);
        if cache.short_entries() > share
            || cache.packed_slots() > slots
            || cache.map_capacity() > 2 * share
        {
            cache.clear_short(share);
        }
        cache
    }

    /// Whether a piece the thread's cache and the seed missed can still be
    /// an entry of the vocabulary: a short piece (`key`) only when the seed
    /// does not hold every short entry; a long piece only when the filter
    /// over the long entries, when the seed's pass read them all, may hold
    /// it (see `long_entries`).
    fn may_be_entry(&self, key: Option<ShortPieceKey>, bytes: &[u8]) -> bool {
        match key {
            Some(_) => !self.seed_covers_short_entries,
            None => self
                .long_entries
                .as_ref()
                .is_none_or(|filter| filter.may_hold(bytes)),
        }
    }

    /// The piece in the byte-level alphabet, the form the model's vocabulary
    /// is keyed by, written into `mapped`.
    fn mapped<'m>(&self, bytes: &[u8], mapped: &'m mut String) -> &'m str {
        mapped.clear();
        mapped.extend(bytes.iter().map(|&b| self.alphabet[usize::from(b)]));
        mapped
    }

    /// Entries one thread's piece cache may hold: the total shared among the
    /// threads that have encoded with this encoder, `PIECE_CACHE_CAPACITY` at
    /// most.
    fn piece_cache_share(&self) -> usize {
        (PIECE_CACHE_TOTAL_CAPACITY / self.threads.load(Ordering::Relaxed).max(1))
            .clamp(1, PIECE_CACHE_CAPACITY)
    }

    /// Bytes of long pieces one thread may hold: the total shared among the
    /// threads that have encoded with this encoder, `LONG_PIECE_BYTES_CAPACITY`
    /// at most.
    fn long_piece_bytes_share(&self) -> usize {
        (LONG_PIECE_BYTES_TOTAL / self.threads.load(Ordering::Relaxed).max(1))
            .min(LONG_PIECE_BYTES_CAPACITY)
    }
}

#[cfg(test)]
mod tests;
