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
//! 2. normalization is NFC, NFKC or nothing, read from the tables
//!    `tokenizers` normalizes with (the crate
//!    `unicode-normalization-alignments`, every normalization read of this
//!    module: a newer crate's tables know marks and compositions the
//!    library's do not and would compose other text); a text already in
//!    the normalizer's form (its ASCII stretches are not read at all, the
//!    quick check answers for most of the rest under NFC and normalizing
//!    settles what it cannot, so marks-heavy scripts cost no copy) is used
//!    as it is, any other text is normalized gap by gap between its raw
//!    added tokens, as `tokenizers` normalizes it, and encoded from those
//!    buffers;
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
//! for the text's spaces and, by the `Metaspace` scheme or the `Prepend`
//! normalizer, before it; the pieces take the same cache, seed and merge
//! table, from character symbols.
//!
//! The shape is checked once at load time; a tokenizer that does not fit
//! (another normalizer, a pre-tokenizer or regex construct outside the
//! supported subset, a post-processor other than `ByteLevel` or a template
//! of that shape, truncation or padding, added tokens with
//! `single_word`) simply has no native path.
//!
//! A tiktoken vocabulary is a byte-level BPE of this shape with no
//! normalizer, one pattern, special tokens cut out of the raw text and no
//! frame, its merges implied by its ranks: [`NativeEncoder::from_ranks`]
//! reads one into the same encoder (the merge table from the ranks, see
//! [`merge`]), and a file whose ranks or pattern do not fit keeps its own
//! encoder.

use std::{
    borrow::{Borrow, Cow},
    cell::{RefCell, RefMut},
    hash::Hash,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex, OnceLock, PoisonError,
    },
    thread::ThreadId,
    time::{Duration, Instant},
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

mod block;
mod byte_split;
mod cased;
mod chain;
mod learned_pieces;
mod long_entries;
mod merge;
mod nfc;
mod pattern;
mod scan;
mod sentencepiece;
mod short_pieces;
mod unicode;
mod wide_pieces;

use learned_pieces::LearnedPieces;
use long_entries::LongEntryFilter;
use merge::{ByteTokens, MergeTable};
use nfc::NfcQuick;
use pattern::Pattern;
use sentencepiece::{CharSymbols, Marker, WORD_MARKER};
use short_pieces::{
    Entry, ShortPieceKey, ShortPieceTable, Vacancy, INLINE_IDS, MIN_SLOTS, SHORT_PIECE_MAX_BYTES,
    THREAD_BACKING_SLOTS,
};
use wide_pieces::{WideKey, WidePieceTable};

use crate::{
    encode_path::{Declined, Fallback},
    word_cache::{Listing, ModelView, TokenizerOf},
};

/// The split the `ByteLevel` pre-tokenizer applies itself when `use_regex`
/// is on (GPT-2 and its descendants).
const BYTE_LEVEL_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// A class of the code points of `ranges` as a pattern: one character a
/// piece, or one run of them under `runs`. ASCII members other than letters
/// and digits are escaped (the class syntax's own characters among them),
/// the rest stand as the ranges they are.
fn class_pattern(ranges: &[(char, char)], runs: bool) -> String {
    let mut pattern = String::with_capacity(4 * ranges.len() + 4);
    pattern.push('[');
    for &(start, end) in ranges {
        if start.is_ascii() {
            for c in start..=end {
                if !c.is_ascii_alphanumeric() {
                    pattern.push('\\');
                }
                pattern.push(c);
            }
        } else {
            pattern.push(start);
            if end > start {
                pattern.push('-');
                pattern.push(end);
            }
        }
    }
    pattern.push(']');
    if runs {
        pattern.push('+');
    }
    pattern
}

/// Pieces up to this many bytes are cached per thread by their count (see
/// `PIECE_CACHE_CAPACITY`): those of up to `SHORT_PIECE_MAX_BYTES` in a
/// packed table, those of up to 32 bytes with up to seven ids in a second
/// one (`wide_pieces`), the rest in a map; longer pieces still, up to
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

/// Bytes of an input from which the scanners walk a letter or white-space
/// run eight bytes at a time (`Roles::run_end_as`, and `run_end_as_in` for
/// the chain scanner's runs, which also end at a stage class): the scanner
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
/// the scans on every unit it is cut into. The chain scanner takes them on
/// a shorter unit too when it opens as ASCII text
/// (`NativeEncoder::opens_as_ascii`).
const WORD_SCAN_MIN_BYTES: usize = PARALLEL_CHUNK_BYTES;
/// Bytes at the head of a unit that `NativeEncoder::opens_as_ascii` reads.
const ASCII_HEAD_BYTES: usize = 256;
/// A thread's share of one encoder's piece cache, at most; see
/// `PIECE_CACHE_TOTAL_CAPACITY`. A share cut for more than sixteen threads,
/// or to a part where the threads share what they merge,
/// is a count of entries over the three tiers. At this full share the wide
/// table and the map hold a share between them and the packed table is
/// counted apart, by the fill bound of the slots the share takes (one and
/// a half shares), because one count over the three cleared them all while
/// those slots were at most half full: two passes over 33 documents of
/// 500k tokens (277,000 distinct pieces, 154,000 of them packed) cleared a
/// thread's tables twice. Either bound is the cliff of a
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
/// spill: 22.5 MiB more at most, emptied with the table, its capacity kept).
/// The pieces of 16 to 64 bytes count against the share in a map
/// (`PieceCache::short`): its slot with the table's slack, the piece and
/// the ids as two boxes, about 140 bytes for a typical entry (a 20-byte
/// piece with four ids) and about 430 at most (a 64-byte piece whose ids
/// are its bytes), so a thread whose entries were all such pieces holds
/// about 35 MiB, 108 MiB in that worst case. A piece of 16 to 32 bytes
/// with up to seven ids takes a slot of the wide table instead
/// (`PieceCache::wide`: 65 bytes a slot, the table between three eighths
/// and three quarters full past 4,096 slots, so 87 to 173 bytes an
/// entry and 32.5 MiB at the full bound, inside the map's figures), and
/// allocates nothing per entry. The packed table fills beside the other
/// two, and a clear at a bound keeps
/// the tiers' arrays (the packed table's slots and its spill's capacity,
/// the wide table's slots and the map's buckets, unless the share fell
/// below what they were grown
/// for) and drops their entries, so a thread can hold a full packed table
/// and a full map, about 51 MiB
/// with typical map entries and 147 MiB at the worst case the bounds
/// allow; a thread that filled its wide table before the map keeps those
/// slots as well, 32.5 MiB more (about 84 MiB and 180 MiB).
/// Measured on one thread after a corpus of 7,015 prompts: the peak RSS
/// grew by 8 MB on a 128k-token vocabulary and did not move on three
/// larger ones, whose peak is the file load.
const PIECE_CACHE_CAPACITY: usize = 262_144;
/// Packed entries a thread at the full share holds before its tiers are
/// cleared: the fill bound of the slots that share takes.
const FULL_SHARE_PACKED: usize = ShortPieceTable::slots_for(PIECE_CACHE_CAPACITY) / 4 * 3;
/// Entries all threads together hold for one encoder once more than sixteen
/// have encoded with it: a thread's own cap is
/// this divided by the number of threads that have encoded with the encoder
/// (and never more than `PIECE_CACHE_CAPACITY`). Up to sixteen threads that
/// do not share what they merge are
/// each at the full cap, where the packed table is counted apart and a
/// thread holds up to two and a half caps of entries: their total is the
/// bytes below, not this count. Where the threads share
/// what they merge (once `LEARNED_FROM_THREADS` threads call an encoder
/// whose misses wait), a thread keeps a quarter of that cap and the encoder keeps
/// what the threads merge in one table for all of them, half of the entries
/// (see `GROWN_SHARE_PART`): the figures below are then ceilings the threads
/// stay under, three quarters of them in packed slots, the table's
/// included, and a quarter in the maps and the wide tables. A thread's packed
/// table starts
/// at 64 slots (2 KiB) and grows as the thread meets pieces (to 1,024 slots
/// in its first step, then by four up to
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
/// about 140 bytes a typical entry (430 at most, a wide table's entry within
/// that; see `PIECE_CACHE_CAPACITY`),
/// and a clear keeps the packed slots and the map's buckets (the map made anew,
/// the packed table cut back, for a smaller share), so the count-bounded tiers
/// of one tokenizer hold at most the sum of the two: about 820 MiB at sixteen
/// threads with typical map entries (about 2.3 GiB, sixteen times the 147 MiB
/// of a thread, were every map entry the worst case), and the slots of wide
/// tables that grew to the bound as well, kept through a clear like the packed
/// slots (65 bytes a slot: 520 MiB more at sixteen threads, 693 MiB at the most
/// however many threads share the total), toward 352 + 560 MiB with
/// the rounding of more threads (the packed part toward twice its 352 MiB past
/// some 680 threads, as above); the long pieces add `LONG_PIECE_BYTES_TOTAL` at
/// most once every thread has added one under its present share, and can
/// add more until then, see that constant (a thread's long tier keeps its map's
/// slots through an eviction's clear when they fit four times its share and
/// gives them back when they do not; a share that fell below a quarter of
/// what the tier holds, entries and empty slots together, makes it anew at
/// the thread's next call). A
/// thread reads its share at the start of every encode call and keeps
/// it for that call's misses (a call that was running when threads joined keeps
/// its earlier share until it ends; the long tier reads the live share at every
/// miss), so after threads join it is within the smaller share from its next
/// call on, entries, packed slots, wide slots and map buckets alike (the packed table's
/// pages already touched inside its reservation stay resident,
/// `THREAD_BACKING_SLOTS` slots at most, 33 bytes each, about 528 KiB per
/// thread: more than the share's slots only once those fall under the
/// reservation, a share under 6,144 entries, some 680 threads on this total),
/// and holds at most its earlier share until then, or while the encoder
/// lives if it never encodes again (its exit frees nothing, the encoder owns
/// the caches): right after a join the total is the earlier share times
/// the threads that had it plus the new share times the rest, and settles as
/// they encode. Before this bound the same ceilings were a sixteenth of these.
/// All of it is freed with the encoder.
const PIECE_CACHE_TOTAL_CAPACITY: usize = 16 * PIECE_CACHE_CAPACITY;
/// The part of its share a thread keeps where the threads share what they
/// merge, which is once `LEARNED_FROM_THREADS` threads call an encoder
/// whose misses wait (`NativeEncoder::misses_wait`: no other miss asks the
/// learned table, so no other encoder's threads give up a part of their
/// share for it): a quarter, from
/// a thread's first call (a share cut only at a thread's first clear left
/// the memory of sixteen threads where it was: measured). What such
/// threads merge is kept for all of them in the encoder's learned table
/// ([`LearnedPieces`]), which is made at the first ask for twice the
/// entries the threads that have encoded by then keep themselves: half of
/// what they held before the table (`PIECE_CACHE_TOTAL_CAPACITY` at the
/// most, so 2,097,152 entries: its 64 parts of 65,536 slots are 132 MiB of
/// packed slots, and the ids past four an entry in their spills). The
/// quarter of every thread and that half are three quarters of the
/// packed slots the threads held before the table, and three tenths of the
/// entries a full share lets a thread hold with its packed table counted
/// apart: a
/// thread of eight to sixteen holds 131,072 packed slots, and the table
/// 262,144 for each of them, where a thread held 524,288; the maps and the
/// wide tables, bounded by the same share, hold a quarter. An encoder that
/// fewer threads call, or whose `misses_wait` flag is off (the sentencepiece
/// path waits by its own rule on a far seed, but its threads do not share;
/// a byte-level table that does not merge every short piece does not wait),
/// is as it was: the whole share, and no learned table.
const GROWN_SHARE_PART: usize = 4;
/// The threads that must call an encoder before its threads share what
/// they merge: eight, and only on an encoder whose misses wait
/// (`NativeEncoder::misses_wait`), since no other miss asks the learned
/// table: the threads of any other encoder keep their whole share and
/// share nothing, however many call. A thread is counted at its first
/// encode with the encoder, where it takes its piece cache
/// (`NativeEncoder::thread_piece_cache`), unless it is a thread of a pool
/// then: the
/// threads of a pool encode the cuts of a caller's long text, and a wide
/// pool behind few callers is no reason to share (measured where the
/// pool's threads counted: four callers whose long texts a pool encoded, a
/// fifth of the text new, 3.8 % fewer tokens a second). So two kinds of
/// thread encode and are not counted. A caller whose every text is long
/// enough to be cut for the pool (`PARALLEL_MIN_BYTES`) takes no piece
/// cache of its own: the pool's threads take theirs. And a thread of any
/// pool is left out, so the workers of a batch encode, each of which
/// encodes whole inputs, are not counted. An encoder that only such
/// callers and threads use shares nothing and is as it was, every thread
/// with its whole share; counting them is left for a change of its own
/// (where the pool's threads counted, the cell above read slower). The
/// count never falls: it is of the
/// threads that have called, not of those that call at one time, so a
/// process whose eight threads have each encoded once shares from then on,
/// however little it encodes, until the encoder is dropped. Once eight
/// threads are counted, a
/// thread, a pool's too, keeps a part of its share (`GROWN_SHARE_PART`),
/// and one that has cleared asks the learned table before it merges a piece
/// and tells it after (`PieceCache::shares`). The learned table of fewer
/// threads is small (131,072 entries a thread), and on a stream a fifth of
/// whose text is new it answered under four misses in ten, which cost more
/// than the merges it saved (measured at one, two and four threads); from
/// eight threads on it answers enough.
const LEARNED_FROM_THREADS: usize = 8;
/// The asks of the learned table a thread judges together
/// (`PieceCache::judge_asks`). An ask is of use only when the table
/// answers it: a miss it does not answer has waited and asked for
/// nothing, whether or not the table takes what the thread then merges
/// and tells.
const ASKS_JUDGED: u32 = 256;
/// A thread rests when the table answers fewer than one in this many of
/// the asks it judges, which is under one ask in two: it shares no more
/// (`PieceCache::shares`), so its misses neither wait for the learned
/// table nor ask it nor tell it, as before the thread shared. An ask the
/// table does not answer pays the wait and the ask and saves no merge:
/// sixteen callers of text that is all new encode faster where their
/// threads rest so than where none rests (measured).
const ASKS_OF_USE_PART: u32 = 2;
/// The entries the count-bounded tiers of a resting thread take before
/// the thread asks again and judges the next asks as it judged the last:
/// of the misses the table never answers, one in seventeen still asks. A
/// clear ends the rest, since the thread has lost its own words: the
/// thread shares again from its next call, because sharing is turned on
/// only at the start of a call (`PieceCache::shares`).
const ASKS_RESTED: usize = 4_096;
/// Bytes of long pieces (each piece, its ids and their slot) a thread holds
/// for one encoder before its long entries are cleared, at most; see
/// `LONG_PIECE_BYTES_TOTAL`. This is the cliff of a repeated document for
/// its long pieces, as `PIECE_CACHE_CAPACITY` is for the others: a text
/// whose long pieces fit is encoded again at the cost of their lookups, one
/// with more merges them again on every pass, since the clear takes the
/// pieces that recur with the rest. By this count the 33 documents of
/// 128,000 tokens of the README's benchmark hold 2.8 to 3.0 MiB of long
/// pieces in all and its 33 documents of 500,000 tokens 5.6 to 6.0 MiB
/// (three byte-level vocabularies, some 14,000 and 29,000 entries of about
/// 220 bytes), so the bound holds either set whole; at 2 MiB the tier was
/// cleared once or twice in a pass over the first set and five or six
/// times in a pass over the second, and a second pass merged most of the
/// long pieces again (measured).
/// Memory: a thread holds what the long pieces it met take, so one that
/// meets few holds little; the count leaves out what the allocator rounds
/// each box up by and the empty slots of the map.
const LONG_PIECE_BYTES_CAPACITY: usize = 8 << 20;
/// Bytes of long pieces the threads of one encoder share, the way they
/// share `PIECE_CACHE_TOTAL_CAPACITY`: a thread's share is this total
/// divided by the threads that have encoded with the encoder,
/// `LONG_PIECE_BYTES_CAPACITY` at most, so sixteen threads have the full
/// bound each and more threads a smaller share. A thread is held to its
/// share in two places: where it adds a long piece, its long entries are
/// cleared first when the piece would take them past the share read at
/// that miss, and the start of a call makes anew a long tier that holds
/// more than four times the share. Nothing reaches into another thread's
/// cache, so the threads together are within this total only once each
/// has added a long piece under the present share. A thread that filled a
/// larger share and has added none since keeps what it holds, up to four
/// times its share while it encodes and all of it while it does not, for
/// as long as the encoder lives (a thread's exit frees nothing), and the
/// threads together can stand above this total for that time: sixteen
/// threads that filled 8 MiB each and a seventeenth that fills its share
/// hold 135.5 MiB. At the worst, every thread filling its share as it
/// joins and adding nothing after, the n-th keeps the total divided by n
/// and the sum is 215 MiB at 32 threads, 302 MiB at 64. The long pieces
/// of a prompt (its clauses, in a script without word spaces) are few in
/// number and large, and each one that misses goes through the model's
/// merges, so they are bounded by what they take rather than counted.
const LONG_PIECE_BYTES_TOTAL: usize = 128 << 20;
/// What an entry of the long tier takes beyond its piece and ids.
const LONG_PIECE_ENTRY_BYTES: usize = size_of::<(Box<[u8]>, Box<[u32]>)>();

/// Inputs at least this long are encoded unit by unit on the thread pool (a
/// text between added tokens, or a chunk of one cut at a stable boundary)
/// when the pool has more than one thread (see `worth_the_pool`); shorter
/// inputs would spend more on the hand-offs than they save.
const PARALLEL_MIN_BYTES: usize = 64 << 10;
/// A thread's first text at least this long has the merge table read before
/// it (`NativeEncoder::piece_cache_for_text`): the read is half the time of
/// a first 9 KiB text and a small part of a first 150 KiB one, which runs
/// as before (measured). It is `PARALLEL_MIN_BYTES`: the pool takes only a
/// text that long, so a thread of the pool reads before its first unit.
const READ_AT_ONCE_BYTES: usize = PARALLEL_MIN_BYTES;
/// How long the read of the merge table that ends a load stands in for the
/// loading thread's own (`NativeEncoder::load_read_holds`): the span of a
/// process that loads and encodes at once. A thread that did anything of
/// weight in between, the load of another tokenizer for one, is past it
/// and reads the table itself.
const LOAD_READ_HOLDS: Duration = Duration::from_millis(10);
/// A text at least this long may have the thread's packed table read back
/// before it (`PieceCache::reads_back`).
const READ_BACK_MIN_BYTES: usize = 4096;
/// The time a long text is given to encode, this and `READ_BACK_PER_BYTE`
/// for each of its bytes: a thread whose next long text starts later than
/// that is taken to have lost its processor in between.
const READ_BACK_BASE: Duration = Duration::from_micros(20);
const READ_BACK_PER_BYTE: Duration = Duration::from_nanos(32);
/// The longest gap the read answers: a wait for a processor lasts a few
/// slices of the scheduler, a longer one is a thread at rest.
const READ_BACK_WITHIN: Duration = Duration::from_millis(20);

/// The processors this process may run on, or `usize::MAX` where the
/// system does not say, so that no encoder has more threads inside a long
/// text than that.
fn processors() -> usize {
    std::thread::available_parallelism().map_or(usize::MAX, usize::from)
}

/// A thread inside a long text (`NativeEncoder::inside`). The count gives
/// it up on drop, so an encode that unwinds leaves the count right.
struct Inside<'a>(&'a AtomicUsize);

impl Drop for Inside<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}
/// The size a long text unit is cut into for the pool, at the first stable
/// boundary past each multiple of it.
const PARALLEL_CHUNK_BYTES: usize = 32 << 10;
/// The fewest bytes the units of a text may average for the pool's threads to
/// take their own units of it. A text cut finer, as a chat transcript is by
/// its added tokens, is read as the iterator splits it, a run of neighbours
/// a thread: threads that each take every n-th of many small units write the
/// same cache lines.
const OWN_UNIT_MIN_BYTES: usize = 1 << 10;
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
/// Ids the vocabulary seed's pass reads at a time: one list of this many
/// slots, three quarters of a megabyte, is all the pass holds of its reads
/// beside the tables, the long spellings and the model's own answers of
/// more than four ids, each in a block of its own (see `vocabulary_seed`).
const SEED_WINDOW: usize = 1 << 14;
/// Text on each side of a candidate cut that every stage's pattern is run
/// over to confirm a piece starts there (widened to the spaces around it:
/// the window starts where the text's match loop begins a piece and ends
/// just after such a space whatever the margin, see `window_start` and
/// `window_end`). The check walks the window three times (`piece_starts_at`),
/// serially in the cut search before the units encode, so the margin is
/// its cost: at 512 bytes it took a fifth of a long text's pooled encode
/// (2.3 MB, four threads: 8.4 -> 9.4 ms against the one-pass check before
/// it); at 128 the same cuts, the same ids, 8.0 ms.
const CUT_CHECK_WINDOW: usize = 128;
/// How far the cut search moves past a candidate without a window: at
/// least the margin (`CUT_CHECK_WINDOW`), so that the next candidate's walk
/// back can stop at this one's space and a blob followed by prose keeps
/// the prose's cuts, and no shorter than the margin was before it shrank:
/// each skipped candidate costs the walks back and forward over the reach,
/// and on a text whose candidates all lack a window (letters joined by
/// single spaces under a pattern that reads the join inside a match) a
/// skip of a quarter of this would read every byte of it four times over.
const CUT_SEARCH_SKIP: usize = 512;
/// How far before and after a candidate cut the window's start and end may
/// be looked for when a stage can read a space after a non-whitespace
/// character inside one match (a letter run joined by spaces, a class read
/// before a space), so that spaces lie inside pieces; a candidate without a
/// start or an end within this reach has no window and the cut search
/// skips it (a stretch without one is one piece under such a stage, so no
/// cut is lost). Without such a stage
/// the start and the end are looked for within a chunk instead (see
/// `cut_check_reach`): such a stretch is many pieces there, and a text
/// whose spaces stand further apart than this reach keeps its cuts. Two
/// kibibytes, on its own: the margin of the window (`CUT_CHECK_WINDOW`) is
/// the check's cost, this reach is how far a space may be from a candidate
/// under such a stage, and the one does not follow the other.
const CUT_CHECK_REACH: usize = 2 << 10;
/// Candidates one cut search may reject before it gives up: the byte scan for
/// a candidate is cheap, the window check on each one is not, so a text whose
/// candidates keep failing the check costs at most this many checks per cut.
const MAX_REJECTED_CUT_CANDIDATES: usize = 64;
/// Spans a thread's cut-check lists may keep between searches (64 KiB a
/// list): a window over prose holds tens, so prose never meets this; a
/// window over a long blob under a digit or punctuation split (up to a
/// chunk each side of the candidate, one span per piece) can grow a list
/// to tens of thousands, and the search gives such a list back once it is
/// done rather than keep it for the encoder's life (`stable_cut`).
const CUT_SPANS_KEPT: usize = 4 << 10;

/// Piece bytes to the ids they tokenize to.
type PieceIds = FxHashMap<Box<[u8]>, Box<[u32]>>;

/// The cut check's three span lists (the window's pieces, those of the
/// text before the candidate, those of the text from it on), one set per
/// thread that cuts (`NativeEncoder::cut_spans`).
type CutSpans = [Vec<(usize, usize)>; 3];

/// The ids of one vocabulary entry as a seed thread hands them to the
/// loading thread: their count and the first four, all the seed's table
/// keeps in an entry, so a slot of the window is 48 bytes; the loading
/// thread reads an entry of more ids again.
#[derive(Clone, Copy)]
struct SeededIds {
    count: u8,
    ids: [u32; INLINE_IDS],
}

impl SeededIds {
    fn of(ids: &[u32]) -> Self {
        let mut first = [0; INLINE_IDS];
        for (to, &id) in first.iter_mut().zip(ids) {
            *to = id;
        }
        Self {
            count: ids.len() as u8,
            ids: first,
        }
    }

    /// The ids, when the slot holds them all.
    fn inline(&self) -> Option<&[u32]> {
        self.ids.get(..usize::from(self.count))
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
    /// A short piece the model gave more ids than a slot holds, all of
    /// them: the loading thread asks the model for no entry, since a call
    /// leaves a record of the model in the calling thread's word cache.
    Asked(ShortPieceKey, Box<[u32]>),
    Long(Box<[u8]>),
}

const _: () = assert!(size_of::<Option<Option<Seeded>>>() == 48);

/// Most short pieces that wait for the seed's answer together (see
/// `Native::waits`): the far lines of that many are read at once, and more
/// buy nothing (measured on first sights of word lists: 4 buy a fifth of
/// what 16 do, 64 what 32 do).
const WAITING_PIECES: usize = 32;
/// Ids between two misses of a thread's packed table up to which the second
/// waits: misses that close together are text the thread has not met. A
/// miss further behind the last one stands alone and is answered where it
/// is met (measured: without the bound a thread that knows its text kept a
/// few pieces waiting over hundreds of tokens, 3 to 7 % of the time of a
/// text of 2k tokens).
const WAITING_GAP: usize = 32;
/// Slots of a seed up to which its lines are near a thread (4 MiB of
/// entries and tags): a first sight of the sentencepiece path waits only on
/// a larger seed (measured: with the wait, new text of 256 and of 2k tokens
/// was 11 to 13 % slower on a vocabulary of 32k entries and 5 to 15 %
/// faster by its median text on one of 262k).
const NEAR_SEED_SLOTS: usize = 131_072;

/// Pieces a unit cut by blocks holds and answers together
/// (`Native::answer_asked`).
const ASK_AHEAD: usize = 32;
/// Slots of a thread's packed table past which a unit cut by blocks holds
/// its pieces. A choice by workload: holding costs 25 to 30 instructions a
/// token, and under a bound of 16,384 slots a text of 32 thousand tokens
/// read 3 % slower, new and read again (measured).
const ASK_AHEAD_SLOTS: usize = 8 * THREAD_BACKING_SLOTS;

/// A piece held for `Native::answer_asked`.
#[derive(Clone, Copy)]
struct Asked {
    start: usize,
    end: usize,
    key: Option<ShortPieceKey>,
    hash: u64,
}

/// A short piece the thread's packed table missed, waiting for its answer
/// (see `Native::waits`): where the probe ended, with the piece's key, and
/// where its hole of one id stands in the output.
struct Waiting {
    vacancy: Vacancy,
    at: usize,
}

/// One thread's cache of the pieces it met with one encoder, each with the
/// ids the encoder's seed or the model gave it.
struct PieceCache {
    /// Pieces of up to `SHORT_PIECE_MAX_BYTES`, packed: a table made with
    /// the fewest slots when the cache is, grown with the pieces the thread
    /// meets, up to its share, and held plain (not in an `Option`): the hit
    /// path reads it at an offset of the cache, with no test of its
    /// presence per piece. Bounded by count together with `wide` and
    /// `short`.
    packed: ShortPieceTable,
    /// Pieces of 16 to 32 bytes whose ids fit an entry, packed, a line a
    /// piece (`wide_pieces`): nothing is allocated per entry. Bounded by
    /// their count together with `packed` and `short`.
    wide: WidePieceTable,
    /// The other pieces of more than `SHORT_PIECE_MAX_BYTES`, up to
    /// `CACHED_PIECE_MAX_BYTES`, bounded by their count together with
    /// `packed` and `wide`.
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
    /// The short pieces the packed table missed that wait for their answer
    /// (see `Native::waits`), in the order of their holes; empty between
    /// encode calls.
    waiting: Vec<Waiting>,
    /// The ids of the waiting pieces whose answer is not one id, from their
    /// answer until one pass places them in the output: a scratch list of
    /// the thread.
    answer: Vec<u32>,
    /// The length of the output at the packed table's last miss: a miss
    /// close behind it waits.
    last_miss: usize,
    /// How often the count-bounded tiers were cleared (`Self::clear_short`):
    /// a waiting piece's vacancy is of the table as it was before the next
    /// clear.
    clears: usize,
    /// `clears` when the waiting pieces were last answered: after a clear
    /// since then a waiting piece's vacancy may name no slot of the table
    /// as it is, and every waiting piece asks the table again.
    answered_clears: usize,
    /// Whether the thread's misses ask the encoder's learned table before
    /// they are merged and tell it after (`NativeEncoder::learned_pieces`):
    /// the thread has cleared, and the encoder's threads share what they
    /// merge (`NativeEncoder::enough_callers`), and the thread does not
    /// rest (`Self::judge_asks`). Set at the start of a call, and cleared
    /// where the thread starts to rest; the share does not hang on it.
    shares: bool,
    /// The calls up to the one that reads the merge table through
    /// (`NativeEncoder::piece_cache_for_text`): two in a new cache, zero
    /// once the table is read, or once the load's read stood in for it
    /// (`NativeEncoder::load_read_holds`).
    calls_to_read: u8,
    /// The asks of the learned table since the thread last judged them
    /// (`Self::judge_asks`), and those of them the table answered.
    asked: u32,
    of_use: u32,
    /// The entries the count-bounded tiers must hold before a thread that
    /// rests asks again; zero for a thread that does not rest.
    asks_from: usize,
    /// The vocabulary's form of a waiting piece being merged on the
    /// sentencepiece path: a scratch string of the thread.
    mapped: String,
    /// When the thread's last long text started, and its bytes
    /// (`Self::reads_back`); none when it ran with a processor to spare.
    last_long: Option<(Instant, usize)>,
}

impl PieceCache {
    /// The empty cache of a thread whose share of the count-bounded tiers
    /// is `share`: the packed table with the fewest slots, the wide table
    /// with none, the maps empty.
    fn with_share(share: usize) -> Self {
        Self {
            packed: ShortPieceTable::new(),
            wide: WidePieceTable::new(),
            short: PieceIds::default(),
            long: PieceIds::default(),
            long_bytes: 0,
            symbols: Vec::new(),
            unit_ascii: false,
            share,
            waiting: Vec::new(),
            answer: Vec::new(),
            last_miss: 0,
            clears: 0,
            answered_clears: 0,
            shares: false,
            calls_to_read: 2,
            asked: 0,
            of_use: 0,
            asks_from: 0,
            mapped: String::new(),
            last_long: None,
        }
    }

    /// Whether a long text of `bytes` bytes that starts at `now` has the
    /// packed table read back before it, and notes the text. It has when
    /// the long text before it started longer ago than that one is given
    /// to encode and within [`READ_BACK_WITHIN`]: the thread is taken to
    /// have lost its processor to another caller, and the table's lines
    /// to have left the near caches with it; read as a stream they cost
    /// less than fetched one probe at a time. A thread that did other
    /// work in between, a text the pool encoded for one, reads as well.
    fn reads_back(&mut self, now: Instant, bytes: usize) -> bool {
        let last = self.last_long.replace((now, bytes));
        last.is_some_and(|(then, last_bytes)| {
            let gap = now.saturating_duration_since(then);
            let last_bytes = u32::try_from(last_bytes).unwrap_or(u32::MAX);
            gap > READ_BACK_BASE + READ_BACK_PER_BYTE.saturating_mul(last_bytes)
                && gap < READ_BACK_WITHIN
        })
    }

    /// Bytes the long tier holds, used or not: its entries' pieces, ids and
    /// slots (`long_bytes`) and the empty slots of its map, which a clear
    /// in place keeps.
    fn long_held_bytes(&self) -> usize {
        self.long_bytes + (self.long.capacity() - self.long.len()) * LONG_PIECE_ENTRY_BYTES
    }

    /// Empties the long tier at an eviction for a share of `share` bytes: in
    /// place when its map's slots fit four times the share (the next misses
    /// fill them again without a growth), anew when they do not (a map grown
    /// under a wider share gives its slots back).
    fn clear_long_within(&mut self, share: usize) {
        if self.long.capacity() * LONG_PIECE_ENTRY_BYTES > 4 * share {
            self.remake_long();
        } else {
            self.long.clear();
            self.long_bytes = 0;
        }
    }

    /// Makes the long tier anew, its map's slots given back. The check at a
    /// call's start takes this rather than a clear in place: what the tier
    /// holds is never under its slots alone, so a map grown under a wider
    /// share to just under the bound would be over it again with its first
    /// entries back, and the check would empty the tier at every call.
    fn remake_long(&mut self) {
        self.long = PieceIds::default();
        self.long_bytes = 0;
    }

    /// Entries of the tiers bounded by count: `packed`, `wide` and `short`.
    fn short_entries(&self) -> usize {
        self.packed.len() + self.wide.len() + self.short.len()
    }

    /// The entries counted against `share`: all three tiers under a share
    /// that was cut, the wide table and the map alone at the full share,
    /// where the packed table is bounded by its slots.
    fn counted_entries(&self, share: usize) -> usize {
        if share < PIECE_CACHE_CAPACITY {
            self.short_entries()
        } else {
            self.wide.len() + self.short.len()
        }
    }

    /// Whether the tiers bounded by count are full under `share`. At the
    /// full share the packed table is counted by its slots, apart from the
    /// other two: one count over the three cleared them all while those
    /// slots were at most half full. A share that was cut keeps the one
    /// count, so those threads hold together what they held.
    fn short_full(&self, share: usize) -> bool {
        self.counted_entries(share) >= share
            || (share >= PIECE_CACHE_CAPACITY && self.packed.len() >= FULL_SHARE_PACKED)
    }

    /// Counts `asked` asks of the learned table, `of_use` of them answered
    /// by it, and judges every `ASKS_JUDGED` of them: where the table
    /// answered fewer than one in `ASKS_OF_USE_PART`, under one ask in
    /// two, the thread rests, which is to share no more until its
    /// count-bounded tiers hold `ASKS_RESTED` entries more
    /// (`NativeEncoder::piece_cache_for_call` reads it) or are cleared
    /// (`Self::clear_short`). The counts are the thread's own, so a caller
    /// and a thread of a pool are judged alike.
    fn judge_asks(&mut self, asked: u32, of_use: u32) {
        self.asked += asked;
        self.of_use += of_use;
        if self.asked >= ASKS_JUDGED {
            if self.of_use < self.asked / ASKS_OF_USE_PART {
                self.shares = false;
                self.asks_from = self.short_entries() + ASKS_RESTED;
            }
            self.asked = 0;
            self.of_use = 0;
        }
    }

    fn wide_slots(&self) -> usize {
        self.wide.slots()
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
    /// and a clear at the bound keeps the buckets); the wide table keeps its
    /// slots, or gives them back when it has more than the share takes.
    fn clear_short(&mut self, share: usize) {
        self.clears += 1;
        // The thread has lost its own words: a rest ends, and the asks it
        // makes from here are judged by themselves. `shares` is turned on
        // only at the start of a call, so a thread that rested shares
        // again from its next call.
        self.asked = 0;
        self.of_use = 0;
        self.asks_from = 0;
        self.packed.clear(share);
        self.wide.clear(share);
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
/// has combining class 0, no canonical or compatibility decomposition and
/// no part in any canonical composition
/// (`the_ideograph_range_is_settled_by_the_crates_own_tables` walks the
/// range against the crate's tables), so a run of them is in NFC and NFKC
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
/// ASCII stretch is in NFC and NFKC in every version of Unicode and no
/// normalization changes it; so is a run of ideographs (see
/// [`ideograph_run`]), and nothing composes with an ideograph on either
/// side. A run of other non-ASCII characters goes with the ASCII character
/// before it, the letter an accent at its start composes with; nothing else
/// crosses a cut: no composition has an ASCII character or an ideograph as
/// its second element, no decomposition (canonical or compatibility)
/// reaches across one, and both are starters, so the canonical order of
/// marks never spans one.
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

/// Whether `text` is in `form` by the tables `tokenizers` normalizes with:
/// the crate's check on each stretch that is not settled (the quick check
/// answers for most text; where it cannot, marks that might compose or
/// reorder, the stretch is normalized and compared), none on the ASCII and
/// ideograph ones.
fn is_normalized(text: &str, form: Normalizer) -> bool {
    stretches(text).all(|(_, stretch, settled)| settled || form.holds(stretch))
}

/// `text` in `form` by the tables `tokenizers` normalizes with, borrowed
/// when it is in that form already: only a stretch that is not settled and
/// not in the form goes through the crate's normalizer (its iterator
/// reports a byte-offset shift with each character, which this path has no
/// use for), the other stretches are copied. A stretch is read by the NFC
/// quick check where the form is NFC and the caller has one (its tables
/// place most characters; the library reads the windows they cannot), by
/// the library's check for the form otherwise.
fn normalize<'a>(text: &'a str, form: Normalizer, quick: Option<&NfcQuick>) -> Cow<'a, str> {
    let in_form = |stretch: &str| match (form, quick) {
        (Normalizer::Nfc, Some(quick)) => quick.is_nfc(stretch),
        _ => form.holds(stretch),
    };
    let mut owned: Option<String> = None;
    for (start, stretch, settled) in stretches(text) {
        if settled || in_form(stretch) {
            if let Some(out) = &mut owned {
                out.push_str(stretch);
            }
        } else {
            let out = owned.get_or_insert_with(|| {
                let mut out = String::with_capacity(text.len() + 16);
                out.push_str(&text[..start]);
                out
            });
            form.normalize_into(stretch, out);
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

/// The normalizer the path applies before the cut after normalization:
/// none, or one of the two composed Unicode forms `tokenizers` has
/// (compatibility characters, fullwidth forms and ligatures among them,
/// spell out under NFKC and stay as they are under NFC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Normalizer {
    None,
    Nfc,
    Nfkc,
}

impl Normalizer {
    /// Whether `stretch` is in this form by the library's tables (every
    /// text is, under no normalizer).
    fn holds(self, stretch: &str) -> bool {
        match self {
            Self::None => true,
            Self::Nfc => unicode_normalization_alignments::is_nfc(stretch),
            Self::Nfkc => unicode_normalization_alignments::is_nfkc(stretch),
        }
    }

    /// `stretch` in this form, appended to `out`.
    fn normalize_into(self, stretch: &str, out: &mut String) {
        match self {
            Self::None => out.push_str(stretch),
            Self::Nfc => out.extend(stretch.nfc().map(|(ch, _)| ch)),
            Self::Nfkc => out.extend(stretch.nfkc().map(|(ch, _)| ch)),
        }
    }
}

/// One `Split` of the pre-tokenizer chain.
struct Stage {
    pattern: Pattern,
    /// The spans of the stage's text that go on as pieces.
    pieces: Pieces,
    /// A `Digits` split: from it on, every piece that holds a decimal digit
    /// lies inside a run of numeric characters, which holds no whitespace
    /// (`NativeEncoder::check_anchors` reads it).
    digits: bool,
}

/// The spans of a stage's text that go on as its pieces: the reference's
/// split behaviors the path reproduces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pieces {
    /// `Isolated`: the matches and the text between them.
    MatchesAndGaps,
    /// `Removed` inverted: the matches alone.
    Matches,
    /// `Removed`: the text between the matches alone; a match is dropped
    /// and the pieces on either side of it stay apart.
    Gaps,
    /// `MergedWithNext`: each match with the text after it up to the next
    /// match or the end, as one piece; the text before the first match
    /// stands alone.
    MatchesWithNextGap,
}

impl Pieces {
    /// How a walk cuts a text under this kind: the cursor loop over the
    /// matches with its two switches, or the merged loop. Every walker asks
    /// this first, so a kind the cursor loop does not know never falls
    /// through its switches silently.
    fn walk(self) -> Walk {
        match self {
            Self::MatchesAndGaps => Walk::Apart {
                gaps: true,
                matches: true,
            },
            Self::Matches => Walk::Apart {
                gaps: false,
                matches: true,
            },
            Self::Gaps => Walk::Apart {
                gaps: true,
                matches: false,
            },
            Self::MatchesWithNextGap => Walk::Merged,
        }
    }
}

/// The loop a walk runs under a [`Pieces`] kind (see [`Pieces::walk`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Walk {
    /// Each match and each gap on its own: `gaps` says the text between and
    /// around the matches goes on as pieces, `matches` says the matches do.
    Apart { gaps: bool, matches: bool },
    /// Each match with the text after it up to the next match or the end.
    Merged,
}

/// An added token the path cuts out: its content and id, and whether the
/// whitespace before or after a match is the token's (`lstrip`, `rstrip`,
/// as `AddedVocabulary` reads them). Ordered by content, so a list sorts
/// into a stable order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Added {
    content: String,
    id: u32,
    lstrip: bool,
    rstrip: bool,
}

impl Added {
    /// A token that takes no whitespace: its content and id.
    pub(crate) fn plain(content: String, id: u32) -> Self {
        Self {
            content,
            id,
            lstrip: false,
            rstrip: false,
        }
    }
}

/// Added tokens, the list a matcher is built from.
type AddedTokenList = Vec<Added>;

/// One set of added tokens, matched leftmost-longest as `AddedVocabulary`
/// matches them. The automaton's own search scans a text for the tokens'
/// first bytes when there are at most three of them, all ASCII, each `<`,
/// `[` or `]`; a set of two or three with another first byte takes a byte
/// search of its own ([`Search::FirstBytes`]), unless a token of it holds
/// more than one place of that search past its start. A set the scan cannot
/// take (a token starting beyond ASCII, or more than three first bytes) the
/// automaton searches by the tokens' rarest bytes instead, which for a
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
    /// Per token, whether the whitespace before / after a match is the
    /// token's; none when no token of the set takes any.
    strips: Option<Vec<(bool, bool)>>,
    prefixes: Option<Prefixes>,
}

/// The distinct prefixes of a set's tokens as a packed searcher: the longest
/// length up to [`Self::MAX_LEN`] bytes at which there are at most
/// [`Self::MAX_COUNT`] of them (a token shorter than the length stands
/// whole), so a hit is as close to a token as the set allows; a set that
/// comes to at most [`Self::NARROW_COUNT`] with [`Self::NARROW_MIN_LEN`]
/// bytes or more is cut to that length, for the library's narrow form. A
/// set whose prefixes would be shorter than [`Self::MIN_LEN`] bytes keeps the
/// automaton's own search. When the library declines the prefixes that fit
/// (its SIMD searcher takes a one-byte pattern among at most sixteen, and a
/// set with newline or tab runs among its tokens has such prefixes), the
/// tokens' distinct first bytes stand in when there are at most
/// [`Self::MAX_FIRST_BYTES`]: a hit then is a byte that starts some token,
/// and the automaton anchored there decides, where the unanchored automaton
/// walked every byte of the text (measured at about thirty instructions per
/// byte on a 6,415-token set). A set of two or three first bytes, all ASCII,
/// takes no prefixes: a byte search for them finds its places
/// ([`Search::FirstBytes`]).
struct Prefixes {
    searcher: Search,
}

impl Prefixes {
    const MAX_LEN: usize = 16;
    const MIN_LEN: usize = 3;
    const MAX_COUNT: usize = 64;
    /// The most patterns the library's packed searcher takes in its narrow
    /// form (as of its version 1.1), which reads up to twice the bytes a
    /// step that its wide form reads.
    const NARROW_COUNT: usize = 32;
    /// The shortest prefixes the narrow form is preferred with: a prefix of
    /// this many bytes is nearly as rare in text as its token, where a
    /// shorter one would stop the pass at places no token stands.
    const NARROW_MIN_LEN: usize = 8;
    const MAX_FIRST_BYTES: usize = 16;
    /// The most places of the byte search a token may hold past its own
    /// start: one, the slash of `</think>` in a set with `/think`.
    const MAX_INNER: usize = 1;

    /// Whether a search of our own beats the automaton's on `tokens`: a first
    /// byte beyond ASCII or more than three distinct ones, which its
    /// first-byte scan declines; and two or three unless each is `<`, `[` or
    /// `]`: beside a first byte more common in text (a slash) the automaton
    /// scans for the tokens' rarest bytes instead, letters among them, and
    /// text is full of those.
    fn wanted(tokens: &[Added]) -> bool {
        let mut firsts: Vec<u8> = tokens
            .iter()
            .filter_map(|token| token.content.bytes().next())
            .collect();
        firsts.sort_unstable();
        firsts.dedup();
        firsts.len() > 3
            || firsts.iter().any(|first| !first.is_ascii())
            || (firsts.len() > 1 && firsts.iter().any(|first| !b"<[]".contains(first)))
    }

    /// The distinct prefixes of `tokens` of up to `len` bytes.
    fn distinct(tokens: &[Added], len: usize) -> Vec<&[u8]> {
        let mut prefixes: Vec<&[u8]> = tokens
            .iter()
            .map(|token| &token.content.as_bytes()[..token.content.len().min(len)])
            .collect();
        prefixes.sort_unstable();
        prefixes.dedup();
        prefixes
    }

    /// The prefixes of `tokens` a searcher is made of: the longest of at
    /// most [`Self::NARROW_COUNT`] when their length is
    /// [`Self::NARROW_MIN_LEN`] bytes or more, so that the library takes its
    /// narrow form; else the longest of at most [`Self::MAX_COUNT`]. None
    /// when the prefixes that fit are shorter than [`Self::MIN_LEN`] bytes
    /// for a token longer than them (a set of added words: the pass would
    /// stop at nearly every byte of ordinary text, where the automaton's own
    /// walk is cheaper), or when even the one-byte prefixes exceed
    /// [`Self::MAX_COUNT`].
    fn chosen(tokens: &[Added]) -> Option<Vec<&[u8]>> {
        // The longest length from `shortest` up with at most `most` distinct
        // prefixes. Their count never grows as the length shrinks, so the
        // first length that fits, read from the longest down, is the one.
        let longest = |shortest: usize, most: usize| {
            (shortest..=Self::MAX_LEN)
                .rev()
                .map(|len| (len, Self::distinct(tokens, len)))
                .find(|(_, prefixes)| prefixes.len() <= most)
        };
        let (len, prefixes) = longest(Self::NARROW_MIN_LEN, Self::NARROW_COUNT)
            .or_else(|| longest(1, Self::MAX_COUNT))?;
        let cut_short = tokens.iter().any(|token| token.content.len() > len);
        (!cut_short || len >= Self::MIN_LEN).then_some(prefixes)
    }

    /// What searches for `tokens`: the byte search for two or three first
    /// bytes, all ASCII; else the searcher over the prefixes
    /// ([`Self::chosen`]), none when there are none to take or the library
    /// declines the set, and none where the byte search would have a text
    /// walked over and over.
    fn of(tokens: &[Added]) -> Option<Self> {
        let firsts = Self::distinct(tokens, 1);
        let bytes: Vec<u8> = firsts
            .iter()
            .filter_map(|first| first.first().copied())
            .collect();
        if let [a, .., c] = bytes[..] {
            if bytes.len() <= 3 && bytes.iter().all(u8::is_ascii) {
                let bytes = [a, bytes[1], c];
                let mut seconds = [[false; 256]; 3];
                for token in tokens {
                    let content = token.content.as_bytes();
                    let Some(which) = bytes.iter().position(|byte| content.first() == Some(byte))
                    else {
                        continue;
                    };
                    match content.get(1) {
                        Some(&second) => seconds[which][usize::from(second)] = true,
                        None => seconds[which] = [true; 256],
                    }
                }
                // A token that holds places of this search past its own start
                // has the anchored automaton walk them again from each. With
                // at most `MAX_INNER` in a token, no byte of a text is walked
                // more than that plus three times; a set beyond it (`aaaab`: a
                // text of `a` walked from every byte) keeps the automaton's
                // own search.
                let inner = |token: &Added| {
                    let content = token.content.as_bytes();
                    let pairs = content.iter().zip(content.iter().skip(1));
                    pairs
                        .skip(1)
                        .filter(|(first, second)| {
                            bytes
                                .iter()
                                .position(|byte| byte == *first)
                                .is_some_and(|which| seconds[which][usize::from(**second)])
                        })
                        .count()
                };
                if tokens.iter().any(|token| inner(token) > Self::MAX_INNER) {
                    return None;
                }
                let pair = Pair::of(bytes, &seconds);
                return Some(Self {
                    searcher: Search::FirstBytes(bytes, Box::new(seconds), pair),
                });
            }
        }
        let prefixes = Self::chosen(tokens)?;
        let searcher = Self::searcher(&prefixes).or_else(|| {
            (firsts.len() <= Self::MAX_FIRST_BYTES)
                .then(|| Self::searcher(&firsts))
                .flatten()
        })?;
        Some(Self {
            searcher: Search::Packed(searcher),
        })
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

    /// The first position at or after `from` where a token may start.
    /// `sought` is this text's: through it the byte search counts the places
    /// of a common first byte that start nothing and hands the text to the
    /// split search ([`Pair`]).
    fn next(&self, text: &str, from: usize, sought: &mut Sought) -> Option<usize> {
        match &self.searcher {
            Search::Packed(searcher) => searcher
                .find_in(text, Span::from(from..text.len()))
                .map(|found| found.start()),
            Search::FirstBytes([a, b, c], seconds, pair) => {
                let bytes = text.as_bytes();
                let mut from = from;
                loop {
                    if let Some(pair) = pair {
                        if sought.misses >= Pair::MISSES {
                            return pair.next(bytes, from, [*a, *b, *c], seconds, sought);
                        }
                    }
                    let mut at = from + memchr::memchr3(*a, *b, *c, bytes.get(from..)?)?;
                    // A first byte right behind one that starts no token (a
                    // doubled slash) is tested in place: a new search call
                    // would only find it again.
                    loop {
                        let which = match bytes.get(at)? {
                            byte if byte == a => 0,
                            byte if byte == b => 1,
                            _ => 2,
                        };
                        match bytes.get(at + 1) {
                            Some(&second) if !seconds[which][usize::from(second)] => {
                                if pair.as_ref().is_some_and(|pair| pair.common == which) {
                                    sought.misses = sought.misses.saturating_add(1);
                                    if sought.misses >= Pair::MISSES {
                                        from = at + 1;
                                        break;
                                    }
                                }
                                if [*a, *b, *c].contains(&second) {
                                    at += 1;
                                } else {
                                    from = at + 2;
                                    break;
                                }
                            }
                            _ => return Some(at),
                        }
                    }
                }
            }
        }
    }
}

/// What finds the next place a token may start: the packed prefixes, or,
/// for two or three ASCII first bytes, a byte search for them (the last
/// stands twice for two) that passes a place only when the byte after it
/// follows that first byte in some token, so that the automaton is not
/// asked at every slash or bracket of a text. Where one of the first bytes
/// is common in text, a text dense in it goes on by the split search
/// ([`Pair`]).
#[expect(
    clippy::large_enum_variant,
    reason = "one per tokenizer, never moved once built; a box adds a load to a search call"
)]
enum Search {
    Packed(packed::Searcher),
    FirstBytes([u8; 3], Box<[[bool; 256]; 3]>, Option<Pair>),
}

/// The split search of a set with one first byte common in text (a slash)
/// that every token of it follows with the same second byte: that pair is
/// searched as two bytes and the other first bytes by the byte search, so the
/// common byte alone stops nothing. A text takes it after [`Self::MISSES`]
/// places of the common byte that start no token; until then it pays what
/// the search for all three bytes pays.
struct Pair {
    common: usize,
    rare: [u8; 2],
    finder: memchr::memmem::Finder<'static>,
}

impl Pair {
    /// Two: a text with one such place keeps the one search, and the two
    /// searches of the split start where the common byte has stopped it twice.
    const MISSES: u8 = 2;

    /// The split of `firsts` (sorted, the last standing twice for two), none
    /// unless one of them alone is common and one byte follows it.
    fn of(firsts: [u8; 3], seconds: &[[bool; 256]; 3]) -> Option<Self> {
        let mut distinct = firsts.to_vec();
        distinct.dedup();
        let (rare, commons): (Vec<u8>, Vec<u8>) = distinct
            .iter()
            .copied()
            .partition(|first| b"<[]".contains(first));
        let [common] = commons[..] else {
            return None;
        };
        let which = firsts.iter().position(|first| *first == common)?;
        let mut follows = (0..=u8::MAX).filter(|second| seconds[which][usize::from(*second)]);
        let (Some(second), None) = (follows.next(), follows.next()) else {
            return None;
        };
        Some(Self {
            common: which,
            rare: [*rare.first()?, *rare.last()?],
            finder: memchr::memmem::Finder::new(&[common, second]).into_owned(),
        })
    }

    /// The first place at or after `from`: the earlier of the next rare first
    /// byte a token can go on from and the next pair, each kept in `sought`
    /// until the search has passed it. `from` is past the start of the text
    /// here (a place was missed), so the zeros of a new `sought` are stale.
    fn next(
        &self,
        bytes: &[u8],
        from: usize,
        firsts: [u8; 3],
        seconds: &[[bool; 256]; 3],
        sought: &mut Sought,
    ) -> Option<usize> {
        if sought.rare < from {
            sought.rare = self.rare_from(bytes, from, firsts, seconds);
        }
        if sought.pair < from {
            sought.pair = bytes
                .get(from..)
                .and_then(|rest| self.finder.find(rest))
                .map_or(usize::MAX, |found| from + found);
        }
        let at = sought.rare.min(sought.pair);
        (at != usize::MAX).then_some(at)
    }

    /// The first rare first byte at or after `from` that a token can go on
    /// from, `usize::MAX` for none.
    fn rare_from(
        &self,
        bytes: &[u8],
        mut from: usize,
        firsts: [u8; 3],
        seconds: &[[bool; 256]; 3],
    ) -> usize {
        let [a, b] = self.rare;
        while let Some(found) = bytes
            .get(from..)
            .and_then(|rest| memchr::memchr2(a, b, rest))
        {
            let at = from + found;
            let which = firsts
                .iter()
                .position(|first| bytes.get(at) == Some(first))
                .unwrap_or(0);
            match bytes.get(at + 1) {
                Some(&second) if !seconds[which][usize::from(second)] => from = at + 1,
                _ => return at,
            }
        }
        usize::MAX
    }
}

/// What the search of one text remembers between its calls.
#[derive(Default)]
struct Sought {
    misses: u8,
    rare: usize,
    pair: usize,
}

impl AddedTokens {
    /// The matcher over `tokens`, none for no tokens.
    pub(crate) fn build(tokens: &[Added]) -> Result<Option<Self>, Declined> {
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
            .build(tokens.iter().map(|token| token.content.as_bytes()))
            .map_err(|e| Declined::from(format!("the added tokens form no automaton: {e}")))?;
        Ok(Some(Self {
            automaton,
            ids: tokens.iter().map(|token| token.id).collect(),
            strips: tokens
                .iter()
                .any(|token| token.lstrip || token.rstrip)
                .then(|| {
                    tokens
                        .iter()
                        .map(|token| (token.lstrip, token.rstrip))
                        .collect()
                }),
            prefixes,
        }))
    }

    /// The added tokens in `text`, in order, each as its byte range and its
    /// index in the list: the leftmost-longest matches, each continuing after
    /// the last (after the token's own end: a range widened over whitespace,
    /// [`Self::stripped`], does not move the search, as the reference's loop
    /// does not). With prefixes, a token starts only where a prefix stands, so
    /// the first prefix at or after the last match that begins a token begins
    /// the next one, and the longest token there is the match. A match the
    /// widening before it leaves empty (a whitespace token inside the
    /// whitespace a stripped neighbour took) is no match: the reference drops
    /// the empty split it makes of it; it leaves the end of the last match
    /// where it was, so that no later widening reaches into an earlier range
    /// (the reference's own loop fails on such a text).
    fn matches<'a>(&'a self, text: &'a str) -> impl Iterator<Item = (Range<usize>, usize)> + 'a {
        let mut at = 0;
        let mut taken = 0;
        let mut sought = Sought::default();
        std::iter::from_fn(move || loop {
            let Some(prefixes) = &self.prefixes else {
                let input = Input::new(text).range(at..);
                let found = self.automaton.find(input)?;
                at = found.end();
                let index = found.pattern().as_usize();
                let range = self.stripped(text, found.range(), index, taken);
                if range.is_empty() {
                    continue;
                }
                taken = range.end;
                return Some((range, index));
            };
            let start = prefixes.next(text, at, &mut sought)?;
            let input = Input::new(text).range(start..).anchored(Anchored::Yes);
            if let Some(found) = self.automaton.find(input) {
                at = found.end();
                let index = found.pattern().as_usize();
                let range = self.stripped(text, found.range(), index, taken);
                if range.is_empty() {
                    continue;
                }
                taken = range.end;
                return Some((range, index));
            }
            at = start + 1;
        })
    }

    /// The match `found` of the token at `index`, widened over the whitespace
    /// the token takes: with `rstrip` the whitespace characters after it, with
    /// `lstrip` those before it, not past `taken`, the end of the previous
    /// match (`AddedVocabulary::find_matches`; its whitespace is Unicode
    /// `White_Space`, which is `char::is_whitespace`).
    fn stripped(
        &self,
        text: &str,
        mut found: Range<usize>,
        index: usize,
        taken: usize,
    ) -> Range<usize> {
        let Some(strips) = &self.strips else {
            return found;
        };
        let (lstrip, rstrip) = strips[index];
        if lstrip {
            let before: usize = text[..found.start]
                .chars()
                .rev()
                .take_while(|c| c.is_whitespace())
                .map(char::len_utf8)
                .sum();
            found.start = (found.start - before).max(taken);
        }
        if rstrip {
            let after: usize = text[found.end..]
                .chars()
                .take_while(|c| c.is_whitespace())
                .map(char::len_utf8)
                .sum();
            found.end += after;
        }
        found
    }

    /// The byte ranges of the added tokens in `text`, in order.
    pub(crate) fn ranges<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Range<usize>> + 'a {
        self.matches(text).map(|(range, _)| range)
    }

    /// The added tokens in `text`, in order, each as its byte range and its
    /// id.
    pub(crate) fn tokens<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = (Range<usize>, u32)> + 'a {
        self.matches(text)
            .map(|(range, index)| (range, self.ids[index]))
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

/// A raw added token, or a text between raw added tokens in the
/// normalizer's form: borrowed where it was in that form already, owned
/// where normalizing changed it (see [`NativeEncoder::normalized_gaps`]).
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
    /// check every encode under it starts with (see [`Self::in_form`]; the
    /// NFKC form has no quick check and reads the library's).
    nfc_quick: Option<NfcQuick>,
    /// Whether a stage holds the end anchor `$`: such a text is never cut
    /// into units for the thread pool (see [`Self::stable_cut`]).
    end_anchored: bool,
    /// The stages from this index on run over each piece's byte-level
    /// image, the alphabet character of each of its bytes, rather than
    /// over the raw text: a stage after the `ByteLevel` whose pattern can
    /// read beyond printable ASCII cuts the image where the reference does,
    /// which is not where it would cut the raw text (see
    /// [`Self::pieces_in_alphabet`]). `None` when every stage runs over the
    /// raw text. Such a text is never cut into units for the thread pool
    /// either (see [`Self::stable_cut`]).
    mapped_from: Option<usize>,
    /// For the stages over the image: whether the image of each byte is a
    /// character one of their matches can begin with. A piece that holds no
    /// such byte holds no match of any of them and goes through every one
    /// whole, so it is final as it stands, with no image built and no search
    /// started (see [`Self::pieces_mapping`]). `None` when no stage runs
    /// over the image, or one of them names no opening class or drops the
    /// text outside its matches.
    image_opens: Option<Box<[bool; 256]>>,
    /// Added tokens matched on the raw text (`normalized: false`).
    raw_added: Option<AddedTokens>,
    /// Added tokens matched after normalization (`normalized: true`).
    normalized_added: Option<AddedTokens>,
    stages: Vec<Stage>,
    /// Whether a stage can read a space after a non-whitespace character
    /// inside one match ([`Pattern::reads_space_inside`]): the thread
    /// pool's cut check then chooses its window's start with care
    /// ([`Self::window_start`]).
    space_inside: bool,
    /// Whether a stage is the byte-level split, which its scanner matches
    /// in the walk: the walk then runs in the copy that carries the scanner
    /// ([`Self::staged_pieces`]).
    byte_split: bool,
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
    /// What the threads have merged, kept for all of them: asked by a
    /// thread that shares (`PieceCache::shares`), for a piece its own table
    /// and the seed do not hold, before the merge
    /// ([`Self::learned_pieces`]).
    learned: OnceLock<LearnedPieces>,
    /// Whether the seed holds every vocabulary entry of up to
    /// `SHORT_PIECE_MAX_BYTES` raw bytes: the ids `0..vocab_size` were all
    /// entries when it was read. Then a short piece the seed misses is no
    /// entry, and the whole-piece lookup `ignore_merges` asks for is spared
    /// (see [`Self::tokenize_piece`]). A vocabulary with gaps in its ids has
    /// entries past that range the seed did not read, and keeps the lookup.
    seed_covers_short_entries: bool,
    /// Whether a young thread's misses wait for their answers
    /// ([`Self::waits`]). The waiting pieces are answered with no model at
    /// hand, so they wait only where none is asked: on an encoder whose
    /// table merges every short piece
    /// ([`Self::merges_every_short_piece`]). Only a miss that waits asks
    /// the learned table, so only such an encoder's threads share what
    /// they merge ([`Self::enough_callers`]). The sentencepiece path, whose
    /// character symbols merge every piece, waits by its own rule
    /// ([`Self::seeded_raw`]) with this flag off: its threads do not share.
    misses_wait: bool,
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
    /// The cut check's span lists of each thread that cuts a text for the
    /// pool: cleared per check and grown once per thread
    /// ([`Self::piece_starts_at`]), given back after a search that grew one
    /// past [`CUT_SPANS_KEPT`] ([`Self::stable_cut`]), apart from the piece
    /// cache so that a thread that only cuts gets none and is not counted
    /// among the threads that encode.
    cut_spans: ThreadLocal<RefCell<CutSpans>>,
    /// Threads that hold a piece cache of this encoder.
    threads: AtomicUsize,
    /// Those of them that call: the threads that were no thread of a pool
    /// at their first encode with this encoder ([`Self::thread_piece_cache`]
    /// counts them, `LEARNED_FROM_THREADS` asks for them).
    callers: AtomicUsize,
    /// The thread that loaded this encoder, and when its rehearsal had read
    /// the merge table through ([`Self::rehearse`]).
    load_read: OnceLock<(ThreadId, Instant)>,
    /// [`processors`] when the encoder was made: while more threads than
    /// this are inside a long text, a thread reads its table back
    /// (`PieceCache::reads_back`).
    processors: usize,
    /// The threads that encode the units of a long text of their own now
    /// (`READ_BACK_MIN_BYTES` or more): no unit on the pool, and no thread
    /// that still normalizes or cuts its text or waits for the pool.
    inside: AtomicUsize,
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
        // the flag keeps the byte-level path, wherever its `ByteLevel` sits.
        if bpe.byte_fallback && !Self::has_byte_level(&pre_tokenizer) {
            return Self::sentencepiece_from(tokenizer, &pre_tokenizer);
        }
        let normalizer =
            Self::normalizer_from(&serialized("normalizer", tokenizer.get_normalizer())?)?;
        let (stages, mapped_from) = Self::stages_from(&pre_tokenizer)?;
        let frame = Self::frame_from(&serialized(
            "post-processor",
            tokenizer.get_post_processor(),
        )?)?;
        let (raw, normalized) = Self::added_tokens_of(tokenizer, normalizer)?;
        let merges = match MergeTable::from_model(
            tokenizer.get_model(),
            ByteTokens::Alphabet,
            tokenizer.get_model().read(),
        ) {
            Ok(table) => {
                tracing::debug!(
                    merges = table.merges(),
                    bytes = table.bytes(),
                    preformed_characters = table.preformed_characters(),
                    missing_bytes = table.missing_bytes(),
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
        // The one-pass chain scanner reads the raw text: not for stages
        // over the image.
        let chain = match mapped_from {
            None => chain::Scanner::recognize(&stages),
            Some(_) => None,
        };
        // One alphabet for the walk's map and for the table read through it.
        let alphabet = bytes_char();
        Ok(Self {
            end_anchored: stages.iter().any(|stage| stage.pattern.end_anchored()),
            mapped_from,
            image_opens: Self::image_opening_bytes(&stages, mapped_from, &alphabet),
            nfc_quick: (normalizer == Normalizer::Nfc).then(NfcQuick::read),
            normalizer,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            byte_split: stages.iter().any(|stage| stage.pattern.has_byte_split()),
            space_inside: stages
                .iter()
                .any(|stage| stage.pattern.reads_space_inside()),
            stages,
            chain,
            frame,
            alphabet,
            seed: ShortPieceTable::for_entries(0),
            learned: OnceLock::new(),
            seed_covers_short_entries: false,
            misses_wait: false,
            long_entries: None,
            merges,
            chars: None,
            piece_cache: ThreadLocal::new(),
            cut_spans: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
            callers: AtomicUsize::new(0),
            load_read: OnceLock::new(),
            processors: processors(),
            inside: AtomicUsize::new(0),
        })
    }

    /// This path's encoder for a vocabulary with byte fallback (see
    /// [`sentencepiece`]), or why it declines: the spaces of the text become
    /// the word marker by a `Replace` normalizer alone (a `Split` on the
    /// space behind it cuts nothing then, and is accepted) or by the
    /// `Metaspace` pre-tokenizer of the Llama class with no normalizer, whose
    /// prepend scheme places the marker, or by a `Prepend` of the marker
    /// before the `Replace` (the legacy conversion's shape), which places it
    /// before every text; anything else declines, as does an added token
    /// matched after the `Replace` with a space or a marker in it, or with
    /// `lstrip` / `rstrip` (matched on the raw text here, its spelling and
    /// the whitespace it takes would not be the ones `tokenizers` sees after
    /// the normalizer), and any added token matched after the `Prepend` (a
    /// text after it gets no marker there; one after a raw added token does).
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
        if marker == Marker::Every && !normalized.is_empty() {
            return Err(Declined::from(
                "a byte-fallback vocabulary with an added token matched after the Prepend \
                 normalizer",
            ));
        }
        // The tokens matched after the Replace are matched on the raw text
        // here; one whose strip takes whitespace would take a raw space,
        // which the reference, matching the normalized piece, sees as the
        // word marker and leaves to the text.
        if replaces
            && normalized.iter().any(|token| {
                token.content.contains([' ', WORD_MARKER]) || token.lstrip || token.rstrip
            })
        {
            return Err(Declined::from(
                "a byte-fallback vocabulary with an added token matched after the Replace \
                 normalizer that holds a space or the word marker, or takes the whitespace \
                 beside it",
            ));
        }
        let chars =
            CharSymbols::from_model(tokenizer.get_model(), marker, tokenizer.get_model().read())?;
        tracing::debug!(
            merges = chars.merges(),
            bytes = chars.bytes(),
            bytes_without_a_token = chars.bytes_without_a_token(),
            ?marker,
            "the native encode path takes the byte-fallback vocabulary from character symbols"
        );
        Ok(Self {
            end_anchored: false,
            mapped_from: None,
            image_opens: None,
            normalizer: Normalizer::None,
            nfc_quick: None,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            stages: Vec::new(),
            space_inside: false,
            byte_split: false,
            chain: None,
            frame,
            alphabet: bytes_char(),
            seed: ShortPieceTable::for_entries(0),
            learned: OnceLock::new(),
            seed_covers_short_entries: false,
            misses_wait: false,
            long_entries: None,
            merges: None,
            chars: Some(Box::new(chars)),
            piece_cache: ThreadLocal::new(),
            cut_spans: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
            callers: AtomicUsize::new(0),
            load_read: OnceLock::new(),
            processors: processors(),
            inside: AtomicUsize::new(0),
        })
    }

    /// Whether a pre-tokenizer (serialized) holds a `ByteLevel` stage, the
    /// shape of the byte-level path ([`Self::stages_from`], which finds the
    /// stage by position and takes stages after it).
    fn has_byte_level(pre_tokenizer: &Value) -> bool {
        let is_byte_level =
            |item: &Value| item.get("type").and_then(Value::as_str) == Some("ByteLevel");
        match pre_tokenizer.get("pretokenizers").and_then(Value::as_array) {
            Some(items) => items.iter().any(is_byte_level),
            None => is_byte_level(pre_tokenizer),
        }
    }

    /// Where the word marker comes from on a byte-fallback vocabulary, and
    /// whether a `Replace` normalizer puts it for the spaces (true in the
    /// answer; the `Metaspace` pre-tokenizer does so itself): `Never` under
    /// the normalizer, `Every` under a `Prepend` of the marker before it, the
    /// scheme of the pre-tokenizer without one; or why the pipeline declines.
    fn marker_from(normalizer: &Value, pre_tokenizer: &Value) -> Result<(Marker, bool), Declined> {
        let replace = serde_json::json!(
            {"type": "Replace", "pattern": {"String": " "}, "content": WORD_MARKER.to_string()}
        );
        let prepend = serde_json::json!({"type": "Prepend", "prepend": WORD_MARKER.to_string()});
        // Alone, as the one step of a `Sequence`, as some conversions write
        // it, or after a `Prepend` of the marker (the legacy conversion's
        // shape, which puts the marker before every text).
        let steps = match normalizer.get("type").and_then(Value::as_str) {
            Some("Sequence") => normalizer
                .get("normalizers")
                .and_then(Value::as_array)
                .map(Vec::as_slice),
            _ => None,
        };
        let prepends =
            matches!(steps, Some([first, second]) if *first == prepend && *second == replace);
        let replaces =
            prepends || *normalizer == replace || matches!(steps, Some([step]) if *step == replace);
        if !(normalizer.is_null() || replaces) {
            return Err(Declined::from(
                "a byte-fallback vocabulary with a normalizer other than Replace(space, marker), \
                 alone or after Prepend(marker), or none",
            ));
        }
        if pre_tokenizer.is_null() {
            return if prepends {
                Ok((Marker::Every, true))
            } else if replaces {
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
                    && !prepends
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
            other if prepends => Err(Declined::from(format!(
                "a byte-fallback vocabulary with the pre-tokenizer {other} under the Prepend \
                 normalizer (the path takes that normalizer with none)"
            ))),
            other => Err(Declined::from(format!(
                "a byte-fallback vocabulary with the pre-tokenizer {other} (only none or a Split \
                 on the space after the Replace normalizer, or Metaspace without split)"
            ))),
        }
    }

    /// The normalizer this path applies: none, or NFC or NFKC alone.
    fn normalizer_from(normalizer: &Value) -> Result<Normalizer, Declined> {
        if normalizer.is_null() {
            return Ok(Normalizer::None);
        }
        match type_of("normalizer", normalizer)? {
            "NFC" => Ok(Normalizer::Nfc),
            "NFKC" => Ok(Normalizer::Nfkc),
            "Sequence" => {
                let steps = normalizer
                    .get("normalizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Declined::from("a normalizer sequence without a list"))?;
                if steps.is_empty() {
                    Ok(Normalizer::None)
                } else {
                    Err(Declined::from(
                        "a normalizer sequence with steps (only NFC or NFKC alone, or none)",
                    ))
                }
            }
            other => Err(Declined::from(format!(
                "normalizer {other} (only NFC, NFKC or none)"
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
    /// each in a stable order so leftmost-longest ties stay deterministic;
    /// a token's `lstrip` / `rstrip` ride along, `single_word` declines.
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
            if token.single_word || token.content.is_empty() {
                return Err(Declined::from(format!(
                    "added token {:?} with single_word, or without content",
                    token.content
                )));
            }
            let added = |content: String| Added {
                content,
                id,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            };
            if token.normalized {
                normalized.push(added(match normalizer {
                    Normalizer::None => token.content.clone(),
                    form => normalize(&token.content, form, None).into_owned(),
                }));
            } else {
                raw.push(added(token.content.clone()));
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
    /// no byte-level piece; one the model declines, whose ids the table
    /// cannot hold, or that holds a byte the table has no token for, is left
    /// to the miss path. A model with merge dropout
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
        self.misses_wait =
            Self::merges_every_short_piece(self.merges.as_ref(), self.seed_covers_short_entries);
    }

    /// On the sentencepiece path, whose encode reads no model, the merges'
    /// operands in rank order.
    pub(crate) fn sentencepiece_pairs_by_rank(&self) -> Option<Vec<[u32; 2]>> {
        self.chars.as_deref().map(CharSymbols::pairs_by_rank)
    }

    /// Whether this is the sentencepiece path.
    pub(crate) fn is_sentencepiece(&self) -> bool {
        self.chars.is_some()
    }

    /// Whether the merge table answers, with no model, every short piece
    /// that neither a thread's table nor the seed holds: it has a token for
    /// every byte, and the whole-piece lookup `ignore_merges` asks for is
    /// spared for a short piece. The slow call asks the model in those two
    /// cases alone (see [`Self::tokenize_piece_slow`]).
    fn merges_every_short_piece(
        merges: Option<&MergeTable>,
        seed_covers_short_entries: bool,
    ) -> bool {
        merges.is_some_and(|table| {
            table.missing_bytes() == 0 && (!table.ignore_merges || seed_covers_short_entries)
        })
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
            if bpe.ignore_merges {
                return Some(SeededIds::of(&[id]));
            }
            scratch.ids.clear();
            match (chars, merges) {
                (Some(chars), _) => chars.merge(entry, &mut scratch.symbols, &mut scratch.ids),
                (None, Some(table)) => {
                    if table.lacks_a_token_for(piece) {
                        return None;
                    }
                    table.merge(piece, &mut scratch.ids);
                }
                (None, None) => {
                    let tokens = model.tokenize(entry).ok()?;
                    // One id past the limit tells that the entry is over it.
                    let ids = tokens.iter().take(SHORT_PIECE_MAX_BYTES + 1);
                    scratch.ids.extend(ids.map(|token| token.id));
                }
            }
            (scratch.ids.len() <= SHORT_PIECE_MAX_BYTES).then(|| SeededIds::of(&scratch.ids))
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
                let ids = ids_of(id, entry, &piece[..len], scratch)?;
                Some(if merges.is_none() && ids.inline().is_none() {
                    Seeded::Asked(key, scratch.ids.as_slice().into())
                } else {
                    Seeded::Short(key, ids)
                })
            })();
            Some(seeded)
        };
        // A listing is the whole vocabulary, and a model that was not built
        // has no size.
        let vocab_size =
            listing.map_or_else(|| bpe.get_vocab_size(), |listing| listing.len() as usize);
        // The pass reads a window of ids at a time into one list of slots
        // the calling thread owns and reuses, and the calling thread puts
        // each window's short pieces straight into the table: no list of
        // them as long as the vocabulary stands beside it (the long
        // spellings wait for the filter). What the pass holds for
        // a time stays behind after the load as free memory: the system
        // allocator gives back no free top below the threshold that a freed
        // block as large as the vocabulary file has raised. The table is
        // made before the pass, so in the slots for a count the listing
        // gives without a merge: a short piece has no more characters than
        // its key has bytes.
        let mut table = match listing {
            Some(listing) => ShortPieceTable::for_entries(if chars.is_some() {
                // A longer entry is short by its key's length: its bytes,
                // each marker's three as one. One with a space has no key.
                let extra = WORD_MARKER.len_utf8() - 1;
                listing.count(SHORT_PIECE_MAX_BYTES, |entry| {
                    !entry.contains(' ')
                        && entry.len() - extra * entry.matches(WORD_MARKER).count()
                            <= SHORT_PIECE_MAX_BYTES
                })
            } else {
                listing.count(SHORT_PIECE_MAX_BYTES, |entry| {
                    entry.chars().nth(SHORT_PIECE_MAX_BYTES).is_none()
                })
            }),
            // No count before the pass: the table starts small and grows
            // in place, in a reservation for the whole vocabulary, its only
            // bound: 33 bytes a slot of address space, written only as far
            // as pieces are stored. Below 4,096 slots a step of growth is
            // more than a doubling (64 to 1,024 at the 49th piece), so
            // `fitted` may make a smaller table of what this one holds.
            None => ShortPieceTable::with_slots(MIN_SLOTS, ShortPieceTable::slots_for(vocab_size)),
        };
        let mut slots: Vec<Option<Option<Seeded>>> =
            Vec::with_capacity(SEED_WINDOW.min(vocab_size));
        let mut entries = 0usize;
        let mut long_entries: Vec<Box<[u8]>> = Vec::new();
        let mut again = SeedScratch::default();
        // An id that was an entry has a slot that is `Some`; the seeded
        // pieces are the inner `Some`s. A read leaves a short entry's ids
        // in its scratch: one of more ids than a slot holds is read again
        // (the model's come whole, see `Seeded::Asked`).
        let mut keep = |first: u32, slots: &mut Vec<Option<Option<Seeded>>>| {
            for (id, slot) in (first..).zip(slots.drain(..)) {
                let Some(seeded) = slot else { continue };
                entries += 1;
                match seeded {
                    Some(Seeded::Short(key, ids)) => {
                        if let Some(ids) = ids.inline() {
                            table.insert(key, ids, vocab_size);
                        } else if let Some(Some(Seeded::Short(..))) = read(id, &mut again) {
                            table.insert(key, &again.ids, vocab_size);
                        }
                    }
                    Some(Seeded::Asked(key, ids)) => {
                        table.insert(key, &ids, vocab_size);
                    }
                    Some(Seeded::Long(bytes)) => long_entries.push(bytes),
                    None => {}
                }
            }
        };
        let windows = || {
            (0..vocab_size)
                .step_by(SEED_WINDOW)
                .map(|start| start as u32..(start + SEED_WINDOW).min(vocab_size) as u32)
        };
        // One slot per id of the window, written in place (the read keeps
        // its index, so the collect is one list and no concatenation); one
        // scratch per rayon job. Rayon halves a window into jobs, none of
        // fewer than `job` entries: the window's share of a thread of the
        // pool the read runs on, held between 256 and 2048. On the seed's
        // own pool a full window is eight jobs of 2048 for its four
        // threads. On a caller's pool of eight threads or more it is as
        // many jobs as the largest power of two that is not above the
        // pool's threads, sixty-four at the most, and a thread reads a
        // job: no more threads than there are jobs read a window, however
        // wide the pool (its width is the `threads` of the debug line).
        let read_window = |window: Range<u32>, slots: &mut Vec<Option<Option<Seeded>>>| {
            let job = (SEED_WINDOW / rayon::current_num_threads().max(1)).clamp(256, 2048);
            window
                .into_par_iter()
                .with_min_len(job)
                .map_init(SeedScratch::default, |scratch, id| read(id, scratch))
                .collect_into_vec(slots);
        };
        // A caller that is a worker of a rayon pool already (the Python
        // binding loads on its own, sized by its host) reads on that pool,
        // which never builds the global one. Any other caller reads on this
        // load's own pool: its threads start here and are joined before the
        // entries are returned, so the load leaves no pool behind; a process
        // that cannot spawn them reads on the calling thread and says so.
        // (The name fits the fifteen bytes a thread's name keeps.)
        let threads = if rayon::current_thread_index().is_some() {
            for window in windows() {
                let first = window.start;
                read_window(window, &mut slots);
                keep(first, &mut slots);
            }
            rayon::current_num_threads()
        } else {
            let pooled = rayon::ThreadPoolBuilder::new()
                .num_threads(SEED_THREADS)
                .thread_name(|index| format!("vocab-seed-{index}"))
                .build_scoped(
                    |thread| thread.run(),
                    |pool| {
                        for window in windows() {
                            let first = window.start;
                            pool.install(|| read_window(window, &mut slots));
                            keep(first, &mut slots);
                        }
                    },
                );
            match pooled {
                Ok(()) => SEED_THREADS,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "could not build the vocabulary seed's pool; reading on the calling thread"
                    );
                    let mut scratch = SeedScratch::default();
                    for window in windows() {
                        let first = window.start;
                        slots.extend(window.map(|id| read(id, &mut scratch)));
                        keep(first, &mut slots);
                    }
                    1
                }
            }
        };
        let covers = entries == vocab_size;
        let long = long_entries.len();
        let table = table.fitted();
        let mut filter = LongEntryFilter::for_entries(long);
        for bytes in &long_entries {
            filter.insert(bytes);
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
    pub(crate) fn from_ranks<K: Borrow<[u8]> + Eq + Hash>(
        ranks: &FxHashMap<K, u32>,
        pattern: &str,
        specials: &[(String, u32)],
    ) -> Result<Self, Declined> {
        let parsed = Pattern::parse(pattern).ok_or_else(|| {
            Declined::from(format!(
                "a split pattern outside the supported regex subset: {pattern}"
            ))
        })?;
        // The file's own encoder reads `$` as the end of the text only, where
        // the subset reads it as the Hugging Face engine does (before a
        // newline too): a pattern with the anchor is declined on this path.
        if parsed.end_anchored() {
            return Err(Declined::from(format!(
                "a split pattern with the end anchor, which the file's own encoder reads as the \
                 end of the text only: {pattern}"
            )));
        }
        // The file's own encoder reads the word boundary with another word
        // class (the UTS#18 set, without the other numbers such as `²`),
        // where the subset reads it as the Hugging Face engine does: a
        // pattern with the digit-grouping lookahead is declined on this path.
        if parsed.reads_word_boundary() {
            return Err(Declined::from(format!(
                "a split pattern with the digit-grouping lookahead, whose word boundary the file's \
                 own encoder reads with another word class: {pattern}"
            )));
        }
        // The one stage of this path runs over a unit of the thread pool as
        // over the text between two specials, and would read the unit's
        // bounds as the text's: a pattern anchored to its text is declined
        // on this path (see `Self::check_anchors`).
        if parsed.anchored() {
            return Err(Declined::from(format!(
                "a split pattern anchored to its text (\\A, \\z or \\G): {pattern}"
            )));
        }
        // The file's own encoder reads a `+` after a bare interval as a
        // possessive mark (one interval, nothing given back), where the
        // subset reads it as the Hugging Face engine does (a repeat of the
        // interval): a pattern with one inside the lookahead is declined on
        // this path.
        if parsed.repeats_bare_interval() {
            return Err(Declined::from(format!(
                "a split pattern with a `+` after a bare interval, which the file's own encoder \
                 reads as a possessive mark: {pattern}"
            )));
        }
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
            missing_bytes = merges.missing_bytes(),
            "the native encode path merges a tiktoken vocabulary's piece-cache misses itself"
        );
        let stages = vec![Stage {
            pattern: parsed,
            pieces: Pieces::Matches,
            digits: false,
        }];
        let chain = chain::Scanner::recognize(&stages);
        let specials: Vec<Added> = specials
            .iter()
            .map(|(content, id)| Added::plain(content.clone(), *id))
            .collect();
        Ok(Self {
            normalizer: Normalizer::None,
            raw_added: AddedTokens::build(&specials)?,
            normalized_added: None,
            nfc_quick: None,
            end_anchored: false,
            mapped_from: None,
            image_opens: None,
            byte_split: stages.iter().any(|stage| stage.pattern.has_byte_split()),
            space_inside: stages
                .iter()
                .any(|stage| stage.pattern.reads_space_inside()),
            stages,
            chain,
            frame: Frame::default(),
            alphabet: bytes_char(),
            seed: Self::ranks_seed(ranks),
            learned: OnceLock::new(),
            seed_covers_short_entries: false,
            misses_wait: Self::merges_every_short_piece(Some(&merges), false),
            long_entries: None,
            merges: Some(merges),
            chars: None,
            piece_cache: ThreadLocal::new(),
            cut_spans: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
            callers: AtomicUsize::new(0),
            load_read: OnceLock::new(),
            processors: processors(),
            inside: AtomicUsize::new(0),
        })
    }

    /// The seed of a tiktoken vocabulary: every token the key holds, with
    /// its own id (what a miss on it merges to, the table being complete).
    fn ranks_seed<K: Borrow<[u8]>>(ranks: &FxHashMap<K, u32>) -> ShortPieceTable {
        let mut table = ShortPieceTable::for_entries(ranks.len());
        for (token, &rank) in ranks {
            if let Some(key) = ShortPieceKey::of(token.borrow()) {
                table.insert(key, &[rank], ranks.len());
            }
        }
        table
    }

    /// The stages of a pre-tokenizer that is zero or more `Split`s (or
    /// `Digits`, a split on numeric characters, or `Punctuation`s) around
    /// one `ByteLevel` without prefix space: those before it as they are,
    /// those after it over the raw text when they cut it where they cut its
    /// byte-level image, else over the image itself (the index of the first
    /// of them is the second half of the answer, see `mapped_from`).
    fn stages_from(pre_tokenizer: &Value) -> Result<(Vec<Stage>, Option<usize>), Declined> {
        if pre_tokenizer.is_null() {
            return Err(Declined::from(
                "no pre-tokenizer (the path needs a ByteLevel without a prefix space)",
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
        let at = items
            .iter()
            .position(|item| item.get("type").and_then(Value::as_str) == Some("ByteLevel"))
            .filter(|&at| items[at].get("add_prefix_space").and_then(Value::as_bool) == Some(false))
            .ok_or_else(|| {
                Declined::from("the pre-tokenizer has no ByteLevel without a prefix space")
            })?;
        let mut stages = Vec::with_capacity(items.len());
        for split in &items[..at] {
            let stage = Self::split_stage(split)?;
            Self::check_anchors(&stage, &stages)?;
            stages.push(stage);
        }
        if items[at].get("use_regex").and_then(Value::as_bool) == Some(true) {
            stages.push(Stage {
                pattern: Pattern::parse(BYTE_LEVEL_PATTERN)
                    .ok_or_else(|| Declined::from("the ByteLevel split pattern"))?,
                pieces: Pieces::MatchesAndGaps,
                digits: false,
            });
        }
        // A stage after the `ByteLevel` cuts the mapped text. When every
        // such stage's pattern can only ever match or reject printable
        // ASCII characters, it cuts the raw text at the same places (the
        // map leaves those in place and takes every other byte beyond
        // ASCII), so the stages run over the raw text as the ones before
        // the map do. Otherwise the stages after the map run over each
        // piece's byte-level image, from the first of them on.
        let after = stages.len();
        for split in &items[at + 1..] {
            let stage = Self::split_stage(split)?;
            Self::check_anchors(&stage, &stages)?;
            stages.push(stage);
        }
        let mapped_from = stages[after..]
            .iter()
            .any(|stage| !stage.pattern.within_printable_ascii())
            .then_some(after);
        // Over the image an ordinary text is one run of letters (a space
        // maps to one), and the search passes over a failed run only when
        // the program opens with it (see `Matches::next`): a stage whose
        // program holds an unbounded run anywhere else would rescan that
        // run from every start, so it is declined.
        if let Some(from) = mapped_from {
            if stages[from..]
                .iter()
                .any(|stage| stage.pattern.rescans_a_run())
            {
                return Err(Declined::from(
                    "a pre-tokenizer stage over the byte-level image whose pattern rescans a run from every start",
                ));
            }
        }
        // A merged-with-next stage joins each match with the text after it
        // (see `Pieces::walk`); the walk over the image takes matches and
        // gaps apart, so such a stage among those over the image is declined
        // (no covered vocabulary has one there). One after the `ByteLevel`
        // still loads when every stage after the map stays within printable
        // ASCII: those stages run over the raw text, where `pieces` glues it.
        if let Some(from) = mapped_from {
            if stages[from..]
                .iter()
                .any(|stage| stage.pieces == Pieces::MatchesWithNextGap)
            {
                return Err(Declined::from(
                    "a merged-with-next pre-tokenizer stage over the byte-level image",
                ));
            }
        }
        Ok((stages, mapped_from))
    }

    /// The bytes whose image a match of a stage over the image can begin
    /// with (the `image_opens` field), read through `alphabet`, the map the
    /// walk builds each image with: every stage from `mapped_from` on must
    /// name the class its matches begin with and keep the text outside its
    /// matches, or there is no table.
    fn image_opening_bytes(
        stages: &[Stage],
        mapped_from: Option<usize>,
        alphabet: &[char; 256],
    ) -> Option<Box<[bool; 256]>> {
        let mut opens = Box::new([false; 256]);
        for stage in &stages[mapped_from?..] {
            if !matches!(stage.pieces.walk(), Walk::Apart { gaps: true, .. }) {
                return None;
            }
            let class = stage.pattern.opening_class()?;
            for (open, &c) in opens.iter_mut().zip(alphabet) {
                *open |= class.contains(c);
            }
        }
        Some(opens)
    }

    /// A stage anchored to its text (`\A`, `\z` or `\G`) reads the bounds of
    /// the piece the stage before it cut, and where its search in that piece
    /// began, as the reference runs it ([`Self::pieces`] runs every stage
    /// over the pieces of the one before). Three things keep it honest: the
    /// first stage runs over a unit of the thread pool as over the text, and
    /// would read the unit's bounds as the text's; the pool's cut check runs
    /// the stages over a window of the text ([`Self::piece_starts_at`]), each
    /// over the pieces of the one before, from a start an earlier stage's
    /// gap can span ([`Self::window_start`]) to an end that can sit inside a
    /// whitespace run of the text ([`Self::window_end`]), which stays right
    /// only when the stage's matches are runs of decimal digits, which begin
    /// at neither of the window's bounds nor at a cut candidate (all spaces)
    /// and span none of them; and the stage's own input pieces must meet
    /// the same bounds in a unit as in the text, which a gap of an earlier
    /// stage that holds digits and spans a cut would not (it ends at the
    /// unit's end, where the text goes on): after a `Digits` split every
    /// piece that holds a decimal digit lies inside a run of numeric
    /// characters, which holds no whitespace and so spans no cut candidate,
    /// and the stages after it only cut pieces finer. An anchored first
    /// stage, an anchored stage over anything but decimal digits and an
    /// anchored stage without a `Digits` split before it decline.
    fn check_anchors(stage: &Stage, earlier: &[Stage]) -> Result<(), Declined> {
        if !stage.pattern.anchored() {
            return Ok(());
        }
        if earlier.is_empty() {
            return Err(Declined::from(
                "a first split pattern anchored to its text (\\A, \\z or \\G)",
            ));
        }
        if !stage.pattern.decimal_digits_only() {
            return Err(Declined::from(
                "a split pattern anchored to its text (\\A, \\z or \\G) over more than decimal \
                 digits",
            ));
        }
        if !earlier.iter().any(|stage| stage.digits) {
            return Err(Declined::from(
                "a split pattern anchored to its text (\\A, \\z or \\G) without a Digits split \
                 before it",
            ));
        }
        Ok(())
    }

    /// One `Split` as a stage: a regex or a literal the pattern engine takes,
    /// `Isolated`, `Removed`, `Removed` inverted, or `MergedWithNext`. A
    /// `Digits` is the `Isolated` split on numeric characters, each one a
    /// piece under `individual_digits`, else each run: the reference splits on
    /// `char::is_numeric`, the toolchain's numeric set
    /// ([`unicode::STD_NUMERIC`]), which is not the engine's `\p{N}` (the
    /// regex engine's Unicode version). A `Punctuation` is the `Isolated`
    /// split on the reference's punctuation set, one character a piece, or
    /// one run under `Contiguous`: the reference pre-tokenizer's own set
    /// ([`unicode::REFERENCE_PUNCTUATION`]: ASCII punctuation, or the
    /// punctuation category of the Unicode tables it carries), not the
    /// engine's `\p{P}`, which has moved on since.
    fn split_stage(split: &Value) -> Result<Stage, Declined> {
        match type_of("pre-tokenizer stage", split)? {
            "Split" => {}
            "Digits" => {
                let each = split
                    .get("individual_digits")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| {
                        Declined::from("a Digits without a boolean individual_digits")
                    })?;
                return Pattern::parse(&class_pattern(unicode::STD_NUMERIC, !each))
                    .map(|pattern| Stage {
                        pattern,
                        pieces: Pieces::MatchesAndGaps,
                        digits: true,
                    })
                    .ok_or_else(|| Declined::from("the Digits split pattern"));
            }
            "Punctuation" => {
                // The reference reads a missing `behavior` as `Isolated`.
                let runs = match split.get("behavior").and_then(Value::as_str) {
                    None | Some("Isolated") => false,
                    Some("Contiguous") => true,
                    behavior => {
                        return Err(Declined::from(format!(
                            "Punctuation behavior {behavior:?} (only Isolated or Contiguous)"
                        )))
                    }
                };
                return Pattern::parse(&class_pattern(unicode::REFERENCE_PUNCTUATION, runs))
                    .map(|pattern| Stage {
                        pattern,
                        pieces: Pieces::MatchesAndGaps,
                        digits: false,
                    })
                    .ok_or_else(|| Declined::from("the Punctuation split pattern"));
            }
            other => {
                return Err(Declined::from(format!(
                    "pre-tokenizer {other} beside the ByteLevel (only Split, Digits or Punctuation)"
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
        let pieces = match (
            split.get("behavior").and_then(Value::as_str),
            split.get("invert").and_then(Value::as_bool),
        ) {
            (Some("Isolated"), Some(false)) => Pieces::MatchesAndGaps,
            (Some("Removed"), Some(false)) => Pieces::Gaps,
            (Some("Removed"), Some(true)) => Pieces::Matches,
            (Some("MergedWithNext"), Some(false)) => Pieces::MatchesWithNextGap,
            (behavior, invert) => {
                return Err(Declined::from(format!(
                    "Split behavior {behavior:?} with invert {invert:?} (only Isolated, Removed, \
                     Removed inverted, or MergedWithNext)"
                )))
            }
        };
        let pattern = Pattern::parse(&regex).ok_or_else(|| {
            Declined::from(format!(
                "a split pattern outside the supported regex subset: {regex}"
            ))
        })?;
        Ok(Stage {
            pattern,
            pieces,
            digits: false,
        })
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
    /// same either way (the units are encoded by the same code, and the
    /// pool joins their ids in input order).
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
        // them; a short input walks its runs byte by byte, the chain
        // scanner's units that open as ASCII text apart (`encode_unit`).
        let words = text.len() >= WORD_SCAN_MIN_BYTES;
        // Text not in the normalizer's form (by the library's tables, as
        // every normalization read here) is normalized gap by gap first;
        // the units then borrow from those gaps.
        let gaps = (self.normalizer != Normalizer::None && !self.in_form(text))
            .then(|| self.normalized_gaps(text, added));
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
            if units.len() * OWN_UNIT_MIN_BYTES > text.len() {
                units
                    .par_iter()
                    .zip(parts.par_iter_mut())
                    .enumerate()
                    .try_for_each(|(index, (unit, ids))| {
                        let marker = self.marker_before(units, index, at_start);
                        let mut mapped = String::new();
                        let mut cache = self.piece_cache_for_text(text.len());
                        // Filled through a Vec in this thread's frame: the
                        // slots of `parts` share a cache line or two, and
                        // every piece appended writes its Vec's length.
                        let mut filled = std::mem::take(ids);
                        self.encode_unit(
                            model,
                            unit,
                            marker,
                            words,
                            &mut mapped,
                            &mut filled,
                            &mut cache,
                        )?;
                        self.answer_rest(&mut filled, &mut cache);
                        *ids = filled;
                        Ok(())
                    })?;
            } else {
                // One callback a unit, as the iterator has, but the thread
                // that runs a callback chooses its unit: unit k is the own
                // unit of pool thread k (modulo the pool's size), and a
                // thread takes its own units first, then any still free. A
                // text read again so finds a unit's pieces in the cache of
                // the thread that learned them, when that thread comes for
                // it (none is sent for; a thread beyond the unit count owns
                // none). The choice costs two allocations a call more than
                // counted above (the units' claims and locks, the threads'
                // counts). A unit costs one winning swap and one lock that
                // no other thread asks for; every probe on the way to it
                // costs a count, and a load where it names a unit already
                // taken.
                let slots: Vec<(AtomicBool, Mutex<&mut Vec<u32>>)> = parts
                    .iter_mut()
                    .map(|ids| (AtomicBool::new(false), Mutex::new(ids)))
                    .collect();
                let threads = rayon::current_num_threads();
                // How many of its own units each thread has come for (a
                // scan for the next one grows with the square of a thread's
                // units).
                let came: Vec<AtomicUsize> = (0..threads.min(units.len()))
                    .map(|_| AtomicUsize::new(0))
                    .collect();
                // The units any thread may take are offered once each, in
                // order.
                let offered = AtomicUsize::new(0);
                let take = || {
                    let me = rayon::current_thread_index().unwrap_or(0);
                    let own = std::iter::from_fn(|| {
                        let index = me + came.get(me)?.fetch_add(1, Ordering::Relaxed) * threads;
                        (index < units.len()).then_some(index)
                    });
                    let rest = std::iter::from_fn(|| {
                        let index = offered.fetch_add(1, Ordering::Relaxed);
                        (index < units.len()).then_some(index)
                    });
                    let claim = |index: usize| {
                        let taken = &slots[index].0;
                        !taken.load(Ordering::Relaxed) && !taken.swap(true, Ordering::Relaxed)
                    };
                    let Some(index) = own.chain(rest).find(|&index| claim(index)) else {
                        // Every offer is out: a unit still free is held by
                        // the callback that fetched its offer, and that one
                        // takes it. No unit is left unread.
                        return Ok(());
                    };
                    // Only the thread that took the unit asks for its lock.
                    let (_, ids) = &slots[index];
                    let mut ids = ids.lock().unwrap_or_else(PoisonError::into_inner);
                    let marker = self.marker_before(units, index, at_start);
                    let mut mapped = String::new();
                    let mut cache = self.piece_cache_for_text(text.len());
                    let mut filled = std::mem::take(&mut **ids);
                    self.encode_unit(
                        model,
                        &units[index],
                        marker,
                        words,
                        &mut mapped,
                        &mut filled,
                        &mut cache,
                    )?;
                    self.answer_rest(&mut filled, &mut cache);
                    **ids = filled;
                    Ok(())
                };
                // The range splits, and a decline stops it, as the iterator
                // over the units did.
                (0..units.len()).into_par_iter().try_for_each(|_| take())?;
                debug_assert!(
                    slots.iter().all(|(taken, _)| taken.load(Ordering::Relaxed)),
                    "a unit of the text was left unread"
                );
            }
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
        let mut cache = self.piece_cache_for_text(text.len());
        // Only while more threads are inside a long text than the process
        // has processors, and the clock is read only then. Not for a unit
        // on the pool: it is no text of its thread.
        let inside = (text.len() >= READ_BACK_MIN_BYTES).then(|| {
            let others = self.inside.fetch_add(1, Ordering::Relaxed);
            (Inside(&self.inside), others)
        });
        match &inside {
            Some((_, others)) if *others >= self.processors => {
                if cache.reads_back(Instant::now(), text.len()) {
                    cache.packed.read_through();
                }
            }
            // A long text run with a processor to spare: the next gap is
            // not measured across it.
            Some(_) => cache.last_long = None,
            None => {}
        }
        self.encode_serial(model, units, words, at_start, &mut ids, &mut cache)?;
        ids.extend_from_slice(after);
        Ok(ids)
    }

    /// The units of one call encoded in order onto `ids`, on `cache`: the
    /// calling thread's for a request, one of its own for [`Self::rehearse`].
    fn encode_serial(
        &self,
        model: Option<&ModelWrapper>,
        units: &[Segment<'_>],
        words: bool,
        at_start: bool,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let mut mapped = String::new();
        for (index, unit) in units.iter().enumerate() {
            let marker = self.marker_before(units, index, at_start);
            self.encode_unit(model, unit, marker, words, &mut mapped, ids, cache)?;
        }
        self.answer_rest(ids, cache);
        Ok(())
    }

    /// The last step of a load: two short built-in texts encoded on a piece
    /// cache of its own, which is dropped, then the merge table read through
    /// once (`MergeTable::read_through`). The first request of a process
    /// finds the encode path mapped and the table's lines in the caches. The
    /// loading thread does not read the table again while this read is
    /// fresh ([`Self::load_read_holds`]); every other thread's second call
    /// still reads it itself ([`Self::piece_cache_for_text`]): a thread may
    /// start long after the load. No thread's cache is made, so the loading
    /// thread counts toward no share ([`Self::piece_cache_share`]).
    pub(crate) fn rehearse(&self, model: Option<&ModelWrapper>) {
        // Prose, digits, punctuation and JSON, then letters past ASCII.
        const TEXTS: [&str; 2] = [
            "The quick brown fox jumps over 12 lazy dogs. It's 3:45 pm, isn't it?\n\nHello, World! Please summarize the following text in one sentence: {\"key\": [1, 2.5, true]}",
            "Caf\u{e9} na\u{ef}ve \u{4f60}\u{597d} \u{2014} the user's question: what is 2 + 2?\n",
        ];
        let mut cache = PieceCache::with_share(self.piece_cache_share());
        for text in TEXTS {
            let units = self.units(text, false, true);
            let mut ids = Vec::with_capacity(text.len() / 3 + 8);
            if self
                .encode_serial(model, units.as_slice(), false, true, &mut ids, &mut cache)
                .is_err()
            {
                break;
            }
        }
        if let Some(merges) = &self.merges {
            merges.read_through();
        }
        let _ = self
            .load_read
            .set((std::thread::current().id(), Instant::now()));
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

    /// Whether `text` is in the normalizer's form by the library's tables:
    /// the NFC quick check's own tables first, the library's stretch-wise
    /// check where they cannot tell and under NFKC (an encoder without a
    /// normalizer never asks).
    fn in_form(&self, text: &str) -> bool {
        match &self.nfc_quick {
            Some(quick) => quick.is_nfc(text),
            None => is_normalized(text, self.normalizer),
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

    /// [`Self::units`] over the gaps of a text that was not in the
    /// normalizer's form ([`Self::normalized_gaps`]): the raw cut is done
    /// and every text is in that form, what remains is the cut after
    /// normalization and the chunking.
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
    /// each text between them in the normalizer's form: as it is where it
    /// was, normalized into an owned buffer where it was not. `tokenizers`
    /// normalizes each such text on its own, after the raw cut and before
    /// the cut after normalization: a mark right after a raw token has
    /// nothing to compose with, and a token's spelling is never matched
    /// across what normalization changed.
    fn normalized_gaps<'t>(&self, text: &'t str, added: bool) -> Vec<Gap<'t>> {
        let mut gaps = Vec::new();
        let (form, quick) = (self.normalizer, self.nfc_quick.as_ref());
        let push = |text: &'t str, gaps: &mut Vec<Gap<'t>>| {
            gaps.push(Gap::Text(normalize(text, form, quick)));
        };
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
    /// it, cut cleanly ([`Self::piece_starts_at`]). Under a pattern that reads
    /// no space inside a match ([`Pattern::reads_space_inside`]) no piece spans
    /// such a point (letters, digits and punctuation stop at the space, a
    /// whitespace run cannot start before it); under one that does (a letter run
    /// joined by spaces, `\p{L}+ ?`) the check refuses a cut a piece spans. The
    /// patterns read nothing before a match's start (the digit-grouping
    /// lookahead's word boundary reads the character before it, one its match
    /// consumed: the parser declines the lookahead where it could lead a
    /// match), so the pieces of the two halves are the pieces of the whole.
    /// A candidate without a window within the check's reach
    /// ([`Self::window_start`], [`Self::window_end`]: a stretch without a
    /// space longer than the reach on one side of it) is skipped, with the
    /// bytes after it up to [`CUT_SEARCH_SKIP`] (the candidates there may
    /// have a window, this one's space among their starts, but the step
    /// bounds the walks' cost and moves the cut on by less than that), and
    /// not counted: without a stage that reads a space inside a match its
    /// own space begins a whitespace run, where the next candidate's walk
    /// back can stop, so a blob followed by prose keeps its cuts past the
    /// blob. `None` when
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
        // A stage that reads the end of the text (`$`) would read the
        // window's end as one, and the cut check's window is read in the
        // raw text while a stage over the byte-level image cuts elsewhere:
        // no cut, the text stays one unit.
        if self.stages.is_empty() || self.end_anchored || self.mapped_from.is_some() {
            return None;
        }
        // The check's span lists: this thread's scratch, grown at its first
        // check and kept (the search runs on the calling thread, before the
        // pool, and borrows nothing else of the thread); a list a window
        // over a long blob grew past the cap is given back after the search.
        let scratch = self.cut_spans.get_or_default();
        let mut spans = scratch.borrow_mut();
        let found = self.search_cut(text, from, &mut spans);
        for list in spans.iter_mut() {
            if list.capacity() > CUT_SPANS_KEPT {
                *list = Vec::new();
            }
        }
        found
    }

    /// The search of [`Self::stable_cut`] over the stage path, with the
    /// check's three span lists lent by the caller.
    fn search_cut(&self, text: &str, from: usize, spans: &mut CutSpans) -> Option<usize> {
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
                let Some(start) = self.window_start(text, p) else {
                    // No window within the check's reach before the candidate: a stretch
                    // without a space there. The candidates within the skip of this one
                    // may have a window (this one's space among their starts); skipping
                    // them bounds the walks' cost and only moves the cut on. A skip, not
                    // a rejection: the rejections count the candidates the check read.
                    p += CUT_SEARCH_SKIP;
                    continue;
                };
                let Some(end) = self.window_end(text, p) else {
                    // The same after the candidate; the walk forward runs only once the
                    // walk back has found a start.
                    p += CUT_SEARCH_SKIP;
                    continue;
                };
                if self.piece_starts_at(&text[start..end], p - start, spans) {
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

    /// Whether the stages, run over the cut check's `window` of a candidate
    /// at `at` inside it, cut cleanly there. The window starts where the
    /// whole text's match loop begins a piece and ends just after such a
    /// space ([`Self::window_start`], [`Self::window_end`]), so that its
    /// pieces around the candidate are the text's. Over the window the
    /// stages run as the walk runs them over a piece ([`Self::pieces`]), each
    /// over the piece of the stage before it, never over the whole window
    /// (two isolated stages, `aa` then a space with 999 of the space-or-a
    /// class, over a run of ` a` with `aa` inside it: the second stage's match
    /// begins where `aa` ends, while a run from the window's start would end
    /// one match on the candidate and begin the next there). A piece that
    /// holds `at` inside it would be split, so that is no cut. The text
    /// before `at` on its own, as the unit before the cut reads it, must cut
    /// into the same pieces as it does inside the window (a lookahead at
    /// `at` reads the unit's end where the text reads on: ` a{1000}(?!\s)`
    /// holds at the one and fails at the other); and the text from `at` on
    /// its own, as the unit after the cut reads it, must cut into the same
    /// pieces too (an earlier stage's match that a later stage splits at
    /// `at`, or that a dropped stage drops, leaves no final piece over `at`,
    /// while the next unit's first stage begins inside that match and may
    /// match otherwise: `[ a]{999}` then ` a` over a run of ` a`); else that
    /// is no cut. The three span lists are the caller's (`spans`, the
    /// thread's scratch through `stable_cut`): cleared here and grown once
    /// per thread, so a check over prose allocates nothing after the
    /// thread's first (a search whose window grew a list past
    /// [`CUT_SPANS_KEPT`] gives that list back, and the next check grows it
    /// again).
    fn piece_starts_at(&self, window: &str, at: usize, spans: &mut CutSpans) -> bool {
        let [whole, left, right] = spans;
        whole.clear();
        self.piece_spans(window, whole);
        if whole.iter().any(|&(from, to)| from < at && at < to) {
            return false;
        }
        left.clear();
        self.piece_spans(&window[..at], left);
        if !left
            .iter()
            .eq(whole.iter().take_while(|&&(_, to)| to <= at))
        {
            return false;
        }
        right.clear();
        self.piece_spans(&window[at..], right);
        right
            .iter()
            .map(|&(from, to)| (from + at, to + at))
            .eq(whole.iter().skip_while(|&&(_, to)| to <= at).copied())
    }

    /// The final pieces of `text` under the stages, as the walk cuts them
    /// ([`Self::pieces`]), as spans of byte offsets into `text`.
    fn piece_spans(&self, text: &str, out: &mut Vec<(usize, usize)>) {
        let base = text.as_ptr() as usize;
        self.staged_pieces(text, |piece| {
            let from = piece.as_ptr() as usize - base;
            out.push((from, from + piece.len()));
        });
    }

    /// How far before and after a candidate the cut check's window may start
    /// and end ([`Self::window_start`], [`Self::window_end`]): the reach
    /// ([`CUT_CHECK_REACH`]) when a stage reads a space after a
    /// non-whitespace character inside one match, where a stretch without a
    /// piece-beginning space is one piece and the refusal loses no cut; a
    /// chunk ([`PARALLEL_CHUNK_BYTES`]) otherwise, where such a stretch
    /// (digits, letters, line breaks, a whitespace run) is many pieces and
    /// its candidates are cuts the pool wants - a text whose spaces stand
    /// thousands of bytes apart keeps them - while the chunk keeps the
    /// window, and the spans the check collects over it (one per piece),
    /// within two chunks whatever the text.
    fn cut_check_reach(&self) -> usize {
        if self.space_inside {
            CUT_CHECK_REACH
        } else {
            PARALLEL_CHUNK_BYTES
        }
    }

    /// Where the cut check's window for a candidate at `p` starts: the first
    /// space at or before `p - CUT_CHECK_WINDOW`, or the text's start, where
    /// the whole text's match loop begins a piece, so that the window's
    /// pieces up to `p` are the text's. A space begins no piece when a
    /// whitespace run took it from a whitespace character before it (the
    /// text then lands on the character after the run, where a letter run
    /// joined by spaces can begin, which a window begun on the space would
    /// read by another alternative, a step behind the text for the whole
    /// run), when it is the join of such a run
    /// ([`Pattern::spans_space_between`]), or when some alternative of a
    /// stage reads a space after a character like the one before it inside
    /// one match ([`Pattern::reads_space_after`]: `\p{N}+ ?` takes `1 ` and
    /// lands the text on the letter after it the same way); a space after
    /// any other character begins a piece, since no alternative of the
    /// subset reads a space inside its match but as the join, in a
    /// whitespace run or after a character of such a class. When no stage
    /// reads a space after a non-whitespace character inside one match
    /// ([`Pattern::reads_space_inside`]: no such run, and the before-space
    /// classes whitespace alone), a piece holds a space only at its start
    /// or in a whitespace run, so the nearest space back that begins a
    /// whitespace run is the start ([`Self::whitespace_run_begins_at`]: a
    /// window begun inside the run would take a fixed-count whitespace
    /// alternative, `\s{2}`, from another offset than the text and leave
    /// another remainder at the run's end, one space before a letter run
    /// that a space-led alternative with a lookahead at its end,
    /// ` a{100}(?!\s)`, reads one way inside the text and another at a
    /// unit's end), when it lies within a chunk of `p`
    /// ([`Self::cut_check_reach`]); a stretch without one before a
    /// candidate - a longer whitespace run, line breaks, digits, letters
    /// without a space - leaves the candidate without a window (`None`; the
    /// search skips it, [`Self::stable_cut`]), which keeps the window, and
    /// the spans the check collects over it (one per piece: one per byte
    /// under a whitespace or digit split), within a chunk. Otherwise
    /// `None` when no such space lies within [`CUT_CHECK_REACH`] of `p`: a
    /// long text without one stays one unit (under a letter run joined by
    /// spaces such a stretch is one piece, so no cut is lost there).
    fn window_start(&self, text: &str, p: usize) -> Option<usize> {
        let mut start = p.saturating_sub(CUT_CHECK_WINDOW);
        let floor = p.saturating_sub(self.cut_check_reach());
        while start > 0 {
            let begins = if self.space_inside {
                self.piece_begins_at_space(text, start)
            } else {
                Self::whitespace_run_begins_at(text, start)
            };
            if begins {
                return Some(start);
            }
            if start <= floor {
                return None;
            }
            start -= 1;
        }
        Some(0)
    }

    /// Where the cut check's window for a candidate at `p` ends: just after
    /// the first space at or after `p + CUT_CHECK_WINDOW` where the whole
    /// text's match loop begins a piece, by the rule of the start
    /// ([`Self::window_start`]), or at the text's end, so that the window's
    /// pieces around `p` are the text's. A window ended at a space inside a
    /// piece cuts that piece short, and an alternative the end no longer
    /// holds gives way to a shorter one that fits: ` [ a]{1999}| [ a]{999}| a`
    /// over a run of ` a` ends its shorter alternative on a candidate the
    /// text's longer one spans, which would approve a cut inside a piece. The
    /// space itself lies inside the window, so that a lookahead at it reads
    /// the space as the text does, not the window's end (` a{1000}(?!\s)`
    /// holds at the one and fails at the other); nothing from before `p`
    /// reads past that space (no piece spans it, and `p`'s own left
    /// neighbour is no whitespace, so no whitespace run spans `p`). When no
    /// stage reads a space inside a match the first space on is the end: no
    /// piece spans `p` then, so an end inside a later piece (a whitespace
    /// run) changes nothing up to `p`; `None` when no space lies within a
    /// chunk of `p` ([`Self::cut_check_reach`]) and the text runs on (the
    /// search skips the candidate, [`Self::stable_cut`]), which keeps the
    /// window, and the spans the check collects over it, within a chunk.
    /// Otherwise `None` when no such space lies within
    /// [`CUT_CHECK_REACH`] of `p` and the text runs on: no cut there (under
    /// a pattern that reads a space inside a match such a stretch is one
    /// piece, so no cut is lost).
    fn window_end(&self, text: &str, p: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let mut end = (p + CUT_CHECK_WINDOW).min(text.len());
        let ceiling = (p + self.cut_check_reach()).min(text.len());
        while end < text.len() {
            let ends = if self.space_inside {
                self.piece_begins_at_space(text, end)
            } else {
                text.is_char_boundary(end) && bytes[end] == b' '
            };
            if ends {
                return Some(end + 1);
            }
            if end >= ceiling {
                return None;
            }
            end += 1;
        }
        Some(text.len())
    }

    /// Whether the whole text's match loop begins a piece at `at` under a
    /// pattern that reads a space inside a match: `at` holds a space that
    /// no piece before it took, in a whitespace run from a whitespace
    /// character before it, as the join of a letter run
    /// ([`Pattern::spans_space_between`]) or read after a character like the
    /// one before it inside one match ([`Pattern::reads_space_after`]). The
    /// cut check's window starts and ends at such a space.
    fn piece_begins_at_space(&self, text: &str, at: usize) -> bool {
        if !(text.is_char_boundary(at) && text.as_bytes()[at] == b' ') {
            return false;
        }
        let after = text[at + 1..].chars().next();
        !text[..at].chars().next_back().is_some_and(|before| {
            before.is_whitespace()
                || self.stages.iter().any(|stage| {
                    stage.pattern.spans_space_between(before, after)
                        || stage.pattern.reads_space_after(before)
                })
        })
    }

    /// Whether a whitespace run begins at `at` with a space: `at` holds a
    /// space and no whitespace precedes it. Under a pattern that reads no
    /// space inside a match the whole text's match loop attempts a match
    /// there (nothing before the run takes the space), so the cut check's
    /// window begins at such a space: a window begun inside the run would
    /// take a fixed-count whitespace alternative (`\s{2}`) from another
    /// offset than the text and leave another remainder at the run's end.
    fn whitespace_run_begins_at(text: &str, at: usize) -> bool {
        text.is_char_boundary(at)
            && text.as_bytes()[at] == b' '
            && !text[..at]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace)
    }

    /// Whether `text` opens as ASCII text: at most one byte in eight of its
    /// first `ASCII_HEAD_BYTES` is beyond ASCII. The chain scanner takes the
    /// word scans on such a unit whatever its length: on a text the branch
    /// predictor has not seen its byte loops' exits mispredict about once
    /// per run, and on a text it has seen its scanned walk costs no more
    /// than the byte loops (the single-stage scanners' scanned loops do, so
    /// they keep `WORD_SCAN_MIN_BYTES`). A unit in another script has few
    /// runs the scans can pass over and keeps the byte loops. The bytes are
    /// counted a word at a time: 32 steps for a full head.
    fn opens_as_ascii(text: &str) -> bool {
        /// The low bit of every byte of a word.
        const EACH_BYTE: u64 = 0x0101_0101_0101_0101;
        let head = &text.as_bytes()[..text.len().min(ASCII_HEAD_BYTES)];
        let (words, tail) = head.as_chunks::<8>();
        let mut beyond = tail.iter().filter(|byte| !byte.is_ascii()).count() as u64;
        for word in words {
            // Each byte's top bit, moved to its low bit; the multiply sums
            // the eight into the word's top byte.
            let tops = (u64::from_le_bytes(*word) >> 7) & EACH_BYTE;
            beyond += tops.wrapping_mul(EACH_BYTE) >> 56;
        }
        beyond <= (head.len() / 8) as u64
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
        // probes each piece in place (no recursive walk); the chain's walk
        // probes its own pieces, offsets into the unit, with the probe
        // compiled into the walk; any other shape walks, a lone `Removed`
        // stage among them (its matches are no pieces: the scanner loop
        // stays the loop of a stage whose matches are).
        match (&self.chain, self.stages.as_slice(), self.mapped_from) {
            (_, _, Some(from)) => {
                // The stages from `mapped_from` on run over each piece's
                // byte-level image; a final piece is a byte range of the
                // unit, not always a `str`.
                let mut image = String::new();
                let per_piece = |piece: &[u8]| {
                    if let Err(declined) = self.tokenize_piece(model, piece, mapped, ids, cache) {
                        fallback = Some(declined);
                    }
                };
                self.staged_pieces_mapping(from, text, &mut image, per_piece);
            }
            (Some(chain), _, None) => {
                // The chain's scanned walk costs a repeated text no more
                // than its byte loops, so a short unit that opens as ASCII
                // text takes it too. An all-ASCII unit does by definition:
                // the head is counted only for a unit with a byte beyond
                // ASCII.
                let words = words || cache.unit_ascii || Self::opens_as_ascii(text);
                // The probe compiles into the walk where nearly every piece
                // ends (the template's match, the block pass's cut): out of
                // line it costs 16.5 to 18.2 of a token's 155 to 177
                // instructions. Only its hit is compiled in; a miss is a call
                // (`tokenize_piece_in_place`): with the whole probe there the
                // walk kept less of its state in registers around a hit.
                chain.pieces(
                    text,
                    words,
                    #[inline(always)]
                    |start, end| {
                        if let Err(declined) = self.tokenize_piece_in_place(
                            model,
                            text,
                            start..end,
                            mapped,
                            ids,
                            cache,
                        ) {
                            fallback = Some(declined);
                        }
                    },
                );
            }
            (None, [stage], None)
                if matches!(stage.pieces.walk(), Walk::Apart { matches: true, .. }) =>
            {
                // An all-ASCII unit has a copy of the stage's loop compiled
                // for it (see `encode_single_stage`). Any other unit goes by
                // blocks where its pattern gives that reader. The all-ASCII
                // unit begins on its lean scan, which reads long words in
                // fewer instructions than the block pass (measured: the pass
                // costs a sixth more on a text of 18-byte words), and the
                // scan hands a unit that opens as short pieces to the block
                // pass (`block::is_broken`). The cased-letter template has no
                // lean scan: its unit goes by blocks either way, the
                // all-ASCII one from `encode_single_stage`.
                let encoded = if cache.unit_ascii {
                    self.encode_single_stage::<true>(model, stage, text, words, mapped, ids, cache)
                } else if let Some(blocks) = stage.pattern.by_blocks(text) {
                    self.encode_unit_by_blocks(model, blocks, text, words, mapped, ids, cache)
                } else if let Some(blocks) = stage.pattern.by_cased_blocks(text) {
                    self.encode_unit_by_blocks(model, blocks, text, words, mapped, ids, cache)
                } else {
                    self.encode_single_stage::<false>(model, stage, text, words, mapped, ids, cache)
                };
                if let Err(declined) = encoded {
                    fallback = Some(declined);
                }
            }
            (None, _, None) => {
                // The probe compiles into the walk, and the attribute holds
                // it there in both copies of the walk (`staged_pieces`):
                // left to the optimiser, a probe with two callers stays a
                // call a piece (counted: 8 to 14 instructions a token).
                self.staged_pieces(
                    text,
                    #[inline(always)]
                    |piece: &str| {
                        if let Err(declined) =
                            self.tokenize_piece(model, piece.as_bytes(), mapped, ids, cache)
                        {
                            fallback = Some(declined);
                        }
                    },
                );
            }
        }
        fallback.map_or(Ok(()), Err)
    }

    /// [`Self::pieces`] from the first stage on, in the copy these stages
    /// take: with the byte-level split's scanner when one of them is that
    /// split, else without it.
    #[inline(always)]
    fn staged_pieces<F: FnMut(&str)>(&self, text: &str, mut f: F) {
        if self.byte_split {
            self.pieces::<true, F>(0, text, &mut f);
        } else {
            self.pieces::<false, F>(0, text, &mut f);
        }
    }

    /// [`Self::pieces_mapping`] from the first stage on, in the copy these
    /// stages take ([`Self::staged_pieces`]).
    #[inline(always)]
    fn staged_pieces_mapping<F: FnMut(&[u8])>(
        &self,
        from: usize,
        text: &str,
        image: &mut String,
        mut f: F,
    ) {
        if self.byte_split {
            self.pieces_mapping::<true, F>(0, from, text, image, &mut f);
        } else {
            self.pieces_mapping::<false, F>(0, from, text, image, &mut f);
        }
    }

    /// Applies the stages from `depth` on to `text`, calling `f` with each
    /// final piece. `f` is taken by static dispatch so that the per-piece
    /// work compiles into the walk. `SPLIT` says this copy of the walk
    /// carries the byte-level split's scanner: a stage that is that split
    /// then takes its matches from the scanner. It is a constant and no
    /// test in the walk, so the copy for stages without the split holds
    /// nothing for it (one more test a piece in the patterns' iterator
    /// costs a pattern the program runs 0.6 % of a warm encode's
    /// instructions, counted).
    fn pieces<const SPLIT: bool, F: FnMut(&str)>(&self, depth: usize, text: &str, f: &mut F) {
        if text.is_empty() {
            return;
        }
        let Some(stage) = self.stages.get(depth) else {
            f(text);
            return;
        };
        let (gaps, matches) = match stage.pieces.walk() {
            Walk::Merged => {
                // Each match takes the text after it up to the next match or
                // the end; the text before the first match stands alone, and
                // a text without a match is one piece.
                let mut merged = None;
                for (start, _) in stage.pattern.find_iter(text) {
                    match merged {
                        Some(from) => self.pieces::<SPLIT, F>(depth + 1, &text[from..start], f),
                        None if start > 0 => self.pieces::<SPLIT, F>(depth + 1, &text[..start], f),
                        None => {}
                    }
                    merged = Some(start);
                }
                self.pieces::<SPLIT, F>(depth + 1, &text[merged.unwrap_or(0)..], f);
                return;
            }
            Walk::Apart { gaps, matches } => (gaps, matches),
        };
        let mut cursor = 0;
        if SPLIT {
            if let Some(found) = stage.pattern.split_matches(text) {
                // The byte-level split: the same walk over its scanner's
                // matches, in a loop of its own so that the one below
                // holds no test for it.
                for (start, end) in found {
                    if gaps && cursor < start {
                        self.pieces::<SPLIT, F>(depth + 1, &text[cursor..start], f);
                    }
                    if matches {
                        self.pieces::<SPLIT, F>(depth + 1, &text[start..end], f);
                    }
                    cursor = end;
                }
                if gaps && cursor < text.len() {
                    self.pieces::<SPLIT, F>(depth + 1, &text[cursor..], f);
                }
                return;
            }
        }
        for (start, end) in stage.pattern.find_iter(text) {
            if gaps && cursor < start {
                self.pieces::<SPLIT, F>(depth + 1, &text[cursor..start], f);
            }
            if matches {
                self.pieces::<SPLIT, F>(depth + 1, &text[start..end], f);
            }
            cursor = end;
        }
        if gaps && cursor < text.len() {
            self.pieces::<SPLIT, F>(depth + 1, &text[cursor..], f);
        }
    }

    /// [`Self::pieces`] for a chain with stages over the byte-level image
    /// from the stage `from` on ([`Self::mapped_from`]): the stages before
    /// it run over the raw text as there; at it each piece is mapped into
    /// `image` once and the rest run over the image
    /// ([`Self::pieces_in_alphabet`]). `f` takes each final piece's raw
    /// bytes: a cut inside a multi-byte character is one the reference
    /// makes too, so a piece is not always a `str`.
    fn pieces_mapping<const SPLIT: bool, F: FnMut(&[u8])>(
        &self,
        depth: usize,
        from: usize,
        text: &str,
        image: &mut String,
        f: &mut F,
    ) {
        if text.is_empty() {
            return;
        }
        if depth >= from {
            // No match of a stage over the image begins in a piece that
            // holds none of their opening bytes: the piece is final.
            if let Some(opens) = &self.image_opens {
                if !text.bytes().any(|byte| opens[usize::from(byte)]) {
                    f(text.as_bytes());
                    return;
                }
            }
            image.clear();
            image.extend(text.bytes().map(|b| self.alphabet[usize::from(b)]));
            self.pieces_in_alphabet(depth, image, text.as_bytes(), f);
            return;
        }
        let stage = &self.stages[depth];
        let (gaps, matches) = match stage.pieces.walk() {
            Walk::Merged => {
                // As in `pieces`: each match takes the text after it up to
                // the next match or the end; the text before the first
                // match stands alone, and a text without a match is one
                // piece.
                let mut merged = None;
                for (start, _) in stage.pattern.find_iter(text) {
                    match merged {
                        Some(at) => {
                            self.pieces_mapping::<SPLIT, F>(
                                depth + 1,
                                from,
                                &text[at..start],
                                image,
                                f,
                            );
                        }
                        None if start > 0 => {
                            self.pieces_mapping::<SPLIT, F>(
                                depth + 1,
                                from,
                                &text[..start],
                                image,
                                f,
                            );
                        }
                        None => {}
                    }
                    merged = Some(start);
                }
                self.pieces_mapping::<SPLIT, F>(
                    depth + 1,
                    from,
                    &text[merged.unwrap_or(0)..],
                    image,
                    f,
                );
                return;
            }
            Walk::Apart { gaps, matches } => (gaps, matches),
        };
        let mut cursor = 0;
        if SPLIT {
            if let Some(found) = stage.pattern.split_matches(text) {
                // The byte-level split: the same walk over its scanner's
                // matches, in a loop of its own so that the one below
                // holds no test for it.
                for (start, end) in found {
                    if gaps && cursor < start {
                        self.pieces_mapping::<SPLIT, F>(
                            depth + 1,
                            from,
                            &text[cursor..start],
                            image,
                            f,
                        );
                    }
                    if matches {
                        self.pieces_mapping::<SPLIT, F>(
                            depth + 1,
                            from,
                            &text[start..end],
                            image,
                            f,
                        );
                    }
                    cursor = end;
                }
                if gaps && cursor < text.len() {
                    self.pieces_mapping::<SPLIT, F>(depth + 1, from, &text[cursor..], image, f);
                }
                return;
            }
        }
        for (start, end) in stage.pattern.find_iter(text) {
            if gaps && cursor < start {
                self.pieces_mapping::<SPLIT, F>(depth + 1, from, &text[cursor..start], image, f);
            }
            if matches {
                self.pieces_mapping::<SPLIT, F>(depth + 1, from, &text[start..end], image, f);
            }
            cursor = end;
        }
        if gaps && cursor < text.len() {
            self.pieces_mapping::<SPLIT, F>(depth + 1, from, &text[cursor..], image, f);
        }
    }

    /// Applies the stages from `depth` on to `image`, the byte-level image
    /// of `raw` (one alphabet character per byte, in order), calling `f`
    /// with each final piece's raw bytes: a match's bounds in characters of
    /// the image are its bounds in bytes of the raw text.
    fn pieces_in_alphabet<F: FnMut(&[u8])>(
        &self,
        depth: usize,
        image: &str,
        raw: &[u8],
        f: &mut F,
    ) {
        debug_assert_eq!(image.chars().count(), raw.len());
        if raw.is_empty() {
            return;
        }
        let Some(stage) = self.stages.get(depth) else {
            f(raw);
            return;
        };
        // `stages_from` declines a merged-with-next stage among those over
        // the image, so the walk over the image takes the pieces apart.
        let Walk::Apart { gaps, matches } = stage.pieces.walk() else {
            debug_assert!(false, "a merged-with-next stage over the byte-level image");
            f(raw);
            return;
        };
        let mut cursor = 0;
        let mut raw_cursor = 0;
        for (start, end) in stage.pattern.find_iter(image) {
            let raw_start = raw_cursor + image[cursor..start].chars().count();
            let raw_end = raw_start + image[start..end].chars().count();
            if gaps && cursor < start {
                self.pieces_in_alphabet(
                    depth + 1,
                    &image[cursor..start],
                    &raw[raw_cursor..raw_start],
                    f,
                );
            }
            if matches {
                self.pieces_in_alphabet(depth + 1, &image[start..end], &raw[raw_start..raw_end], f);
            }
            cursor = end;
            raw_cursor = raw_end;
        }
        if gaps && cursor < image.len() {
            self.pieces_in_alphabet(depth + 1, &image[cursor..], &raw[raw_cursor..], f);
        }
    }

    /// Appends the ids of one piece: from the thread's cache, else from the
    /// encoder's seed, else merged over the model's merges (the whole piece
    /// looked up first under `ignore_merges`), else, for a model the merge
    /// table declined or a piece holding a byte the table has no token for,
    /// from the model itself over the piece's byte-level alphabet form (a
    /// raw piece of the sentencepiece path never comes here:
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
        piece: &[u8],
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        // The hot path of a warm encode, and nothing else in this function:
        // the key of a short piece and the packed table's probe.
        let vacancy = match ShortPieceKey::of(piece) {
            Some(key) => match cache.packed.lookup(key, ids) {
                None => return Ok(()),
                Some(vacancy) if self.seeded(vacancy, ids, cache) => return Ok(()),
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
    ///
    /// Compiled twice. `ASCII` is the copy an all-ASCII unit takes (the
    /// flag [`Self::encode_unit`] reads once per unit): in it the template
    /// scanner ends its runs by the lean scan (`Roles::run_end_ascii`),
    /// which has no character beyond ASCII to care for, whatever the
    /// input's length, until that scan hands a unit that opens as short
    /// pieces (`block::is_broken`) to
    /// [`Self::encode_single_stage_by_blocks`], which reads the rest. It is
    /// a copy and no third loop of the one function:
    /// compiled in here, the lean loop moved the code of the other loops,
    /// and the byte loops and the cased scanner cost 1 to 2 % more
    /// operations per token on text the lean scan never reads (counted). A
    /// pattern with no lean scan leaves this copy at once: through
    /// [`Self::encode_single_stage_by_blocks`] when it gives a reader of the
    /// text by blocks (the cased-letter template, `Pattern::by_cased_blocks`),
    /// else on to the other copy, which is the function as it was.
    #[inline(never)]
    #[expect(clippy::too_many_arguments)]
    fn encode_single_stage<const ASCII: bool>(
        &self,
        model: Option<&ModelWrapper>,
        stage: &Stage,
        text: &str,
        words: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        if ASCII && !stage.pattern.has_lean_scan() {
            // No lean scan: the block pass where the pattern gives that
            // reader (the cased-letter template), else the walk as it was.
            if let Some(blocks) = stage.pattern.by_cased_blocks(text) {
                return self.encode_unit_by_blocks(model, blocks, text, words, mapped, ids, cache);
            }
            return self
                .encode_single_stage::<false>(model, stage, text, words, mapped, ids, cache);
        }
        let gaps = matches!(stage.pieces.walk(), Walk::Apart { gaps: true, .. });
        let mut cursor = 0;
        let mut fallback = None;
        // The probe compiles into the scanner's loop. As a closure of its
        // own it was a call per piece: a six-register frame, a reload of
        // every captured reference and the spills the slow path needs, 135
        // instructions for a hit of which the probe is about 50 (counted on
        // the long warm cells). The gap piece stays out of line: a scanner's
        // alternatives tile the text, so a gap is the rare case there.
        let stopped = stage.pattern.for_each_match::<ASCII, _>(
            text,
            words,
            #[inline(always)]
            |start, end| {
                if gaps && cursor < start {
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
        if stopped < text.len() {
            // The lean scan left at a piece's end: the unit opens as short
            // pieces of more than one kind, which the block pass reads in
            // fewer missed branches (see `for_each_match_to_marks`).
            if let Some(declined) = fallback {
                return Err(declined);
            }
            let rest = &text[stopped..];
            return match stage.pattern.by_blocks(rest) {
                Some(blocks) => {
                    self.encode_unit_by_blocks(model, blocks, rest, words, mapped, ids, cache)
                }
                None => {
                    self.encode_single_stage::<false>(model, stage, rest, words, mapped, ids, cache)
                }
            };
        }
        if gaps && cursor < text.len() {
            if let Err(declined) =
                self.tokenize_piece_at(model, text, cursor..text.len(), mapped, ids, cache)
            {
                fallback = Some(declined);
            }
        }
        fallback.map_or(Ok(()), Err)
    }

    /// [`Self::encode_single_stage`] for a unit its pattern reads by blocks,
    /// a routine of its own: the block pass and the walk between its
    /// stretches compile in here with a probe of their own, and
    /// [`Self::encode_single_stage`] keeps its text and its loops as they
    /// are. A pattern that gives a reader by blocks leaves no gap in any
    /// text (`Pattern::by_blocks`), so there is none to tokenize here and
    /// no cursor to keep. Compiled once a template (`CASED`, the reader's
    /// type): with the loops of both in one copy, the loop of the GPT-4
    /// family's template cost one instruction a token more (counted).
    #[inline(never)]
    #[expect(clippy::too_many_arguments)]
    fn encode_single_stage_by_blocks<const CASED: bool>(
        &self,
        model: Option<&ModelWrapper>,
        blocks: pattern::ByBlocks<'_, CASED>,
        text: &str,
        words: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let mut fallback = None;
        blocks.for_each_match(
            text,
            words,
            #[inline(always)]
            |start, end| {
                if let Err(declined) =
                    self.tokenize_piece_at(model, text, start..end, mapped, ids, cache)
                {
                    fallback = Some(declined);
                }
            },
        );
        fallback.map_or(Ok(()), Err)
    }

    /// The table's size is asked here, at the call: asked inside the routine
    /// by blocks, its loop ran one instruction a token more (counted).
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn encode_unit_by_blocks<const CASED: bool>(
        &self,
        model: Option<&ModelWrapper>,
        blocks: pattern::ByBlocks<'_, CASED>,
        text: &str,
        words: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        if cache.packed.slots() > ASK_AHEAD_SLOTS {
            self.encode_single_stage_asked_ahead(model, blocks, text, words, mapped, ids, cache)
        } else {
            self.encode_single_stage_by_blocks(model, blocks, text, words, mapped, ids, cache)
        }
    }

    /// [`Self::encode_single_stage_by_blocks`] for a table past
    /// [`ASK_AHEAD_SLOTS`].
    #[inline(never)]
    #[expect(clippy::too_many_arguments)]
    fn encode_single_stage_asked_ahead<const CASED: bool>(
        &self,
        model: Option<&ModelWrapper>,
        blocks: pattern::ByBlocks<'_, CASED>,
        text: &str,
        words: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let mut fallback = None;
        let bytes = text.as_bytes();
        let mut held = [Asked {
            start: 0,
            end: 0,
            key: None,
            hash: 0,
        }; ASK_AHEAD];
        let mut count = 0usize;
        blocks.for_each_match(
            text,
            words,
            #[inline(always)]
            |start, end| {
                let key = ShortPieceKey::at(bytes, start, end);
                // A piece without a key holds hash 0: its touch reads the
                // table's first lines again, cheaper than a test a piece.
                let hash = key.map_or(0, ShortPieceTable::hash_of);
                held[count] = Asked {
                    start,
                    end,
                    key,
                    hash,
                };
                count += 1;
                if count == ASK_AHEAD {
                    count = 0;
                    if let Err(declined) = self.answer_asked(model, text, &held, mapped, ids, cache)
                    {
                        fallback = Some(declined);
                    }
                }
            },
        );
        if count != 0 {
            let tail = self.answer_asked(model, text, &held[..count], mapped, ids, cache);
            fallback = tail.err().or(fallback);
        }
        fallback.map_or(Ok(()), Err)
    }

    /// Answers `pieces` of `text`, in order. The touches go before the
    /// probes, with nothing between them, so that the far lines of all the
    /// pieces are in flight together.
    #[inline(never)]
    fn answer_asked(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        pieces: &[Asked],
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let bytes = text.as_bytes();
        let mut touched = 0u8;
        for piece in pieces {
            touched ^= cache.packed.touch_hashed(piece.hash);
        }
        std::hint::black_box(touched);
        let mut fallback = None;
        for piece in pieces {
            let vacancy = match piece.key {
                Some(key) => match cache.packed.lookup_hashed(key, piece.hash, ids) {
                    None => continue,
                    Some(vacancy) if self.seeded(vacancy, ids, cache) => continue,
                    vacancy => vacancy,
                },
                None => None,
            };
            let piece = &bytes[piece.start..piece.end];
            if let Err(declined) =
                self.tokenize_piece_slow(model, piece, vacancy, mapped, ids, cache)
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
                Some(vacancy) if self.seeded(vacancy, ids, cache) => return Ok(()),
                vacancy => vacancy,
            },
            None => None,
        };
        self.tokenize_piece_slow(model, &text.as_bytes()[piece], vacancy, mapped, ids, cache)
    }

    /// [`Self::tokenize_piece_at`] for a walk that compiles the probe into
    /// its loop, with only the packed table's answer compiled in there: the
    /// key by its one read of the text, the probe, an entry's inline ids.
    /// Every other piece leaves the loop by a call, so the walk keeps its
    /// registers for the hit: a miss asks the seed, which holds nearly every
    /// piece a new text misses; a piece longer than a key goes to the slow
    /// path as it does from [`Self::tokenize_piece_at`] (the wide table or
    /// the map answers it on a warm encode, so it is no rare piece); the
    /// rest takes one cold call ([`Self::tokenize_missed_piece`]).
    #[inline(always)]
    fn tokenize_piece_in_place(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        piece: Range<usize>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let bytes = text.as_bytes();
        match ShortPieceKey::in_window(bytes, piece.start, piece.end) {
            Some(key) => match cache.packed.lookup_inline(key, ids) {
                Ok(()) => Ok(()),
                Err(Some(vacancy)) if self.seeded(vacancy, ids, cache) => Ok(()),
                Err(missed) => {
                    let unseeded = missed.is_some();
                    self.tokenize_missed_piece(model, text, piece, unseeded, mapped, ids, cache)
                }
            },
            None if piece.len() > SHORT_PIECE_MAX_BYTES => {
                self.tokenize_piece_slow(model, &bytes[piece], None, mapped, ids, cache)
            }
            None => self.tokenize_missed_piece(model, text, piece, false, mapped, ids, cache),
        }
    }

    /// The rest of [`Self::tokenize_piece_in_place`], for a short piece its
    /// loop did not answer. `unseeded` is a miss the seed's call did not
    /// answer either: it goes to the slow path from its vacancy, found by a
    /// second probe, since a loop that kept the vacancy across the seed's
    /// call ran more instructions on a hit. A piece the loop did not probe
    /// to the end (no key by the one read, near the end of the text, or ids
    /// in the spill) takes [`Self::tokenize_piece_at`] whole.
    #[cold]
    #[inline(never)]
    #[expect(clippy::too_many_arguments)]
    fn tokenize_missed_piece(
        &self,
        model: Option<&ModelWrapper>,
        text: &str,
        piece: Range<usize>,
        unseeded: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        if !unseeded {
            return self.tokenize_piece_at(model, text, piece, mapped, ids, cache);
        }
        let bytes = text.as_bytes();
        let vacancy = ShortPieceKey::at(bytes, piece.start, piece.end)
            .and_then(|key| cache.packed.probe(key).err());
        self.tokenize_piece_slow(model, &bytes[piece], vacancy, mapped, ids, cache)
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

    /// The wait of a miss of a young thread or of one that shares
    /// (`PieceCache::shares`; [`Self::seeded`] and [`Self::seeded_raw`] ask).
    /// A first sight reads
    /// lines of the seed that are far from the thread, as are the lines of
    /// the learned table a thread that shares asks, and a piece answered
    /// where it is met waits for its own lines alone. A miss within
    /// [`WAITING_GAP`] ids of the last one, or any miss of a thread that
    /// shares (each asks the learned table, however far apart they fall),
    /// takes a hole of
    /// one id in the output and waits, and up to [`WAITING_PIECES`] are
    /// answered together ([`Self::answer_waiting`]), their lines read at once
    /// (measured: a first sight of a word list 15 to 17 % faster, new text
    /// of 16 tokens 5 to 10 %, of 256 tokens 8 to 12 %). [`Self::seeded`]
    /// puts a miss close behind the last one (`near`) on the list itself
    /// while both lists have room and no answer is due; this is the rest of
    /// the wait. False for a miss that stands alone: the seed answers it
    /// where it is met. True for a piece that waits; the waiting pieces are
    /// answered here, this one the last of them, when they are as many as
    /// are answered together or the miss is not close behind the last one.
    #[cold]
    #[inline(never)]
    fn waits(
        &self,
        vacancy: Vacancy,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
        near: bool,
    ) -> bool {
        let close = near || cache.shares;
        if !close && cache.waiting.is_empty() {
            return false;
        }
        let at = ids.len();
        cache.waiting.push(Waiting { vacancy, at });
        ids.push(0);
        if !close || cache.waiting.len() >= WAITING_PIECES {
            self.answer_waiting(ids, cache);
        }
        true
    }

    /// Answers the waiting pieces ([`Self::waits`]). First one byte of the
    /// tag and of the entry every piece's walk of the seed begins on is
    /// read, one piece after the other with nothing between the reads, so
    /// the far lines of all the pieces are in flight together. The bytes go
    /// through `std::hint::black_box`, which keeps the loop: the hint is a
    /// request, and where it were not honoured the pieces would be answered
    /// one by one, with the same ids (measured: a first loop that does more
    /// for a piece keeps less of the gain: one bit kept for the answer keeps
    /// half to seven tenths of it on a first sight; a whole walk keeps seven
    /// tenths to all of it there and half of it or less on new text of 16
    /// tokens; no first loop keeps none). Then the pieces are answered, the
    /// last hole first, and each is learned as [`Self::seeded`] and the slow
    /// call learn it: at its vacancy, unless the cache was cleared since the
    /// waiting pieces were last answered (`PieceCache::answered_clears`),
    /// when the piece asks the table again. An answer of one id is
    /// written into its hole. An answer that is not one id stays in the
    /// thread's scratch list until the last piece is answered; one pass
    /// from the last hole down then places them all: the output grows once,
    /// the ids behind each hole move to their place once, and the holes
    /// before it stay where they are. (Spliced in one at a time, each such
    /// answer moved every id behind its hole: with new words in front of a
    /// long text the thread knows, that text moved once for each word.) A
    /// piece the seed does not hold asks the thread's table once more
    /// before it is merged: one met twice while it waited is merged once.
    /// A thread that shares (`PieceCache::shares`) then asks what the
    /// threads have merged
    /// ([`Self::learned_pieces`], its lines read early behind the seed's):
    /// a piece it holds is kept the way a seeded piece is, and one it does
    /// not hold is merged and told to it. The thread counts these asks and
    /// those the table answers, and rests where it answers under one in
    /// two (`PieceCache::judge_asks`).
    /// The merge is the table's, or the character symbols' on the
    /// sentencepiece path, and asks no model, so the answer comes wherever
    /// it is due: a piece waits only where every short piece is merged so
    /// ([`Self::seeded`], [`Self::seeded_raw`]).
    #[cold]
    #[inline(never)]
    fn answer_waiting(&self, ids: &mut Vec<u32>, cache: &mut PieceCache) {
        let waiting = std::mem::take(&mut cache.waiting);
        let mut answer = std::mem::take(&mut cache.answer);
        // A sentencepiece piece in the vocabulary's form, for its merge.
        let mut mapped = std::mem::take(&mut cache.mapped);
        let mut touched = 0u8;
        for piece in &waiting {
            touched ^= self.seed.touch(piece.vacancy.key());
        }
        let learned = cache.shares.then(|| self.learned_pieces());
        if let Some(learned) = learned {
            for piece in &waiting {
                touched ^= learned.touch(piece.vacancy.key());
            }
        }
        std::hint::black_box(touched);
        // The answers that are not one id: the hole of each and where its ids
        // begin in the scratch list, the last hole first.
        let mut late = [(0usize, 0usize); WAITING_PIECES];
        let mut late_len = 0;
        // The asks of the learned table here, and those it answers.
        let (mut asked, mut of_use) = (0u32, 0u32);
        answer.clear();
        for piece in waiting.iter().rev() {
            let key = piece.vacancy.key();
            // By reference: as a copy the vacancy was written to the stack
            // for every piece, for the calls that few pieces make.
            let anew;
            let mut vacancy = Some(&piece.vacancy);
            let mut one = None;
            let from = answer.len();
            if cache.answered_clears != cache.clears {
                match cache.packed.probe(key) {
                    Ok(entry) => {
                        one = cache.packed.one_or_append(entry, &mut answer);
                        vacancy = None;
                    }
                    Err(fresh) => {
                        anew = fresh;
                        vacancy = Some(&anew);
                    }
                }
            }
            if let Some(vacancy) = vacancy {
                if let Ok(entry) = self.seed.probe(key) {
                    one = self.keep_answer(&self.seed, entry, vacancy, from, &mut answer, cache);
                } else {
                    // The seed does not hold it: the thread's table (a
                    // copy of a piece of this batch), the learned table
                    // or a merge answers. The hint lays the loop out for
                    // the pieces the seed answers; all three ride the
                    // cold edge.
                    std::hint::cold_path();
                    match cache.packed.probe(key) {
                        Ok(entry) => one = cache.packed.one_or_append(entry, &mut answer),
                        Err(fresh) => {
                            let part = learned.and_then(|learned| learned.read(key));
                            let held = part
                                .as_deref()
                                .and_then(|table| Some((table, table.probe(key).ok()?)));
                            asked += 1;
                            if let Some((table, entry)) = held {
                                // Another thread, or this one before a
                                // clear, has merged it.
                                one = self.keep_answer(
                                    table,
                                    entry,
                                    &fresh,
                                    from,
                                    &mut answer,
                                    cache,
                                );
                                of_use += 1;
                            } else {
                                // A writer needs the part to itself.
                                drop(part);
                                // Nobody holds it: it is merged as the slow
                                // call of its path merges a piece (the table
                                // from its pre-formed characters when it has
                                // them, the sentencepiece path from the
                                // character symbols of the raw piece the key
                                // holds), and the thread remembers it.
                                let (bytes, len) = key.bytes();
                                let piece = &bytes[..len];
                                debug_assert!(
                                    self.misses_wait || self.chars.is_some(),
                                    "every short piece is merged"
                                );
                                if let Some(chars) = &self.chars {
                                    // A piece is cut between characters:
                                    // nothing of it is replaced here.
                                    let raw = String::from_utf8_lossy(piece);
                                    chars.merge(
                                        CharSymbols::normalized(&raw, &mut mapped),
                                        &mut cache.symbols,
                                        &mut answer,
                                    );
                                } else if let Some(table) = &self.merges {
                                    if piece.is_ascii()
                                        || !table.merge_preformed(piece, &mut answer)
                                    {
                                        table.merge(piece, &mut answer);
                                    }
                                }
                                self.remember_piece(piece, Some(fresh), from, &answer, cache);
                                if let Some(learned) = learned {
                                    learned.learn(key, &answer[from..]);
                                }
                            }
                        }
                    }
                }
            }
            match one {
                Some(id) => ids[piece.at] = id,
                None if answer.len() == from => {
                    // No id (a table that merges every short piece gives no
                    // such answer): the hole closes, and the holes behind it
                    // move with the ids.
                    ids.remove(piece.at);
                    for (at, _) in &mut late[..late_len] {
                        *at -= 1;
                    }
                }
                None => {
                    late[late_len] = (piece.at, from);
                    late_len += 1;
                }
            }
        }
        // One pass from the last hole down: the output grows once by what
        // the answers of several ids add, the ids behind each hole move to
        // their place once, and the answer goes in front of them.
        let mut rest = ids.len();
        let mut to = rest + answer.len() - late_len;
        ids.resize(to, 0);
        for index in 0..late_len {
            let (at, from) = late[index];
            let end = if index + 1 < late_len {
                late[index + 1].1
            } else {
                answer.len()
            };
            let behind = rest - at - 1;
            if to != rest {
                ids.copy_within(at + 1..rest, to - behind);
            }
            to -= behind + (end - from);
            ids[to..to + end - from].copy_from_slice(&answer[from..end]);
            rest = at;
        }
        cache.waiting = waiting;
        cache.waiting.clear();
        cache.answered_clears = cache.clears;
        cache.answer = answer;
        cache.mapped = mapped;
        if learned.is_some() {
            cache.judge_asks(asked, of_use);
        }
    }

    /// The answer of a waiting piece that `table` holds as `entry` (the
    /// seed, or a part of the learned table): its one id returned, or its
    /// ids appended to `answer`, and the entry kept in the thread's table
    /// at `vacancy` as a copy ([`ShortPieceTable::keep`]) or, where that
    /// table takes no copy, remembered from its ids.
    #[inline(always)]
    fn keep_answer(
        &self,
        table: &ShortPieceTable,
        entry: &Entry,
        vacancy: &Vacancy,
        from: usize,
        answer: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Option<u32> {
        let one = table.one_or_append(entry, answer);
        if cache.short_full(cache.share) || !cache.packed.keep(*vacancy, entry) {
            // Not kept as a copy: remembered from its ids.
            answer.extend(one);
            self.remember_seeded(*vacancy, from, answer, cache);
            if one.is_some() {
                answer.truncate(from);
            }
        }
        one
    }

    /// The pieces that still wait when the units of an encode call are done
    /// (one unit on the pool) take their ids ([`Self::answer_waiting`]).
    /// Here and not at each unit's end: `encode_unit` holds the loops over
    /// the pieces, and it keeps the text and the frame it had.
    #[inline(always)]
    fn answer_rest(&self, ids: &mut Vec<u32>, cache: &mut PieceCache) {
        if !cache.waiting.is_empty() {
            self.answer_waiting(ids, cache);
        }
    }

    /// A short piece the thread's table missed (`vacancy`), answered when
    /// the encoder's seed holds it, as it holds every short vocabulary word:
    /// the ids appended, and the seed's entry kept in the thread's table at
    /// the vacancy as a copy ([`ShortPieceTable::keep`]), so a first-sight
    /// vocabulary word pays neither the slow path's frame nor a second walk
    /// and an insert from its ids (70 to 75 instructions fewer a word,
    /// counted on three vocabularies). False when the seed does not hold
    /// the piece: the slow path merges it. Out of line and cold: inlined at
    /// the probe's miss it cost every hit up to ten instructions, as a plain
    /// call up to twenty-three on one vocabulary; cold, a hit costs what it
    /// did or less (counted).
    #[cold]
    #[inline(never)]
    fn seeded(&self, vacancy: Vacancy, ids: &mut Vec<u32>, cache: &mut PieceCache) -> bool {
        // A young thread's miss close behind another waits (`Self::waits`);
        // a thread whose table has outgrown its reservation knows the common
        // words, most of what it misses is merged, and a merge gains nothing
        // by waiting (counted: waiting cost new text of 32k tokens 15
        // instructions a token and of 500k tokens 2 to 5), until the thread
        // shares (`PieceCache::shares`): from then on a miss asks the
        // learned table before its merge, whose lines are far, and waits
        // for the others. Nor
        // does a miss wait where a short piece may need the model
        // (`Self::misses_wait`).
        // The tests are three compares in front of the seed's read and not a
        // call of their own: two first sights in a row read their far lines
        // together only while both fit the processor's window, and a first
        // sight with a few dozen instructions more in front of the read was
        // 4 to 25 % slower (measured on first sights of word lists).
        if (cache.packed.slots() <= THREAD_BACKING_SLOTS || cache.shares) && self.misses_wait {
            let at = ids.len();
            let near = at.wrapping_sub(cache.last_miss) <= WAITING_GAP;
            cache.last_miss = at;
            // A miss close behind the last one takes its place on the list
            // here while both lists have room and no answer is due: one call
            // a piece and not two (counted: 54 instructions a piece to its
            // place on the list, 82 through `Self::waits`).
            if near
                && cache.waiting.len() < cache.waiting.capacity()
                && cache.waiting.len() + 1 < WAITING_PIECES
                && at < ids.capacity()
            {
                cache.waiting.push(Waiting { vacancy, at });
                ids.push(0);
                return true;
            }
            if self.waits(vacancy, ids, cache, near) {
                return true;
            }
        }
        self.seeded_alone(vacancy, ids, cache)
    }

    /// [`Self::seeded`] for a piece of the sentencepiece path, whose
    /// character symbols merge every piece with no model: a young thread's
    /// miss waits as well ([`Self::waits`]), where the seed is far
    /// (`NEAR_SEED_SLOTS`); else the seed answers where the piece is met.
    #[cold]
    #[inline(never)]
    fn seeded_raw(&self, vacancy: Vacancy, ids: &mut Vec<u32>, cache: &mut PieceCache) -> bool {
        if cache.packed.slots() <= THREAD_BACKING_SLOTS && self.seed.slots() > NEAR_SEED_SLOTS {
            let at = ids.len();
            let near = at.wrapping_sub(cache.last_miss) <= WAITING_GAP;
            cache.last_miss = at;
            if self.waits(vacancy, ids, cache, near) {
                return true;
            }
        }
        self.seeded_alone(vacancy, ids, cache)
    }

    /// The seed's answer where the piece is met: the body of
    /// [`Self::seeded`] and of [`Self::seeded_raw`], compiled into both.
    #[inline(always)]
    fn seeded_alone(&self, vacancy: Vacancy, ids: &mut Vec<u32>, cache: &mut PieceCache) -> bool {
        let Ok(entry) = self.seed.probe(vacancy.key()) else {
            return false;
        };
        let kept = !cache.short_full(cache.share) && cache.packed.keep(vacancy, entry);
        let start = ids.len();
        self.seed.append(entry, ids);
        if !kept {
            self.remember_seeded(vacancy, start, ids, cache);
        }
        true
    }

    /// [`Self::remember_piece`] for a seeded piece the thread's table did
    /// not keep as a copy: its ids are in the seed's spill, or the table is
    /// at its fill bound or the thread at its share. The rare case, out of
    /// line so that [`Self::seeded`] keeps no frame for it.
    #[cold]
    #[inline(never)]
    fn remember_seeded(&self, vacancy: Vacancy, start: usize, ids: &[u32], cache: &mut PieceCache) {
        self.remember_piece(&[], Some(vacancy), start, ids, cache);
    }

    /// The rest of [`Self::tokenize_piece`], out of line, for a piece the
    /// thread's packed table and the seed do not hold: the longer pieces'
    /// cache, the merge or the model, and the cache insert.
    #[inline(never)]
    fn tokenize_piece_slow(
        &self,
        model: Option<&ModelWrapper>,
        bytes: &[u8],
        vacancy: Option<Vacancy>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> Result<(), Fallback> {
        let cacheable = bytes.len() <= LONG_PIECE_MAX_BYTES;
        // A piece of 16 to 32 bytes is the wide table's when its ids fit an
        // entry; one the table does not hold may still be the map's.
        let wide = match vacancy {
            None => WideKey::of(bytes),
            Some(_) => None,
        };
        if wide.is_some_and(|key| cache.wide.ids_into(key, ids)) {
            return Ok(());
        }
        if vacancy.is_none() && cacheable && (wide.is_none() || !cache.short.is_empty()) {
            if let Some(found) = cache.get(bytes) {
                ids.extend_from_slice(found);
                return Ok(());
            }
        }
        let start = ids.len();
        // A piece this thread has not met and the seed does not hold (the
        // probe's miss asked it, see `Self::seeded`): merged, by the table
        // or by the model for a table that declined. A raw piece of the
        // sentencepiece path never comes here: it has its own slow path.
        debug_assert!(
            self.chars.is_none(),
            "a sentencepiece piece takes tokenize_raw_piece_slow"
        );
        // A piece holding a byte the table has no token for is the
        // model's (see `MergeTable::lacks_a_token_for`).
        if let Some(table) = self
            .merges
            .as_ref()
            .filter(|table| !table.lacks_a_token_for(bytes))
        {
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
        self.remember_piece(bytes, vacancy, start, ids, cache);
        Ok(())
    }

    /// Remembers `ids[start..]` as the encoding of `piece` in the thread's
    /// cache: the packed table for a piece with a key, the wide table or the
    /// map for one of up to `CACHED_PIECE_MAX_BYTES` (the table when it is
    /// of at most 32 bytes and seven ids), the long tier, by bytes, for one of up
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
            if cache.short_full(share) {
                // The vacancy named a slot of the table as it was: the cleared
                // table is probed again.
                cache.clear_short(share);
                cache.packed.insert(vacancy.key(), &ids[start..], share);
            } else {
                cache.packed.insert_at(vacancy, &ids[start..], share);
            }
        } else if piece.len() <= CACHED_PIECE_MAX_BYTES {
            let share = cache.share;
            if cache.short_full(share) {
                cache.clear_short(share);
            }
            let kept =
                WideKey::of(piece).is_some_and(|key| cache.wide.insert(key, &ids[start..], share));
            if !kept {
                cache.short.insert(piece.into(), ids[start..].into());
            }
        } else if piece.len() <= LONG_PIECE_MAX_BYTES {
            let entry = LONG_PIECE_ENTRY_BYTES + piece.len() + size_of_val(&ids[start..]);
            let share = self.long_piece_bytes_share();
            if entry <= share {
                if cache.long_bytes + entry > share {
                    cache.clear_long_within(share);
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
        // A unit under this length keeps the walk below, which asks each
        // piece whether it is the first and reads a key near the end of the
        // text inside the loop: the other walk's scan to the first cut and
        // its call for each piece near the end cost a short unit more than
        // its few pieces gain (counted on units between added tokens).
        const LONG_UNIT_BYTES: usize = 256;
        if text.len() >= LONG_UNIT_BYTES
            && self.encode_long_sentencepiece_unit(chars, text, marker, mapped, ids, cache)
        {
            return;
        }
        let bytes = text.as_bytes();
        // A text that starts with a space or a marker has its marker already,
        // unless the normalizer prepends one regardless.
        let mut leading = marker
            && (chars.prepends_to_a_marked_text()
                || !(text.starts_with(' ') || text.starts_with(WORD_MARKER)));
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
                    let key = ShortPieceKey::at(bytes, start, end);
                    self.tokenize_keyed_raw_piece(chars, key, text, start..end, mapped, ids, cache);
                }
            },
        );
    }

    /// [`Self::encode_sentencepiece`] for a long unit: the marker's piece is
    /// encoded before the walk, which then asks no piece whether it is the
    /// first and reads a piece's key by one read of the text where sixteen
    /// bytes stand from its start on. False, with nothing encoded, when a
    /// prepended marker joins the text and no cut ends its piece near the
    /// start: the caller's walk takes that unit. Out of line, so that each
    /// walk keeps its own loop.
    #[inline(never)]
    fn encode_long_sentencepiece_unit(
        &self,
        chars: &CharSymbols,
        text: &str,
        marker: bool,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) -> bool {
        // The marker piece's end is searched a byte at a time, so only this
        // near the start: the caller's walk finds a later one by the block.
        const FIRST_CUT_BYTES: usize = 32;
        let bytes = text.as_bytes();
        let mut rest = text;
        // A text that starts with a space or a marker has its marker already,
        // unless the normalizer prepends one regardless.
        if marker
            && (chars.prepends_to_a_marked_text()
                || !(text.starts_with(' ') || text.starts_with(WORD_MARKER)))
        {
            // The marker's piece goes before the walk, which then tests no
            // piece for it: the marker alone when no token bridges it to the
            // first byte, else the marker and the text up to its first cut.
            let first = if chars.cuts_at(&[b' ', bytes[0]], 1) {
                0
            } else {
                let near = FIRST_CUT_BYTES.min(bytes.len());
                match (1..near).find(|&p| chars.cuts_at(bytes, p)) {
                    Some(first) => first,
                    None => return false,
                }
            };
            self.tokenize_leading_piece(chars, &text[..first], mapped, ids, cache);
            rest = &text[first..];
        }
        chars.pieces(
            rest.as_bytes(),
            #[inline(always)]
            |start, end| self.tokenize_raw_piece(chars, rest, start..end, mapped, ids, cache),
        );
        true
    }

    /// The piece `text[piece]` of a long sentencepiece unit, compiled into
    /// its walk's loop: the key by its one read of the text, as in
    /// [`Self::tokenize_piece_in_place`], the packed table's probe and a
    /// hit's ids, a spilled entry's too. A miss, a piece longer than a key
    /// and a short piece the one read does not cover leave the loop by a
    /// call.
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
        let key = match ShortPieceKey::in_window(text.as_bytes(), piece.start, piece.end) {
            None if piece.len() <= SHORT_PIECE_MAX_BYTES => {
                return self.tokenize_raw_piece_off_loop(chars, text, piece, mapped, ids, cache)
            }
            key => key,
        };
        self.tokenize_keyed_raw_piece(chars, key, text, piece, mapped, ids, cache);
    }

    /// [`Self::tokenize_raw_piece`] for a short piece its loop's one read
    /// does not cover: one with fewer than sixteen bytes of the text from its
    /// start on, whose key is read on its own.
    #[cold]
    #[inline(never)]
    fn tokenize_raw_piece_off_loop(
        &self,
        chars: &CharSymbols,
        text: &str,
        piece: Range<usize>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) {
        let key = ShortPieceKey::of(&text.as_bytes()[piece.clone()]);
        self.tokenize_keyed_raw_piece(chars, key, text, piece, mapped, ids, cache);
    }

    /// The piece `text[piece]` of a sentencepiece unit by its key, when it
    /// is short enough to have one: the packed table's probe and the seed.
    /// The rest goes out of line, and the piece is cut out of the text only
    /// there: a hit of the packed table or of the seed never pays for it.
    #[inline(always)]
    #[expect(clippy::too_many_arguments)]
    fn tokenize_keyed_raw_piece(
        &self,
        chars: &CharSymbols,
        key: Option<ShortPieceKey>,
        text: &str,
        piece: Range<usize>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
        cache: &mut PieceCache,
    ) {
        let vacancy = match key {
            Some(key) => match cache.packed.lookup(key, ids) {
                None => return,
                Some(vacancy) if self.seeded_raw(vacancy, ids, cache) => return,
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
                Some(vacancy) if self.seeded_raw(vacancy, ids, cache) => return,
                vacancy => vacancy,
            },
            None => None,
        };
        self.tokenize_raw_piece_slow(chars, &spaced, vacancy, mapped, ids, cache);
    }

    /// The rest of [`Self::tokenize_raw_piece`], out of line, for a piece
    /// the thread's packed table and the seed do not hold: the longer
    /// pieces' cache (by the raw bytes), the merge from the piece's
    /// character symbols in the vocabulary's form, and the cache insert.
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
        if vacancy.is_none() && WideKey::of(bytes).is_some_and(|key| cache.wide.ids_into(key, ids))
        {
            return;
        }
        if vacancy.is_none() && bytes.len() <= LONG_PIECE_MAX_BYTES {
            if let Some(found) = cache.get(bytes) {
                ids.extend_from_slice(found);
                return;
            }
        }
        let start = ids.len();
        chars.merge(
            CharSymbols::normalized(raw, mapped),
            &mut cache.symbols,
            ids,
        );
        self.remember_piece(bytes, vacancy, start, ids, cache);
    }

    /// The calling thread's piece cache for this encoder, made at the
    /// thread's first encode with it (from then on the thread counts toward
    /// every thread's share, see [`Self::piece_cache_share`], and, unless it
    /// is a thread of a pool then, among the threads that call, see
    /// `LEARNED_FROM_THREADS`).
    fn thread_piece_cache(&self) -> &RefCell<PieceCache> {
        self.piece_cache.get_or(|| {
            self.threads.fetch_add(1, Ordering::Relaxed);
            if rayon::current_thread_index().is_none() {
                self.callers.fetch_add(1, Ordering::Relaxed);
            }
            RefCell::new(PieceCache::with_share(self.piece_cache_share()))
        })
    }

    /// The calling thread's piece cache, borrowed for one encode call and
    /// within the thread's share at the start of it: a thread whose share
    /// fell since it last read it (threads joined) clears its count-bounded
    /// tiers here, by their entries, by the slots of its packed or of its
    /// wide table or by the capacity of its map (arrays grown for the earlier share keep
    /// their size through a clear at the bound; here the map is made anew for
    /// the share and the packed table is cut back to the share's slots inside
    /// its reservation, see `ShortPieceTable::clear`), and makes its long
    /// tier anew when it holds more than four times the long share, entries
    /// and empty slots together (see `PieceCache::remake_long`), so
    /// every thread that still encodes is within the share
    /// from its next call on, its long tier within four times the long
    /// share until it next adds a long piece (the share is read here once
    /// and kept in the
    /// cache for the call's misses, so a call that was running when threads
    /// joined keeps the share it started with until it ends; a thread that
    /// never encodes again holds what it had while the encoder lives, see
    /// `LONG_PIECE_BYTES_TOTAL`). Whether the
    /// thread shares what it merges is set here as well
    /// (`PieceCache::shares`): a thread that rests
    /// (`PieceCache::judge_asks`) shares again once its tiers hold the
    /// entries its rest lasts. It is set before the clear below and turned
    /// on nowhere else, so it holds for the rest of the call unless the
    /// thread starts to rest: a thread whose rest a clear ends, here or
    /// where it remembers a piece, shares again from its next call.
    /// Four relaxed loads, three stores and eleven
    /// compares per call at the most. A call that ended on
    /// a declined piece or unwound left its waiting pieces behind
    /// (`Self::waits`): they are dropped here, their holes stood in ids
    /// that are gone.
    fn piece_cache_for_call(&self) -> RefMut<'_, PieceCache> {
        let mut cache = self.thread_piece_cache().borrow_mut();
        let share = self.piece_cache_share();
        cache.shares =
            cache.clears > 0 && self.enough_callers() && cache.short_entries() >= cache.asks_from;
        cache.share = share;
        cache.waiting.clear();
        let slots = ShortPieceTable::slots_for(share);
        if cache.counted_entries(share) > share
            || cache.packed_slots() > slots
            || cache.wide_slots() > WidePieceTable::slots_for(share)
            || cache.map_capacity() > 2 * share
        {
            cache.clear_short(share);
        }
        let long_share = self.long_piece_bytes_share();
        if cache.long_held_bytes() > 4 * long_share {
            cache.remake_long();
        }
        cache
    }

    /// [`Self::piece_cache_for_call`] for a call that encodes a text of
    /// `bytes` bytes, or a unit of one. The thread's second call reads the
    /// merge table through once (`MergeTable::read_through`), so the
    /// first-sight pieces of the calls after it find the table's lines in
    /// the caches; its first call does not, the read is most of the time of
    /// a short first encode and only the encodes after it pay it back. A
    /// first text of [`READ_AT_ONCE_BYTES`] or more has the read before it.
    /// A call is one call of the encoder, not one prompt: a prompt encoded
    /// by segments is several, and has the read before its second. The
    /// thread that loaded skips its read while the load's own is fresh
    /// ([`Self::load_read_holds`]).
    fn piece_cache_for_text(&self, bytes: usize) -> RefMut<'_, PieceCache> {
        let mut cache = self.piece_cache_for_call();
        if cache.calls_to_read > 0 {
            cache.calls_to_read = if bytes >= READ_AT_ONCE_BYTES {
                0
            } else {
                cache.calls_to_read - 1
            };
            if cache.calls_to_read == 0 && !self.load_read_holds(Instant::now()) {
                if let Some(merges) = &self.merges {
                    merges.read_through();
                }
            }
        }
        cache
    }

    /// Whether the read that ended the load ([`Self::rehearse`]) stands in
    /// for the calling thread's own at `now`: the thread is the one that
    /// loaded, and the read is less than [`LOAD_READ_HOLDS`] old.
    fn load_read_holds(&self, now: Instant) -> bool {
        self.load_read.get().is_some_and(|&(loader, read_at)| {
            loader == std::thread::current().id() && now.duration_since(read_at) < LOAD_READ_HOLDS
        })
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

    /// One thread's share of the piece cache (`PIECE_CACHE_CAPACITY` says
    /// what a share bounds): the total shared among the
    /// threads that have encoded with this encoder, `PIECE_CACHE_CAPACITY` at
    /// most, and where the threads share what they merge
    /// ([`Self::enough_callers`]) a part of that (`GROWN_SHARE_PART`),
    /// beside the table that keeps it for all of them
    /// ([`Self::learned_pieces`]).
    fn piece_cache_share(&self) -> usize {
        let threads = self.threads.load(Ordering::Relaxed).max(1);
        let share = (PIECE_CACHE_TOTAL_CAPACITY / threads).clamp(1, PIECE_CACHE_CAPACITY);
        if self.enough_callers() {
            // The threads share what they merge: each keeps a part.
            (share / GROWN_SHARE_PART).max(1)
        } else {
            share
        }
    }

    /// Whether the threads of this encoder share what they merge: a miss of
    /// theirs can ask the learned table, which only a miss that waits does
    /// (`Self::misses_wait`), and enough threads call
    /// (`LEARNED_FROM_THREADS`). The cut of a thread's share
    /// ([`Self::piece_cache_share`]) and `PieceCache::shares` both follow
    /// it, so an encoder whose misses never reach the table keeps every
    /// thread's whole share.
    fn enough_callers(&self) -> bool {
        self.misses_wait && self.callers.load(Ordering::Relaxed) >= LEARNED_FROM_THREADS
    }

    /// What the threads have merged ([`LearnedPieces`]), made at the first
    /// ask, which is the first miss of a thread that shares
    /// (`PieceCache::shares`): for twice the entries the threads that have
    /// encoded with this encoder by then keep themselves, half of what they
    /// held before the table (see `GROWN_SHARE_PART`). Threads that join
    /// later share the table as it was made.
    fn learned_pieces(&self) -> &LearnedPieces {
        self.learned.get_or_init(|| {
            let threads = self.threads.load(Ordering::Relaxed).max(1);
            LearnedPieces::for_entries(2 * self.piece_cache_share() * threads)
        })
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
