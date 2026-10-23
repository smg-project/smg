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
//! 2. normalization is NFC or nothing; NFC is applied only as a check: the
//!    text must already be in NFC (the quick check says so for most prompts
//!    and normalizing settles the rest, so marks-heavy scripts stay on this
//!    path when they are in NFC) and anything else goes back to `tokenizers`;
//! 3. each `Split` pre-tokenizer runs as a [`pattern::Pattern`] over `&str`,
//!    with the same leftmost-first semantics and Unicode tables as the regex
//!    engine `tokenizers` uses, producing byte ranges instead of strings;
//! 4. each piece is mapped to the byte-level alphabet and tokenized by the
//!    tokenizer's own BPE model (so `ignore_merges` and the rest keep their
//!    meaning), through a small per-thread cache from piece bytes to ids.
//!
//! The shape is checked once at load time; a tokenizer that does not fit
//! (another normalizer, a pre-tokenizer or regex construct outside the
//! supported subset, a post-processor that adds tokens, truncation or
//! padding, added tokens with `single_word`/`lstrip`/`rstrip`) simply has no
//! native path. Since no post-processor the path accepts adds tokens, a
//! request for special tokens gets the same ids and is served by the path
//! as well.

use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Serialize;
use serde_json::Value;
use thread_local::ThreadLocal;
use tokenizers::{models::ModelWrapper, Model};
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

mod pattern;
mod unicode;

use pattern::Pattern;

use crate::{
    encode_path::{Declined, Fallback},
    word_cache::{ModelView, TokenizerOf},
};

/// The split the `ByteLevel` pre-tokenizer applies itself when `use_regex`
/// is on (GPT-2 and its descendants).
const BYTE_LEVEL_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Pieces up to this many bytes are cached per thread by their count (see
/// `PIECE_CACHE_CAPACITY`); longer ones, up to `LONG_PIECE_MAX_BYTES`, by
/// the bytes they take (see `LONG_PIECE_BYTES_CAPACITY`).
const CACHED_PIECE_MAX_BYTES: usize = 64;
/// Longest piece cached at all (a clause of some 80 characters in a script
/// without word spaces, 256 of ASCII): a longer piece is rare and tokenized
/// every time.
const LONG_PIECE_MAX_BYTES: usize = 256;
/// Entries a thread's piece cache holds for one encoder before it is cleared,
/// at most; see `PIECE_CACHE_TOTAL_CAPACITY`. This is also the cliff of a
/// repeated document: one with fewer distinct pieces is encoded again at the
/// cost of its lookups, one with more at first-pass cost on every pass, since
/// the clear takes the pieces that recur with the rest (the crate bench
/// `encode_volume` shows both, and that the cost of text no pass has seen
/// does not depend on how much a thread encoded before).
const PIECE_CACHE_CAPACITY: usize = 16_384;
/// Entries all threads together hold for one encoder: a thread's own cap is
/// this divided by the number of threads that have encoded with the encoder
/// (and never more than `PIECE_CACHE_CAPACITY`), so the piece caches of one
/// tokenizer stay under about 40 MB however many runtime and blocking threads
/// encode with it, and they are freed with the encoder.
const PIECE_CACHE_TOTAL_CAPACITY: usize = 262_144;
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
/// text between added tokens, or a chunk of one cut at a stable boundary);
/// shorter inputs would spend more on the hand-offs than they save.
const PARALLEL_MIN_BYTES: usize = 64 << 10;
/// The size a long text unit is cut into for the pool, at the first stable
/// boundary past each multiple of it.
const PARALLEL_CHUNK_BYTES: usize = 32 << 10;
/// Text on each side of a candidate cut that every stage's pattern is run
/// over to confirm a piece starts there (widened to the spaces around it).
const CUT_CHECK_WINDOW: usize = 512;
/// Candidates one cut search may reject before it gives up: the byte scan for
/// a candidate is cheap, the window check on each one is not, so a text whose
/// candidates keep failing the check costs at most this many checks per cut.
const MAX_REJECTED_CUT_CANDIDATES: usize = 64;

/// Piece bytes to the ids they tokenize to.
type PieceIds = FxHashMap<Box<[u8]>, Box<[u32]>>;

/// One thread's cache of the pieces it tokenized with one encoder.
#[derive(Default)]
struct PieceCache {
    /// Pieces of up to `CACHED_PIECE_MAX_BYTES`, bounded by their count.
    short: PieceIds,
    /// Longer pieces, up to `LONG_PIECE_MAX_BYTES`, bounded by their bytes.
    long: PieceIds,
    /// What `long` holds: each entry's piece, ids and slot.
    long_bytes: usize,
}

impl PieceCache {
    /// The ids cached for `piece`, if any.
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

/// Whether `text` is already in NFC. The quick check answers for most text;
/// where it cannot (marks that might compose or reorder: Indic vowel signs
/// and nuktas, an acute after a letter, Hangul jamo, ...) the text is
/// normalized and compared, which costs far less than the encode it keeps on
/// the native path.
fn is_nfc(text: &str) -> bool {
    match is_nfc_quick(text.chars()) {
        IsNormalized::Yes => true,
        IsNormalized::No => false,
        IsNormalized::Maybe => text.nfc().eq(text.chars()),
    }
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
/// matches them.
pub(crate) struct AddedTokens {
    automaton: AhoCorasick,
    ids: Vec<u32>,
}

impl AddedTokens {
    /// The matcher over `tokens`, none for no tokens.
    pub(crate) fn build(tokens: &[(String, u32)]) -> Result<Option<Self>, Declined> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens.iter().map(|(content, _)| content.as_bytes()))
            .map_err(|e| Declined::from(format!("the added tokens form no automaton: {e}")))?;
        Ok(Some(Self {
            automaton,
            ids: tokens.iter().map(|(_, id)| *id).collect(),
        }))
    }

    /// The byte ranges of the added tokens in `text`, in order.
    pub(crate) fn ranges<'a>(
        &'a self,
        text: &'a str,
    ) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
        self.automaton
            .find_iter(text)
            .map(|found| found.start()..found.end())
    }

    /// Calls `f` with each segment in order: an added token's id, or a text
    /// span between added tokens.
    fn split<'t>(&self, text: &'t str, mut f: impl FnMut(Segment<'t>)) {
        let mut cursor = 0;
        for found in self.automaton.find_iter(text) {
            if cursor < found.start() {
                f(Segment::Text(&text[cursor..found.start()]));
            }
            f(Segment::Added(self.ids[found.pattern().as_usize()]));
            cursor = found.end();
        }
        if cursor < text.len() {
            f(Segment::Text(&text[cursor..]));
        }
    }
}

enum Segment<'t> {
    Text(&'t str),
    Added(u32),
}

/// The native encode path of one tokenizer; see the module documentation.
pub(crate) struct NativeEncoder {
    normalizer: Normalizer,
    /// Added tokens matched on the raw text (`normalized: false`).
    raw_added: Option<AddedTokens>,
    /// Added tokens matched after normalization (`normalized: true`).
    normalized_added: Option<AddedTokens>,
    stages: Vec<Stage>,
    alphabet: [char; 256],
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
            .field("stages", &self.stages.len())
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
    /// outside what the module documentation lists.
    pub(crate) fn from_tokenizer<M: Model + ModelView>(
        tokenizer: &TokenizerOf<M>,
    ) -> Result<Self, Declined> {
        if tokenizer.get_truncation().is_some() {
            return Err(Declined::from("truncation is configured"));
        }
        if tokenizer.get_padding().is_some() {
            return Err(Declined::from("padding is configured"));
        }
        if !matches!(tokenizer.get_model().wrapper(), ModelWrapper::BPE(_)) {
            return Err(Declined::from("the model is not BPE"));
        }
        let normalizer =
            Self::normalizer_from(&serialized("normalizer", tokenizer.get_normalizer())?)?;
        let stages =
            Self::stages_from(&serialized("pre-tokenizer", tokenizer.get_pre_tokenizer())?)?;
        Self::post_processor_check(&serialized(
            "post-processor",
            tokenizer.get_post_processor(),
        )?)?;
        let (raw, normalized) = Self::added_tokens_of(tokenizer, normalizer)?;
        Ok(Self {
            normalizer,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            stages,
            alphabet: bytes_char(),
            piece_cache: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
        })
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

    /// Only a post-processor that leaves the ids alone passes: none, or
    /// `ByteLevel`, which reshapes offsets.
    fn post_processor_check(post_processor: &Value) -> Result<(), Declined> {
        if post_processor.is_null() {
            return Ok(());
        }
        match type_of("post-processor", post_processor)? {
            "ByteLevel" => Ok(()),
            other => Err(Declined::from(format!(
                "post-processor {other} (only ByteLevel or none)"
            ))),
        }
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
                    Normalizer::Nfc => token.content.nfc().collect(),
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

    /// The `Split` stages of a pre-tokenizer that is zero or more `Split`s
    /// followed by one `ByteLevel` without prefix space.
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
    /// the pattern engine takes, `Isolated`, or `Removed` inverted.
    fn split_stage(split: &Value) -> Result<Stage, Declined> {
        match type_of("pre-tokenizer stage", split)? {
            "Split" => {}
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

    /// The ids of `text` (the same whether special tokens were requested or
    /// not: no post-processor the path accepts adds any), or why this text
    /// goes through `tokenizers` instead, for the caller to count: not in
    /// NFC under an NFC normalizer, found before any work, or a piece the
    /// model declines, found after the pieces before it were tokenized, so
    /// that the error surfaces from the `tokenizers` encode. A long input is
    /// encoded on the thread pool, unit by unit (see [`Self::units`]), with
    /// the same code per unit.
    pub(crate) fn encode(&self, model: &ModelWrapper, text: &str) -> Result<Vec<u32>, Fallback> {
        self.encode_with(model, text, text.len() >= PARALLEL_MIN_BYTES)
    }

    /// The ids of `text` with no added token cut out of it: every spelling
    /// in it, a special token's included, goes through the stages and the
    /// model as text. The plain segments of a prompt rendered by segments
    /// are encoded this way (see `encoders::marks`); otherwise as
    /// [`Self::encode`].
    pub(crate) fn encode_plain(
        &self,
        model: &ModelWrapper,
        text: &str,
    ) -> Result<Vec<u32>, Fallback> {
        self.encode_units(model, text, text.len() >= PARALLEL_MIN_BYTES, false)
    }

    /// [`Self::encode`] with the thread pool used or not; the ids are the
    /// same either way (the units are encoded by the same code in the same
    /// order).
    pub(crate) fn encode_with(
        &self,
        model: &ModelWrapper,
        text: &str,
        parallel: bool,
    ) -> Result<Vec<u32>, Fallback> {
        self.encode_units(model, text, parallel, true)
    }

    /// The encode behind [`Self::encode_with`] and [`Self::encode_plain`]:
    /// `added` says whether the added tokens are cut out of the text first.
    fn encode_units(
        &self,
        model: &ModelWrapper,
        text: &str,
        parallel: bool,
        added: bool,
    ) -> Result<Vec<u32>, Fallback> {
        if self.normalizer == Normalizer::Nfc && !is_nfc(text) {
            return Err(Fallback::NotNfc);
        }
        let units = self.units(text, parallel, added);
        if parallel && units.len() >= 2 {
            let encoded: Result<Vec<Vec<u32>>, Fallback> = units
                .par_iter()
                .map(|unit| {
                    let mut ids = Vec::new();
                    let mut mapped = String::new();
                    self.encode_unit(model, unit, &mut mapped, &mut ids)
                        .map(|()| ids)
                })
                .collect();
            return encoded.map(|parts| parts.concat());
        }
        let mut ids = Vec::with_capacity(text.len() / 3 + 8);
        let mut mapped = String::new();
        for unit in &units {
            self.encode_unit(model, unit, &mut mapped, &mut ids)?;
        }
        Ok(ids)
    }

    /// The input as the units its encode is made of: the added tokens
    /// matched on the raw text, then the ones matched after normalization on
    /// the texts between them (the text is in its normal form here, see
    /// [`Self::encode_units`]), and the texts between all of them; for the
    /// thread pool a long text is cut further at stable boundaries
    /// ([`Self::stable_cut`]). Both kinds of added token come out before any
    /// cut: a cut inside one would turn it into pieces. With `added` off no
    /// token is cut out: the whole text is text.
    fn units<'t>(&self, text: &'t str, chunked: bool, added: bool) -> Vec<Segment<'t>> {
        let mut units = Vec::new();
        let push_text = |text: &'t str, units: &mut Vec<Segment<'t>>| {
            let mut rest = text;
            while chunked && rest.len() > 2 * PARALLEL_CHUNK_BYTES {
                let Some(cut) = self.stable_cut(rest, PARALLEL_CHUNK_BYTES) else {
                    break;
                };
                units.push(Segment::Text(&rest[..cut]));
                rest = &rest[cut..];
            }
            units.push(Segment::Text(rest));
        };
        let push_normalized =
            |text: &'t str, units: &mut Vec<Segment<'t>>| match &self.normalized_added {
                Some(added) => added.split(text, |segment| match segment {
                    Segment::Added(id) => units.push(Segment::Added(id)),
                    Segment::Text(text) => push_text(text, units),
                }),
                None => push_text(text, units),
            };
        if !added {
            push_text(text, &mut units);
            return units;
        }
        match &self.raw_added {
            Some(added) => added.split(text, |segment| match segment {
                Segment::Added(id) => units.push(Segment::Added(id)),
                Segment::Text(text) => push_normalized(text, &mut units),
            }),
            None => push_normalized(text, &mut units),
        }
        units
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
    /// encode).
    fn encode_unit(
        &self,
        model: &ModelWrapper,
        unit: &Segment<'_>,
        mapped: &mut String,
        ids: &mut Vec<u32>,
    ) -> Result<(), Fallback> {
        let mut fallback = None;
        let mut tokenize = |segment: &str, ids: &mut Vec<u32>| {
            self.pieces(0, segment, &mut |piece| {
                if let Err(declined) = self.tokenize_piece(model, piece, mapped, ids) {
                    fallback = Some(declined);
                }
            });
        };
        match *unit {
            Segment::Added(id) => ids.push(id),
            Segment::Text(text) => tokenize(text, ids),
        }
        fallback.map_or(Ok(()), Err)
    }

    /// Applies the stages from `depth` on to `text`, calling `f` with each
    /// final piece.
    fn pieces(&self, depth: usize, text: &str, f: &mut dyn FnMut(&str)) {
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

    /// Appends the ids of one piece, from the thread's cache or from the
    /// model over the piece's byte-level alphabet form;
    /// [`Fallback::PieceDeclined`] when the model declines the piece.
    fn tokenize_piece(
        &self,
        model: &ModelWrapper,
        piece: &str,
        mapped: &mut String,
        ids: &mut Vec<u32>,
    ) -> Result<(), Fallback> {
        let bytes = piece.as_bytes();
        let cacheable = bytes.len() <= LONG_PIECE_MAX_BYTES;
        let cache = self.piece_cache.get_or(|| {
            self.threads.fetch_add(1, Ordering::Relaxed);
            RefCell::default()
        });
        if cacheable {
            if let Some(found) = cache.borrow().get(bytes) {
                ids.extend_from_slice(found);
                return Ok(());
            }
        }
        mapped.clear();
        mapped.extend(bytes.iter().map(|&b| self.alphabet[usize::from(b)]));
        let start = ids.len();
        match model.tokenize(mapped) {
            Ok(tokens) => ids.extend(tokens.iter().map(|token| token.id)),
            Err(_) => return Err(Fallback::PieceDeclined),
        }
        if bytes.len() <= CACHED_PIECE_MAX_BYTES {
            let mut pieces = cache.borrow_mut();
            if pieces.short.len() >= self.piece_cache_share() {
                pieces.short.clear();
            }
            pieces.short.insert(bytes.into(), ids[start..].into());
        } else if cacheable {
            let entry = LONG_PIECE_ENTRY_BYTES + bytes.len() + size_of_val(&ids[start..]);
            let share = self.long_piece_bytes_share();
            if entry <= share {
                let mut pieces = cache.borrow_mut();
                if pieces.long_bytes + entry > share {
                    pieces.long.clear();
                    pieces.long_bytes = 0;
                }
                pieces.long_bytes += entry;
                pieces.long.insert(bytes.into(), ids[start..].into());
            }
        }
        Ok(())
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
