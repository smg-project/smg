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
struct AddedTokens {
    automaton: AhoCorasick,
    ids: Vec<u32>,
}

impl AddedTokens {
    /// The matcher over `tokens`, none for no tokens.
    fn build(tokens: &[(String, u32)]) -> Result<Option<Self>, Declined> {
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

    /// [`Self::encode`] with the thread pool used or not; the ids are the
    /// same either way (the units are encoded by the same code in the same
    /// order).
    pub(crate) fn encode_with(
        &self,
        model: &ModelWrapper,
        text: &str,
        parallel: bool,
    ) -> Result<Vec<u32>, Fallback> {
        if self.normalizer == Normalizer::Nfc && !is_nfc(text) {
            return Err(Fallback::NotNfc);
        }
        let units = self.units(text, parallel);
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
    /// [`Self::encode_with`]), and the texts between all of them; for the
    /// thread pool a long text is cut further at stable boundaries
    /// ([`Self::stable_cut`]). Both kinds of added token come out before any
    /// cut: a cut inside one would turn it into pieces.
    fn units<'t>(&self, text: &'t str, chunked: bool) -> Vec<Segment<'t>> {
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
mod tests {
    use tokenizers::{
        decoders::byte_level::ByteLevel as ByteLevelDecoder,
        models::{
            bpe::{Merges, Vocab, BPE},
            wordlevel::WordLevel,
        },
        normalizers::{
            unicode::{NFC, NFKC},
            NormalizerWrapper,
        },
        pre_tokenizers::{
            byte_level::ByteLevel as ByteLevelPreTokenizer,
            sequence::Sequence,
            split::{Split, SplitPattern},
            PreTokenizerWrapper,
        },
        processors::{byte_level::ByteLevel as ByteLevelProcessor, template::TemplateProcessing},
        AddedToken, SplitDelimiterBehavior, Tokenizer as HfTokenizer,
    };

    use super::*;
    use crate::{
        encode_path::{encode_fallbacks, EncodeFallbacks, EncodePath},
        huggingface::HuggingFaceTokenizer,
    };

    const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const DIGITS: &str = r"\p{N}{1,3}";
    const CJK: &str = "[一-龥぀-ゟ゠-ヿ]+";
    const DEEPSEEK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    const CASED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    const POOL: &[char] = &[
        'a',
        'b',
        'e',
        'd',
        'l',
        'm',
        'r',
        's',
        't',
        'v',
        'A',
        'E',
        'L',
        'S',
        'T',
        'Z',
        'ſ',
        'ß',
        'é',
        'Ö',
        'ǅ',
        'ʰ',
        'ª',
        '0',
        '1',
        '9',
        '٣',
        '๒',
        '²',
        '½',
        ' ',
        '\t',
        '\n',
        '\r',
        '\u{0B}',
        '\u{85}',
        '\u{A0}',
        '\u{2003}',
        '\u{3000}',
        '\'',
        '"',
        ',',
        '.',
        '-',
        '/',
        '_',
        '~',
        '[',
        ']',
        '\\',
        '^',
        '`',
        '{',
        '|',
        '}',
        '<',
        '>',
        '!',
        '(',
        ')',
        '€',
        '→',
        '\u{301}',
        '\u{93E}',
        '\u{E31}',
        '\u{E48}',
        '\u{9BE}',
        'ก',
        'ข',
        'ा',
        'क',
        'ب',
        'я',
        'Я',
        'α',
        'Ω',
        '中',
        '文',
        '龥',
        'あ',
        'ん',
        'ア',
        'ー',
        '한',
        '😀',
        '👍',
        '\u{1F3FD}',
        '\u{200D}',
        '\u{FE0F}',
        '\u{FFFD}',
        '\u{0}',
        '\u{7F}',
    ];

    fn pseudo_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Random text from the pool, with added tokens dropped in now and then
    /// (whole, and cut so only a prefix of one appears).
    fn random_text(state: &mut u64, len: usize, added: &[&str]) -> String {
        let mut text = String::new();
        for _ in 0..len {
            let roll = pseudo_random(state);
            if !added.is_empty() && roll.is_multiple_of(11) {
                let token = added[(roll / 11) as usize % added.len()];
                if roll.is_multiple_of(3) {
                    let half = token
                        .char_indices()
                        .nth(token.chars().count() / 2)
                        .map_or(token.len(), |(i, _)| i);
                    text.push_str(&token[..half]);
                } else {
                    text.push_str(token);
                }
            } else {
                text.push(POOL[(roll % POOL.len() as u64) as usize]);
            }
        }
        text
    }

    /// A byte-level BPE over the whole alphabet with merges for the pairs
    /// that occur in a corpus of pool text, so pieces really do merge.
    fn model(state: &mut u64, ignore_merges: bool) -> BPE {
        let alphabet: Vec<char> = {
            let table = bytes_char();
            let mut chars: Vec<char> = table.to_vec();
            chars.sort_unstable();
            chars
        };
        let mut vocab = Vocab::default();
        for ch in &alphabet {
            let next = vocab.len() as u32;
            vocab.insert(ch.to_string(), next);
        }
        let mut merges = Merges::new();
        let table = bytes_char();
        let corpus = random_text(state, 4000, &[]);
        let mapped: String = corpus.bytes().map(|b| table[usize::from(b)]).collect();
        let chars: Vec<char> = mapped.chars().collect();
        let mut seen = std::collections::HashSet::new();
        for pair in chars.windows(2) {
            let (a, b) = (pair[0].to_string(), pair[1].to_string());
            if a == " " || b == " " || !seen.insert((a.clone(), b.clone())) {
                continue;
            }
            let merged = format!("{a}{b}");
            if vocab.contains_key(&merged) {
                continue;
            }
            let next = vocab.len() as u32;
            vocab.insert(merged, next);
            merges.push((a, b));
            if merges.len() >= 600 {
                break;
            }
        }
        // A few longer merges on top of the pairs.
        for (a, b) in [
            ("Ġt", "h"),
            ("Ġth", "e"),
            ("ĠT", "he"),
            ("Ã", "©"),
            ("ð", "Ł"),
        ] {
            if vocab.contains_key(a) && vocab.contains_key(b) {
                let merged = format!("{a}{b}");
                if !vocab.contains_key(&merged) {
                    let next = vocab.len() as u32;
                    vocab.insert(merged, next);
                    merges.push((a.to_string(), b.to_string()));
                }
            }
        }
        BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(ignore_merges)
            .build()
            .expect("bpe")
    }

    fn split(pattern: &str, behavior: SplitDelimiterBehavior, invert: bool) -> PreTokenizerWrapper {
        PreTokenizerWrapper::Split(
            Split::new(SplitPattern::Regex(pattern.to_owned()), behavior, invert).expect("split"),
        )
    }

    fn byte_level(use_regex: bool) -> PreTokenizerWrapper {
        PreTokenizerWrapper::ByteLevel(
            ByteLevelPreTokenizer::default()
                .add_prefix_space(false)
                .use_regex(use_regex),
        )
    }

    struct Shape {
        name: &'static str,
        nfc: bool,
        pre_tokenizers: Vec<PreTokenizerWrapper>,
        ignore_merges: bool,
        special: &'static [&'static str],
        added: &'static [(&'static str, bool)],
    }

    fn shapes() -> Vec<Shape> {
        vec![
            Shape {
                name: "qwen3.5",
                nfc: true,
                pre_tokenizers: vec![
                    split(QWEN35, SplitDelimiterBehavior::Isolated, false),
                    byte_level(false),
                ],
                ignore_merges: false,
                special: &[
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|endoftext|>",
                    "<think>",
                    "</think>",
                ],
                added: &[("<tool_call>", false), ("</tool_call>", false)],
            },
            Shape {
                name: "cl100k-ignore-merges",
                nfc: false,
                pre_tokenizers: vec![
                    split(CL100K, SplitDelimiterBehavior::Isolated, false),
                    byte_level(false),
                ],
                ignore_merges: true,
                special: &["<|endoftext|>", "[gMASK]", "<sop>"],
                added: &[],
            },
            Shape {
                name: "three-splits",
                nfc: false,
                pre_tokenizers: vec![
                    split(DIGITS, SplitDelimiterBehavior::Isolated, false),
                    split(CJK, SplitDelimiterBehavior::Isolated, false),
                    split(DEEPSEEK, SplitDelimiterBehavior::Isolated, false),
                    byte_level(false),
                ],
                ignore_merges: false,
                special: &[
                    "<｜begin▁of▁sentence｜>",
                    "<｜end▁of▁sentence｜>",
                    "<｜User｜>",
                ],
                added: &[("<｜tool▁call▁begin｜>", true), ("<｜fim▁hole｜>", true)],
            },
            Shape {
                name: "inverted-removed",
                nfc: true,
                pre_tokenizers: vec![
                    split(CASED, SplitDelimiterBehavior::Removed, true),
                    byte_level(false),
                ],
                ignore_merges: false,
                special: &["<|im_start|>", "<|im_end|>", "[e~[", "]~b]"],
                added: &[],
            },
            Shape {
                name: "byte-level-regex",
                nfc: false,
                pre_tokenizers: vec![byte_level(true)],
                ignore_merges: false,
                special: &["<|endoftext|>"],
                added: &[],
            },
            // Added tokens with a space inside, matched after normalization
            // (the default of `add_tokens`): a cut at that space would turn
            // the token into pieces.
            Shape {
                name: "normalized-words",
                nfc: false,
                pre_tokenizers: vec![
                    split(CL100K, SplitDelimiterBehavior::Isolated, false),
                    byte_level(false),
                ],
                ignore_merges: false,
                special: &["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                added: &[
                    ("foo bar", true),
                    ("### Instruction", true),
                    ("<tool_call>", false),
                ],
            },
        ]
    }

    fn tokenizer_of(shape: &Shape, state: &mut u64) -> HfTokenizer {
        let mut tokenizer = HfTokenizer::new(model(state, shape.ignore_merges));
        if shape.nfc {
            tokenizer
                .with_normalizer(Some(NormalizerWrapper::NFC(NFC)))
                .expect("normalizer");
        }
        let pre = if shape.pre_tokenizers.len() == 1 {
            shape.pre_tokenizers[0].clone()
        } else {
            PreTokenizerWrapper::Sequence(Sequence::new(shape.pre_tokenizers.clone()))
        };
        tokenizer.with_pre_tokenizer(Some(pre));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        tokenizer.with_post_processor(Some(ByteLevelProcessor::default().trim_offsets(false)));
        tokenizer
            .add_special_tokens(shape.special.iter().map(|s| AddedToken::from(*s, true)))
            .expect("special tokens");
        tokenizer
            .add_tokens(
                shape
                    .added
                    .iter()
                    .map(|(s, normalized)| AddedToken::from(*s, false).normalized(*normalized)),
            )
            .expect("added tokens");
        tokenizer
    }

    #[test]
    fn every_shape_gets_a_native_path() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for shape in shapes() {
            let tokenizer = tokenizer_of(&shape, &mut state);
            let native = NativeEncoder::from_tokenizer(&tokenizer);
            assert!(
                native.is_ok(),
                "{}: {:?}",
                shape.name,
                native.as_ref().err()
            );
            let expected_stages = shape
                .pre_tokenizers
                .iter()
                .filter(|p| matches!(p, PreTokenizerWrapper::Split(_)))
                .count()
                + usize::from(shape.name == "byte-level-regex");
            assert_eq!(
                native.expect("native").stages.len(),
                expected_stages,
                "{}",
                shape.name
            );
        }
    }

    #[test]
    fn shapes_outside_the_path_are_declined() {
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let base = &shapes()[0];
        let declined = |tokenizer: &HfTokenizer| {
            NativeEncoder::from_tokenizer(tokenizer)
                .expect_err("declined")
                .to_string()
        };
        // Prefix space.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(QWEN35, SplitDelimiterBehavior::Isolated, false),
            PreTokenizerWrapper::ByteLevel(ByteLevelPreTokenizer::default().add_prefix_space(true)),
        ]))));
        assert_eq!(
            declined(&tokenizer),
            "the pre-tokenizer does not end in ByteLevel without a prefix space"
        );
        // A normalizer other than NFC.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer
            .with_normalizer(Some(NormalizerWrapper::NFKC(NFKC)))
            .expect("normalizer");
        assert_eq!(declined(&tokenizer), "normalizer NFKC (only NFC or none)");
        // A split behaviour the path does not reproduce.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(QWEN35, SplitDelimiterBehavior::MergedWithNext, false),
            byte_level(false),
        ]))));
        assert_eq!(
            declined(&tokenizer),
            "Split behavior Some(\"MergedWithNext\") with invert Some(false) (only Isolated, or \
             Removed inverted)"
        );
        // An added token with lstrip.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer
            .add_special_tokens([AddedToken::from("<pad>", true).lstrip(true)])
            .expect("special token");
        assert_eq!(
            declined(&tokenizer),
            "added token \"<pad>\" with single_word, lstrip or rstrip, or without content"
        );
        // A regex outside the subset.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(r"\w+|\s+", SplitDelimiterBehavior::Isolated, false),
            byte_level(false),
        ]))));
        assert_eq!(
            declined(&tokenizer),
            r"a split pattern outside the supported regex subset: \w+|\s+"
        );
        // A post-processor that adds tokens, and a model that is not BPE.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<|im_start|> $A")
                .expect("template")
                .special_tokens(vec![("<|im_start|>", 0)])
                .build()
                .expect("post-processor"),
        ));
        assert_eq!(
            declined(&tokenizer),
            "post-processor TemplateProcessing (only ByteLevel or none)"
        );
        let tokenizer = HfTokenizer::new(WordLevel::default());
        assert_eq!(declined(&tokenizer), "the model is not BPE");
        // No pre-tokenizer at all.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(None::<PreTokenizerWrapper>);
        assert_eq!(
            declined(&tokenizer),
            "no pre-tokenizer (the path needs ByteLevel, after any Split stages)"
        );
    }

    #[test]
    fn a_batch_takes_the_direct_path_input_by_input() {
        use crate::traits::{Encoder, Encoding};
        let mut state = 0x5A5A_1234_5678_9ABCu64;
        for shape in shapes() {
            let tokenizer = HuggingFaceTokenizer::from_tokenizer(tokenizer_of(&shape, &mut state));
            let added: Vec<&str> = shape.special.to_vec();
            let mut texts: Vec<String> = (0..8)
                .map(|round| random_text(&mut state, 1 + round * 5, &added))
                .filter(|text| !shape.nfc || text.nfc().eq(text.chars()))
                .collect();
            // Under an NFC shape, one input the direct encoder declines sits
            // in the middle of the batch: it takes the `tokenizers` path on
            // its own, and its neighbours stay on the direct path.
            let declined = "e\u{301}";
            if shape.nfc {
                texts.insert(texts.len() / 2, declined.to_string());
            }
            let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let batch = tokenizer.encode_batch(&inputs, false).expect(shape.name);
            assert_eq!(batch.len(), inputs.len(), "{}", shape.name);
            for (input, encoding) in inputs.iter().zip(&batch) {
                let single = tokenizer.encode(input, false).expect("encode");
                assert_eq!(
                    encoding.token_ids(),
                    single.token_ids(),
                    "{}: {input:?}",
                    shape.name
                );
                if shape.nfc && *input == declined && !tokenizer.uses_v1_encoder() {
                    assert!(
                        matches!(encoding, Encoding::Hf(_)),
                        "{}: the declined input takes the tokenizers path",
                        shape.name
                    );
                } else {
                    assert!(
                        matches!(encoding, Encoding::Plain(_)),
                        "{}: the batch left the direct path for {input:?}",
                        shape.name
                    );
                }
            }
        }
    }

    /// A text long enough for the pool, with spaces, words, digits, CJK and
    /// added tokens, so that stable cuts exist and the units are several.
    fn long_text(state: &mut u64, added: &[&str], bytes: usize) -> String {
        let mut text = String::with_capacity(bytes + 64);
        while text.len() < bytes {
            let words = 1 + (pseudo_random(state) % 12) as usize;
            text.push_str(&random_text(state, words, added));
            text.push(' ');
            text.push_str(
                ["word", "Wort", "parole", "palabra", "slovo"][(pseudo_random(state) % 5) as usize],
            );
            if pseudo_random(state).is_multiple_of(7) {
                text.push_str(" 12345 ");
            }
        }
        text
    }

    #[test]
    fn a_long_text_encodes_the_same_on_the_pool_as_in_one_piece() {
        let mut state = 0x5A5A_1234_5678_9ABCu64;
        for shape in shapes() {
            let tokenizer = tokenizer_of(&shape, &mut state);
            let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
            let added: Vec<&str> = shape
                .special
                .iter()
                .copied()
                .chain(shape.added.iter().map(|(s, _)| *s))
                .collect();
            let mut compared = 0;
            for _ in 0..3 {
                let mut text = long_text(&mut state, &added, 3 * PARALLEL_MIN_BYTES);
                if shape.nfc {
                    text = text.nfc().collect();
                }
                let units = native.units(&text, true).len();
                assert!(
                    units >= 4,
                    "{}: {units} units for {} bytes",
                    shape.name,
                    text.len()
                );
                let (Ok(pooled), Ok(serial)) = (
                    native.encode_with(tokenizer.get_model(), &text, true),
                    native.encode_with(tokenizer.get_model(), &text, false),
                ) else {
                    assert!(
                        shape.nfc && text.nfc().ne(text.chars()),
                        "{}: declined",
                        shape.name
                    );
                    continue;
                };
                compared += 1;
                assert_eq!(pooled, serial, "{}: the pool changed the ids", shape.name);
                let reference = tokenizer
                    .encode(text.as_str(), false)
                    .expect("encode")
                    .get_ids()
                    .to_vec();
                assert_eq!(pooled, reference, "{}: against the reference", shape.name);
            }
            assert!(
                compared >= 2,
                "{}: {compared} long texts compared",
                shape.name
            );
        }
    }

    /// An added token matched after normalization whose inner space is the
    /// first cut candidate past the chunk size: it comes out as a unit of its
    /// own before any cut, so the pooled ids are the one-piece ids.
    #[test]
    fn a_normalized_added_token_across_a_cut_stays_whole() {
        let mut state = 0x0F1E_2D3C_4B5A_6978u64;
        let shape = shapes()
            .into_iter()
            .find(|shape| shape.name == "normalized-words")
            .expect("the shape");
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let foo_bar = tokenizer.token_to_id("foo bar").expect("the added token");
        // No candidate before the token (letters only), the token's space as
        // the first candidate past the chunk size, then text enough for cuts.
        let mut text = "x".repeat(PARALLEL_CHUNK_BYTES + 5);
        text.push_str("foo bar");
        text.push_str(&long_text(&mut state, &[], 3 * PARALLEL_CHUNK_BYTES));
        let units = native.units(&text, true);
        assert!(
            units
                .iter()
                .any(|unit| matches!(unit, Segment::Added(id) if *id == foo_bar)),
            "the added token is a unit of its own ({} units)",
            units.len()
        );
        assert!(units.len() >= 4, "{} units", units.len());
        let pooled = native
            .encode_with(tokenizer.get_model(), &text, true)
            .expect("pooled");
        let serial = native
            .encode_with(tokenizer.get_model(), &text, false)
            .expect("serial");
        let reference = tokenizer
            .encode(text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        assert_eq!(pooled, serial, "the pool changed the ids");
        assert_eq!(pooled, reference, "against the reference");
        assert!(pooled.contains(&foo_bar), "the token's id is in the ids");
    }

    /// A stretch without a cut candidate (a table of numbers: spaces before
    /// digits, never before a letter) longer than two chunks, then prose: the
    /// search goes on past the stretch and the prose is chunked.
    #[test]
    fn chunking_goes_on_past_a_stretch_without_candidates() {
        let mut state = 0x7E57_C0DE_0BAD_F00Du64;
        let shape = shapes()
            .into_iter()
            .find(|shape| shape.name == "cl100k-ignore-merges")
            .expect("the shape");
        let tokenizer = tokenizer_of(&shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let mut text = "0.125, -3.5, 42, ".repeat(PARALLEL_CHUNK_BYTES / 8);
        assert!(text.len() > 2 * PARALLEL_CHUNK_BYTES);
        text.push_str(&long_text(&mut state, &[], 4 * PARALLEL_CHUNK_BYTES));
        let units = native.units(&text, true);
        assert!(
            units.len() > 2,
            "{} units: the prose after the table is chunked",
            units.len()
        );
        let pooled = native
            .encode_with(tokenizer.get_model(), &text, true)
            .expect("pooled");
        let serial = native
            .encode_with(tokenizer.get_model(), &text, false)
            .expect("serial");
        let reference = tokenizer
            .encode(text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        assert_eq!(pooled, serial, "the pool changed the ids");
        assert_eq!(pooled, reference, "against the reference");
    }

    #[test]
    fn stable_cuts_leave_the_pieces_as_they_are() {
        let mut state = 0x0BAD_CAFE_F00D_0001u64;
        for shape in shapes() {
            let tokenizer = tokenizer_of(&shape, &mut state);
            let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
            let mut text = long_text(&mut state, &[], 2 * PARALLEL_MIN_BYTES);
            if shape.nfc {
                text = text.nfc().collect();
            }
            let whole: Vec<String> = {
                let mut out = Vec::new();
                native.pieces(0, &text, &mut |piece| out.push(piece.to_owned()));
                out
            };
            let mut chunked = Vec::new();
            let units = native.units(&text, true);
            assert!(
                units.len() >= 2,
                "{}: no cut in {} bytes",
                shape.name,
                text.len()
            );
            for unit in &units {
                let Segment::Text(chunk) = unit else {
                    panic!("no added tokens in this text");
                };
                native.pieces(0, chunk, &mut |piece| chunked.push(piece.to_owned()));
            }
            assert_eq!(chunked, whole, "{}: a cut split a piece", shape.name);
        }
    }

    #[test]
    fn ids_match_tokenizers_on_random_text_for_every_shape() {
        let mut state = 0x0F0F_1234_ABCD_9876u64;
        for shape in shapes() {
            let tokenizer = tokenizer_of(&shape, &mut state);
            let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
            let added: Vec<&str> = shape
                .special
                .iter()
                .copied()
                .chain(shape.added.iter().map(|(s, _)| *s))
                .collect();
            let mut compared = 0;
            for round in 0..2000 {
                let text = random_text(&mut state, 1 + round % 40, &added);
                let reference = tokenizer
                    .encode(text.as_str(), false)
                    .expect("encode")
                    .get_ids()
                    .to_vec();
                match native.encode(tokenizer.get_model(), &text) {
                    Ok(ids) => {
                        compared += 1;
                        assert_eq!(ids, reference, "{}: {text:?}", shape.name);
                    }
                    Err(fallback) => assert!(
                        fallback == Fallback::NotNfc && shape.nfc && text.nfc().ne(text.chars()),
                        "{}: declined {text:?} as {fallback:?}",
                        shape.name
                    ),
                }
            }
            assert!(
                compared > 1000,
                "{}: only {compared} texts compared",
                shape.name
            );
        }
    }

    #[test]
    fn fixed_texts_match_tokenizers_for_every_shape() {
        let texts = [
            "",
            " ",
            "\n\n",
            "Hello, world!",
            "I'm testing: 12345\nsecond line.",
            "Cafe\u{301} 中文🙂 — naïve",
            "  leading  and trailing  ",
            "<|im_start|>assistant\nHello<|im_end|>",
            "<|im_start|>user\n<tool_call>{\"a\": 1}</tool_call><|im_end|>\n",
            "WE'LL see what IT'S worth; they'd know, I'VE heard, you're, he's, isn'T",
            "x'ſ y'ſ",
            "1234567890 ١٢٣ ๑๒๓๔",
            "日本語のテキスト、한국어 텍스트, 中文文本。",
            "ครับ ผม ชื่อ นาย คำ ไทย",
            "कृपया यह पढ़ें और समझें",
            "একটি নতুন প্রোগ্রাম 16টি",
            "ఒక కొత్త కార్యక్రమం",
            "يَتَلَقَّى مَرْحَبًا",
            "fn main() { println!(\"hi\"); }\n\n\n// done\r\n",
            "a\tb\u{A0}c\u{3000}d\u{2028}e",
            "<｜begin▁of▁sentence｜>hello<｜User｜>q<｜tool▁call▁begin｜>x<｜fim▁hole｜>",
            "[e~[hello]~b] and [e~ half",
        ];
        let mut state = 0x7777_1111_2222_3333u64;
        for shape in shapes() {
            let tokenizer = tokenizer_of(&shape, &mut state);
            let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
            for text in texts {
                let reference = tokenizer
                    .encode(text, false)
                    .expect("encode")
                    .get_ids()
                    .to_vec();
                if let Ok(ids) = native.encode(tokenizer.get_model(), text) {
                    assert_eq!(ids, reference, "{}: {text:?}", shape.name);
                }
            }
        }
    }

    /// The letters-only split of the Qwen2 family, under which a combining
    /// mark (`\p{M}`) starts a new piece; the Qwen3.5-family files carry
    /// [`QWEN35`], whose `[\p{L}\p{M}]+` keeps a letter and its marks together.
    const LETTERS_ONLY: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// A byte-level BPE that learned `letter` and `marks` byte by byte and
    /// then their merge, the way a vocabulary that keeps letters and marks
    /// together holds a cluster: the letter, the marks and the cluster are
    /// tokens, and the cluster is reached only from the other two.
    fn cluster_model(letter: &str, marks: &str) -> BPE {
        let table = bytes_char();
        let mut alphabet: Vec<char> = table.to_vec();
        alphabet.sort_unstable();
        let mut vocab = Vocab::default();
        for ch in alphabet {
            let next = vocab.len() as u32;
            vocab.insert(ch.to_string(), next);
        }
        let mut merges = Merges::new();
        let mut learn = |left: String, right: String| {
            let merged = format!("{left}{right}");
            if !vocab.contains_key(&merged) {
                let next = vocab.len() as u32;
                vocab.insert(merged.clone(), next);
                merges.push((left, right));
            }
            merged
        };
        let mut chain = |word: &str| {
            let mut mapped = word.bytes().map(|b| table[usize::from(b)].to_string());
            let mut token = mapped.next().expect("a non-empty word");
            for right in mapped {
                token = learn(token, right);
            }
            token
        };
        let letter = chain(letter);
        let marks = chain(marks);
        learn(letter, marks);
        BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .expect("bpe")
    }

    /// An NFC byte-level tokenizer over `model` with one split `pattern`.
    fn split_tokenizer(pattern: &str, model: BPE) -> HfTokenizer {
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer
            .with_normalizer(Some(NormalizerWrapper::NFC(NFC)))
            .expect("normalizer");
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(pattern, SplitDelimiterBehavior::Isolated, false),
            byte_level(false),
        ]))));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        tokenizer.with_post_processor(Some(ByteLevelProcessor::default().trim_offsets(false)));
        tokenizer
    }

    /// The pieces the native pre-tokenizer cuts `text` into.
    fn native_pieces(native: &NativeEncoder, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        native.pieces(0, text, &mut |piece| pieces.push(piece.to_owned()));
        pieces
    }

    /// A letter followed by its combining `marks` is one piece under a split
    /// that admits marks, and so one token when the vocabulary has it, and two
    /// pieces and two tokens under the letters-only split: the native path
    /// follows the file's pattern either way and gives `tokenizers`' ids on
    /// the cluster and on `sentence`.
    fn marks_stay_with_their_letter(letter: &str, marks: &str, sentence: &str) {
        let cluster = format!("{letter}{marks}");
        let model = || cluster_model(letter, marks);
        let with_marks = split_tokenizer(QWEN35, model());
        let letters_only = split_tokenizer(LETTERS_ONLY, model());
        let native_with_marks = NativeEncoder::from_tokenizer(&with_marks).expect("native path");
        let native_letters_only =
            NativeEncoder::from_tokenizer(&letters_only).expect("native path");
        assert_eq!(
            native_pieces(&native_with_marks, &cluster),
            std::slice::from_ref(&cluster)
        );
        assert_eq!(
            native_pieces(&native_letters_only, &cluster),
            [letter.to_owned(), marks.to_owned()]
        );
        let mut lengths = Vec::new();
        for (tokenizer, native) in [
            (&with_marks, &native_with_marks),
            (&letters_only, &native_letters_only),
        ] {
            for text in [cluster.as_str(), sentence] {
                let reference = tokenizer
                    .encode(text, false)
                    .expect("encode")
                    .get_ids()
                    .to_vec();
                let ids = native
                    .encode(tokenizer.get_model(), text)
                    .expect("the native path takes NFC text");
                assert_eq!(ids, reference, "{text:?}");
            }
            lengths.push(
                native
                    .encode(tokenizer.get_model(), &cluster)
                    .unwrap()
                    .len(),
            );
        }
        assert_eq!(lengths, [1, 2], "{cluster:?}");
    }

    #[test]
    fn bengali_marks_stay_with_their_letter() {
        marks_stay_with_their_letter("ট", "\u{9BF}", "একটি নতুন প্রোগ্রাম 16টি");
    }

    #[test]
    fn telugu_marks_stay_with_their_letter() {
        marks_stay_with_their_letter("క", "\u{C3E}", "ఒక కొత్త కార్యక్రమం");
    }

    #[test]
    fn thai_marks_stay_with_their_letter() {
        marks_stay_with_their_letter("น", "\u{E47}", "โปรแกรมใหม่หนึ่ง เน็ต");
    }

    #[test]
    fn devanagari_marks_stay_with_their_letter() {
        marks_stay_with_their_letter("क", "\u{948}", "चाय कैसे बनाएँ");
    }

    #[test]
    fn arabic_marks_stay_with_their_letter() {
        marks_stay_with_their_letter("ق", "\u{64E}\u{651}", "يَتَلَقَّى مَرْحَبًا");
    }

    #[test]
    fn text_not_in_nfc_is_declined_under_an_nfc_normalizer() {
        let mut state = 0x5555_6666_7777_8888u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        // "e" + combining acute composes under NFC: not our business.
        assert_eq!(
            native.encode(tokenizer.get_model(), "Cafe\u{301}"),
            Err(Fallback::NotNfc)
        );
        assert!(native.encode(tokenizer.get_model(), "Caf\u{E9}").is_ok());
    }

    #[test]
    fn a_piece_the_model_declines_is_reported_after_the_work() {
        // A vocabulary without the character for byte 0x7F whose `unk_token`
        // is not in it either: `tokenizers` fails such a piece, and so does
        // this path, once the pieces before it have been tokenized.
        let mut state = 0x4444_3333_2222_1111u64;
        let shape = &shapes()[0];
        let mut tokenizer = tokenizer_of(shape, &mut state);
        let table = bytes_char();
        let mut vocab = Vocab::default();
        for ch in table.iter().filter(|&&ch| ch != table[0x7F]) {
            let id = vocab.len() as u32;
            vocab.insert(ch.to_string(), id);
        }
        let mut bpe = BPE::builder()
            .vocab_and_merges(vocab, Merges::new())
            .build()
            .expect("bpe");
        bpe.unk_token = Some("<unk>".to_owned());
        tokenizer.with_model(bpe);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        assert!(native.encode(tokenizer.get_model(), "plain text").is_ok());
        assert_eq!(
            native.encode(tokenizer.get_model(), "plain text\u{7F}"),
            Err(Fallback::PieceDeclined)
        );
        assert!(tokenizer.encode("plain text\u{7F}", false).is_err());
    }

    #[test]
    fn an_input_the_direct_encoder_gives_up_is_counted_and_encoded_by_tokenizers() {
        use crate::traits::Encoder;
        let mut state = 0x8888_9999_AAAA_BBBBu64;
        let shape = &shapes()[0];
        let reference = tokenizer_of(shape, &mut state);
        let ours = HuggingFaceTokenizer::from_tokenizer(reference.clone());
        // The count applies to the direct path: with the v1 feature on, the
        // shape's split pattern is one the pipeline declines today.
        assert_eq!(ours.encode_path(), EncodePath::Native);
        let not_nfc = |stats: &[EncodeFallbacks; 2]| {
            stats
                .iter()
                .find(|s| s.reason == "not_nfc")
                .map_or(0, |s| s.encodes)
        };
        let before = not_nfc(&encode_fallbacks());
        for text in ["Cafe\u{301}", "a\u{323}\u{301}", "\u{1100}\u{1161}"] {
            let ids = ours.encode(text, false).expect("encode");
            assert_eq!(
                ids.token_ids(),
                reference.encode(text, false).expect("encode").get_ids(),
                "{text:?}"
            );
        }
        // Three inputs left the direct path here; other tests may add theirs.
        assert!(not_nfc(&encode_fallbacks()) >= before + 3);
    }

    #[test]
    fn the_piece_cache_returns_the_same_ids() {
        let mut state = 0xABCD_EF01_2345_6789u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let text = "the same words the same words, again and again and again";
        let first = native.encode(tokenizer.get_model(), text).expect("ids");
        let second = native.encode(tokenizer.get_model(), text).expect("ids");
        assert_eq!(first, second);
        assert_eq!(
            first,
            tokenizer.encode(text, false).expect("encode").get_ids()
        );
    }

    #[test]
    fn a_split_pattern_that_can_match_empty_declines_the_tokenizer() {
        let mut state = 0x2222_3333_4444_5555u64;
        let base = &shapes()[0];
        let mut tokenizer = tokenizer_of(base, &mut state);
        // `\p{N}*` matches empty between letters, and the engine splits there.
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(r"\p{N}*|\s+", SplitDelimiterBehavior::Isolated, false),
            byte_level(false),
        ]))));
        if let Ok(native) = NativeEncoder::from_tokenizer(&tokenizer) {
            for text in ["ab1", "the same words 12"] {
                let reference = tokenizer.encode(text, false).expect("encode");
                assert_eq!(
                    native.encode(tokenizer.get_model(), text),
                    Ok(reference.get_ids().to_vec()),
                    "{text:?}"
                );
            }
        }
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_err());
    }

    #[test]
    fn piece_caches_belong_to_their_encoder() {
        // Each thread that encodes gets its own cache, held by the encoder
        // rather than by the thread, so a removed tokenizer frees all of them.
        let mut state = 0x1357_9BDF_2468_ACE0u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let mut native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let model = tokenizer.get_model();
        let threads = 4;
        let barrier = std::sync::Barrier::new(threads);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (native, barrier) = (&native, &barrier);
                scope.spawn(move || {
                    native.encode(model, "the same words again").expect("ids");
                    barrier.wait();
                });
            }
        });
        let caches: Vec<usize> = native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().short.len())
            .collect();
        assert_eq!(caches.len(), threads, "{caches:?}");
        assert!(caches.iter().all(|&entries| entries > 0), "{caches:?}");
        drop(native);
    }

    /// Distinct words for distinct `i`: distinct pieces.
    fn word(mut i: u32) -> String {
        let mut word = String::new();
        loop {
            word.push(char::from(b'a' + (i % 26) as u8));
            i /= 26;
            if i == 0 {
                return word;
            }
        }
    }

    #[test]
    fn piece_caches_stay_within_their_bound_across_threads() {
        let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let mut native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let model = tokenizer.get_model();
        let threads = 24u32;
        let barrier = std::sync::Barrier::new(threads as usize);
        std::thread::scope(|scope| {
            for t in 0..threads {
                let (native, barrier) = (&native, &barrier);
                scope.spawn(move || {
                    native.encode(model, "warm").expect("ids");
                    barrier.wait();
                    // More distinct pieces than one thread's share, and
                    // together far more than the total.
                    let text: String = (0..15_000)
                        .map(|i| format!(" {}", word(i * threads + t)))
                        .collect();
                    // On this thread, not the pool: the test is about the
                    // caches the encoding threads hold.
                    native.encode_with(model, &text, false).expect("ids");
                });
            }
        });
        assert_eq!(native.threads.load(Ordering::Relaxed), threads as usize);
        let share = native.piece_cache_share();
        assert!(share < PIECE_CACHE_CAPACITY, "{share}");
        let long_share = native.long_piece_bytes_share();
        assert!(long_share < LONG_PIECE_BYTES_CAPACITY, "{long_share}");
        let caches: Vec<usize> = native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().short.len())
            .collect();
        assert_eq!(caches.len(), threads as usize, "{caches:?}");
        assert!(
            caches.iter().all(|&entries| entries <= share),
            "share {share}, caches {caches:?}"
        );
        assert!(
            caches.iter().sum::<usize>() <= PIECE_CACHE_TOTAL_CAPACITY,
            "{caches:?}"
        );
    }

    /// `len` letters, distinct for distinct `i`: one piece of `len` bytes
    /// under the byte-level split (letters only, no space). The padding is
    /// a letter `word` never uses, so no two differ only in where it starts.
    fn letters(len: usize, i: u32) -> String {
        let mut text = word(i);
        while text.len() < len {
            text.push('X');
        }
        text
    }

    /// What the long tier accounts for one entry.
    fn long_entry_bytes(piece: &[u8], ids: &[u32]) -> usize {
        LONG_PIECE_ENTRY_BYTES + piece.len() + size_of_val(ids)
    }

    #[test]
    fn long_pieces_are_cached_by_their_bytes() {
        let mut state = 0x0BAD_F00D_1234_5678u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let model = tokenizer.get_model();
        let short = letters(CACHED_PIECE_MAX_BYTES, 1);
        let long = letters(CACHED_PIECE_MAX_BYTES + 1, 2);
        let longest = letters(LONG_PIECE_MAX_BYTES, 3);
        let beyond = letters(LONG_PIECE_MAX_BYTES + 1, 4);
        // Newlines split the words without joining a space to them.
        let text = format!("{short}\n{long}\n{longest}\n{beyond}");
        let reference = tokenizer.encode(text.as_str(), false).expect("encode");
        let first = native.encode(model, &text).expect("ids");
        assert_eq!(first, reference.get_ids());
        {
            let cache = native
                .piece_cache
                .get()
                .expect("this thread's cache")
                .borrow();
            assert!(cache.short.contains_key(short.as_bytes()));
            assert!(!cache.long.contains_key(short.as_bytes()));
            assert!(cache.long.contains_key(long.as_bytes()));
            assert!(cache.long.contains_key(longest.as_bytes()));
            assert!(!cache.long.contains_key(beyond.as_bytes()));
            assert!(!cache.short.contains_key(beyond.as_bytes()));
            assert_eq!(cache.long.len(), 2);
            let accounted: usize = cache
                .long
                .iter()
                .map(|(piece, ids)| long_entry_bytes(piece, ids))
                .sum();
            assert_eq!(cache.long_bytes, accounted);
        }
        // The second encode is answered from the cache: the same ids, and
        // the cache as it was.
        let second = native.encode(model, &text).expect("ids");
        assert_eq!(second, first);
        let cache = native
            .piece_cache
            .get()
            .expect("this thread's cache")
            .borrow();
        assert_eq!(cache.long.len(), 2);
        assert_eq!(
            cache.long_bytes,
            long_entry_bytes(long.as_bytes(), &cache.long[long.as_bytes()])
                + long_entry_bytes(longest.as_bytes(), &cache.long[longest.as_bytes()])
        );
    }

    #[test]
    fn long_pieces_stay_within_their_byte_share() {
        let mut state = 0x7A7A_5B5B_3C3C_1D1Du64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let model = tokenizer.get_model();
        // Distinct long pieces worth more than one thread's share: each
        // entry takes its 80 bytes, its ids (at least 27, the model merges
        // pairs) and its slot, so 16,000 of them are over 3 MB.
        let words = 16_000u32;
        let text: String = (0..words)
            .map(|i| format!("{}\n", letters(80, i)))
            .collect();
        native.encode(model, &text).expect("ids");
        let share = native.long_piece_bytes_share();
        let cache = native
            .piece_cache
            .get()
            .expect("this thread's cache")
            .borrow();
        let accounted: usize = cache
            .long
            .iter()
            .map(|(piece, ids)| long_entry_bytes(piece, ids))
            .sum();
        assert_eq!(cache.long_bytes, accounted);
        assert!(cache.long_bytes <= share, "{} > {share}", cache.long_bytes);
        assert!(cache.long.len() < words as usize, "{}", cache.long.len());
        assert!(!cache.long.is_empty());
    }

    #[test]
    fn text_in_nfc_takes_the_native_path_when_the_quick_check_is_unsure() {
        let mut state = 0x6666_7777_8888_9999u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        // Marks the quick check cannot vouch for, on text that is already in
        // NFC: a Bengali vowel sign, a Devanagari nukta (its composition is
        // excluded), an acute on a letter with no precomposed form, a lone
        // Hangul trailing consonant, a prompt in Thai and Hindi.
        for text in [
            "কা",
            "क़ पढ़ें",
            "x\u{301}",
            "\u{11A8}",
            "<|im_start|>user\nครับ ผม ชื่อ นาย คำ ไทย, कृपया यह पढ़ें<|im_end|>\n",
        ] {
            assert_ne!(
                is_nfc_quick(text.chars()),
                IsNormalized::Yes,
                "{text:?}: the quick check should be unsure here"
            );
            let ids = native
                .encode(tokenizer.get_model(), text)
                .unwrap_or_else(|fallback| {
                    panic!("{text:?} is in NFC and must take the native path, not {fallback:?}")
                });
            assert_eq!(
                ids,
                tokenizer.encode(text, false).expect("encode").get_ids(),
                "{text:?}"
            );
        }
        // Not in NFC: composition or reordering would change the text, so
        // these stay with `tokenizers`.
        for text in [
            "Cafe\u{301}",
            "a\u{323}\u{301}",
            "\u{1100}\u{1161}",
            "\u{AC00}\u{11A8}",
            "ก\u{E48}\u{E38}",
        ] {
            assert_eq!(
                native.encode(tokenizer.get_model(), text),
                Err(Fallback::NotNfc),
                "{text:?}"
            );
        }
    }
}
