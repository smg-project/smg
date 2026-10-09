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
//! 2. normalization is NFC or nothing; NFC is applied only as a check, the
//!    text must already be in NFC (the quick check says so for practically
//!    every prompt) and anything else goes back to `tokenizers`;
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
//! native path. `add_special_tokens = true` also stays with `tokenizers`.

use std::{
    cell::RefCell,
    sync::atomic::{AtomicU64, Ordering},
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value;
use tokenizers::{models::ModelWrapper, Model, Tokenizer as HfTokenizer};
use unicode_normalization::{
    char::canonical_combining_class, is_nfc_quick, IsNormalized, UnicodeNormalization,
};

mod pattern;
mod unicode;

use pattern::Pattern;

/// The split the `ByteLevel` pre-tokenizer applies itself when `use_regex`
/// is on (GPT-2 and its descendants).
const BYTE_LEVEL_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Pieces up to this many bytes are cached per thread; longer ones are rare
/// and tokenized every time.
const CACHED_PIECE_MAX_BYTES: usize = 64;
/// Entries per encoder in a thread's piece cache before it is cleared.
const PIECE_CACHE_CAPACITY: usize = 16_384;
/// Texts at least this long (between added tokens) are encoded across the
/// thread pool; shorter ones are not worth the hand-off.
const PARALLEL_MIN_BYTES: usize = 128 * 1024;
/// The least a parallel chunk gets, so threads do not fight over scraps.
const PARALLEL_CHUNK_MIN_BYTES: usize = 64 * 1024;

static NEXT_ENCODER_ID: AtomicU64 = AtomicU64::new(1);

/// Piece bytes to the ids they tokenize to.
type PieceIds = FxHashMap<Box<[u8]>, Box<[u32]>>;

thread_local! {
    /// Each encoder's piece cache on this thread, by encoder id.
    static PIECE_CACHE: RefCell<FxHashMap<u64, PieceIds>> = RefCell::new(FxHashMap::default());
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

/// One set of added tokens, matched leftmost-longest as `AddedVocabulary`
/// matches them.
struct AddedTokens {
    automaton: AhoCorasick,
    ids: Vec<u32>,
}

impl AddedTokens {
    fn build(tokens: &[(String, u32)]) -> Option<Option<Self>> {
        if tokens.is_empty() {
            return Some(None);
        }
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens.iter().map(|(content, _)| content.as_bytes()))
            .ok()?;
        Some(Some(Self {
            automaton,
            ids: tokens.iter().map(|(_, id)| *id).collect(),
        }))
    }

    /// Calls `f` with each segment in order: an added token's id, or a text
    /// span between added tokens.
    fn split<'t>(&self, text: &'t str, f: &mut dyn FnMut(Segment<'t>)) {
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

/// One unit of the parallel path's tokenization: a first-stage piece of the
/// text, by byte range, or an added token's id.
#[derive(Clone, Copy)]
enum Item {
    Piece(usize, usize),
    Added(u32),
}

/// The native encode path of one tokenizer; see the module documentation.
pub(crate) struct NativeEncoder {
    id: u64,
    normalizer: Normalizer,
    /// Added tokens matched on the raw text (`normalized: false`).
    raw_added: Option<AddedTokens>,
    /// Added tokens matched after normalization (`normalized: true`).
    normalized_added: Option<AddedTokens>,
    stages: Vec<Stage>,
    alphabet: [char; 256],
}

impl std::fmt::Debug for NativeEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEncoder")
            .field("normalizer", &self.normalizer)
            .field("stages", &self.stages.len())
            .finish()
    }
}

impl NativeEncoder {
    /// `None` unless the tokenizer has the shape this path reproduces.
    pub(crate) fn from_tokenizer(tokenizer: &HfTokenizer) -> Option<Self> {
        if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
            return None;
        }
        if !matches!(tokenizer.get_model(), ModelWrapper::BPE(_)) {
            return None;
        }
        let normalizer = match serde_json::to_value(tokenizer.get_normalizer()).ok()? {
            Value::Null => Normalizer::None,
            value => match value.get("type").and_then(Value::as_str)? {
                "NFC" => Normalizer::Nfc,
                "Sequence" => {
                    let inner = value.get("normalizers").and_then(Value::as_array)?;
                    if !inner.is_empty() {
                        return None;
                    }
                    Normalizer::None
                }
                _ => return None,
            },
        };
        let stages = Self::stages(&serde_json::to_value(tokenizer.get_pre_tokenizer()).ok()?)?;
        match serde_json::to_value(tokenizer.get_post_processor()).ok()? {
            Value::Null => {}
            // Only reshapes offsets; the ids are untouched.
            value if value.get("type").and_then(Value::as_str) == Some("ByteLevel") => {}
            _ => return None,
        }

        let added = tokenizer.get_added_vocabulary();
        if added.get_encode_special_tokens() {
            return None;
        }
        let mut raw = Vec::new();
        let mut normalized = Vec::new();
        for (&id, token) in added.get_added_tokens_decoder() {
            if token.single_word || token.lstrip || token.rstrip || token.content.is_empty() {
                return None;
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
        // A stable order keeps leftmost-longest ties deterministic.
        raw.sort();
        normalized.sort();
        Some(Self {
            id: NEXT_ENCODER_ID.fetch_add(1, Ordering::Relaxed),
            normalizer,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            stages,
            alphabet: bytes_char(),
        })
    }

    /// The `Split` stages of a pre-tokenizer that is zero or more `Split`s
    /// followed by one `ByteLevel` without prefix space.
    fn stages(pre_tokenizer: &Value) -> Option<Vec<Stage>> {
        let items: Vec<&Value> = match pre_tokenizer.get("type").and_then(Value::as_str)? {
            "Sequence" => pre_tokenizer
                .get("pretokenizers")
                .and_then(Value::as_array)?
                .iter()
                .collect(),
            _ => vec![pre_tokenizer],
        };
        let (byte_level, splits) = items.split_last()?;
        if byte_level.get("type").and_then(Value::as_str) != Some("ByteLevel")
            || byte_level.get("add_prefix_space").and_then(Value::as_bool) != Some(false)
        {
            return None;
        }
        let mut stages = Vec::with_capacity(splits.len() + 1);
        for split in splits {
            if split.get("type").and_then(Value::as_str) != Some("Split") {
                return None;
            }
            let pattern = split.get("pattern")?;
            let regex = match (pattern.get("Regex"), pattern.get("String")) {
                (Some(regex), None) => regex.as_str()?.to_owned(),
                (None, Some(literal)) => escape_literal(literal.as_str()?),
                _ => return None,
            };
            let keep_gaps = match (
                split.get("behavior").and_then(Value::as_str)?,
                split.get("invert").and_then(Value::as_bool)?,
            ) {
                ("Isolated", false) => true,
                ("Removed", true) => false,
                _ => return None,
            };
            stages.push(Stage {
                pattern: Pattern::parse(&regex)?,
                keep_gaps,
            });
        }
        if byte_level.get("use_regex").and_then(Value::as_bool) == Some(true) {
            stages.push(Stage {
                pattern: Pattern::parse(BYTE_LEVEL_PATTERN)?,
                keep_gaps: true,
            });
        }
        Some(stages)
    }

    /// The ids of `text` without special tokens added, or `None` when this
    /// text must take the `tokenizers` path: not in NFC under an NFC
    /// normalizer, or a piece the model would not tokenize (so that the
    /// error surfaces from there).
    pub(crate) fn encode(&self, model: &ModelWrapper, text: &str) -> Option<Vec<u32>> {
        if self.normalizer == Normalizer::Nfc && !is_nfc(text) {
            return None;
        }
        let threads = rayon::current_num_threads();
        if text.len() >= PARALLEL_MIN_BYTES && !self.stages.is_empty() && threads > 1 {
            let chunk = (text.len() / threads).max(PARALLEL_CHUNK_MIN_BYTES);
            return self.encode_parallel(model, text, chunk);
        }
        let mut ids = Vec::with_capacity(text.len() / 3 + 8);
        let mut scratch = String::new();
        let mut ok = true;
        self.split_added(text, &mut |segment| match segment {
            Segment::Added(id) => ids.push(id),
            Segment::Text(segment) => {
                if ok && !self.encode_text(model, segment, &mut scratch, &mut ids) {
                    ok = false;
                }
            }
        });
        ok.then_some(ids)
    }

    /// [`Self::encode`] through the parallel path in chunks of about `chunk`
    /// bytes, whatever the text's length.
    #[cfg(test)]
    fn encode_with_chunk(
        &self,
        model: &ModelWrapper,
        text: &str,
        chunk: usize,
    ) -> Option<Vec<u32>> {
        if self.normalizer == Normalizer::Nfc && !is_nfc_chunked(text, chunk) {
            return None;
        }
        if self.stages.is_empty() {
            return self.encode(model, text);
        }
        self.encode_parallel(model, text, chunk)
    }

    /// Calls `f` with each added-token id and each text span between them,
    /// in order: the non-normalized set is matched on the raw text, the
    /// normalized set on what is left.
    fn split_added<'t>(&self, text: &'t str, f: &mut dyn FnMut(Segment<'t>)) {
        let mut after_raw = |segment: Segment<'t>| match (&segment, &self.normalized_added) {
            (Segment::Text(text), Some(added)) => added.split(text, &mut *f),
            _ => f(segment),
        };
        match &self.raw_added {
            Some(added) => added.split(text, &mut after_raw),
            None => after_raw(Segment::Text(text)),
        }
    }

    /// Appends the ids of one text span (no added tokens inside); `false`
    /// when the model declined a piece.
    fn encode_text(
        &self,
        model: &ModelWrapper,
        text: &str,
        scratch: &mut String,
        ids: &mut Vec<u32>,
    ) -> bool {
        let mut ok = true;
        self.pieces(0, text, &mut |piece| {
            if ok && !self.tokenize_piece(model, piece, scratch, ids) {
                ok = false;
            }
        });
        ok
    }

    /// The long-prompt path. The text spans between added tokens are scanned
    /// by the first `Split` in parallel, a span longer than `chunk` in chunks
    /// of about that size: a chunk's pieces are exact except where its scan
    /// touched the chunk's end, so at every seam the span is rescanned
    /// serially from the start of the left chunk's last piece until that scan
    /// lands on a piece end the right chunk's scan also produced, from where
    /// the two agree (a scan is a function of the text from a piece end on).
    /// The pieces, now exactly the serial ones, and the added tokens between
    /// them are then tokenized in parallel batches, each through its thread's
    /// piece cache.
    fn encode_parallel(&self, model: &ModelWrapper, text: &str, chunk: usize) -> Option<Vec<u32>> {
        let mut segments = Vec::new();
        self.split_added(text, &mut |segment| segments.push(segment));
        let base = text.as_ptr() as usize;
        let per_segment: Vec<Vec<(usize, usize)>> = segments
            .par_iter()
            .map(|segment| match *segment {
                Segment::Added(_) => Vec::new(),
                Segment::Text(span) => {
                    let start = span.as_ptr() as usize - base;
                    let end = start + span.len();
                    if span.len() > chunk {
                        let bounds: Vec<usize> = chunk_bounds(span, chunk)
                            .into_iter()
                            .map(|offset| start + offset)
                            .collect();
                        let per_chunk: Vec<Vec<(usize, usize)>> = bounds
                            .par_windows(2)
                            .map(|window| {
                                self.first_stage_pieces(text, window[0], window[1])
                                    .collect()
                            })
                            .collect();
                        self.splice(text, &bounds, per_chunk)
                    } else {
                        self.first_stage_pieces(text, start, end).collect()
                    }
                }
            })
            .collect();
        let mut items: Vec<Item> = Vec::with_capacity(per_segment.iter().map(Vec::len).sum());
        for (segment, pieces) in segments.iter().zip(per_segment) {
            match *segment {
                Segment::Added(id) => items.push(Item::Added(id)),
                Segment::Text(_) => {
                    items.extend(pieces.into_iter().map(|(s, e)| Item::Piece(s, e)));
                }
            }
        }
        let batch = (items.len() / (rayon::current_num_threads() * 4)).max(256);
        let batches: Option<Vec<Vec<u32>>> = items
            .par_chunks(batch)
            .map(|batch| {
                let mut out = Vec::with_capacity(batch.len() * 2);
                let mut scratch = String::new();
                let mut ok = true;
                for item in batch {
                    match *item {
                        Item::Added(id) => out.push(id),
                        Item::Piece(start, end) => {
                            self.pieces(1, &text[start..end], &mut |piece| {
                                if ok && !self.tokenize_piece(model, piece, &mut scratch, &mut out)
                                {
                                    ok = false;
                                }
                            });
                            if !ok {
                                return None;
                            }
                        }
                    }
                }
                Some(out)
            })
            .collect();
        let batches = batches?;
        let mut ids = Vec::with_capacity(batches.iter().map(Vec::len).sum());
        for batch in batches {
            ids.extend_from_slice(&batch);
        }
        Some(ids)
    }

    /// The first stage's pieces of `text[start..end]`, as absolute ranges,
    /// produced one at a time.
    fn first_stage_pieces<'t>(
        &self,
        text: &'t str,
        start: usize,
        end: usize,
    ) -> FirstStagePieces<'_, 't> {
        let stage = &self.stages[0];
        FirstStagePieces {
            matches: stage.pattern.find_iter(&text[start..end]),
            keep_gaps: stage.keep_gaps,
            base: start,
            end,
            cursor: start,
            pending: None,
            done: false,
        }
    }

    /// Joins the chunks' first-stage pieces of one span into the serial
    /// scan's pieces, rescanning across each seam as described on
    /// [`Self::encode_parallel`]. `bounds` holds the chunk starts and the
    /// span's end.
    fn splice(
        &self,
        text: &str,
        bounds: &[usize],
        chunks: Vec<Vec<(usize, usize)>>,
    ) -> Vec<(usize, usize)> {
        let span_end = bounds[bounds.len() - 1];
        let mut result: Vec<(usize, usize)> = Vec::with_capacity(chunks.iter().map(Vec::len).sum());
        let mut chunks = chunks.into_iter().enumerate();
        if let Some((_, first)) = chunks.next() {
            result.extend(first);
        }
        'seams: while let Some((index, chunk)) = chunks.next() {
            let seam = bounds[index];
            // Everything before the start of the last piece so far is exact
            // (the span's start always is): rescan from there.
            let rescan_from = result.last().map_or(bounds[0], |&(start, _)| start);
            result.truncate(result.partition_point(|&(_, end)| end <= rescan_from));
            let mut current = chunk;
            let mut current_end = bounds[index + 1];
            for (s, e) in self.first_stage_pieces(text, rescan_from, span_end) {
                result.push((s, e));
                if e < seam {
                    continue;
                }
                // The rescan may run past whole chunks; those are dropped.
                while e > current_end {
                    match chunks.next() {
                        Some((next_index, next_chunk)) => {
                            current = next_chunk;
                            current_end = bounds[next_index + 1];
                        }
                        None => break 'seams,
                    }
                }
                if let Ok(at) = current.binary_search_by_key(&e, |&(_, end)| end) {
                    result.extend_from_slice(&current[at + 1..]);
                    continue 'seams;
                }
            }
            // The rescan reached the end of the span.
            break;
        }
        result
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
    /// model over the piece's byte-level alphabet form; `false` when the
    /// model declines the piece.
    fn tokenize_piece(
        &self,
        model: &ModelWrapper,
        piece: &str,
        mapped: &mut String,
        ids: &mut Vec<u32>,
    ) -> bool {
        let bytes = piece.as_bytes();
        let cacheable = bytes.len() <= CACHED_PIECE_MAX_BYTES;
        if cacheable {
            let hit = PIECE_CACHE.with(|cache| {
                let cache = cache.borrow();
                cache
                    .get(&self.id)
                    .and_then(|pieces| pieces.get(bytes))
                    .map(|found| ids.extend_from_slice(found))
            });
            if hit.is_some() {
                return true;
            }
        }
        mapped.clear();
        mapped.extend(bytes.iter().map(|&b| self.alphabet[usize::from(b)]));
        let start = ids.len();
        match model.tokenize(mapped) {
            Ok(tokens) => ids.extend(tokens.iter().map(|token| token.id)),
            Err(_) => return false,
        }
        if cacheable {
            PIECE_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                let pieces = cache.entry(self.id).or_default();
                if pieces.len() >= PIECE_CACHE_CAPACITY {
                    pieces.clear();
                }
                pieces.insert(bytes.into(), ids[start..].into());
            });
        }
        true
    }
}

/// The pieces of one first-stage scan over a text span: its matches and,
/// when the stage keeps them, the gaps between them.
struct FirstStagePieces<'p, 't> {
    matches: pattern::Matches<'p, 't>,
    keep_gaps: bool,
    base: usize,
    end: usize,
    cursor: usize,
    pending: Option<(usize, usize)>,
    done: bool,
}

impl Iterator for FirstStagePieces<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if let Some(piece) = self.pending.take() {
            return Some(piece);
        }
        if self.done {
            return None;
        }
        match self.matches.next() {
            Some((s, e)) => {
                let (s, e) = (self.base + s, self.base + e);
                let gap = (self.keep_gaps && self.cursor < s).then_some((self.cursor, s));
                self.cursor = e;
                match gap {
                    Some(gap) => {
                        self.pending = Some((s, e));
                        Some(gap)
                    }
                    None => Some((s, e)),
                }
            }
            None => {
                self.done = true;
                let gap =
                    (self.keep_gaps && self.cursor < self.end).then_some((self.cursor, self.end));
                self.cursor = self.end;
                gap
            }
        }
    }
}

/// Chunk starts and the text end for a text split into pieces of about
/// `chunk` bytes at character boundaries.
fn chunk_bounds(text: &str, chunk: usize) -> Vec<usize> {
    let mut bounds = vec![0usize];
    let mut next = chunk;
    while next < text.len() {
        while !text.is_char_boundary(next) {
            next += 1;
        }
        if next >= text.len() {
            break;
        }
        bounds.push(next);
        next += chunk;
    }
    bounds.push(text.len());
    bounds
}

/// NFC quick check, over the thread pool for long texts.
fn is_nfc(text: &str) -> bool {
    if text.len() < PARALLEL_MIN_BYTES || rayon::current_num_threads() < 2 {
        return is_nfc_quick(text.chars()) == IsNormalized::Yes;
    }
    let chunk = (text.len() / rayon::current_num_threads()).max(PARALLEL_CHUNK_MIN_BYTES);
    is_nfc_chunked(text, chunk)
}

/// The quick check in chunks: every chunk must pass, and at each seam the
/// canonical ordering must hold across it (a non-starter may not follow a
/// character of higher combining class).
fn is_nfc_chunked(text: &str, chunk: usize) -> bool {
    let bounds = chunk_bounds(text, chunk);
    let chunks_ok = bounds
        .par_windows(2)
        .all(|window| is_nfc_quick(text[window[0]..window[1]].chars()) == IsNormalized::Yes);
    chunks_ok
        && bounds[1..bounds.len() - 1].iter().all(|&seam| {
            let before = text[..seam]
                .chars()
                .next_back()
                .map_or(0, canonical_combining_class);
            let after = text[seam..]
                .chars()
                .next()
                .map_or(0, canonical_combining_class);
            after == 0 || before <= after
        })
}

impl Drop for NativeEncoder {
    fn drop(&mut self) {
        // This thread's share of the cache; other threads drop theirs when
        // they next see an id they do not know, which never happens for a
        // retired id, so they keep it until they exit. Encoders live as long
        // as the tokenizer registry, so that is the process lifetime in
        // practice.
        let _ = PIECE_CACHE.try_with(|cache| {
            if let Ok(mut cache) = cache.try_borrow_mut() {
                cache.remove(&self.id);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        decoders::byte_level::ByteLevel as ByteLevelDecoder,
        models::bpe::{Merges, Vocab, BPE},
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
        processors::byte_level::ByteLevel as ByteLevelProcessor,
        AddedToken, SplitDelimiterBehavior,
    };

    use super::*;

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
            assert!(native.is_some(), "{}", shape.name);
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
        // Prefix space.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(QWEN35, SplitDelimiterBehavior::Isolated, false),
            PreTokenizerWrapper::ByteLevel(ByteLevelPreTokenizer::default().add_prefix_space(true)),
        ]))));
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_none());
        // A normalizer other than NFC.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer
            .with_normalizer(Some(NormalizerWrapper::NFKC(NFKC)))
            .expect("normalizer");
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_none());
        // A split behaviour the path does not reproduce.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(QWEN35, SplitDelimiterBehavior::MergedWithNext, false),
            byte_level(false),
        ]))));
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_none());
        // An added token with lstrip.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer
            .add_special_tokens([AddedToken::from("<pad>", true).lstrip(true)])
            .expect("special token");
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_none());
        // A regex outside the subset.
        let mut tokenizer = tokenizer_of(base, &mut state);
        tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
            split(r"\w+|\s+", SplitDelimiterBehavior::Isolated, false),
            byte_level(false),
        ]))));
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_none());
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
                    Some(ids) => {
                        compared += 1;
                        assert_eq!(ids, reference, "{}: {text:?}", shape.name);
                    }
                    None => assert!(
                        shape.nfc && is_nfc_quick(text.chars()) != IsNormalized::Yes,
                        "{}: declined {text:?} for no reason",
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
                if let Some(ids) = native.encode(tokenizer.get_model(), text) {
                    assert_eq!(ids, reference, "{}: {text:?}", shape.name);
                }
            }
        }
    }

    #[test]
    fn text_not_in_nfc_is_declined_under_an_nfc_normalizer() {
        let mut state = 0x5555_6666_7777_8888u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        // "e" + combining acute composes under NFC: not our business.
        assert!(native
            .encode(tokenizer.get_model(), "Cafe\u{301}")
            .is_none());
        assert!(native.encode(tokenizer.get_model(), "Caf\u{E9}").is_some());
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

    /// Long text with the shapes that stress chunk seams: runs of one
    /// character class far longer than a chunk, whitespace runs with and
    /// without newlines, CJK, marks, clitics, and added tokens.
    fn long_text(state: &mut u64, added: &[&str]) -> String {
        let mut text = random_text(state, 3000, added);
        text.push_str(&"a".repeat(700));
        text.push_str(&" ".repeat(500));
        text.push('x');
        text.push_str(&"\n".repeat(300));
        text.push_str(&" \n".repeat(200));
        text.push_str(&"中".repeat(400));
        text.push_str(&"7".repeat(350));
        text.push_str(&"ก\u{E31}".repeat(200));
        text.push_str(&random_text(state, 3000, added));
        text.push_str(&"-".repeat(300));
        text.push('\n');
        text.push_str(&random_text(state, 2000, added));
        text
    }

    #[test]
    fn the_parallel_path_gives_the_serial_ids_for_every_chunk_size() {
        let mut state = 0x1357_9BDF_2468_ACE0u64;
        for shape in shapes() {
            let tokenizer = tokenizer_of(&shape, &mut state);
            let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
            let added: Vec<&str> = shape
                .special
                .iter()
                .copied()
                .chain(shape.added.iter().map(|(s, _)| *s))
                .collect();
            for round in 0..3 {
                let text = long_text(&mut state, &added);
                let Some(serial) = native.encode(tokenizer.get_model(), &text) else {
                    continue;
                };
                let reference = tokenizer
                    .encode(text.as_str(), false)
                    .expect("encode")
                    .get_ids()
                    .to_vec();
                assert_eq!(serial, reference, "{} round {round}: serial", shape.name);
                for chunk in [5usize, 17, 64, 333, 1000, 4096, 1 << 20] {
                    let parallel = native
                        .encode_with_chunk(tokenizer.get_model(), &text, chunk)
                        .expect("same NFC verdict");
                    assert_eq!(
                        parallel, serial,
                        "{} round {round}: chunk {chunk}",
                        shape.name
                    );
                }
            }
        }
    }

    #[test]
    fn a_long_prompt_takes_the_parallel_path_and_matches_tokenizers() {
        let mut state = 0xFEDC_BA98_7654_3210u64;
        let shape = &shapes()[0];
        let tokenizer = tokenizer_of(shape, &mut state);
        let native = NativeEncoder::from_tokenizer(&tokenizer).expect("native path");
        let mut text = String::new();
        while text.len() < 3 * PARALLEL_MIN_BYTES {
            // Without the two pool characters that compose with a preceding
            // base, so the text is in NFC and takes the native path.
            text.extend(
                long_text(&mut state, &["<|im_start|>", "<|im_end|>"])
                    .chars()
                    .filter(|c| !matches!(c, '\u{301}' | '\u{9BE}')),
            );
        }
        assert_eq!(is_nfc_quick(text.chars()), IsNormalized::Yes);
        let Some(ids) = native.encode(tokenizer.get_model(), &text) else {
            panic!("the long prompt was declined");
        };
        let reference = tokenizer
            .encode(text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        assert_eq!(ids, reference);
    }

    #[test]
    fn the_chunked_nfc_check_agrees_with_the_whole_check() {
        const MARKS: &[char] = &[
            'a', 'e', 'o', '\u{301}', '\u{300}', '\u{323}', '\u{31B}', '\u{E9}', '\u{1EA1}', 'ก',
            '\u{E31}', 'क', '\u{93E}', '\u{93C}', '\u{1100}', '\u{1161}', '\u{11A8}', ' ', '\n',
            'x', '\u{FFFD}', '中',
        ];
        let mut state = 0x0BAD_CAFE_F00D_1234u64;
        for _ in 0..3000 {
            let len = 1 + (pseudo_random(&mut state) % 24) as usize;
            let text: String = (0..len)
                .map(|_| MARKS[(pseudo_random(&mut state) % MARKS.len() as u64) as usize])
                .collect();
            let whole = is_nfc_quick(text.chars()) == IsNormalized::Yes;
            for chunk in [1usize, 2, 3, 5, 8, 64] {
                assert_eq!(
                    is_nfc_chunked(&text, chunk),
                    whole,
                    "{text:?} chunk {chunk}"
                );
            }
        }
    }
}
