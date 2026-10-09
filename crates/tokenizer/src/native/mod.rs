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
//! native path. `add_special_tokens = true` also stays with `tokenizers`.

use std::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rustc_hash::FxHashMap;
use serde_json::Value;
use thread_local::ThreadLocal;
use tokenizers::{models::ModelWrapper, Model, Tokenizer as HfTokenizer};
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

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
/// Entries a thread's piece cache holds for one encoder before it is cleared,
/// at most; see `PIECE_CACHE_TOTAL_CAPACITY`.
const PIECE_CACHE_CAPACITY: usize = 16_384;
/// Entries all threads together hold for one encoder: a thread's own cap is
/// this divided by the number of threads that have encoded with the encoder
/// (and never more than `PIECE_CACHE_CAPACITY`), so the piece caches of one
/// tokenizer stay under about 40 MB however many runtime and blocking threads
/// encode with it, and they are freed with the encoder.
const PIECE_CACHE_TOTAL_CAPACITY: usize = 262_144;

/// Piece bytes to the ids they tokenize to.
type PieceIds = FxHashMap<Box<[u8]>, Box<[u32]>>;

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
    piece_cache: ThreadLocal<RefCell<PieceIds>>,
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
            normalizer,
            raw_added: AddedTokens::build(&raw)?,
            normalized_added: AddedTokens::build(&normalized)?,
            stages,
            alphabet: bytes_char(),
            piece_cache: ThreadLocal::new(),
            threads: AtomicUsize::new(0),
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
        let mut ids = Vec::with_capacity(text.len() / 3 + 8);
        let mut mapped = String::new();
        let mut failed = false;
        let mut tokenize = |segment: &str, ids: &mut Vec<u32>| {
            self.pieces(0, segment, &mut |piece| {
                if !self.tokenize_piece(model, piece, &mut mapped, ids) {
                    failed = true;
                }
            });
        };
        let mut after_raw = |segment: Segment<'_>, ids: &mut Vec<u32>| match segment {
            Segment::Added(id) => ids.push(id),
            Segment::Text(text) => match &self.normalized_added {
                Some(added) => added.split(text, |inner| match inner {
                    Segment::Added(id) => ids.push(id),
                    Segment::Text(text) => tokenize(text, ids),
                }),
                None => tokenize(text, ids),
            },
        };
        match &self.raw_added {
            Some(added) => added.split(text, |segment| after_raw(segment, &mut ids)),
            None => after_raw(Segment::Text(text), &mut ids),
        }
        (!failed).then_some(ids)
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
        let cache = self.piece_cache.get_or(|| {
            self.threads.fetch_add(1, Ordering::Relaxed);
            RefCell::default()
        });
        if cacheable {
            if let Some(found) = cache.borrow().get(bytes) {
                ids.extend_from_slice(found);
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
            let mut pieces = cache.borrow_mut();
            if pieces.len() >= self.piece_cache_share() {
                pieces.clear();
            }
            pieces.insert(bytes.into(), ids[start..].into());
        }
        true
    }

    /// Entries one thread's piece cache may hold: the total shared among the
    /// threads that have encoded with this encoder, `PIECE_CACHE_CAPACITY` at
    /// most.
    fn piece_cache_share(&self) -> usize {
        (PIECE_CACHE_TOTAL_CAPACITY / self.threads.load(Ordering::Relaxed).max(1))
            .clamp(1, PIECE_CACHE_CAPACITY)
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
                        shape.nfc && text.nfc().ne(text.chars()),
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
        if let Some(native) = NativeEncoder::from_tokenizer(&tokenizer) {
            for text in ["ab1", "the same words 12"] {
                let reference = tokenizer.encode(text, false).expect("encode");
                assert_eq!(
                    native.encode(tokenizer.get_model(), text),
                    Some(reference.get_ids().to_vec()),
                    "{text:?}"
                );
            }
        }
        assert!(NativeEncoder::from_tokenizer(&tokenizer).is_none());
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
            .map(|cache| cache.get_mut().len())
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
                    native.encode(model, &text).expect("ids");
                });
            }
        });
        assert_eq!(native.threads.load(Ordering::Relaxed), threads as usize);
        let share = native.piece_cache_share();
        assert!(share < PIECE_CACHE_CAPACITY, "{share}");
        let caches: Vec<usize> = native
            .piece_cache
            .iter_mut()
            .map(|cache| cache.get_mut().len())
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
                .unwrap_or_else(|| panic!("{text:?} is in NFC and must take the native path"));
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
            assert!(
                native.encode(tokenizer.get_model(), text).is_none(),
                "{text:?}"
            );
        }
    }
}
