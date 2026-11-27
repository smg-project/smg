//! Incremental detokenization for byte-level BPE vocabularies.
//!
//! The `tokenizers` `ByteLevel` decoder maps every token string back to the
//! bytes it stands for, concatenates them, and runs `String::from_utf8_lossy`
//! over the result; a tiktoken rank is such a byte string to begin with. That
//! makes decoding compositional: a token's bytes never depend on its
//! neighbours. So instead of the generic incremental algorithm, which decodes
//! the retained window twice per generated token and diffs the two strings,
//! we precompute each id's bytes once per tokenizer and, per token, append
//! them to a small pending buffer and emit its lossy decode once that decode
//! no longer ends in U+FFFD.
//!
//! That is the rule `tokenizers`' `DecodeStream` applies to its window, and
//! the window's already-emitted prefix always consists of complete
//! sequences, so the lossy decode of the pending bytes alone is the text the
//! reference would emit at each step. The pending buffer holds at most the
//! bytes since the last emission, normally one partial character; the
//! stream's end emits its complete prefix and drops an incomplete suffix.

use std::{
    borrow::Cow,
    hash::BuildHasher,
    ops::Range,
    sync::{Arc, LazyLock},
};

use anyhow::{anyhow, Result};
use rustc_hash::FxBuildHasher;
use tokenizers::{decoders::DecoderWrapper, Tokenizer as HfTokenizer};

use crate::{
    traits::{IncrementalDecoder, TokenIdType},
    word_cache::{token_of, Listing},
};

/// Highest code point in the byte-level alphabet plus one (the 68 bytes that
/// are not printable Latin-1 map to U+0100..=U+0143).
const ALPHABET_END: usize = 0x144;

/// GPT-2's `bytes_to_unicode`, inverted: byte value for each alphabet char.
/// Mirrors `tokenizers::pre_tokenizers::byte_level::bytes_char`.
static CHAR_BYTES: LazyLock<[Option<u8>; ALPHABET_END]> = LazyLock::new(|| {
    let mut table = [None; ALPHABET_END];
    let mut next_escape = 0x100u32;
    for byte in 0..=u8::MAX {
        let printable = (b'!'..=b'~').contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte);
        let code = if printable {
            u32::from(byte)
        } else {
            next_escape += 1;
            next_escape - 1
        };
        table[code as usize] = Some(byte);
    }
    table
});

/// What the table knows about an id: whether it is in the vocabulary, which
/// store holds its bytes, and whether `skip_special_tokens` strips it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Not in the vocabulary.
    Absent,
    /// A token kept in decoded text, whole: in the text store.
    Text,
    /// A token kept in decoded text, not whole on its own: in the byte store.
    Bytes,
    /// A token `skip_special_tokens` strips, in the text store.
    SpecialText,
    /// A token `skip_special_tokens` strips, in the byte store.
    SpecialBytes,
}

impl Kind {
    fn special(self) -> bool {
        matches!(self, Self::SpecialText | Self::SpecialBytes)
    }

    /// Whether the token's bytes are in the text store.
    fn text(self) -> bool {
        matches!(self, Self::Text | Self::SpecialText)
    }
}

/// The bytes of one token as the table holds them: text when the token is
/// whole on its own (valid UTF-8 that does not end in the replacement
/// character), bytes otherwise.
#[derive(Clone, Copy)]
enum Piece<'a> {
    Text(&'a str),
    Bytes(&'a [u8]),
}

impl Piece<'_> {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Piece::Text(text) => text.as_bytes(),
            Piece::Bytes(bytes) => bytes,
        }
    }
}

/// Per-tokenizer table of the bytes each id decodes to.
///
/// Built once at load time: by [`ByteLevelTable::build`] for a `tokenizers`
/// vocabulary whose decoder is exactly `ByteLevel`, through
/// [`ByteLevelTableBuilder`] for a tiktoken vocabulary, whose ranks are byte
/// strings already. The tokens that are whole on their own (valid UTF-8 not
/// ending in U+FFFD: the common case, the step emits them as they are) live
/// in a text store, so the decoder reads each as `&str` with no validation
/// per token; the others live in a byte store. About 9 bytes per id plus the
/// token bytes themselves.
#[derive(Default)]
pub(crate) struct ByteLevelTable {
    /// `spans[id]` indexes the store `kind[id]` names for token `id`.
    spans: Vec<(u32, u32)>,
    text: String,
    bytes: Vec<u8>,
    kind: Vec<Kind>,
}

/// A store's span as the table keeps it. The spans are `u32` to keep the
/// table small, and nothing caps a store's size, so a store past 4 GiB is
/// refused here, at load, rather than served through wrapped spans later.
fn span(start: usize, end: usize) -> (u32, u32) {
    assert!(
        end <= u32::MAX as usize,
        "the byte-level table's store exceeds what its u32 spans index"
    );
    (start as u32, end as u32)
}

/// The largest id of the vocabulary, added tokens included: the model's as
/// the file's one-pass read saw it, against the added tokens' own; through
/// the whole vocabulary (a clone of its map and every token) only where no
/// read gave it, a model built in memory say.
pub(crate) fn max_token_id(tokenizer: &HfTokenizer, model_max_id: Option<u32>) -> Option<u32> {
    match model_max_id {
        Some(model) => {
            let added = tokenizer
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .keys()
                .copied()
                .max();
            Some(added.map_or(model, |added| added.max(model)))
        }
        None => tokenizer.get_vocab(true).values().copied().max(),
    }
}

/// Builds a [`ByteLevelTable`] one id at a time, from 0 up.
pub(crate) struct ByteLevelTableBuilder {
    spans: Vec<(u32, u32)>,
    text: String,
    bytes: Vec<u8>,
    kind: Vec<Kind>,
}

impl ByteLevelTableBuilder {
    pub(crate) fn with_capacity(ids: usize) -> Self {
        Self {
            spans: Vec::with_capacity(ids),
            text: String::with_capacity(ids * 4),
            bytes: Vec::new(),
            kind: Vec::with_capacity(ids),
        }
    }

    /// The next id decodes to `bytes`; `special` marks a token that
    /// `skip_special_tokens` strips. A token that is whole on its own goes to
    /// the text store, classified here once rather than on every step.
    pub(crate) fn push(&mut self, bytes: &[u8], special: bool) {
        match std::str::from_utf8(bytes) {
            Ok(text) if settled(text) => {
                let start = self.text.len();
                self.text.push_str(text);
                self.spans.push(span(start, self.text.len()));
                self.kind.push(if special {
                    Kind::SpecialText
                } else {
                    Kind::Text
                });
            }
            _ => {
                let start = self.bytes.len();
                self.bytes.extend_from_slice(bytes);
                self.spans.push(span(start, self.bytes.len()));
                self.kind.push(if special {
                    Kind::SpecialBytes
                } else {
                    Kind::Bytes
                });
            }
        }
    }

    /// The next id is not in the vocabulary.
    pub(crate) fn push_absent(&mut self) {
        self.spans.push((0, 0));
        self.kind.push(Kind::Absent);
    }

    pub(crate) fn finish(mut self) -> Arc<ByteLevelTable> {
        self.text.shrink_to_fit();
        self.bytes.shrink_to_fit();
        Arc::new(ByteLevelTable {
            spans: self.spans,
            text: self.text,
            bytes: self.bytes,
            kind: self.kind,
        })
    }
}

/// The decoder kind of a `tokenizers` vocabulary as the load's log names it:
/// the variant, with a `Sequence`'s members listed, `none` without a decoder.
pub(crate) fn decoder_kind(decoder: Option<&DecoderWrapper>) -> String {
    fn variant(decoder: &DecoderWrapper) -> String {
        let debug = format!("{decoder:?}");
        debug
            .split(['(', ' ', '{'])
            .next()
            .unwrap_or(debug.as_str())
            .to_owned()
    }
    match decoder {
        None => "none".to_owned(),
        Some(DecoderWrapper::Sequence(sequence)) => {
            let members: Vec<String> = sequence.get_decoders().iter().map(variant).collect();
            format!("Sequence({})", members.join(", "))
        }
        Some(decoder) => variant(decoder),
    }
}

impl ByteLevelTable {
    /// `None` unless the tokenizer decodes with a plain `ByteLevel` decoder.
    pub(crate) fn build(tokenizer: &HfTokenizer) -> Option<Arc<Self>> {
        Self::build_with_max_id(tokenizer, None, None)
    }

    /// [`Self::build`] with the model's largest id as the file's read saw
    /// it, when it did (see [`max_token_id`]).
    pub(crate) fn build_with_max_id(
        tokenizer: &HfTokenizer,
        model_max_id: Option<u32>,
        listing: Option<&Listing>,
    ) -> Option<Arc<Self>> {
        if !matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_))) {
            return None;
        }
        let max_id = max_token_id(tokenizer, model_max_id)?;
        let count = max_id as usize + 1;
        let added = tokenizer.get_added_vocabulary();
        let mut table = ByteLevelTableBuilder::with_capacity(count);
        let mut bytes = Vec::new();
        for id in 0..count {
            match token_of(tokenizer, listing, id as u32) {
                Some(token) => {
                    bytes.clear();
                    push_token_bytes(&token, &mut bytes);
                    table.push(&bytes, added.is_special_token(&token));
                }
                None => table.push_absent(),
            }
        }
        Some(table.finish())
    }

    /// Decode a complete HF sequence using the same bytes as incremental decode.
    /// Unknown IDs are dropped, as in tokenizers; bytes that are not whole on
    /// their own are converted per run, so characters may span token boundaries.
    pub(crate) fn decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> String {
        self.concat(ids, skip_special_tokens, false)
            .unwrap_or_default()
            .0
    }

    /// Bytes the decode copies at once for a short token.
    const WINDOW: usize = 16;

    /// The ids of one streamed chunk: few enough that the exact sizing pass
    /// of [`Self::decode_capacity`] reads the entries the concatenation reads
    /// right after, from cache.
    /// Measured on the crate's decode bench over the four pinned vocabularies
    /// with sliding windows of 1 to 64 ids (the sweep in #5556's Numbers
    /// table): the exact sizing wins or ties at every length up to
    /// eight, 19 to 23 percent at one id and 2 to 5 percent at eight, and loses
    /// from sixteen on by 9 percent and more, so eight is the last length at
    /// which it pays on every vocabulary.
    const CHUNK_IDS: usize = 8;

    /// The buffer [`Self::decode`] starts with for `ids`. A chunk's worth of
    /// ids is sized from the pieces themselves: the table knows every id's
    /// length, and a guess of four bytes an id made the one-token decode of
    /// a stream's chunk grow its buffer on most words (a word with its
    /// leading space is longer than four bytes), a second allocator call per
    /// streamed event. A longer decode keeps the four-byte estimate and its
    /// amortized regrows: a pass over its entries is a second walk of the
    /// table, one to three nanoseconds an id on the crate's decode bench,
    /// more than the regrows it would save.
    fn decode_capacity(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> usize {
        if ids.len() <= Self::CHUNK_IDS {
            self.decoded_len(ids, skip_special_tokens)
        } else {
            ids.len().saturating_mul(4)
        }
    }

    /// The length of the bytes [`Self::decode`] concatenates for `ids`: the
    /// spans of the known ids, less the special ones when they are skipped.
    /// Out of line: inlined (twice) it doubled the body every decode shares,
    /// and a long decode, which never takes this pass, paid about seventy
    /// instructions more per call for the larger frame.
    #[inline(never)]
    fn decoded_len(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> usize {
        ids.iter()
            .filter_map(|&id| self.token_bytes(id))
            .filter(|(_, special)| !skip_special_tokens || !special)
            .map(|(token_bytes, _)| token_bytes.len())
            .sum()
    }

    /// [`Self::decode`] for a vocabulary that does not drop unknown ids:
    /// `None` at the first id without an entry, before anything is rendered,
    /// so the caller decodes the whole sequence its own way (the tiktoken
    /// backend's crate decode, with its unknown-id error and its lossy
    /// fallback); otherwise the same text, with whether a conversion was
    /// lossy, for the caller's log.
    pub(crate) fn decode_known(
        &self,
        ids: &[TokenIdType],
        skip_special_tokens: bool,
    ) -> Option<(String, bool)> {
        self.concat(ids, skip_special_tokens, true)
    }

    /// The text of `ids` and whether a conversion was lossy; `None` at an id
    /// the vocabulary lacks when `known` is set (dropped otherwise). A token
    /// that is whole on its own is appended as text, so nothing validates it
    /// again. The others wait in a run that is converted when a whole token
    /// or the end closes it: a whole token neither ends nor continues a
    /// character, so the runs convert as the whole sequence would.
    #[inline(always)]
    fn concat(&self, ids: &[TokenIdType], skip: bool, known: bool) -> Option<(String, bool)> {
        let mut text = String::with_capacity(self.decode_capacity(ids, skip));
        let mut run = Vec::new();
        let mut lossy = false;
        for &id in ids {
            let Some((kind, span)) = self.locate(id) else {
                if known {
                    return None;
                }
                continue;
            };
            if skip && kind.special() {
                continue;
            }
            if !kind.text() {
                run.extend_from_slice(&self.bytes[span]);
                continue;
            }
            if !run.is_empty() {
                lossy |= close_run(&mut run, &mut text);
            }
            // A short token goes as one fixed window with the length cut
            // back, where the buffer has the room: a copy call for a few
            // bytes costs more than the bytes.
            let room = text.capacity() - text.len() >= Self::WINDOW;
            match self.text.get(span.start..span.start + Self::WINDOW) {
                Some(window) if room && span.len() <= Self::WINDOW => {
                    let end = text.len() + span.len();
                    text.push_str(window);
                    text.truncate(end);
                }
                _ => text.push_str(&self.text[span]),
            }
        }
        if !run.is_empty() {
            lossy |= close_run(&mut run, &mut text);
        }
        Some((text, lossy))
    }

    /// Where `id` lives: its kind and its span in the store the kind names;
    /// `None` for an id the vocabulary lacks.
    #[inline]
    fn locate(&self, id: TokenIdType) -> Option<(Kind, Range<usize>)> {
        let id = id as usize;
        let kind = *self.kind.get(id)?;
        if kind == Kind::Absent {
            return None;
        }
        let (start, end) = self.spans[id];
        Some((kind, start as usize..end as usize))
    }

    /// The bytes of `id` from whichever store holds them, and whether it is
    /// special. The sizing pass of a chunk's decode ([`Self::decoded_len`])
    /// reads every id through this rather than through [`Self::entry`]: one
    /// kind test, one span and one slice of the store the kind picks; the
    /// step's piece, with the text slice's boundary checks, is not built for
    /// a pass that only sums lengths.
    #[inline]
    pub(crate) fn token_bytes(&self, id: TokenIdType) -> Option<(&[u8], bool)> {
        let (kind, span) = self.locate(id)?;
        let store = if kind.text() {
            self.text.as_bytes()
        } else {
            self.bytes.as_slice()
        };
        Some((&store[span], kind.special()))
    }

    /// The piece of `id` for a step: text when the token is whole on its own
    /// (borrowed from the text store, nothing validated), bytes otherwise;
    /// and whether it is special. `None` for an id the vocabulary lacks.
    #[inline]
    fn entry(&self, id: TokenIdType) -> Option<(Piece<'_>, bool)> {
        let (kind, span) = self.locate(id)?;
        let piece = if kind.text() {
            Piece::Text(&self.text[span])
        } else {
            Piece::Bytes(&self.bytes[span])
        };
        Some((piece, kind.special()))
    }
}

/// A store of every id's bytes that a [`TextIndex`] is built over and asked
/// through: a [`ByteLevelTable`], the listing of a vocabulary.
pub(crate) trait TextStore {
    /// How many ids the store covers (`0..ids`).
    fn ids(&self) -> u32;

    /// The bytes of `id`; `None` for an id the store lacks.
    fn bytes_of(&self, id: u32) -> Option<&[u8]>;
}

impl TextStore for ByteLevelTable {
    fn ids(&self) -> u32 {
        self.kind.len() as u32
    }

    #[inline]
    fn bytes_of(&self, id: u32) -> Option<&[u8]> {
        self.token_bytes(id).map(|(bytes, _)| bytes)
    }
}

/// The ids of a [`TextStore`] by their bytes, the store holding every id's
/// bytes: one array of 32-bit slots, open-addressed, a slot the id and, in
/// the bits the ids leave free, bits of the hash of its bytes, so that a
/// probe reads the store only where those bits agree. Six bytes an id, where
/// a map by text holds every text a second time.
pub(crate) struct TextIndex {
    slots: Box<[u32]>,
    /// Bits of a slot the id takes: enough for the count of ids itself, so
    /// that the id of all ones is never one of them and [`Self::EMPTY`] is
    /// no slot's value.
    id_bits: u32,
}

impl TextIndex {
    const EMPTY: u32 = u32::MAX;

    /// The index of every id `store` has bytes for, of two ids with the same
    /// bytes the lower; and whether two ids had the same bytes.
    pub(crate) fn build<S: TextStore>(store: &S) -> (Self, bool) {
        let ids = store.ids();
        // Two thirds full: a miss walks about five slots, a hit two.
        let room = ids as usize + ids as usize / 2 + 1;
        let mut index = Self {
            slots: vec![Self::EMPTY; room].into_boxed_slice(),
            id_bits: u32::BITS - ids.leading_zeros(),
        };
        let mask = index.id_mask();
        let mut twice = false;
        for id in 0..ids {
            let Some(bytes) = store.bytes_of(id) else {
                continue;
            };
            let (mut at, tag) = index.place(bytes);
            loop {
                let slot = index.slots[at];
                if slot == Self::EMPTY {
                    index.slots[at] = tag | id;
                    break;
                }
                if slot & !mask == tag && store.bytes_of(slot & mask) == Some(bytes) {
                    twice = true;
                    break;
                }
                at = index.after(at);
            }
        }
        (index, twice)
    }

    /// The id whose bytes in `store`, the store of the build, are `bytes`.
    pub(crate) fn get<S: TextStore>(&self, store: &S, bytes: &[u8]) -> Option<TokenIdType> {
        let (mut at, tag) = self.place(bytes);
        let mask = self.id_mask();
        loop {
            let slot = self.slots[at];
            if slot == Self::EMPTY {
                return None;
            }
            if slot & !mask == tag && store.bytes_of(slot & mask) == Some(bytes) {
                return Some(slot & mask);
            }
            at = self.after(at);
        }
    }

    fn id_mask(&self) -> u32 {
        u32::MAX.checked_shr(u32::BITS - self.id_bits).unwrap_or(0)
    }

    fn after(&self, at: usize) -> usize {
        if at + 1 == self.slots.len() {
            0
        } else {
            at + 1
        }
    }

    /// Where the probe of `bytes` starts, and the bits of their hash as a
    /// slot holds them.
    fn place(&self, bytes: &[u8]) -> (usize, u32) {
        let hash = FxBuildHasher
            .hash_one(bytes)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let at = ((hash >> 32) * self.slots.len() as u64) >> 32;
        let tag = (hash as u32).checked_shl(self.id_bits).unwrap_or(0);
        (at as usize, tag)
    }
}

/// Appends the text of a run of token bytes and empties the run: one UTF-8
/// conversion, lossy only when the run is not valid UTF-8 (a character cut by
/// the run's end, or bytes that never formed one); true when it was lossy.
fn close_run(run: &mut Vec<u8>, text: &mut String) -> bool {
    let lossy = match std::str::from_utf8(run) {
        Ok(whole) => {
            text.push_str(whole);
            false
        }
        Err(_) => {
            text.push_str(&String::from_utf8_lossy(run));
            true
        }
    };
    run.clear();
    lossy
}

/// The bytes a token string stands for: its chars mapped through the
/// byte-level alphabet when every one of them is in it, otherwise the string's
/// own UTF-8 (how `ByteLevel::decode_chain` treats added tokens).
pub(crate) fn push_token_bytes(token: &str, out: &mut Vec<u8>) {
    let start = out.len();
    for ch in token.chars() {
        let mapped = usize::try_from(u32::from(ch))
            .ok()
            .filter(|&code| code < ALPHABET_END)
            .and_then(|code| CHAR_BYTES[code]);
        match mapped {
            Some(byte) => out.push(byte),
            None => {
                out.truncate(start);
                out.extend_from_slice(token.as_bytes());
                return;
            }
        }
    }
}

/// What to do with an id the vocabulary lacks: what the backend's `decode`
/// does with it.
#[derive(Clone, Copy)]
pub(crate) enum UnknownId {
    /// Decode nothing for it (`tokenizers`).
    Drop,
    /// Fail the stream (tiktoken).
    Reject,
}

/// One stream's state: the bytes of a character that is still incomplete.
pub(crate) struct ByteLevelIncremental {
    table: Arc<ByteLevelTable>,
    pending: Vec<u8>,
    /// The text a run of pending bytes decoded to once it settled, kept so
    /// the next run's text needs no allocation.
    text: String,
    skip_special_tokens: bool,
    unknown: UnknownId,
    /// Tokens fed so far, to place the once-per-stream report.
    fed: usize,
    /// Whether this stream's invalid bytes were reported already.
    reported: bool,
}

impl ByteLevelIncremental {
    pub(crate) fn new(
        table: Arc<ByteLevelTable>,
        skip_special_tokens: bool,
        unknown: UnknownId,
    ) -> Self {
        Self {
            table,
            pending: Vec::new(),
            text: String::new(),
            skip_special_tokens,
            unknown,
            fed: 0,
            reported: false,
        }
    }

    /// Bytes withheld because their lossy decode still ends in U+FFFD.
    #[cfg(test)]
    fn pending(&self) -> &[u8] {
        &self.pending
    }
}

/// Whether `text` is something the reference emits: non-empty and not ending
/// in U+FFFD. `DecodeStream` keeps its whole window back while the window's
/// lossy decode ends in the replacement character, whether that character
/// stands for an incomplete sequence, an invalid byte, or is literally in the
/// text; the pending bytes here are exactly that window minus its emitted
/// prefix, so the same test on them gives the same piece at the same token.
#[inline]
fn settled(text: &str) -> bool {
    !text.is_empty() && !text.ends_with(char::REPLACEMENT_CHARACTER)
}

/// Where a character the input's end cut short begins in `bytes`: the index
/// of a trailing sequence that is the valid start of a character and nothing
/// more (`Utf8Error::error_len` is `None` there), or `bytes.len()` when the
/// input ends on a complete character or on an invalid sequence. The bytes
/// before it render as `String::from_utf8_lossy` renders them, a complete
/// character as it is and an invalid sequence (a stray byte the model wrote)
/// as the replacement character; the bytes from it on are the start of a
/// character the model never finished, which a byte-level stream drops at
/// its end and which a ByteFallback vocabulary's window counts to keep off
/// its end.
pub(crate) fn cut_start(bytes: &[u8]) -> usize {
    let mut index = 0;
    while index < bytes.len() {
        match std::str::from_utf8(&bytes[index..]) {
            Ok(_) => break,
            Err(error) => match error.error_len() {
                Some(invalid) => index += error.valid_up_to() + invalid,
                None => return index + error.valid_up_to(),
            },
        }
    }
    bytes.len()
}

impl ByteLevelIncremental {
    /// The text `token_id` makes final: a whole token of valid text borrowed
    /// from the table (the common case), or a pending run's decode in the
    /// decoder's own buffer; `None` while the token completes no character,
    /// is skipped as special or is an id the vocabulary lacks and drops.
    fn piece(&mut self, token_id: TokenIdType) -> Result<Option<&str>> {
        self.fed += 1;
        let Some((piece, special)) = self.table.entry(token_id) else {
            return match self.unknown {
                UnknownId::Drop => Ok(None),
                UnknownId::Reject => Err(anyhow!("unknown token id {token_id}")),
            };
        };
        if self.skip_special_tokens && special {
            return Ok(None);
        }
        if self.pending.is_empty() {
            // The common case: a token whole on its own (the table knows it
            // since load), borrowed as it is, no validation and no copy.
            if let Piece::Text(text) = piece {
                return Ok(Some(text));
            }
        }
        self.pending.extend_from_slice(piece.as_bytes());
        self.text.clear();
        match String::from_utf8_lossy(&self.pending) {
            Cow::Borrowed(text) if settled(text) => self.text.push_str(text),
            // A replacement inside settled text stands for bytes no later
            // token can complete: say so once per stream, not once per token.
            Cow::Owned(text) if settled(&text) => {
                if !self.reported {
                    self.reported = true;
                    tracing::warn!(
                        token_index = self.fed - 1,
                        "invalid UTF-8 bytes in the token stream decoded to U+FFFD"
                    );
                }
                self.text = text;
            }
            _ => return Ok(None),
        }
        self.pending.clear();
        Ok(Some(&self.text))
    }
}

impl IncrementalDecoder for ByteLevelIncremental {
    fn step(&mut self, token_id: TokenIdType) -> Result<String> {
        Ok(self.piece(token_id)?.map(str::to_owned).unwrap_or_default())
    }

    fn step_into(&mut self, token_id: TokenIdType, out: &mut String) -> Result<bool> {
        Ok(match self.piece(token_id)? {
            Some(text) => {
                out.push_str(text);
                true
            }
            None => false,
        })
    }

    fn finish_into(&mut self, out: &mut String) -> Result<bool> {
        if self.pending.is_empty() {
            return Ok(false);
        }
        // Keep complete text and invalid bytes, but not the start of a
        // character for which the stream supplied no remaining bytes.
        let end = cut_start(&self.pending);
        out.push_str(&String::from_utf8_lossy(&self.pending[..end]));
        self.pending.clear();
        Ok(end != 0)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.text.clear();
        self.fed = 0;
        self.reported = false;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use tokenizers::{
        decoders::{byte_level::ByteLevel as ByteLevelDecoder, metaspace::Metaspace},
        models::bpe::{Merges, Vocab, BPE},
        pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
        AddedToken,
    };

    use super::*;
    use crate::traits::Decoder;

    pub(crate) const SPECIAL: &str = "<|im_end|>";
    const ADDED: &str = "<tool>";

    /// The lookup the decode tables take agrees with `id_to_token` on
    /// every id, with the read's listing and without it: an added token
    /// first, then the model's token; and the table built from the
    /// listing is the table built from the model.
    #[test]
    fn the_token_lookup_and_the_table_agree_with_and_without_the_listing() {
        use crate::word_cache::{token_of, Listing};
        let mut tokenizer = byte_level_tokenizer();
        tokenizer
            .add_tokens([AddedToken::from(ADDED, false)])
            .expect("add token");
        let vocab: Vocab = tokenizer.get_vocab(false).into_iter().collect();
        let listing = Listing::of(&vocab).expect("a dense vocabulary lists");
        assert_eq!(listing.len() as usize, vocab.len());
        let model_max_id = Some(listing.len() - 1);
        let max_id = max_token_id(&tokenizer, model_max_id).expect("ids");
        assert!(
            max_id >= listing.len(),
            "the added tokens sit past the model's ids"
        );
        for id in 0..=max_id + 1 {
            let expected = tokenizer.id_to_token(id);
            assert_eq!(
                token_of(&tokenizer, Some(&listing), id).as_deref(),
                expected.as_deref(),
                "id {id} with the listing"
            );
            assert_eq!(
                token_of(&tokenizer, None, id).as_deref(),
                expected.as_deref(),
                "id {id} without it"
            );
        }
        let listed = ByteLevelTable::build_with_max_id(&tokenizer, model_max_id, Some(&listing))
            .expect("a byte-level table");
        let swept = ByteLevelTable::build_with_max_id(&tokenizer, model_max_id, None)
            .expect("a byte-level table");
        for id in 0..=max_id + 1 {
            assert_eq!(listed.token_bytes(id), swept.token_bytes(id), "id {id}");
        }
    }

    fn alphabet() -> Vec<char> {
        let mut chars: Vec<(u8, char)> = CHAR_BYTES
            .iter()
            .enumerate()
            .filter_map(|(code, byte)| {
                byte.map(|b| (b, char::from_u32(code as u32).expect("alphabet char")))
            })
            .collect();
        chars.sort_unstable();
        chars.into_iter().map(|(_, c)| c).collect()
    }

    /// A byte-level BPE with the 256 byte tokens plus a few merges: "hello",
    /// "Ġworld" (space + word), "Ã©" (the two bytes of 'é'), the first two
    /// bytes of a 4-byte emoji, "Ġà" (a space plus the lead byte of a 3-byte
    /// character) and "ï¿½" (the three bytes of a literal U+FFFD), so
    /// characters get split across tokens in every way the fixtures show.
    pub(crate) fn byte_level_tokenizer() -> HfTokenizer {
        let mut vocab = Vocab::default();
        for (id, ch) in alphabet().into_iter().enumerate() {
            vocab.insert(ch.to_string(), id as u32);
        }
        let merge_pairs = [
            ("h", "e"),
            ("he", "l"),
            ("hel", "l"),
            ("hell", "o"),
            ("Ġ", "w"),
            ("Ġw", "o"),
            ("Ġwo", "r"),
            ("Ġwor", "l"),
            ("Ġworl", "d"),
            ("Ã", "©"),
            ("ð", "Ł"),
            ("Ġ", "à"),
            ("ï", "¿"),
            ("ï¿", "½"),
        ];
        let mut merges = Merges::new();
        for (a, b) in merge_pairs {
            let merged = format!("{a}{b}");
            let next = vocab.len() as u32;
            vocab.insert(merged, next);
            merges.push((a.to_string(), b.to_string()));
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .expect("bpe");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevelPreTokenizer::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        tokenizer
            .add_special_tokens([AddedToken::from(SPECIAL, true)])
            .expect("add special token");
        tokenizer
            .add_tokens([AddedToken::from(ADDED, false)])
            .expect("add token");
        tokenizer
    }

    fn stream(table: &Arc<ByteLevelTable>, ids: &[u32], skip: bool) -> (Vec<String>, Vec<u8>) {
        let mut dec = ByteLevelIncremental::new(Arc::clone(table), skip, UnknownId::Drop);
        let steps = ids.iter().map(|&id| dec.step(id).expect("step")).collect();
        (steps, dec.pending().to_vec())
    }

    /// The one-character byte-level token that stands for `byte`.
    pub(crate) fn byte_token(byte: u8) -> String {
        alphabet()
            .into_iter()
            .find(|&c| CHAR_BYTES[c as usize] == Some(byte))
            .expect("every byte has a char")
            .to_string()
    }

    /// The bytes `ids` stand for, in order, a skipped special token left out:
    /// what the full decode renders and what a stream's bytes add up to.
    pub(crate) fn bytes_of(tokenizer: &HfTokenizer, ids: &[u32], skip: bool) -> Vec<u8> {
        let added = tokenizer.get_added_tokens_decoder();
        ids.iter()
            .filter(|id| !(skip && added.get(id).is_some_and(|token| token.special)))
            .flat_map(|&id| {
                let token = tokenizer.id_to_token(id).expect("a vocabulary id");
                token
                    .chars()
                    .map(|c| CHAR_BYTES[c as usize].expect("a byte-level char"))
                    .collect::<Vec<u8>>()
            })
            .collect()
    }

    /// The reference: tokenizers' own incremental algorithm, step by step.
    fn hf_steps(tokenizer: &HfTokenizer, ids: &[u32], skip: bool) -> Vec<String> {
        let mut window = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0;
        ids.iter()
            .map(|&id| {
                tokenizers::tokenizer::step_decode_stream(
                    tokenizer,
                    vec![id],
                    skip,
                    &mut window,
                    &mut prefix,
                    &mut prefix_index,
                )
                .expect("hf step")
                .unwrap_or_default()
            })
            .collect()
    }

    #[test]
    fn cut_start_finds_the_character_the_end_cut() {
        // Complete text, an invalid sequence (a stray continuation byte, a
        // surrogate's bytes, a four-byte start a letter interrupts) and a
        // literal U+FFFD are kept whole: the full decode's rendering stands.
        let kept: [&[u8]; 8] = [
            b"",
            b"abc",
            "\u{E9}".as_bytes(),
            &[0xA9],
            &[0xA9, 0xA9],
            &[0xED, 0xA0],
            "\u{FFFD}".as_bytes(),
            &[0xF0, 0x9F, 0x41],
        ];
        for bytes in kept {
            assert_eq!(cut_start(bytes), bytes.len(), "{bytes:02X?}");
        }
        // The start of a character and nothing more is the cut, whatever
        // precedes it: text, a stray byte, a complete character, a literal
        // U+FFFD.
        let cut: [(&[u8], usize); 8] = [
            (&[0xC3], 0),
            (&[0x41, 0xC3], 1),
            (&[0xF0, 0x9F], 0),
            (&[0xF0, 0x9F, 0x98], 0),
            (&[0x20, 0xE0], 1),
            (&[0xA9, 0xF0, 0x9F], 1),
            (&[0xE0, 0xB5, 0x8D, 0xE0, 0xB4], 3),
            (&[0xEF, 0xBF, 0xBD, 0xE2, 0x80], 3),
        ];
        for (bytes, start) in cut {
            assert_eq!(cut_start(bytes), start, "{bytes:02X?}");
        }
    }

    #[test]
    fn alphabet_matches_tokenizers() {
        let ours: HashMap<char, u8> = alphabet()
            .into_iter()
            .map(|c| (c, CHAR_BYTES[c as usize].expect("mapped")))
            .collect();
        let theirs = ByteLevelPreTokenizer::alphabet();
        assert_eq!(ours.len(), 256);
        assert_eq!(theirs.len(), 256);
        for ch in theirs {
            assert!(ours.contains_key(&ch), "{ch:?} missing from our alphabet");
        }
        // Printable ASCII maps to itself; the first escaped byte (0) is U+0100.
        assert_eq!(ours[&'A'], b'A');
        assert_eq!(ours[&'\u{100}'], 0);
        assert_eq!(ours[&'\u{120}'], b' ');
    }

    #[test]
    fn whole_tokens_live_in_the_text_store_and_the_rest_in_the_byte_store() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("byte-level table");
        let id = |token: &str| tokenizer.token_to_id(token).expect(token);
        let is_text = |token: &str| {
            let (piece, _) = table.entry(id(token)).expect("in the vocabulary");
            matches!(piece, Piece::Text(_))
        };
        // Valid UTF-8 that does not end in U+FFFD: whole on its own.
        assert!(is_text("hello"));
        assert!(is_text("Ġworld"));
        assert!(is_text("Ã©"));
        assert!(is_text("a"));
        // A split character, a lead byte, a literal replacement character at
        // the end: not whole, kept as bytes for the pending window.
        assert!(!is_text("ðŁ"));
        assert!(!is_text("Ġà"));
        assert!(!is_text("ï¿½"));
        // Every piece reads back the bytes the token stands for, and the
        // decode's view of the same id reads the same bytes.
        for token in ["hello", "Ã©", "ðŁ", "ï¿½"] {
            let mut expected = Vec::new();
            push_token_bytes(token, &mut expected);
            let (piece, _) = table.entry(id(token)).expect("in the vocabulary");
            assert_eq!(piece.as_bytes(), expected.as_slice(), "{token}");
            let (bytes, _) = table.token_bytes(id(token)).expect("in the vocabulary");
            assert_eq!(bytes, expected.as_slice(), "{token}");
        }
        // A whole token is the step's text as it is, with nothing pending.
        let mut dec = ByteLevelIncremental::new(Arc::clone(&table), false, UnknownId::Drop);
        assert_eq!(dec.step(id("hello")).unwrap(), "hello");
        assert!(dec.pending().is_empty());
    }

    /// The index answers every byte string of its table with the lowest id
    /// that holds it and no other string, at every count of ids from none
    /// on (the powers of two among them): every fifth id is absent, an odd
    /// id repeats the bytes of the even id below it, and every other pair's
    /// bytes are not UTF-8. Some probe goes round the end of the slots.
    #[test]
    fn the_text_index_answers_every_byte_string_with_its_lowest_id() {
        let mut wrapped = false;
        for ids in 0..=64_u32 {
            let mut builder = ByteLevelTableBuilder::with_capacity(ids as usize);
            let mut lowest = HashMap::new();
            for id in 0..ids {
                if id % 5 == 4 {
                    builder.push_absent();
                    continue;
                }
                let pair = id / 2;
                let bytes = if pair % 2 == 0 {
                    format!("t{pair}").into_bytes()
                } else {
                    vec![0xFF, pair as u8]
                };
                builder.push(&bytes, false);
                lowest.entry(bytes).or_insert(id);
            }
            let table = builder.finish();
            let (index, _) = TextIndex::build(&*table);
            for (bytes, &id) in &lowest {
                assert_eq!(index.get(&*table, bytes), Some(id), "of {ids} ids");
                let at = index
                    .slots
                    .iter()
                    .position(|slot| slot & index.id_mask() == id);
                wrapped |= at.is_some_and(|at| at < index.place(bytes).0);
            }
            for miss in [&b""[..], b"t1", &[0xFF]] {
                assert_eq!(index.get(&*table, miss), None, "of {ids} ids");
            }
        }
        assert!(wrapped, "no probe went round the end of its slots");
    }

    #[test]
    fn only_byte_level_decoders_get_a_table() {
        let tokenizer = byte_level_tokenizer();
        assert!(ByteLevelTable::build(&tokenizer).is_some());
        let mut metaspace = byte_level_tokenizer();
        metaspace.with_decoder(Some(Metaspace::default()));
        assert!(ByteLevelTable::build(&metaspace).is_none());
        let mut none = byte_level_tokenizer();
        none.with_decoder(None::<ByteLevelDecoder>);
        assert!(ByteLevelTable::build(&none).is_none());
    }

    #[test]
    fn text_streams_match_tokenizers_step_for_step() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let texts = [
            "hello world",
            "héllo wörld, naïve café",
            "emoji 😀 and 👍🏽 and ✨",
            "日本語のテキストと한국어",
            "tabs\tand\nnewlines\r\n",
            &format!("hello{SPECIAL} world {ADDED} done{SPECIAL}"),
            "",
        ];
        for text in texts {
            let ids = tokenizer
                .encode(text, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            for skip in [false, true] {
                let (steps, pending) = stream(&table, &ids, skip);
                assert!(pending.is_empty(), "{text:?}: pending {pending:?}");
                assert_eq!(
                    steps,
                    hf_steps(&tokenizer, &ids, skip),
                    "{text:?} skip={skip}"
                );
                assert_eq!(
                    steps.concat(),
                    tokenizer.decode(&ids, skip).expect("decode"),
                    "{text:?} skip={skip}"
                );
            }
        }
    }

    #[test]
    fn arbitrary_id_streams_match_full_decode() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let vocab_len = tokenizer.get_vocab_size(true) as u32;
        let mut state = 0x9E37_79B9u32;
        for round in 0..400 {
            let len = 1 + round % 40;
            let ids: Vec<u32> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    // Mostly real ids, a few beyond the vocabulary.
                    if state.is_multiple_of(50) {
                        vocab_len + state % 7
                    } else {
                        state % vocab_len
                    }
                })
                .collect();
            for skip in [false, true] {
                let (steps, pending) = stream(&table, &ids, skip);
                let full = tokenizer.decode(&ids, skip).expect("decode");
                let ours = steps.concat() + &String::from_utf8_lossy(&pending);
                assert_eq!(ours, full, "ids={ids:?} skip={skip}");
                // The same piece at the same token as tokenizers' algorithm,
                // including what it withholds while its window's lossy decode
                // ends in U+FFFD.
                assert_eq!(
                    steps,
                    hf_steps(&tokenizer, &ids, skip),
                    "ids={ids:?} skip={skip}"
                );
            }
        }
    }

    #[test]
    fn full_decode_matches_hf_at_byte_boundaries() {
        let mut tokenizer = byte_level_tokenizer();
        // Outside the byte alphabet: the entire added token must stay UTF-8.
        tokenizer
            .add_tokens([AddedToken::from("工具Ġ", false)])
            .expect("add Unicode token");
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let smg = crate::HuggingFaceTokenizer::from_tokenizer(tokenizer.clone());
        let special = tokenizer.token_to_id(SPECIAL).expect("special");
        let unicode = tokenizer.token_to_id("工具Ġ").expect("Unicode");
        for a in 0..256 {
            for b in 0..256 {
                for skip in [false, true] {
                    let ids = [a, special, b, u32::MAX, unicode];
                    let expected = tokenizer.decode(&ids, skip).expect("HF decode");
                    assert_eq!(table.decode(&ids, skip), expected, "{ids:?} {skip}");
                    assert_eq!(
                        Decoder::decode(&smg, &ids, skip).expect("SMG decode"),
                        expected
                    );
                }
            }
        }
        for ids in [vec![], vec![u32::MAX], vec![0xF0, 0x9F], vec![0xFF]] {
            assert_eq!(
                table.decode(&ids, false),
                tokenizer.decode(&ids, false).unwrap()
            );
        }
    }

    /// The sizing pass of a short decode applies the filter of both
    /// concatenations it sizes (unknown ids dropped, specials dropped when
    /// skipped), so the buffer is exactly the text: a drift between the pass
    /// and either loop shows up here as a regrow, which the decoded bytes
    /// alone never show.
    #[test]
    fn short_decode_is_sized_from_its_pieces() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let hello = tokenizer.token_to_id("hello").expect("hello");
        let world = tokenizer.token_to_id("Ġworld").expect("Ġworld");
        let special = tokenizer.token_to_id(SPECIAL).expect("special");
        for skip in [false, true] {
            // A known word, a special, an unknown id, a word with its space.
            let text = table.decode(&[hello, special, u32::MAX, world], skip);
            assert_eq!(text.capacity(), text.len(), "skip={skip}");
            // The tiktoken backend's decode: every id known, or it is `None`.
            let (text, _) = table
                .decode_known(&[hello, special, world], skip)
                .expect("every id known");
            assert_eq!(text.capacity(), text.len(), "decode_known skip={skip}");
        }
    }

    #[test]
    fn full_decode_drops_sparse_vocabulary_holes() {
        let model = BPE::builder()
            .vocab_and_merges(
                Vocab::from_iter([("a".to_owned(), 0), ("b".to_owned(), 2)]),
                vec![],
            )
            .build()
            .expect("sparse BPE");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        let smg = crate::HuggingFaceTokenizer::from_tokenizer(tokenizer.clone());
        for skip in [false, true] {
            let ids = [0, 1, 2, u32::MAX];
            assert_eq!(
                Decoder::decode(&smg, &ids, skip).unwrap(),
                tokenizer.decode(&ids, skip).unwrap()
            );
        }
    }

    #[test]
    fn full_decode_preserves_non_byte_level_decoder() {
        let mut tokenizer = byte_level_tokenizer();
        tokenizer.with_decoder(Some(Metaspace::default()));
        let smg = crate::HuggingFaceTokenizer::from_tokenizer(tokenizer.clone());
        let ids = [b'h' as u32, b'e' as u32, u32::MAX];
        assert_eq!(
            Decoder::decode(&smg, &ids, false).unwrap(),
            tokenizer.decode(&ids, false).unwrap()
        );
    }

    #[test]
    fn split_character_is_withheld_until_complete() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        // 'é' is C3 A9; its byte tokens are 'Ã' and '©'.
        let c3 = tokenizer.token_to_id("Ã").expect("C3");
        let a9 = tokenizer.token_to_id("©").expect("A9");
        let mut dec = ByteLevelIncremental::new(table, false, UnknownId::Drop);
        assert_eq!(dec.step(c3).expect("step"), "");
        assert_eq!(dec.pending(), [0xC3]);
        assert_eq!(dec.step(a9).expect("step"), "é");
        assert!(dec.pending().is_empty());
        // A lone continuation byte decodes to U+FFFD, which the reference keeps
        // back until settled text follows it; then both come out together.
        assert_eq!(dec.step(a9).expect("step"), "");
        assert_eq!(dec.pending(), [0xA9]);
        let h = tokenizer.token_to_id("h").expect("h");
        assert_eq!(dec.step(h).expect("step"), "\u{FFFD}h");
        assert!(dec.pending().is_empty());
        dec.step(c3).expect("step");
        dec.reset();
        assert!(dec.pending().is_empty());
    }

    /// A character the end of the stream cut short, two, three and four
    /// bytes long, is dropped; a lone invalid continuation byte is rendered.
    #[test]
    fn the_end_drops_cut_characters_and_renders_invalid_bytes() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let id = |token: &str| tokenizer.token_to_id(token).expect("token");
        let h = id("h");
        let cases: [(Vec<u32>, &str); 4] = [
            (vec![h, id("Ã")], "h"),
            (vec![h, id("Ġà"), id("¤")], "h "),
            (vec![h, id("ðŁ"), id(&byte_token(0x98))], "h"),
            (vec![h, id("©")], "h\u{FFFD}"),
        ];
        for (ids, text) in cases {
            let mut dec = ByteLevelIncremental::new(Arc::clone(&table), false, UnknownId::Drop);
            let mut out = String::new();
            for &id in &ids {
                dec.step_into(id, &mut out).expect("step");
            }
            assert_eq!(out, "h", "{ids:?}: the cut bytes are held");
            assert_eq!(dec.finish_into(&mut out).expect("finish"), text.len() > 1);
            assert_eq!(out, text, "{ids:?}");
            assert!(dec.pending().is_empty());
            assert!(!dec.finish_into(&mut out).expect("a second finish"));
            assert_eq!(out, text);
        }
    }

    #[test]
    fn the_end_drops_a_cut_character_and_keeps_complete_text() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let id = |token: &str| tokenizer.token_to_id(token).expect("token");
        let mut dec = ByteLevelIncremental::new(table, false, UnknownId::Drop);
        let mut out = String::new();

        assert!(dec.step_into(id("h"), &mut out).expect("step"));
        assert!(!dec.step_into(id("Ġà"), &mut out).expect("step"));
        assert_eq!(out, "h");
        assert!(dec.finish_into(&mut out).expect("finish"));
        assert_eq!(out, "h ");
        assert!(!out.contains(char::REPLACEMENT_CHARACTER));
    }

    /// The three piece-timing shapes the reference fixtures exercise, each
    /// checked against tokenizers' own step algorithm: settled text ahead of
    /// a partial character in the same token, a literal U+FFFD token, and an
    /// invalid byte ahead of text.
    #[test]
    fn withholds_exactly_what_the_reference_withholds() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let id = |token: &str| tokenizer.token_to_id(token).expect("token");
        // " " + E0, then A4, then BE: " ा" is emitted whole on the last byte.
        let space_lead = id("Ġà");
        let (a4, be) = (id("¤"), id("¾"));
        // EF BF BD as one token: a literal U+FFFD, held until text follows.
        let fffd = id("ï¿½");
        let h = id("h");
        let cases: [&[u32]; 4] = [
            &[space_lead, a4, be, h],
            &[fffd, h],
            &[h, fffd, fffd, h],
            &[a4, space_lead, a4, be],
        ];
        for ids in cases {
            for skip in [false, true] {
                let (steps, pending) = stream(&table, ids, skip);
                assert_eq!(
                    steps,
                    hf_steps(&tokenizer, ids, skip),
                    "ids={ids:?} skip={skip}"
                );
                let ours = steps.concat() + &String::from_utf8_lossy(&pending);
                assert_eq!(ours, tokenizer.decode(ids, skip).expect("decode"));
            }
        }
        let (steps, _) = stream(&table, &[space_lead, a4, be, h], false);
        assert_eq!(steps, ["", "", " ा", "h"]);
        let (steps, _) = stream(&table, &[fffd, h], false);
        assert_eq!(steps, ["", "\u{FFFD}h"]);
    }

    #[test]
    fn unknown_ids_are_dropped_or_rejected_as_asked() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let beyond = tokenizer.get_vocab_size(true) as u32 + 3;
        let h = tokenizer.token_to_id("h").expect("h");
        let mut drop = ByteLevelIncremental::new(Arc::clone(&table), false, UnknownId::Drop);
        assert_eq!(drop.step(beyond).expect("dropped"), "");
        assert_eq!(drop.step(h).expect("step"), "h");
        let mut reject = ByteLevelIncremental::new(table, false, UnknownId::Reject);
        assert_eq!(reject.step(h).expect("step"), "h");
        let err = reject.step(beyond).expect_err("rejected");
        assert!(err.to_string().contains("unknown token id"), "{err}");
    }

    #[test]
    fn special_tokens_follow_the_skip_flag() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let special = tokenizer.token_to_id(SPECIAL).expect("special id");
        let added = tokenizer.token_to_id(ADDED).expect("added id");
        let mut keep = ByteLevelIncremental::new(Arc::clone(&table), false, UnknownId::Drop);
        assert_eq!(keep.step(special).expect("step"), SPECIAL);
        assert_eq!(keep.step(added).expect("step"), ADDED);
        let mut skip = ByteLevelIncremental::new(table, true, UnknownId::Drop);
        assert_eq!(skip.step(special).expect("step"), "");
        assert_eq!(skip.step(added).expect("step"), ADDED);
    }

    /// `step_into` appends what `step` returns, token for token, across the
    /// characters the fixtures split over tokens, and leaves the same pending bytes.
    #[test]
    fn step_into_appends_the_steps_text() {
        let tokenizer = byte_level_tokenizer();
        let table = ByteLevelTable::build(&tokenizer).expect("table");
        let texts = [
            "hello world",
            "café naïve",
            "日本語のテキストと한국어",
            "emoji 😀👍🏽",
            &format!("hello{SPECIAL} world {ADDED} done{SPECIAL}"),
        ];
        for text in texts {
            let ids = tokenizer
                .encode(text, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            for skip in [false, true] {
                let (steps, pending) = stream(&table, &ids, skip);
                let mut into = ByteLevelIncremental::new(Arc::clone(&table), skip, UnknownId::Drop);
                let mut out = String::new();
                for (&id, step) in ids.iter().zip(&steps) {
                    let before = out.len();
                    let appended = into.step_into(id, &mut out).expect("step_into");
                    assert_eq!(&out[before..], step, "{text:?} skip={skip} token {id}");
                    assert_eq!(
                        appended,
                        !step.is_empty(),
                        "{text:?} skip={skip} token {id}"
                    );
                }
                assert_eq!(out, steps.concat(), "{text:?} skip={skip}");
                assert_eq!(into.pending(), pending.as_slice(), "{text:?} skip={skip}");
            }
        }
    }
}
