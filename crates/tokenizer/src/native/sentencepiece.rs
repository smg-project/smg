//! A sentencepiece-style vocabulary with byte fallback on the native encode
//! path (Gemma, Llama 2 and their kin). Such a vocabulary has no `ByteLevel`
//! stage: the spaces of the text become the word marker U+2581 (a `Replace`
//! normalizer, or the `Metaspace` pre-tokenizer of the Llama class, which
//! also prepends the marker by its scheme), the text between two added
//! tokens is one word of the model, and the model's `merge_word` makes a
//! symbol of each character the vocabulary has and of each byte (`<0xHH>`)
//! of one it has not, then merges. This module gives that word pieces: a cut
//! between two adjacent bytes that no vocabulary token holds adjacent can
//! never be crossed by a merge (a merge's product is a vocabulary token, and
//! it would hold the pair), so the pieces merge on their own to the ids of
//! the whole, and the piece cache, the seed and the merge table of the
//! byte-level path serve them. The byte tokens take no part in any merge (a
//! vocabulary with a token that spells one inside it is declined), so the
//! cuts are read from the bytes of the tokens' own text.

use std::borrow::Cow;

use rustc_hash::FxHashMap;
use tokenizers::models::ModelWrapper;

use super::merge::{ByteTokens, MergeTable};
use crate::encode_path::Declined;

/// The word marker of a sentencepiece vocabulary.
pub(super) const WORD_MARKER: char = '\u{2581}';
/// The marker's UTF-8 bytes.
const MARKER_BYTES: [u8; 3] = [0xE2, 0x96, 0x81];
/// No token for the character.
const NO_ID: u32 = u32::MAX;
/// Words of the bridge table: one bit per pair of bytes.
const BRIDGE_WORDS: usize = 256 * 256 / 64;

/// Where the word marker is prepended: the `Metaspace` prepend scheme
/// (`Never` for a `Replace` normalizer too, which prepends nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Marker {
    Never,
    /// Before the text that starts the input (byte 0), as `Metaspace` reads
    /// it from the split's original offset.
    First,
    /// Before every text between added tokens.
    Always,
}

/// The symbols a piece of a byte-fallback vocabulary starts from, where its
/// text may be cut, and the merges; see the module documentation.
pub(super) struct CharSymbols {
    /// The token of each ASCII character, [`NO_ID`] for none.
    ascii: [u32; 128],
    /// The token of each other character the vocabulary has, by scalar value.
    beyond: FxHashMap<u32, u32>,
    /// Bit `prev << 8 | cur`: some vocabulary token holds the bytes `prev`,
    /// `cur` adjacent.
    bridge: Box<[u64; BRIDGE_WORDS]>,
    marker: Marker,
    /// The model's merges over its own ids.
    merges: MergeTable,
}

impl CharSymbols {
    /// The tables of `model`'s vocabulary (a BPE with byte fallback), or why
    /// the vocabulary declines: `ignore_merges` (it looks the whole text up,
    /// which pieces cannot), a byte token missing or merges the table cannot
    /// stand in for (the table's own declines, with the `<0xHH>` spelling of
    /// the byte tokens), a token that spells a byte token inside it (a merge
    /// could then cross a cut).
    pub(super) fn from_model(model: &ModelWrapper, marker: Marker) -> Result<Self, Declined> {
        let ModelWrapper::BPE(bpe) = model else {
            return Err(Declined::from("the model is not BPE"));
        };
        if bpe.ignore_merges {
            return Err(Declined::from(
                "a byte-fallback vocabulary with ignore_merges (the whole text is looked up)",
            ));
        }
        let merges = MergeTable::from_model(model, ByteTokens::Hex)?;
        let mut symbols = Self {
            ascii: [NO_ID; 128],
            beyond: FxHashMap::default(),
            bridge: Box::new([0; BRIDGE_WORDS]),
            marker,
            merges,
        };
        for (token, id) in bpe.get_vocab() {
            symbols.read_token(&token, id)?;
        }
        Ok(symbols)
    }

    /// Reads one vocabulary token: the pairs of bytes it holds adjacent, and
    /// the character it stands for when it is one.
    fn read_token(&mut self, token: &str, id: u32) -> Result<(), Declined> {
        if token.contains("<0x") && !is_byte_token(token) {
            return Err(Declined::from(format!(
                "the token {token:?} spells a byte token inside it"
            )));
        }
        for pair in token.as_bytes().windows(2) {
            let bit = usize::from(pair[0]) << 8 | usize::from(pair[1]);
            self.bridge[bit >> 6] |= 1 << (bit & 63);
        }
        let mut chars = token.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            let code = u32::from(c);
            if code < 128 {
                self.ascii[code as usize] = id;
            } else {
                self.beyond.insert(code, id);
            }
        }
        Ok(())
    }

    /// Bytes the tables hold, for the load log.
    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.beyond.len() * size_of::<(u32, u32)>()
            + size_of_val(&*self.bridge)
            + self.merges.bytes()
    }

    /// Merges the table holds.
    pub(super) fn merges(&self) -> usize {
        self.merges.merges()
    }

    /// Whether the word marker goes before a text: one that starts a text
    /// between added tokens (`starts_text`), under the `Always` scheme; the
    /// one at the start of the input (`at_input_start`), under `First`.
    pub(super) fn prepends(&self, starts_text: bool, at_input_start: bool) -> bool {
        match self.marker {
            Marker::Never => false,
            Marker::Always => starts_text,
            Marker::First => at_input_start,
        }
    }

    /// Whether some vocabulary token holds `prev` and `cur` adjacent.
    #[inline(always)]
    fn bridged(&self, prev: u8, cur: u8) -> bool {
        let bit = usize::from(prev) << 8 | usize::from(cur);
        (self.bridge[bit >> 6] >> (bit & 63)) & 1 != 0
    }

    /// The token of `c`, or [`NO_ID`].
    #[inline(always)]
    fn char_id(&self, c: char) -> u32 {
        let code = u32::from(c);
        if code < 128 {
            self.ascii[code as usize]
        } else {
            self.beyond.get(&code).copied().unwrap_or(NO_ID)
        }
    }

    /// `text` with its spaces as word markers, and the marker before it when
    /// `prepend` says so and it does not start with one already (a space at
    /// its start is one, replaced); borrowed when nothing changes.
    pub(super) fn marked(text: &str, prepend: bool) -> Cow<'_, str> {
        let prepend = prepend && !(text.starts_with(WORD_MARKER) || text.starts_with(' '));
        if !prepend && !text.as_bytes().contains(&b' ') {
            return Cow::Borrowed(text);
        }
        let mut out = String::with_capacity(text.len() + text.len() / 2 + 3);
        if prepend {
            out.push(WORD_MARKER);
        }
        for c in text.chars() {
            out.push(if c == ' ' { WORD_MARKER } else { c });
        }
        Cow::Owned(out)
    }

    /// Calls `f` with each piece of `text` (normalized): the text cut, at
    /// character boundaries, between two bytes no vocabulary token holds
    /// adjacent.
    pub(super) fn pieces(&self, text: &str, f: &mut dyn FnMut(&str)) {
        let bytes = text.as_bytes();
        let mut start = 0;
        for i in 1..bytes.len() {
            let cur = bytes[i];
            if is_continuation(cur) || self.bridged(bytes[i - 1], cur) {
                continue;
            }
            f(&text[start..i]);
            start = i;
        }
        if start < bytes.len() {
            f(&text[start..]);
        }
    }

    /// Whether a cut of the raw text at `p` (past its first byte, before
    /// its last) is a cut of its normalized form too: the spaces either side
    /// of it read as the marker's bytes.
    pub(super) fn raw_cut_ok(&self, bytes: &[u8], p: usize) -> bool {
        let prev = if bytes[p - 1] == b' ' {
            MARKER_BYTES[2]
        } else {
            bytes[p - 1]
        };
        let cur = if bytes[p] == b' ' {
            MARKER_BYTES[0]
        } else {
            bytes[p]
        };
        !is_continuation(cur) && !self.bridged(prev, cur)
    }

    /// Appends the ids `piece` (normalized text) merges to: its initial
    /// symbols as the model's `merge_word` makes them, a character's token,
    /// else its bytes' tokens, written into `symbols` (a scratch list of the
    /// thread), then merged over the table.
    pub(super) fn merge(&self, piece: &str, symbols: &mut Vec<u32>, ids: &mut Vec<u32>) {
        symbols.clear();
        for c in piece.chars() {
            let id = self.char_id(c);
            if id != NO_ID {
                symbols.push(id);
                continue;
            }
            let mut utf8 = [0u8; 4];
            for &byte in c.encode_utf8(&mut utf8).as_bytes() {
                symbols.push(self.merges.byte_id(byte));
            }
        }
        self.merges.merge_symbols(symbols, ids);
    }
}

/// Whether `token` is one of the 256 `<0xHH>` byte tokens.
fn is_byte_token(token: &str) -> bool {
    token.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && token[3..5]
            .bytes()
            .all(|digit| digit.is_ascii_digit() || (b'A'..=b'F').contains(&digit))
}

/// Whether `byte` continues a UTF-8 character (no cut goes before it).
#[inline(always)]
fn is_continuation(byte: u8) -> bool {
    (0x80..0xC0).contains(&byte)
}
