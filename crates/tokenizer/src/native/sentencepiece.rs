//! A sentencepiece-style vocabulary with byte fallback on the native encode
//! path (Gemma, Llama 2 and their kin). Such a vocabulary has no `ByteLevel`
//! stage: the spaces of the text become the word marker U+2581 (a `Replace`
//! normalizer, alone or after a `Prepend` normalizer that puts the marker
//! before every text between added tokens, or the `Metaspace`
//! pre-tokenizer of the Llama class, which also prepends the marker by its
//! scheme), the text between two added tokens is one word of the model,
//! and the model's `merge_word` makes a symbol of each character the
//! vocabulary has and of each byte (`<0xHH>`) of one it has not, then
//! merges. This module gives that word pieces: a cut
//! between two adjacent bytes that no vocabulary token holds adjacent can
//! never be crossed by a merge (a merge's product is a vocabulary token, and
//! it would hold the pair), so the pieces merge on their own to the ids of
//! the whole, and the piece cache, the seed and the merge table of the
//! byte-level path serve them. The byte tokens take no part in any merge (a
//! vocabulary with a token that spells one inside it is declined), so the
//! cuts are read from the bytes of the tokens' own text. The scan runs over
//! the raw text, a space standing for the marker (the bridge table is built
//! in that form), and a piece is keyed in the caches by its raw bytes, so a
//! hit reads the text once and copies nothing; a piece met for the first
//! time is written in the vocabulary's form for the merge alone.

use rustc_hash::FxHashMap;
use tokenizers::models::ModelWrapper;

use super::{
    merge::{ByteTokens, MergeTable},
    short_pieces::ShortPieceKey,
};
use crate::{
    encode_path::Declined,
    word_cache::{entries_of, ReadBpe},
};

/// The word marker of a sentencepiece vocabulary.
pub(super) const WORD_MARKER: char = '\u{2581}';
/// The marker's UTF-8 bytes.
const MARKER_BYTES: [u8; 3] = [0xE2, 0x96, 0x81];
/// No token for the character.
const NO_ID: u32 = u32::MAX;
/// Words of the bridge table: one bit per pair of bytes.
const BRIDGE_WORDS: usize = 256 * 256 / 64;
/// Symbols the thread's scratch list keeps between merges, 16 KiB: a piece
/// longer than this (a space-free text no vocabulary token cuts can be the
/// whole unit) hands its list back after the merge, so a thread holds this
/// much scratch, not the longest piece it ever met.
pub(super) const MERGE_SCRATCH_SYMBOLS: usize = 4_096;

/// Where the word marker is prepended: the `Metaspace` prepend scheme
/// (`Never` for a `Replace` normalizer too, which prepends nothing), or
/// the `Prepend` normalizer (`Every`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Marker {
    Never,
    /// Before the text that starts the input (byte 0), as `Metaspace` reads
    /// it from the split's original offset.
    First,
    /// Before every text between added tokens.
    Always,
    /// Before every text between added tokens, one that starts with a space
    /// or the marker too: the `Prepend` normalizer, which prepends to the
    /// whole normalized text where `Metaspace` prepends only to one that
    /// does not start with the marker.
    Every,
}

/// The symbols a piece of a byte-fallback vocabulary starts from, where its
/// text may be cut, and the merges; see the module documentation.
pub(super) struct CharSymbols {
    /// The token of each ASCII character, [`NO_ID`] for none.
    ascii: [u32; 128],
    /// The token of each other character the vocabulary has, by scalar value.
    beyond: FxHashMap<u32, u32>,
    /// Bit `prev << 8 | cur`: no piece cut goes between the raw bytes
    /// `prev` and `cur`: some vocabulary token holds them adjacent (a space
    /// read as the marker, its last byte before the cut and its first byte
    /// after it), or `cur` continues a UTF-8 character. Read from the
    /// tokens' text, then rewritten for the raw text by
    /// [`Self::bridge_raw_text`].
    bridge: Box<[u64; BRIDGE_WORDS]>,
    marker: Marker,
    /// The model's merges over its own ids.
    merges: MergeTable,
}

impl CharSymbols {
    /// The tables of `model`'s vocabulary (a BPE with byte fallback), or why
    /// the vocabulary declines: `ignore_merges` (it looks the whole text up,
    /// which pieces cannot), merges the table cannot stand in for (the
    /// table's own declines, with the `<0xHH>` spelling of the byte tokens),
    /// a byte token missing for a byte the fallback could reach (one beyond
    /// ASCII, which may lie inside any character that is no token, or an
    /// ASCII one whose character is no token; an ASCII character that is a
    /// token never falls back, so its byte's token is not needed), a token
    /// that spells a byte token inside it (a merge could then cross a cut).
    ///
    /// `read` is what the file's one-pass read saw of the model: the merge
    /// table takes the merges from it, and the entries are read by id when
    /// it says the ids are the range of the vocabulary (else through the
    /// vocabulary map, as without a read); `None` for a model built in
    /// memory.
    pub(super) fn from_model(
        model: &ModelWrapper,
        marker: Marker,
        read: Option<&ReadBpe>,
    ) -> Result<Self, Declined> {
        let ModelWrapper::BPE(bpe) = model else {
            return Err(Declined::from("the model is not BPE"));
        };
        if bpe.ignore_merges {
            return Err(Declined::from(
                "a byte-fallback vocabulary with ignore_merges (the whole text is looked up)",
            ));
        }
        let merges = MergeTable::from_model(model, ByteTokens::Hex, read)?;
        let mut symbols = Self {
            ascii: [NO_ID; 128],
            beyond: FxHashMap::default(),
            bridge: Box::new([0; BRIDGE_WORDS]),
            marker,
            merges,
        };
        for (id, token) in entries_of(bpe, read) {
            symbols.read_token(token.as_ref(), id)?;
        }
        // A byte without a token: the fallback spells the bytes of a
        // character that is no token, so an ASCII byte whose character is
        // one is never reached (a character beyond ASCII holds no ASCII
        // byte); any other missing byte could be, and declines.
        for byte in 0..=255u8 {
            if symbols.merges.lacks_a_token_for_byte(byte)
                && !(byte.is_ascii() && symbols.ascii[usize::from(byte)] != NO_ID)
            {
                return Err(Declined::from(format!("byte {byte:#04x} has no token")));
            }
        }
        symbols.bridge_raw_text();
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

    /// Rewrites the bridge table, read from the tokens' text, for the raw
    /// text the scan runs over: a space is the marker (what precedes a cut
    /// after a space is the marker's last byte, what follows a cut before
    /// one is its first byte), and no cut goes before a byte that continues
    /// a character.
    fn bridge_raw_text(&mut self) {
        let mut raw = Box::new([0u64; BRIDGE_WORDS]);
        for prev in 0..=255u8 {
            for cur in 0..=255u8 {
                let before = if prev == b' ' { MARKER_BYTES[2] } else { prev };
                let after = if cur == b' ' { MARKER_BYTES[0] } else { cur };
                if is_continuation(cur) || self.bridged(before, after) {
                    let bit = usize::from(prev) << 8 | usize::from(cur);
                    raw[bit >> 6] |= 1 << (bit & 63);
                }
            }
        }
        self.bridge = raw;
    }

    /// The cache key of a vocabulary entry on this path: its text in the raw
    /// form (the marker as a space), when that is a short piece. An entry
    /// that holds a space of its own has no key: no text reaches it (a space
    /// of the text is the marker), and its raw form would be another entry's.
    pub(super) fn entry_key(entry: &str) -> Option<ShortPieceKey> {
        if entry.contains(' ') {
            return None;
        }
        if entry.contains(WORD_MARKER) {
            ShortPieceKey::of(entry.replace(WORD_MARKER, " ").as_bytes())
        } else {
            ShortPieceKey::of(entry.as_bytes())
        }
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

    /// Bytes the vocabulary has no token for (none the fallback reaches,
    /// see [`Self::from_model`]), for the load log.
    pub(super) fn bytes_without_a_token(&self) -> usize {
        self.merges.missing_bytes()
    }

    /// Whether the word marker goes before a text: one that starts a text
    /// between added tokens (`starts_text`), under the `Always` scheme and
    /// the `Prepend` normalizer (`Every`); the one at the start of the
    /// input (`at_input_start`), under `First`.
    pub(super) fn prepends(&self, starts_text: bool, at_input_start: bool) -> bool {
        match self.marker {
            Marker::Never => false,
            Marker::Always | Marker::Every => starts_text,
            Marker::First => at_input_start,
        }
    }

    /// Whether the marker goes before a text that already starts with a
    /// space or the marker (the `Prepend` normalizer puts it there;
    /// `Metaspace` does not).
    pub(super) fn prepends_to_a_marked_text(&self) -> bool {
        self.marker == Marker::Every
    }

    /// Whether no cut goes between the raw bytes `prev` and `cur` (see the
    /// field).
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

    /// `piece` (raw text) in the vocabulary's form, written into `out`: its
    /// spaces as the word marker.
    pub(super) fn normalized<'o>(piece: &str, out: &'o mut String) -> &'o str {
        out.clear();
        let mut parts = piece.split(' ');
        out.push_str(parts.next().unwrap_or_default());
        for part in parts {
            out.push(WORD_MARKER);
            out.push_str(part);
        }
        out
    }

    /// Calls `f` with the start and end of each piece of the raw `text`:
    /// the text cut between two bytes it does not bridge (see the module
    /// documentation; a space reads as the marker). `f` is taken by value so
    /// that the per-piece work compiles into the loop.
    #[inline(always)]
    pub(super) fn pieces<F: FnMut(usize, usize)>(&self, text: &[u8], mut f: F) {
        let mut start = 0;
        for (i, pair) in text.windows(2).enumerate() {
            if !self.bridged(pair[0], pair[1]) {
                f(start, i + 1);
                start = i + 1;
            }
        }
        if start < text.len() {
            f(start, text.len());
        }
    }

    /// Whether the raw text is cut at `p` (past its first byte, before its
    /// end): the bytes either side of it are not bridged.
    pub(super) fn cuts_at(&self, bytes: &[u8], p: usize) -> bool {
        !self.bridged(bytes[p - 1], bytes[p])
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
                // The construction declines a missing byte the fallback
                // could reach (see `from_model`): one spelled here has a
                // token.
                debug_assert!(
                    !self.merges.lacks_a_token_for_byte(byte),
                    "a byte without a token reached the fallback: {byte:#04x} in {c:?}"
                );
                symbols.push(self.merges.byte_id(byte));
            }
        }
        self.merges.merge_symbols(symbols, ids);
        if symbols.capacity() > MERGE_SCRATCH_SYMBOLS {
            symbols.clear();
            symbols.shrink_to(MERGE_SCRATCH_SYMBOLS);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_is_keyed_by_its_raw_form_and_one_with_a_space_of_its_own_has_none() {
        assert_eq!(
            CharSymbols::entry_key("\u{2581}the"),
            ShortPieceKey::of(b" the")
        );
        assert_eq!(
            CharSymbols::entry_key("\u{2581}the\u{2581}"),
            ShortPieceKey::of(b" the ")
        );
        assert_eq!(CharSymbols::entry_key("the"), ShortPieceKey::of(b"the"));
        assert_eq!(CharSymbols::entry_key(" the"), None);
        assert_eq!(CharSymbols::entry_key(&"\u{2581}".repeat(16)), None);
        assert!(CharSymbols::entry_key(&"\u{2581}".repeat(15)).is_some());
    }

    #[test]
    fn a_raw_piece_is_written_in_the_vocabularys_form() {
        let mut out = String::new();
        assert_eq!(
            CharSymbols::normalized(" the cat", &mut out),
            "\u{2581}the\u{2581}cat"
        );
        assert_eq!(CharSymbols::normalized("x", &mut out), "x");
        assert_eq!(CharSymbols::normalized("", &mut out), "");
        assert_eq!(CharSymbols::normalized("  ", &mut out), "\u{2581}\u{2581}");
        assert_eq!(
            CharSymbols::normalized("\u{2581}y z", &mut out),
            "\u{2581}y\u{2581}z"
        );
    }
}
