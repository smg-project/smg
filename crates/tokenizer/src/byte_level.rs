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
//! reference would emit: the same pieces at the same tokens, and over a
//! whole stream exactly `decode(all_ids)`. The pending buffer holds at most
//! the bytes since the last emission, normally one partial character.

use std::{
    borrow::Cow,
    sync::{Arc, LazyLock},
};

use anyhow::{anyhow, Result};
use tokenizers::{decoders::DecoderWrapper, Tokenizer as HfTokenizer};

use crate::traits::{IncrementalDecoder, TokenIdType};

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

/// What the table knows about an id.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Not in the vocabulary.
    Absent,
    /// A token kept in decoded text.
    Ordinary,
    /// A token `skip_special_tokens` strips.
    Special,
}

/// Per-tokenizer table of the bytes each id decodes to.
///
/// Built once at load time: by [`ByteLevelTable::build`] for a `tokenizers`
/// vocabulary whose decoder is exactly `ByteLevel`, through
/// [`ByteLevelTableBuilder`] for a tiktoken vocabulary, whose ranks are byte
/// strings already. About 9 bytes per id plus the token bytes themselves.
pub(crate) struct ByteLevelTable {
    /// `offsets[id]..offsets[id + 1]` indexes `bytes` for token `id`.
    offsets: Vec<usize>,
    bytes: Vec<u8>,
    kind: Vec<Kind>,
}

/// Builds a [`ByteLevelTable`] one id at a time, from 0 up.
pub(crate) struct ByteLevelTableBuilder {
    offsets: Vec<usize>,
    bytes: Vec<u8>,
    kind: Vec<Kind>,
}

impl ByteLevelTableBuilder {
    pub(crate) fn with_capacity(ids: usize) -> Self {
        let mut offsets = Vec::with_capacity(ids + 1);
        offsets.push(0);
        Self {
            offsets,
            bytes: Vec::with_capacity(ids * 4),
            kind: Vec::with_capacity(ids),
        }
    }

    /// The next id decodes to `bytes`; `special` marks a token that
    /// `skip_special_tokens` strips.
    pub(crate) fn push(&mut self, bytes: &[u8], special: bool) {
        self.bytes.extend_from_slice(bytes);
        self.offsets.push(self.bytes.len());
        self.kind.push(if special {
            Kind::Special
        } else {
            Kind::Ordinary
        });
    }

    /// The next id is not in the vocabulary.
    pub(crate) fn push_absent(&mut self) {
        self.offsets.push(self.bytes.len());
        self.kind.push(Kind::Absent);
    }

    pub(crate) fn finish(mut self) -> Arc<ByteLevelTable> {
        self.bytes.shrink_to_fit();
        Arc::new(ByteLevelTable {
            offsets: self.offsets,
            bytes: self.bytes,
            kind: self.kind,
        })
    }
}

impl ByteLevelTable {
    /// `None` unless the tokenizer decodes with a plain `ByteLevel` decoder.
    pub(crate) fn build(tokenizer: &HfTokenizer) -> Option<Arc<Self>> {
        if !matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_))) {
            return None;
        }
        let max_id = tokenizer.get_vocab(true).values().copied().max()?;
        let count = max_id as usize + 1;
        let added = tokenizer.get_added_vocabulary();
        let mut table = ByteLevelTableBuilder::with_capacity(count);
        let mut bytes = Vec::new();
        for id in 0..count {
            match tokenizer.id_to_token(id as u32) {
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

    /// The bytes of `id` and whether it is special; `None` for an id the
    /// vocabulary lacks.
    #[inline]
    fn entry(&self, id: TokenIdType) -> Option<(&[u8], bool)> {
        let id = id as usize;
        let kind = *self.kind.get(id)?;
        if kind == Kind::Absent {
            return None;
        }
        let bytes = &self.bytes[self.offsets[id]..self.offsets[id + 1]];
        Some((bytes, kind == Kind::Special))
    }
}

/// The bytes a token string stands for: its chars mapped through the
/// byte-level alphabet when every one of them is in it, otherwise the string's
/// own UTF-8 (how `ByteLevel::decode_chain` treats added tokens).
fn push_token_bytes(token: &str, out: &mut Vec<u8>) {
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

impl IncrementalDecoder for ByteLevelIncremental {
    fn step(&mut self, token_id: TokenIdType) -> Result<String> {
        self.fed += 1;
        let Some((bytes, special)) = self.table.entry(token_id) else {
            return match self.unknown {
                UnknownId::Drop => Ok(String::new()),
                UnknownId::Reject => Err(anyhow!("unknown token id {token_id}")),
            };
        };
        if self.skip_special_tokens && special {
            return Ok(String::new());
        }
        if self.pending.is_empty() {
            // The common case: a whole token of valid text, no copy.
            if let Ok(text) = std::str::from_utf8(bytes) {
                if settled(text) {
                    return Ok(text.to_owned());
                }
            }
        }
        self.pending.extend_from_slice(bytes);
        let text = match String::from_utf8_lossy(&self.pending) {
            Cow::Borrowed(text) if settled(text) => text.to_owned(),
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
                text
            }
            _ => return Ok(String::new()),
        };
        self.pending.clear();
        Ok(text)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.fed = 0;
        self.reported = false;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tokenizers::{
        decoders::{byte_level::ByteLevel as ByteLevelDecoder, metaspace::Metaspace},
        models::bpe::{Merges, Vocab, BPE},
        pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer,
        AddedToken,
    };

    use super::*;

    const SPECIAL: &str = "<|im_end|>";
    const ADDED: &str = "<tool>";

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
    fn byte_level_tokenizer() -> HfTokenizer {
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
}
