//! The piece table of a sentencepiece-style vocabulary with byte fallback.
//!
//! Such a vocabulary decodes with `Replace(U+2581 -> " ")`, `ByteFallback`,
//! `Fuse` and, usually, `Strip(" ", 1, 0)`: a piece contributes its text with
//! the word marker turned into a space, a `<0xHH>` token one byte of a run
//! that renders as text when the run is valid UTF-8 and as one replacement
//! character per byte otherwise, `Fuse` joins, and `Strip` removes one leading
//! space of the whole text (a chain without the `Replace` keeps the marker,
//! one without the `Strip` the space). None of that depends on anything but the ids, so
//! the text of a stream's window (a handful of ids) is a table walk, where the
//! `tokenizers` decode materializes every token string and runs the chain.
//! The streaming step keeps its algorithm and gets its window decodes here.

use std::sync::Arc;

use tokenizers::Tokenizer as HfTokenizer;

use crate::traits::TokenIdType;

/// What an id contributes to a decode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Piece {
    /// Not in the vocabulary: the decode skips it.
    Absent,
    /// A piece of text (its bytes in the table).
    Text,
    /// A special added token: text when kept, nothing when
    /// `skip_special_tokens` strips it.
    Special,
    /// A `<0xHH>` token: one byte of a byte run.
    Byte(u8),
}

/// Per-id text and kind for a vocabulary whose decoder chain is the one this
/// module describes (see [`PieceTable::build`]).
pub(crate) struct PieceTable {
    /// `offsets[id]..offsets[id + 1]` indexes `text` for token `id`.
    offsets: Vec<usize>,
    text: Vec<u8>,
    piece: Vec<Piece>,
    /// The chain ends in `Strip(" ", 1, 0)`: one leading space of the whole
    /// text goes.
    strips_leading_space: bool,
}

const WORD_MARKER: char = '\u{2581}';

impl PieceTable {
    /// `None` unless the decoder is an optional `Replace(U+2581 -> " ")`,
    /// then `ByteFallback` and `Fuse`, then an optional `Strip(" ", 1, 0)`;
    /// any other chain keeps the `tokenizers` decode.
    pub(crate) fn build(tokenizer: &HfTokenizer) -> Option<Arc<Self>> {
        let chain = serde_json::to_value(tokenizer.get_decoder()?).ok()?;
        let decoders = chain.get("decoders")?.as_array()?;
        let replace = serde_json::json!(
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "}
        );
        let byte_fallback = serde_json::json!({"type": "ByteFallback"});
        let fuse = serde_json::json!({"type": "Fuse"});
        let strip = serde_json::json!({"type": "Strip", "content": " ", "start": 1, "stop": 0});
        // An optional leading Replace, then ByteFallback and Fuse, then an
        // optional Strip: the chains the vocabularies of this kind ship.
        let rest = match decoders.as_slice() {
            [a, rest @ ..] if *a == replace => rest,
            rest => rest,
        };
        let replaces_marker = rest.len() != decoders.len();
        let strips_leading_space = match rest {
            [b, c] if *b == byte_fallback && *c == fuse => false,
            [b, c, d] if *b == byte_fallback && *c == fuse && *d == strip => true,
            _ => return None,
        };
        // Each id through the lookups the decode uses: `id_to_token` (an added
        // token before a model piece that shares its id) and the added
        // vocabulary's own notion of special, as the byte-level table does.
        let max_id = tokenizer.get_vocab(true).values().copied().max()?;
        let added = tokenizer.get_added_vocabulary();
        let ids = max_id as usize + 1;
        let mut offsets = Vec::with_capacity(ids + 1);
        offsets.push(0);
        let mut text = Vec::with_capacity(ids * 4);
        let mut piece = Vec::with_capacity(ids);
        for id in 0..=max_id {
            let kind = match tokenizer.id_to_token(id) {
                None => Piece::Absent,
                Some(token) => match byte_token(&token) {
                    Some(byte) => Piece::Byte(byte),
                    None => {
                        for ch in token.chars() {
                            let ch = if replaces_marker && ch == WORD_MARKER {
                                ' '
                            } else {
                                ch
                            };
                            let mut buf = [0u8; 4];
                            text.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        if added.is_special_token(&token) {
                            Piece::Special
                        } else {
                            Piece::Text
                        }
                    }
                },
            };
            offsets.push(text.len());
            piece.push(kind);
        }
        text.shrink_to_fit();
        Some(Arc::new(Self {
            offsets,
            text,
            piece,
            strips_leading_space,
        }))
    }

    fn piece(&self, id: TokenIdType) -> Piece {
        usize::try_from(id)
            .ok()
            .and_then(|index| self.piece.get(index).copied())
            .unwrap_or(Piece::Absent)
    }

    fn text_of(&self, id: TokenIdType) -> &str {
        let index = id as usize;
        // The table holds what `encode_utf8` wrote: valid UTF-8 per piece.
        std::str::from_utf8(&self.text[self.offsets[index]..self.offsets[index + 1]])
            .unwrap_or_default()
    }

    /// The text of `ids` as the vocabulary's decoder chain renders it.
    pub(crate) fn render(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> String {
        let mut out = String::with_capacity(ids.len() * 4);
        let mut run: Vec<u8> = Vec::new();
        for &id in ids {
            match self.piece(id) {
                Piece::Absent => {}
                Piece::Special if skip_special_tokens => {}
                Piece::Byte(byte) => run.push(byte),
                Piece::Text | Piece::Special => {
                    close_run(&mut out, &mut run);
                    out.push_str(self.text_of(id));
                }
            }
        }
        close_run(&mut out, &mut run);
        if self.strips_leading_space && out.starts_with(' ') {
            out.remove(0);
        }
        out
    }

    /// The text `id` adds to a window that has nothing pending, when that is
    /// all a step needs to know: a plain piece (or a kept special token) with
    /// text that does not end in a replacement character, which the step
    /// algorithm would hold as a possibly incomplete character.
    pub(crate) fn plain_text(&self, id: TokenIdType, skip_special_tokens: bool) -> Option<&str> {
        match self.piece(id) {
            Piece::Text => {}
            Piece::Special if !skip_special_tokens => {}
            Piece::Absent | Piece::Special | Piece::Byte(_) => return None,
        }
        let text = self.text_of(id);
        (!text.is_empty() && !text.ends_with('\u{FFFD}')).then_some(text)
    }
}

/// The byte a `<0xHH>` token stands for, exactly as the `ByteFallback`
/// decoder reads it.
fn byte_token(token: &str) -> Option<u8> {
    (token.len() == 6 && token.starts_with("<0x") && token.ends_with('>'))
        .then(|| u8::from_str_radix(&token[3..5], 16).ok())
        .flatten()
}

/// A byte run ends: valid UTF-8 is its text, anything else one replacement
/// character per byte (the `ByteFallback` decoder's rule).
fn close_run(out: &mut String, run: &mut Vec<u8>) {
    if run.is_empty() {
        return;
    }
    match std::str::from_utf8(run) {
        Ok(text) => out.push_str(text),
        Err(_) => out.extend(std::iter::repeat_n('\u{FFFD}', run.len())),
    }
    run.clear();
}

#[cfg(test)]
pub(crate) mod tests {
    use tokenizers::{
        decoders::{
            byte_fallback::ByteFallback, byte_level::ByteLevel, fuse::Fuse, sequence::Sequence,
            strip::Strip,
        },
        models::bpe::{Merges, Vocab, BPE},
        normalizers::Replace,
        AddedToken, DecoderWrapper, Tokenizer as HfTokenizer,
    };

    use super::*;

    /// A small sentencepiece-style vocabulary: word pieces, the byte tokens of
    /// a CJK character and of a literal U+FFFD, a stray byte, a lone marker, a
    /// piece that ends in U+FFFD, and two special tokens.
    pub(crate) fn synthetic(strip: bool) -> HfTokenizer {
        let pieces = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}",
            "\u{2581}hello",
            "\u{2581}world",
            "he",
            "llo",
            "<0xE6>",
            "<0x97>",
            "<0xA5>",
            "<0xEF>",
            "<0xBF>",
            "<0xBD>",
            "<0xFF>",
            "<0x41>",
            "\u{2581}x\u{FFFD}",
            "\u{2581}a",
            "b",
        ];
        let mut vocab = Vocab::default();
        for (id, piece) in pieces.iter().enumerate() {
            vocab.insert((*piece).to_string(), id as u32);
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, Merges::new())
            .byte_fallback(true)
            .unk_token("<unk>".into())
            .build()
            .expect("model");
        let mut tokenizer = HfTokenizer::new(model);
        let mut chain: Vec<DecoderWrapper> = vec![
            Replace::new("\u{2581}", " ").expect("replace").into(),
            ByteFallback::new().into(),
            Fuse::new().into(),
        ];
        if strip {
            chain.push(Strip::new(' ', 1, 0).into());
        }
        tokenizer.with_decoder(Some(Sequence::new(chain)));
        tokenizer
            .add_special_tokens([
                AddedToken::from("<s>", true),
                AddedToken::from("</s>", true),
            ])
            .expect("specials");
        tokenizer
    }

    /// Ids drawn to make runs valid, incomplete and lost, with specials and
    /// text pieces in between.
    pub(crate) fn random_ids(state: &mut u64, len: usize) -> Vec<u32> {
        (0..len)
            .map(|_| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let roll = (*state >> 33) % 100;
                match roll {
                    0..=44 => 8 + (roll % 8) as u32,  // byte tokens 8..=15
                    45..=49 => 1 + (roll % 2) as u32, // <s>, </s>
                    50..=52 => 3,                     // the lone marker
                    53..=54 => 16,                    // the piece ending in U+FFFD
                    55 => 999,                        // not in the vocabulary
                    _ => 4 + (roll % 4) as u32,       // word pieces 4..=7
                }
            })
            .collect()
    }

    /// Like [`random_ids`] without anything that loses a run: byte tokens come
    /// as whole characters (a CJK character, a literal U+FFFD), no stray byte.
    pub(crate) fn random_ids_without_lost_runs(state: &mut u64, len: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(len * 3);
        while ids.len() < len {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let roll = (*state >> 33) % 100;
            match roll {
                0..=29 => ids.extend([8, 9, 10]),
                30..=39 => ids.extend([11, 12, 13]),
                40..=44 => ids.push(1 + (roll % 2) as u32),
                45..=49 => ids.push(3),
                50..=54 => ids.push(16),
                _ => ids.push(4 + (roll % 4) as u32),
            }
        }
        ids
    }

    #[test]
    fn the_table_renders_what_the_decoder_chain_renders() {
        for strip in [false, true] {
            let tokenizer = synthetic(strip);
            let table = PieceTable::build(&tokenizer).expect("the shape is the supported one");
            assert_eq!(table.strips_leading_space, strip);
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            for round in 0..2_000 {
                let ids = random_ids(&mut state, round % 9);
                for skip in [false, true] {
                    let want = tokenizer.decode(&ids, skip).expect("decode");
                    assert_eq!(table.render(&ids, skip), want, "ids {ids:?} skip {skip}");
                }
            }
        }
    }

    #[test]
    fn a_chain_without_the_replace_keeps_the_marker() {
        let mut tokenizer = synthetic(false);
        let chain: Vec<DecoderWrapper> = vec![ByteFallback::new().into(), Fuse::new().into()];
        tokenizer.with_decoder(Some(Sequence::new(chain)));
        let table = PieceTable::build(&tokenizer).expect("ByteFallback + Fuse alone is supported");
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        for round in 0..500 {
            let ids = random_ids(&mut state, round % 7);
            assert_eq!(
                table.render(&ids, false),
                tokenizer.decode(&ids, false).expect("decode")
            );
        }
        assert_eq!(table.plain_text(4, false), Some("\u{2581}hello"));
    }

    #[test]
    fn plain_text_is_the_text_of_a_plain_piece_only() {
        let table = PieceTable::build(&synthetic(true)).expect("table");
        assert_eq!(table.plain_text(4, false), Some(" hello"));
        assert_eq!(table.plain_text(3, false), Some(" "));
        assert_eq!(table.plain_text(1, false), Some("<s>"));
        assert_eq!(
            table.plain_text(1, true),
            None,
            "a skipped special adds nothing"
        );
        assert_eq!(table.plain_text(8, false), None, "a byte token opens a run");
        assert_eq!(
            table.plain_text(16, false),
            None,
            "text ending in U+FFFD is held"
        );
        assert_eq!(table.plain_text(999, false), None);
    }

    #[test]
    fn other_decoder_chains_keep_the_tokenizers_decode() {
        let mut tokenizer = synthetic(true);
        let chain: Vec<DecoderWrapper> = vec![
            Replace::new("\u{2581}", " ").expect("replace").into(),
            ByteFallback::new().into(),
        ];
        tokenizer.with_decoder(Some(Sequence::new(chain)));
        assert!(PieceTable::build(&tokenizer).is_none(), "no Fuse");
        let mut tokenizer = synthetic(true);
        let chain: Vec<DecoderWrapper> = vec![
            Replace::new("\u{2581}", " ").expect("replace").into(),
            ByteFallback::new().into(),
            Fuse::new().into(),
            Strip::new(' ', 1, 1).into(),
        ];
        tokenizer.with_decoder(Some(Sequence::new(chain)));
        assert!(PieceTable::build(&tokenizer).is_none(), "a trailing strip");
        let mut tokenizer = synthetic(true);
        tokenizer.with_decoder(Some(ByteLevel::default()));
        assert!(
            PieceTable::build(&tokenizer).is_none(),
            "a byte-level decoder"
        );
    }
}
