//! The text of each id under a decoder that reads tokens one at a time.
//!
//! The `WordPiece` and `Metaspace` decoders render a token from the token
//! alone and its place: first or later. `WordPiece` keeps a first token as it
//! is, strips the continuation prefix off a later one or leads it with a
//! space, and takes the space before punctuation and clitics out again;
//! `Metaspace` drops a first token's word markers under a prepend scheme and
//! turns every later marker into a space. The decode is the texts joined. So
//! the text of any ids is a table walk once every id's two texts are known,
//! where the `tokenizers` decode clones every token's string and runs the
//! decoder's passes over each (eleven `replace` calls per token for
//! `WordPiece`). The streaming step keeps its algorithm and gets its window
//! decodes here; a step that adds a plain token to a window with nothing
//! pending takes the token's text without a decode at all.
//!
//! The table is built at load, where its cost sits beside the vocabulary's own
//! parse and no request waits for it: about 3,000 instructions per id (25 ms
//! for a vocabulary of 30 thousand ids, a few MB and a fraction of a second for
//! one of 250 thousand). A tokenizer that only encodes carries it unused.

use std::sync::Arc;

use tokenizers::{
    decoders::{wordpiece, DecoderWrapper},
    pre_tokenizers::metaspace::PrependScheme,
    Tokenizer as HfTokenizer,
};

use crate::traits::TokenIdType;

/// What an id contributes to a decode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Token {
    /// Not in the vocabulary: the decode skips it.
    Absent,
    /// A token with text.
    Text,
    /// A special added token: text when kept, nothing when
    /// `skip_special_tokens` strips it.
    Special,
}

/// Per-id text, as the first token decoded and as a later one, for a
/// vocabulary whose decoder is one this module describes (see
/// [`TokenTextTable::build`]).
pub(crate) struct TokenTextTable {
    /// `later[id]` spans `text`: the token's text after another token.
    later: Vec<(u32, u32)>,
    /// `first[id]` spans `text`: the token's text as the first token decoded
    /// (a slice of its later text when it is one).
    first: Vec<(u32, u32)>,
    token: Vec<Token>,
    text: String,
}

/// How a decoder renders one token.
enum Rule {
    WordPiece {
        prefix: String,
        cleanup: bool,
    },
    Metaspace {
        marker: char,
        first_drops_markers: bool,
    },
}

impl Rule {
    fn of(decoder: &DecoderWrapper) -> Option<Self> {
        match decoder {
            DecoderWrapper::WordPiece(decoder) => Some(Self::WordPiece {
                prefix: decoder.prefix.clone(),
                cleanup: decoder.cleanup,
            }),
            DecoderWrapper::Metaspace(decoder) => Some(Self::Metaspace {
                marker: decoder.get_replacement(),
                first_drops_markers: decoder.get_prepend_scheme() != PrependScheme::Never,
            }),
            _ => None,
        }
    }

    /// `token` as the first token decoded and as a later one.
    fn texts(&self, token: &str) -> (String, String) {
        match self {
            Self::WordPiece { prefix, cleanup } => {
                let later = match token.strip_prefix(prefix.as_str()) {
                    Some(rest) => rest.to_string(),
                    None => format!(" {token}"),
                };
                (clean(token, *cleanup), clean(&later, *cleanup))
            }
            Self::Metaspace {
                marker,
                first_drops_markers,
            } => {
                let later: String = token
                    .chars()
                    .map(|c| if c == *marker { ' ' } else { c })
                    .collect();
                let first = if *first_drops_markers {
                    token.chars().filter(|c| c != marker).collect()
                } else {
                    later.clone()
                };
                (first, later)
            }
        }
    }
}

/// `text` as the `WordPiece` decoder's cleanup leaves it: eleven space-led
/// patterns replaced one after the other. A text without a space is its own
/// cleanup. A text whose one space leads it matches at most one pattern
/// (every pattern starts with the space, and none is left once it is
/// replaced), so its space goes when the rest starts with a pattern's tail.
/// Any other text takes the crate's pass.
fn clean(text: &str, cleanup: bool) -> String {
    const TAILS: [&str; 9] = [".", "?", "!", ",", "n't", "'m", "'s", "'ve", "'re"];
    if !cleanup {
        return text.to_string();
    }
    match text.split_once(' ') {
        None => text.to_string(),
        Some(("", rest)) if !rest.contains(' ') => {
            if TAILS.iter().any(|tail| rest.starts_with(tail)) {
                rest.to_string()
            } else {
                text.to_string()
            }
        }
        Some(_) => wordpiece::cleanup(text),
    }
}

impl TokenTextTable {
    /// Whether [`Self::build`] takes the tokenizer's decoder.
    #[cfg(test)]
    pub(crate) fn applies(tokenizer: &HfTokenizer) -> bool {
        tokenizer
            .get_decoder()
            .is_some_and(|decoder| Rule::of(decoder).is_some())
    }

    /// `None` unless the decoder is `WordPiece` or `Metaspace`; any other
    /// decoder keeps the `tokenizers` decode.
    pub(crate) fn build(tokenizer: &HfTokenizer) -> Option<Arc<Self>> {
        Self::build_with_max_id(tokenizer, None, None)
    }

    /// `build` with the model's largest id as the file's read saw it, when
    /// it did (see `max_token_id` in `byte_level`).
    pub(crate) fn build_with_max_id(
        tokenizer: &HfTokenizer,
        model_max_id: Option<u32>,
        listing: Option<&crate::word_cache::Listing>,
    ) -> Option<Arc<Self>> {
        let rule = Rule::of(tokenizer.get_decoder()?)?;
        // Each id through the lookups the decode uses: `id_to_token` (an added
        // token before a model piece that shares its id) and the added
        // vocabulary's own notion of special, as the other tables do.
        let max_id = crate::byte_level::max_token_id(tokenizer, model_max_id)?;
        let added = tokenizer.get_added_vocabulary();
        let ids = usize::try_from(max_id).ok()? + 1;
        let mut table = Self {
            later: Vec::with_capacity(ids),
            first: Vec::with_capacity(ids),
            token: Vec::with_capacity(ids),
            text: String::with_capacity(ids * 6),
        };
        for id in 0..=max_id {
            match crate::word_cache::token_of(tokenizer, listing, id) {
                None => table.push(Token::Absent, "", "")?,
                Some(token) => {
                    let (first, later) = rule.texts(&token);
                    let kind = if added.is_special_token(&token) {
                        Token::Special
                    } else {
                        Token::Text
                    };
                    table.push(kind, &first, &later)?;
                }
            }
        }
        table.text.shrink_to_fit();
        Some(Arc::new(table))
    }

    /// Appends one id's texts, the shorter inside the longer when it is a
    /// slice of it (the later text led by a space, or the continuation prefix
    /// in front of it: nearly every id). `None` when the text outgrows what
    /// the spans index, which no vocabulary's text does.
    fn push(&mut self, kind: Token, first: &str, later: &str) -> Option<()> {
        let (first_span, later_span) = if let Some(at) = later.find(first) {
            let later_span = self.append(later)?;
            (Self::within(later_span, at, first.len())?, later_span)
        } else if let Some(at) = first.find(later) {
            let first_span = self.append(first)?;
            (first_span, Self::within(first_span, at, later.len())?)
        } else {
            (self.append(first)?, self.append(later)?)
        };
        self.first.push(first_span);
        self.later.push(later_span);
        self.token.push(kind);
        Some(())
    }

    fn append(&mut self, text: &str) -> Option<(u32, u32)> {
        let start = u32::try_from(self.text.len()).ok()?;
        self.text.push_str(text);
        Some((start, u32::try_from(self.text.len()).ok()?))
    }

    fn within(span: (u32, u32), at: usize, len: usize) -> Option<(u32, u32)> {
        let start = span.0.checked_add(u32::try_from(at).ok()?)?;
        Some((start, start.checked_add(u32::try_from(len).ok()?)?))
    }

    fn text_of(&self, (start, end): (u32, u32)) -> &str {
        // The spans were cut at the ends of whole texts and of slices found
        // by `str::find`: character boundaries.
        self.text
            .get(start as usize..end as usize)
            .unwrap_or_default()
    }

    /// The index of a token the decode keeps: in the vocabulary, and not a
    /// special token `skip_special_tokens` strips.
    fn kept(&self, id: TokenIdType, skip_special_tokens: bool) -> Option<usize> {
        let index = usize::try_from(id).ok()?;
        match self.token.get(index)? {
            Token::Absent => None,
            Token::Special if skip_special_tokens => None,
            Token::Text | Token::Special => Some(index),
        }
    }

    /// The text of `ids` as the vocabulary's decoder renders it.
    pub(crate) fn render(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> String {
        let mut out = String::with_capacity(ids.len() * 6);
        let mut at_start = true;
        for &id in ids {
            let Some(index) = self.kept(id, skip_special_tokens) else {
                continue;
            };
            let span = if at_start {
                self.first[index]
            } else {
                self.later[index]
            };
            out.push_str(self.text_of(span));
            at_start = false;
        }
        out
    }

    /// What a step reads for `id` when the window has a kept token and
    /// nothing pending: the text the token adds after it, and the token's
    /// text on its own, the prefix of the window it becomes. `None` for a
    /// token the step algorithm skips or holds: absent, a skipped special,
    /// an empty text, a text ending in the replacement character.
    pub(crate) fn step_texts(
        &self,
        id: TokenIdType,
        skip_special_tokens: bool,
    ) -> Option<(&str, &str)> {
        let index = self.kept(id, skip_special_tokens)?;
        let later = self.text_of(self.later[index]);
        (!later.is_empty() && !later.ends_with('\u{FFFD}'))
            .then(|| (later, self.text_of(self.first[index])))
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::{
        decoders::wordpiece::WordPiece as WordPieceDecoder,
        models::{bpe::Vocab, unigram::Unigram, wordpiece::WordPiece},
        normalizers::BertNormalizer,
        pre_tokenizers::{bert::BertPreTokenizer, byte_level::ByteLevel, metaspace::Metaspace},
        processors::bert::BertProcessing,
        AddedToken,
    };

    use super::*;
    use crate::{huggingface::HuggingFaceTokenizer, sequence::Sequence};

    /// A small WordPiece vocabulary: words and continuations, the
    /// punctuation and clitics the cleanup reads, tokens with spaces in them
    /// (the crate's pass), a continuation with no text, a piece ending in
    /// U+FFFD, and three special tokens; the `[CLS] ... [SEP]` template.
    fn word_piece(cleanup: bool) -> HfTokenizer {
        let pieces = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "hello",
            "##ing",
            "world",
            ".",
            ",",
            "?",
            "!",
            "'",
            "s",
            "'s",
            "'ve",
            "'re",
            "'m",
            "n't",
            "a b",
            " led",
            "##",
            "x\u{FFFD}",
            "##.",
            "do",
            "##'s",
            "' s",
            "do not",
        ];
        let vocab: Vocab = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| ((*piece).to_string(), id as u32))
            .collect();
        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .expect("model");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer
            .with_normalizer(Some(BertNormalizer::new(true, true, None, true)))
            .expect("normalizer");
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".to_string(), 2),
            ("[CLS]".to_string(), 1),
        )));
        tokenizer.with_decoder(Some(WordPieceDecoder::new("##".to_string(), cleanup)));
        tokenizer
            .add_special_tokens([
                AddedToken::from("[UNK]", true),
                AddedToken::from("[CLS]", true),
                AddedToken::from("[SEP]", true),
            ])
            .expect("specials");
        tokenizer
    }

    /// A small Unigram vocabulary under a `Metaspace` decoder: marked and
    /// unmarked pieces, a lone marker, a piece with a marker inside, a piece
    /// ending in U+FFFD, and two special tokens.
    fn metaspace(scheme: PrependScheme) -> HfTokenizer {
        let pieces = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}hello",
            "\u{2581}world",
            "he",
            "llo",
            "\u{2581}",
            "he\u{2581}llo",
            "\u{2581}x\u{FFFD}",
            "\u{2581}\u{2581}",
        ];
        let model = Unigram::from(
            pieces
                .iter()
                .enumerate()
                .map(|(i, piece)| ((*piece).to_string(), -(i as f64) - 1.0))
                .collect(),
            Some(0),
            false,
        )
        .expect("model");
        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', scheme, true)));
        tokenizer.with_decoder(Some(Metaspace::new('\u{2581}', scheme, true)));
        tokenizer
            .add_special_tokens([
                AddedToken::from("<s>", true),
                AddedToken::from("</s>", true),
            ])
            .expect("specials");
        tokenizer
    }

    /// Ids over the whole vocabulary plus one outside it, specials included.
    fn random_ids(state: &mut u64, len: usize, vocabulary: u32) -> Vec<u32> {
        (0..len)
            .map(|_| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let roll = (*state >> 33) as u32;
                if roll.is_multiple_of(23) {
                    999
                } else {
                    roll % vocabulary
                }
            })
            .collect()
    }

    fn every_tokenizer() -> Vec<(&'static str, HfTokenizer)> {
        vec![
            ("word piece, cleanup", word_piece(true)),
            ("word piece, no cleanup", word_piece(false)),
            ("metaspace always", metaspace(PrependScheme::Always)),
            ("metaspace first", metaspace(PrependScheme::First)),
            ("metaspace never", metaspace(PrependScheme::Never)),
        ]
    }

    #[test]
    fn the_table_renders_what_the_decoder_renders() {
        for (name, tokenizer) in every_tokenizer() {
            assert!(TokenTextTable::applies(&tokenizer), "{name}");
            let table = TokenTextTable::build(&tokenizer).expect("the decoder is read");
            let vocabulary = tokenizer.get_vocab_size(true) as u32;
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            for round in 0..3_000 {
                let ids = random_ids(&mut state, round % 9, vocabulary);
                for skip in [false, true] {
                    let want = tokenizer.decode(&ids, skip).expect("decode");
                    assert_eq!(
                        table.render(&ids, skip),
                        want,
                        "{name}: ids {ids:?} skip {skip}"
                    );
                }
            }
        }
    }

    #[test]
    fn clean_is_the_crate_cleanup() {
        let alphabet: Vec<char> = " .?!,'ntmsvredoa".chars().collect();
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for round in 0..20_000 {
            let text: String = (0..round % 9)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    alphabet[(state >> 33) as usize % alphabet.len()]
                })
                .collect();
            assert_eq!(clean(&text, true), wordpiece::cleanup(&text), "{text:?}");
            assert_eq!(clean(&text, false), text);
        }
    }

    #[test]
    fn step_texts_is_the_text_of_a_plain_token_only() {
        let table = TokenTextTable::build(&word_piece(true)).expect("table");
        assert_eq!(table.step_texts(3, false), Some((" hello", "hello")));
        assert_eq!(table.step_texts(4, false), Some(("ing", "##ing")));
        assert_eq!(table.step_texts(6, false), Some((".", ".")));
        assert_eq!(table.step_texts(12, false), Some(("'s", "'s")));
        assert_eq!(table.step_texts(10, false), Some((" '", "'")));
        assert_eq!(table.step_texts(1, false), Some((" [CLS]", "[CLS]")));
        assert_eq!(table.step_texts(1, true), None, "a skipped special");
        assert_eq!(
            table.step_texts(19, false),
            None,
            "a continuation with no text"
        );
        assert_eq!(
            table.step_texts(20, false),
            None,
            "text ending in U+FFFD is held"
        );
        assert_eq!(table.step_texts(999, false), None);
        let table = TokenTextTable::build(&metaspace(PrependScheme::Always)).expect("table");
        assert_eq!(table.step_texts(3, false), Some((" hello", "hello")));
        assert_eq!(table.step_texts(8, false), Some(("he llo", "hello")));
        assert_eq!(table.step_texts(7, false), Some((" ", "")));
    }

    #[test]
    fn a_stream_steps_as_the_crate_steps() {
        for (name, tokenizer) in every_tokenizer() {
            let vocabulary = tokenizer.get_vocab_size(true) as u32;
            let smg: Arc<dyn crate::traits::Tokenizer> =
                Arc::new(HuggingFaceTokenizer::from_tokenizer(tokenizer.clone()));
            let mut state = 0x1234_5678_9ABC_DEF1u64;
            for round in 0..1_500 {
                let ids = random_ids(&mut state, round % 13, vocabulary);
                for skip in [false, true] {
                    let mut window = Vec::new();
                    let mut prefix = String::new();
                    let mut prefix_index = 0;
                    let want: Vec<String> = ids
                        .iter()
                        .map(|&id| {
                            tokenizers::tokenizer::step_decode_stream(
                                &tokenizer,
                                vec![id],
                                skip,
                                &mut window,
                                &mut prefix,
                                &mut prefix_index,
                            )
                            .expect("the crate's step")
                            .unwrap_or_default()
                        })
                        .collect();
                    let mut sequence = Sequence::new_with_options(Arc::clone(&smg), skip);
                    let got: Vec<String> = ids
                        .iter()
                        .map(|&id| sequence.append_token(id).expect("step"))
                        .collect();
                    assert_eq!(got, want, "{name}: ids {ids:?} skip {skip}");
                    let tail = sequence.finish().expect("finish");
                    assert_eq!(
                        got.concat() + &tail,
                        tokenizer.decode(&ids, skip).expect("decode"),
                        "{name}: ids {ids:?} skip {skip}: the stream and the end add up to the decode"
                    );
                }
            }
        }
    }

    #[test]
    fn the_ids_are_those_of_the_full_encode() {
        let texts = [
            "Hello world.",
            "I don't know, do you? I've a résumé!",
            "[CLS] hello [SEP] a b ' s",
            "héllo wörld ,,, '''s",
            "",
        ];
        for (name, tokenizer) in every_tokenizer() {
            let smg = HuggingFaceTokenizer::from_tokenizer(tokenizer.clone());
            for text in texts {
                for specials in [false, true] {
                    let want = tokenizer.encode(text, specials).expect("encode");
                    let got = crate::traits::Encoder::encode(&smg, text, specials).expect("encode");
                    assert_eq!(
                        got.token_ids(),
                        want.get_ids(),
                        "{name}: {text:?} {specials}"
                    );
                    let got = crate::traits::Encoder::encode_batch(&smg, &[text, "do"], specials)
                        .expect("batch");
                    assert_eq!(got[0].token_ids(), want.get_ids(), "{name}: batch {text:?}");
                }
            }
        }
    }

    #[test]
    fn other_decoders_keep_the_tokenizers_decode() {
        let mut tokenizer = word_piece(true);
        tokenizer.with_decoder(Some(ByteLevel::default()));
        assert!(!TokenTextTable::applies(&tokenizer));
        assert!(TokenTextTable::build(&tokenizer).is_none());
        let mut tokenizer = metaspace(PrependScheme::Always);
        tokenizer.with_decoder(None::<DecoderWrapper>);
        assert!(!TokenTextTable::applies(&tokenizer));
        assert!(TokenTextTable::build(&tokenizer).is_none());
    }
}
