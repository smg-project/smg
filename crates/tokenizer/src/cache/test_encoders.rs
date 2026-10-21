//! The test encoders of the cache tests: the atomic strings a tokenizer's
//! added vocabulary extracts leftmost-longest, one id each from a base, and
//! one id per byte otherwise, with what `add_special_tokens` adds around
//! them. The tests' tokenizers are instances of it, or delegate their
//! `encode` to one; the ones that split differently stay their own.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use crate::{traits::Encoder, Encoding, Result, TokenIdType};

/// What the encoder puts after the text when `add_special_tokens` is set.
#[derive(Clone, Copy)]
pub(super) enum Tail {
    /// Nothing.
    None,
    /// One id: an EOS, a frame the cache can apply around a result.
    Id(TokenIdType),
    /// The input's length as an id: text-dependent, no frame.
    TextLength,
}

/// An encoder that extracts its strings leftmost-longest, as a BPE
/// tokenizer's added vocabulary does it, each the id `base + its index`,
/// and gives one id per byte otherwise; a BOS in front and a [`Tail`]
/// behind when `add_special_tokens` is set.
pub(super) struct SplittingEncoder {
    matcher: AhoCorasick,
    base: TokenIdType,
    bos: Option<TokenIdType>,
    tail: Tail,
}

impl SplittingEncoder {
    /// `strings` extracted leftmost-longest, each the id `base + its index`;
    /// nothing added for `add_special_tokens`.
    pub(super) fn new<I, S>(strings: I, base: TokenIdType) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        Self {
            matcher: AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(strings)
                .unwrap(),
            base,
            bos: None,
            tail: Tail::None,
        }
    }

    /// `bos` in front of the ids when `add_special_tokens` is set.
    pub(super) fn with_bos(mut self, bos: TokenIdType) -> Self {
        self.bos = Some(bos);
        self
    }

    /// `tail` after the ids when `add_special_tokens` is set.
    pub(super) fn with_tail(mut self, tail: Tail) -> Self {
        self.tail = tail;
        self
    }

    /// The ids of `input`: the BOS when the flag is set, the strings and the
    /// bytes between them, the tail when the flag is set.
    pub(super) fn ids(&self, input: &str, add_special_tokens: bool) -> Vec<TokenIdType> {
        let mut ids: Vec<TokenIdType> = Vec::new();
        if add_special_tokens {
            if let Some(bos) = self.bos {
                ids.push(bos);
            }
        }
        let mut last = 0;
        for found in self.matcher.find_iter(input) {
            ids.extend(input[last..found.start()].bytes().map(TokenIdType::from));
            ids.push(self.base + found.pattern().as_u32());
            last = found.end();
        }
        ids.extend(input[last..].bytes().map(TokenIdType::from));
        if add_special_tokens {
            match self.tail {
                Tail::None => {}
                Tail::Id(id) => ids.push(id),
                Tail::TextLength => {
                    ids.push(TokenIdType::try_from(input.len()).unwrap_or(TokenIdType::MAX));
                }
            }
        }
        ids
    }
}

impl Encoder for SplittingEncoder {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        Ok(Encoding::Plain(self.ids(input, add_special_tokens)))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|i| self.encode(i, add_special_tokens))
            .collect()
    }
}
