//! The added tokens of a vocabulary: each with the flags that place it in
//! the pipeline, the lists the cache splits on, and all of them as one
//! matcher over raw text.

use std::ops::Range;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use crate::{encode_path::Declined, traits::TokenIdType};

/// One added token as the pipeline holds it: a token cut out of the text
/// before the model sees it, on the raw text (`normalized: false`) or after
/// normalization, and special (stripped by `skip_special_tokens`) or not.
#[derive(Debug, Clone)]
pub(crate) struct AddedToken {
    pub(crate) id: TokenIdType,
    pub(crate) content: String,
    pub(crate) special: bool,
    pub(crate) normalized: bool,
}

/// The added tokens matched on the raw text that are not special: strings
/// the tokenizer takes whole before pre-tokenization (see
/// `Tokenizer::added_token_strings`).
pub(crate) fn raw_pass_added_tokens(added_tokens: &[AddedToken]) -> Vec<String> {
    added_tokens
        .iter()
        .filter(|token| !token.special && !token.normalized)
        .map(|token| token.content.clone())
        .collect()
}

/// The added tokens matched after normalization, special or not (see
/// `Tokenizer::added_tokens_after_normalization`).
pub(crate) fn normalized_pass_added_tokens(added_tokens: &[AddedToken]) -> Vec<String> {
    added_tokens
        .iter()
        .filter(|token| token.normalized)
        .map(|token| token.content.clone())
        .collect()
}

/// Every added token of a vocabulary, special or not, matched
/// leftmost-longest as `AddedVocabulary` matches them. It tells whether the
/// flat encode of a prompt rendered by segments would cut out exactly the
/// renderer's control tokens (see `EncoderCore`).
pub(crate) struct AddedTokens {
    automaton: AhoCorasick,
}

impl AddedTokens {
    /// The matcher over `tokens`, none for no tokens.
    pub(crate) fn build(tokens: &[&str]) -> Result<Option<Self>, Declined> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens)
            .map_err(|e| Declined::from(format!("the added tokens form no automaton: {e}")))?;
        Ok(Some(Self { automaton }))
    }

    /// The byte ranges of the added tokens in `text`, in order.
    pub(crate) fn ranges<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Range<usize>> + 'a {
        self.automaton
            .find_iter(text)
            .map(|found| found.start()..found.end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(id: TokenIdType, content: &str, special: bool, normalized: bool) -> AddedToken {
        AddedToken {
            id,
            content: content.to_string(),
            special,
            normalized,
        }
    }

    /// The raw pass takes the added tokens that are neither special nor
    /// normalized; the normalized pass the normalized ones, special or not.
    #[test]
    fn the_passes_take_the_tokens_their_flags_place_in_them() {
        let tokens = [
            token(3, "<s>", true, false),
            token(4, "<tool>", false, false),
            token(5, "<think>", false, true),
            token(6, "<pad>", true, true),
        ];
        assert_eq!(raw_pass_added_tokens(&tokens), ["<tool>"]);
        assert_eq!(normalized_pass_added_tokens(&tokens), ["<think>", "<pad>"]);
    }

    #[test]
    fn no_tokens_is_no_matcher() {
        assert!(AddedTokens::build(&[]).unwrap().is_none());
    }

    /// The longest token at a position wins, as `AddedVocabulary` cuts the
    /// text: `<ab>` where `<a>` would also match, and nothing inside a
    /// token already cut out.
    #[test]
    fn the_ranges_are_leftmost_longest_in_order() {
        let added = AddedTokens::build(&["<a>", "<ab>", "b>"]).unwrap().unwrap();
        let ranges: Vec<_> = added.ranges("x<ab>y<a>b>").collect();
        assert_eq!(ranges, [1..5, 6..9, 9..11]);
    }
}
