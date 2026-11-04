//! The added tokens of a vocabulary as one matcher over raw text.

use std::ops::Range;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};

use crate::encode_path::Declined;

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
    use super::AddedTokens;

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
