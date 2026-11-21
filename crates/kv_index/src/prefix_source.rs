//! Which of a model's records of cached prefixes places a request.
//!
//! A router keeps up to three records of the prefixes a model's workers
//! hold, and one request reads exactly one of them:
//!
//! 1. Event-driven: the index of the workers' cache events, scored by
//!    overlap (a tokenized request of a model whose workers publish events).
//! 2. Approximate token tree: [`TokenTree`](crate::TokenTree) prefix
//!    matching (a tokenized request, no events).
//! 3. Approximate string tree: [`StringTree`](crate::StringTree) prefix
//!    matching (a request the router did not tokenize).
//!
//! [`PrefixSource::of`] is the one statement of that choice.

/// The record one request is placed by, with what the request brings to it.
/// `I` is the caller's handle of a model's index of cache events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixSource<'a, I> {
    /// Token ids, and the model has an index of cache events: the index
    /// alone places the request, and neither tree is read or written.
    Events {
        /// The request's token ids.
        tokens: &'a [u32],
        /// The model's index.
        index: I,
    },
    /// Token ids and no index: the model's token tree.
    TokenTree(&'a [u32]),
    /// No token ids: the model's string tree, by the request's text. A
    /// request that states no text is the empty text.
    StringTree(&'a str),
}

impl<'a, I> PrefixSource<'a, I> {
    /// The record a request of `tokens`, or else of `text`, is placed by.
    /// `index` answers the model's index of cache events where it has one,
    /// and is asked for a tokenized request alone.
    #[inline]
    pub fn of(
        tokens: Option<&'a [u32]>,
        text: Option<&'a str>,
        index: impl FnOnce() -> Option<I>,
    ) -> Self {
        match tokens {
            Some(tokens) => match index() {
                Some(index) => Self::Events { tokens, index },
                None => Self::TokenTree(tokens),
            },
            None => Self::StringTree(text.unwrap_or("")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixSource;

    #[test]
    fn a_tokenized_request_reads_the_index_where_the_model_has_one() {
        let tokens = [1u32, 2, 3];
        let with_index = PrefixSource::of(Some(&tokens), Some("text"), || Some(7));
        assert_eq!(
            with_index,
            PrefixSource::Events {
                tokens: &tokens,
                index: 7
            }
        );
        let without = PrefixSource::<u8>::of(Some(&tokens), Some("text"), || None);
        assert_eq!(without, PrefixSource::TokenTree(&tokens));
    }

    #[test]
    fn a_request_with_no_tokens_reads_the_string_tree_and_asks_for_no_index() {
        let asked = std::cell::Cell::new(false);
        let index = || {
            asked.set(true);
            Some(7)
        };
        assert_eq!(
            PrefixSource::of(None, Some("text"), index),
            PrefixSource::StringTree("text")
        );
        assert!(!asked.get());
        assert_eq!(
            PrefixSource::<u8>::of(None, None, || None),
            PrefixSource::StringTree("")
        );
    }
}
