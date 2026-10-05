//! What a parser is fed: the prompt, decoded deltas, and the end of the stream.

/// Where one engine token's bytes sit in the decoded text of a delta, as byte offsets into that
/// text.
///
/// The spans of a delta partition its text: they are in order, contiguous, and together cover
/// every byte, which [`TokenSpan::cover`] checks. `end` is exclusive.
///
/// Two cases have no bytes of their own and get `start == end`:
///
/// - a special token the decoder hid (`skip_special_tokens`), which parsers that match markers
///   by token identity still see;
/// - a token whose bytes the decoder is still holding, because they end an incomplete UTF-8
///   sequence or a possible stop string. Such a token sits at the end of the delta with a
///   zero-width span. When its bytes are released in a later delta they are attributed to the
///   same token id again, so one id can appear in more than one delta. Parsers read a span as
///   "these bytes came from token X", never as "token X occurred once".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSpan {
    /// The engine's token id.
    pub token_id: u32,
    /// First byte of this token's text in the delta.
    pub start: usize,
    /// One past the last byte of this token's text in the delta.
    pub end: usize,
}

impl TokenSpan {
    /// Whether `spans` partition `text`: in order, contiguous, starting at zero and ending at
    /// `text.len()`, with every boundary on a character boundary. Zero-width spans are allowed
    /// anywhere. An empty `spans` covers only empty text.
    pub fn cover(text: &str, spans: &[TokenSpan]) -> bool {
        let mut at = 0;
        for span in spans {
            if span.start != at || span.end < span.start || !text.is_char_boundary(span.end) {
                return false;
            }
            at = span.end;
        }
        at == text.len()
    }
}

/// How the engine said the stream ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineFinish {
    /// The model stopped on its own, or a stop sequence matched.
    Stop,
    /// The token limit cut the output.
    Length,
    /// The request was aborted.
    Abort,
    /// A finish reason this crate does not interpret; carried through verbatim.
    Other(String),
}

/// One step of a parser's lifecycle. Fed in order: one `Prompt`, any number of `Delta`s, one `End`.
#[derive(Clone, Debug)]
pub enum Input<'a> {
    /// The prompt the engine was given, before any output. Sets the initial state (is reasoning
    /// already open, which channel is active) and lets the format decide grammar coverage.
    Prompt {
        /// The prompt token ids exactly as sent to the engine.
        token_ids: &'a [u32],
        /// The same prompt decoded with special tokens kept.
        text: &'a str,
    },
    /// One engine chunk, decoded. The gateway owns detokenization, so ids, text and their
    /// alignment are all available.
    Delta {
        /// Token ids in this chunk.
        token_ids: &'a [u32],
        /// Decoded text of this chunk.
        text: &'a str,
        /// One span per entry of `token_ids`, in the same order, locating each token's bytes in
        /// `text`; together they partition `text` (see [`TokenSpan::cover`]).
        spans: &'a [TokenSpan],
    },
    /// The stream ended. The parser flushes what it holds and reports the finish.
    End {
        /// The engine's finish reason, which the parser may refine (for example to `tool_calls`).
        finish: EngineFinish,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(token_id: u32, start: usize, end: usize) -> TokenSpan {
        TokenSpan {
            token_id,
            start,
            end,
        }
    }

    #[test]
    fn spans_that_partition_the_text_cover_it() {
        assert!(TokenSpan::cover("héllo", &[span(1, 0, 3), span(2, 3, 6)]));
        assert!(TokenSpan::cover("", &[]));
        assert!(TokenSpan::cover("", &[span(9, 0, 0)]));
    }

    #[test]
    fn zero_width_spans_for_hidden_or_held_tokens_are_allowed() {
        assert!(TokenSpan::cover(
            "ab",
            &[span(7, 0, 0), span(1, 0, 1), span(2, 1, 2), span(8, 2, 2)]
        ));
    }

    #[test]
    fn gaps_overlaps_short_coverage_and_split_characters_do_not_cover() {
        assert!(!TokenSpan::cover("abc", &[span(1, 0, 1), span(2, 2, 3)]));
        assert!(!TokenSpan::cover("abc", &[span(1, 0, 2), span(2, 1, 3)]));
        assert!(!TokenSpan::cover("abc", &[span(1, 0, 2)]));
        assert!(!TokenSpan::cover("é", &[span(1, 0, 1), span(2, 1, 2)]));
    }
}
