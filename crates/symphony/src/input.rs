//! What a parser is fed: the prompt, decoded deltas, and the end of the stream.

/// Where some bytes of the decoded text of a delta came from, as byte offsets into that text.
///
/// The spans of a delta partition its text: they are in order, contiguous, and together cover
/// every byte, which [`TokenSpan::partitions`] checks. `end` is exclusive. Boundaries fall on
/// character boundaries, because the decoder only releases whole characters.
///
/// Each engine token is listed exactly once in `token_ids`, in the delta the engine produced it
/// in, and has exactly one span with `continued == false` in that same delta. That is the span
/// to count when counting tokens. It is zero-width (`start == end`) in two cases:
///
/// - a special token the decoder hid (`skip_special_tokens`), which parsers that match markers
///   by token identity still see;
/// - a token whose bytes the decoder is still holding, because they end an incomplete UTF-8
///   sequence or a possible stop string.
///
/// Bytes the decoder releases later than the token that produced them appear in the later delta
/// under a span with `continued == true` and the same `token_id`. Such a span is never counted
/// as a token. A character completed by a later token belongs to that later token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSpan {
    /// The engine's token id.
    pub token_id: u32,
    /// First byte of this span in the delta's text.
    pub start: usize,
    /// One past the last byte of this span in the delta's text.
    pub end: usize,
    /// Whether these bytes belong to a token listed in an earlier delta.
    pub continued: bool,
}

impl TokenSpan {
    /// Whether `spans` partition `text`: in order, contiguous, starting at zero and ending at
    /// `text.len()`, with every boundary on a character boundary. Zero-width spans are allowed
    /// anywhere. An empty `spans` covers only empty text. This checks the layout, not the
    /// `continued` flags.
    pub fn partitions(text: &str, spans: &[TokenSpan]) -> bool {
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
        /// The tokens the engine produced in this chunk, each listed once.
        token_ids: &'a [u32],
        /// The text the decoder released for this chunk.
        text: &'a str,
        /// Where the bytes of `text` came from: one span with `continued == false` per entry of
        /// `token_ids`, in the same order, plus `continued` spans for bytes released late from
        /// earlier tokens. Together they partition `text` (see [`TokenSpan::partitions`]).
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
            continued: false,
        }
    }

    fn continued(token_id: u32, start: usize, end: usize) -> TokenSpan {
        TokenSpan {
            continued: true,
            ..span(token_id, start, end)
        }
    }

    #[test]
    fn spans_that_partition_the_text_are_accepted() {
        assert!(TokenSpan::partitions(
            "héllo",
            &[span(1, 0, 3), span(2, 3, 6)]
        ));
        assert!(TokenSpan::partitions("", &[]));
        assert!(TokenSpan::partitions("", &[span(9, 0, 0)]));
    }

    #[test]
    fn zero_width_spans_for_hidden_or_held_tokens_are_allowed() {
        assert!(TokenSpan::partitions(
            "ab",
            &[span(7, 0, 0), span(1, 0, 1), span(2, 1, 2), span(8, 2, 2)]
        ));
    }

    #[test]
    fn bytes_released_late_are_continued_spans_of_the_earlier_token() {
        // Delta 1: token 5 produced "<", which the decoder held as a possible stop string.
        assert!(TokenSpan::partitions("ab", &[span(4, 0, 2), span(5, 2, 2)]));
        // Delta 2: the held "<" is released together with token 6's text.
        assert!(TokenSpan::partitions(
            "<c",
            &[continued(5, 0, 1), span(6, 1, 2)]
        ));
    }

    #[test]
    fn gaps_overlaps_short_coverage_and_split_characters_are_rejected() {
        assert!(!TokenSpan::partitions(
            "abc",
            &[span(1, 0, 1), span(2, 2, 3)]
        ));
        assert!(!TokenSpan::partitions(
            "abc",
            &[span(1, 0, 2), span(2, 1, 3)]
        ));
        assert!(!TokenSpan::partitions("abc", &[span(1, 0, 2)]));
        assert!(!TokenSpan::partitions("é", &[span(1, 0, 1), span(2, 1, 2)]));
    }
}
