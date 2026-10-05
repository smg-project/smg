//! What a parser is fed: the prompt, decoded deltas, and the end of the stream.

/// Where one engine token sits in the decoded text of a delta, as byte offsets into that text.
///
/// `end` is exclusive. A token that decoded to no text (a special token hidden by the decoder)
/// has `start == end`; parsers that match markers by token identity still see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSpan {
    /// The engine's token id.
    pub token_id: u32,
    /// First byte of this token's text in the delta.
    pub start: usize,
    /// One past the last byte of this token's text in the delta.
    pub end: usize,
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
        /// One span per token id, in order, locating each token's bytes in `text`.
        spans: &'a [TokenSpan],
    },
    /// The stream ended. The parser flushes what it holds and reports the finish.
    End {
        /// The engine's finish reason, which the parser may refine (for example to `tool_calls`).
        finish: EngineFinish,
    },
}
