use std::fmt;

/// Result of parsing text for reasoning content.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParserResult {
    /// The normal text outside reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

impl ParserResult {
    /// Create a new ParserResult with the given normal and reasoning text.
    pub fn new(normal_text: String, reasoning_text: String) -> Self {
        Self {
            normal_text,
            reasoning_text,
        }
    }

    /// Create a result with only normal text.
    pub fn normal(text: String) -> Self {
        Self {
            normal_text: text,
            reasoning_text: String::new(),
        }
    }

    /// Create a result with only reasoning text.
    pub fn reasoning(text: String) -> Self {
        Self {
            normal_text: String::new(),
            reasoning_text: text,
        }
    }

    /// Check if this result contains any text.
    pub fn is_empty(&self) -> bool {
        self.normal_text.is_empty() && self.reasoning_text.is_empty()
    }
}

/// Trait for parsing reasoning content from LLM outputs.
pub trait ReasoningParser: Send + Sync {
    /// Detects and parses reasoning from the input text (one-time parsing).
    ///
    /// This method is used for non-streaming scenarios where the complete
    /// text is available at once.
    ///
    /// Returns an error if the text exceeds buffer limits or contains invalid UTF-8.
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError>;

    /// Parses reasoning incrementally from streaming input.
    ///
    /// This method maintains internal state across calls to handle partial
    /// tokens and chunk boundaries correctly.
    ///
    /// Returns an error if the buffer exceeds max_buffer_size.
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError>;

    /// Emit text held at the end of a stream. Repeated calls return no text.
    /// Call `reset` before the next stream.
    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        Ok(ParserResult::default())
    }

    /// Reset the parser state for reuse.
    ///
    /// This should clear any buffers and reset flags to initial state.
    fn reset(&mut self);

    /// Get the model type this parser is designed for.
    fn model_type(&self) -> &str;

    /// Whether detokenization must preserve model control tokens for this parser.
    ///
    /// Most reasoning formats use ordinary text delimiters such as `<think>`.
    /// Typed-output formats can instead use tokenizer special tokens, which must
    /// reach the parser verbatim rather than being removed by detokenization.
    fn requires_special_tokens(&self) -> bool {
        false
    }

    /// Check if the parser is currently in reasoning mode.
    ///
    /// Returns true if the parser is currently parsing reasoning content.
    fn is_in_reasoning(&self) -> bool;

    /// Mark that reasoning has already started (e.g. `<think>` was injected in the prefill).
    ///
    /// Called when the chat template injects `<think>` in the generation prompt,
    /// so the parser should treat output as reasoning from the start without
    /// waiting for a `<think>` tag in the generated output.
    fn mark_reasoning_started(&mut self);

    /// Mark that the `<think>` start token was already consumed (in the prefill).
    ///
    /// Prevents the streaming parser from trying to find and strip `<think>`
    /// from the model output when the template already included it.
    fn mark_think_start_stripped(&mut self);
}

/// Error types for reasoning parsing operations.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Invalid UTF-8 in stream: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("Buffer overflow: {0} bytes exceeds maximum")]
    BufferOverflow(usize),

    #[error("Unknown model type: {0}")]
    UnknownModel(String),

    #[error("Parser configuration error: {0}")]
    ConfigError(String),
}

/// Default maximum buffer size for reasoning parsers (4MB).
pub const DEFAULT_MAX_BUFFER_SIZE: usize = 4 * 1024 * 1024;

/// Configuration for parser behavior.
#[derive(Debug, Clone)]
pub struct ParserConfig {
    /// The token that marks the start of reasoning content.
    pub think_start_token: String,

    /// The token that marks the end of reasoning content.
    pub think_end_token: String,

    /// Whether to stream reasoning content as it arrives.
    pub stream_reasoning: bool,

    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,

    /// Whether this model always starts in reasoning mode (e.g. DeepSeek R1).
    /// For models with a template thinking toggle, this should be `false` —
    /// the runtime will call `mark_reasoning_started()` when appropriate.
    pub always_in_reasoning: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false, // Default to false (explicit reasoning)
        }
    }
}

impl fmt::Display for ParserResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ParserResult {{ normal: {} chars, reasoning: {} chars }}",
            self.normal_text.len(),
            self.reasoning_text.len()
        )
    }
}
