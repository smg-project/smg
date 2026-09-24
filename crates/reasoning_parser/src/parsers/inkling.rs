//! Inkling/TML typed-content reasoning parser.

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const CONTENT_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const END_MESSAGE: &str = "<|end_message|>";
const MESSAGE_MODEL: &str = "<|message_model|>";

// Keep this in sync with the Inkling TMLv0 tokenizer control tokens.
const CONTROL_TOKENS: &[&str] = &[
    "<|endoftext|>",
    "<|message_user|>",
    MESSAGE_MODEL,
    "<|message_system|>",
    "<|message_tool|>",
    CONTENT_TEXT,
    "<|content_image|>",
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_THINKING,
    "<|content_audio_input|>",
    "<|content_tool_error|>",
    "<|content_xml|>",
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    END_MESSAGE,
    "<|audio_end|>",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    /// TML message headers may carry an author/tool name between the role and
    /// content-type tokens. It is protocol metadata, not assistant content.
    Header,
    Reasoning,
    Content,
    Tool,
    UnsupportedTool,
}

#[derive(Debug, Clone, Copy)]
enum ControlCandidate {
    Complete { start: usize, token: &'static str },
    Partial { start: usize },
}

/// Parser for Inkling's TML typed output blocks.
///
/// Unlike `<think>` formats, Inkling emits a sequence of typed blocks.
/// Structured JSON tool blocks remain in normal output verbatim so the
/// tool-call parser can consume them after reasoning separation. Unsupported
/// headerless text-mode invocations are suppressed as protocol data.
#[derive(Debug, Clone)]
pub struct InklingParser {
    block_kind: Option<BlockKind>,
    buffer: String,
    max_buffer_size: usize,
}

impl InklingParser {
    pub fn new() -> Self {
        Self {
            block_kind: None,
            buffer: String::new(),
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    fn find_control_candidate(text: &str) -> Option<ControlCandidate> {
        for (start, _) in text.match_indices('<') {
            let suffix = &text[start..];

            if let Some(token) = CONTROL_TOKENS
                .iter()
                .copied()
                .find(|token| suffix.starts_with(token))
            {
                return Some(ControlCandidate::Complete { start, token });
            }

            if CONTROL_TOKENS.iter().any(|token| token.starts_with(suffix)) {
                return Some(ControlCandidate::Partial { start });
            }
        }

        None
    }

    fn emit_text(&self, text: &str, result: &mut ParserResult) {
        match self.block_kind {
            Some(BlockKind::Reasoning) => result.reasoning_text.push_str(text),
            Some(BlockKind::Header | BlockKind::UnsupportedTool) => {}
            Some(BlockKind::Content | BlockKind::Tool) | None => {
                result.normal_text.push_str(text);
            }
        }
    }

    fn handle_control(&mut self, token: &str, result: &mut ParserResult) {
        if token == MESSAGE_MODEL {
            self.block_kind = Some(BlockKind::Header);
            return;
        }

        if token == CONTENT_INVOKE_TOOL_JSON {
            self.block_kind = Some(BlockKind::Tool);
            result.normal_text.push_str(token);
            return;
        }

        if token == CONTENT_INVOKE_TOOL_TEXT {
            // The OpenAI function-call shape cannot losslessly represent
            // headerless text-mode invocations. Suppress the entire frame so
            // its payload cannot be mistaken for assistant answer text.
            self.block_kind = Some(BlockKind::UnsupportedTool);
            return;
        }

        if self.block_kind == Some(BlockKind::Tool) {
            result.normal_text.push_str(token);
            if Self::is_end_token(token) {
                self.block_kind = None;
            }
            return;
        }

        if self.block_kind == Some(BlockKind::UnsupportedTool) {
            if Self::is_end_token(token) {
                self.block_kind = None;
            }
            return;
        }

        match token {
            CONTENT_THINKING => self.block_kind = Some(BlockKind::Reasoning),
            CONTENT_TEXT => self.block_kind = Some(BlockKind::Content),
            END_MESSAGE | CONTENT_MODEL_END_SAMPLING => self.block_kind = None,
            _ => {}
        }
    }

    fn is_end_token(token: &str) -> bool {
        matches!(token, END_MESSAGE | CONTENT_MODEL_END_SAMPLING)
    }

    fn parse_buffer(&mut self, finalize: bool) -> ParserResult {
        let text = std::mem::take(&mut self.buffer);
        let mut result = ParserResult::default();
        let mut pos = 0;

        while pos < text.len() {
            let remaining = &text[pos..];
            match Self::find_control_candidate(remaining) {
                Some(ControlCandidate::Complete { start, token }) => {
                    self.emit_text(&remaining[..start], &mut result);
                    self.handle_control(token, &mut result);
                    pos += start + token.len();
                }
                Some(ControlCandidate::Partial { start }) => {
                    self.emit_text(&remaining[..start], &mut result);
                    if finalize {
                        self.emit_text(&remaining[start..], &mut result);
                    } else {
                        self.buffer.push_str(&remaining[start..]);
                    }
                    break;
                }
                None => {
                    self.emit_text(remaining, &mut result);
                    break;
                }
            }
        }

        result
    }
}

impl Default for InklingParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for InklingParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        if text.len() > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(text.len()));
        }

        // Complete parsing is independent of any prior streaming state.
        let mut parser = Self::new();
        parser.max_buffer_size = self.max_buffer_size;
        parser.buffer.push_str(text);
        Ok(parser.parse_buffer(true))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        let buffered_size = self.buffer.len() + text.len();
        if buffered_size > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(buffered_size));
        }

        self.buffer.push_str(text);
        Ok(self.parse_buffer(false))
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        Ok(self.parse_buffer(true))
    }

    fn reset(&mut self) {
        self.block_kind = None;
        self.buffer.clear();
    }

    fn model_type(&self) -> &str {
        "inkling"
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.block_kind == Some(BlockKind::Reasoning)
    }

    fn mark_reasoning_started(&mut self) {
        self.block_kind = Some(BlockKind::Reasoning);
    }

    fn mark_think_start_stripped(&mut self) {
        // Inkling uses typed blocks rather than a separately injected start tag.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOL_BLOCK: &str = concat!(
        "<|content_invoke_tool_json|>",
        r#"{"name":"search","args":{"q":"rust"}}"#,
        "<|end_message|>"
    );
    // Canonical typed-content fixture.
    // Structured tool calls carry their tool name in the model-message header
    // as well as in the JSON payload; only the payload reaches the tool parser.
    const TYPED_OUTPUT: &str = concat!(
        "<|message_model|>",
        "<|content_thinking|>check sources<|end_message|>",
        "<|message_model|>",
        "<|content_text|>Here is the answer.<|end_message|>",
        "<|message_model|>search",
        "<|content_invoke_tool_json|>",
        r#"{"name":"search","args":{"q":"rust"}}"#,
        "<|end_message|>",
        "<|content_model_end_sampling|>"
    );

    #[test]
    fn parses_typed_blocks_and_preserves_tool_framing() {
        let mut parser = InklingParser::new();
        let result = parser.detect_and_parse_reasoning(TYPED_OUTPUT).unwrap();

        assert_eq!(result.reasoning_text, "check sources");
        assert_eq!(
            result.normal_text,
            format!("Here is the answer.{TOOL_BLOCK}")
        );
        assert!(parser.requires_special_tokens());
    }

    #[test]
    fn safely_suppresses_headerless_text_tool_frame() {
        let output = concat!(
            "<|content_invoke_tool_text|>search for rust<|end_message|>",
            "<|content_model_end_sampling|>"
        );
        let mut parser = InklingParser::new();
        let result = parser.detect_and_parse_reasoning(output).unwrap();

        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn streaming_buffers_control_tokens_at_every_chunk_boundary() {
        let expected_normal = format!("Here is the answer.{TOOL_BLOCK}");

        for split in TYPED_OUTPUT
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(TYPED_OUTPUT.len()))
        {
            let mut parser = InklingParser::new();
            let first = parser
                .parse_reasoning_streaming_incremental(&TYPED_OUTPUT[..split])
                .unwrap();
            let second = parser
                .parse_reasoning_streaming_incremental(&TYPED_OUTPUT[split..])
                .unwrap();

            assert_eq!(
                format!("{}{}", first.reasoning_text, second.reasoning_text),
                "check sources",
                "reasoning mismatch at split {split}"
            );
            assert_eq!(
                format!("{}{}", first.normal_text, second.normal_text),
                expected_normal,
                "content mismatch at split {split}"
            );
        }
    }

    #[test]
    fn reset_clears_partial_token_and_reasoning_state() {
        let mut parser = InklingParser::new();
        parser
            .parse_reasoning_streaming_incremental("<|content_thinking|>work<|end_mes")
            .unwrap();
        assert!(parser.is_in_reasoning());

        parser.reset();
        let result = parser
            .parse_reasoning_streaming_incremental("plain answer")
            .unwrap();
        assert_eq!(result, ParserResult::normal("plain answer".to_string()));
    }
}
