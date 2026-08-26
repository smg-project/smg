//! Gemma 4 reasoning parser.
//!
//! Gemma 4 delimits chain-of-thought with channel markers rather than think
//! tags: reasoning opens with `<|channel>` followed by a `thought\n` role
//! label and closes with `<channel|>` (format per the family's public
//! reference parsing utilities, which also document that some checkpoints
//! emit a bare `thought\n` label even with thinking disabled). Both markers
//! are special tokens, so detokenization must preserve them —
//! `requires_special_tokens` returns true.
//!
//! Delegates marker handling to [`BaseReasoningParser`] and adds the two
//! Gemma-specific behaviors: stripping the `thought\n` role label from the
//! start of reasoning content (with streaming hold-back while the label is
//! still a possible prefix), and stripping the spurious bare label from
//! non-streaming output when no markers are present.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

const THINK_START: &str = "<|channel>";
const THINK_END: &str = "<channel|>";
const THOUGHT_LABEL: &str = "thought\n";

pub struct Gemma4Parser {
    base: BaseReasoningParser,
    /// Streaming: whether the leading `thought\n` label decision was made.
    label_handled: bool,
    /// Streaming: reasoning held back while it is still a label prefix.
    pending_reasoning: String,
}

impl Gemma4Parser {
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: THINK_START.to_string(),
            think_end_token: THINK_END.to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };
        Self {
            base: BaseReasoningParser::new(config).with_model_type("gemma4".to_string()),
            label_handled: false,
            pending_reasoning: String::new(),
        }
    }
}

impl Default for Gemma4Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for Gemma4Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        // Thinking disabled, spurious role label: no markers anywhere, but
        // the output leads with the bare label — strip it from content.
        if !text.contains(THINK_START) && !text.contains(THINK_END) {
            if let Some(rest) = text.strip_prefix(THOUGHT_LABEL) {
                return Ok(ParserResult::normal(rest.to_string()));
            }
            return self.base.detect_and_parse_reasoning(text);
        }

        let mut result = self.base.detect_and_parse_reasoning(text)?;
        if let Some(rest) = result.reasoning_text.strip_prefix(THOUGHT_LABEL) {
            result.reasoning_text = rest.to_string();
        }
        Ok(result)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        let mut result = self.base.parse_reasoning_streaming_incremental(text)?;
        if self.label_handled {
            return Ok(result);
        }

        if !result.reasoning_text.is_empty() {
            self.pending_reasoning.push_str(&result.reasoning_text);
            result.reasoning_text = String::new();
            if let Some(rest) = self.pending_reasoning.strip_prefix(THOUGHT_LABEL) {
                result.reasoning_text = rest.to_string();
                self.pending_reasoning.clear();
                self.label_handled = true;
            } else if !THOUGHT_LABEL.starts_with(self.pending_reasoning.as_str()) {
                // Diverged from the label: release everything held back.
                result.reasoning_text = std::mem::take(&mut self.pending_reasoning);
                self.label_handled = true;
            }
            // else: still a strict prefix of the label — keep holding.
        }

        // Reasoning block closed while text was still held (a block shorter
        // than the label, possibly with no trailing answer in the chunk):
        // release it as reasoning now — the stream may end here and no later
        // call could recover it.
        if !self.pending_reasoning.is_empty() && !self.base.is_in_reasoning() {
            let held = std::mem::take(&mut self.pending_reasoning);
            result.reasoning_text = format!("{held}{}", result.reasoning_text);
            self.label_handled = true;
        }

        Ok(result)
    }

    fn reset(&mut self) {
        self.base.reset();
        self.label_handled = false;
        self.pending_reasoning.clear();
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn requires_special_tokens(&self) -> bool {
        // The channel markers are special tokens; stripping them during
        // detokenization would leave the parser nothing to split on.
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }

    fn mark_reasoning_started(&mut self) {
        self.base.mark_reasoning_started();
    }

    fn mark_think_start_stripped(&mut self) {
        self.base.mark_think_start_stripped();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_streaming_splits_and_strips_label() {
        let mut parser = Gemma4Parser::new();
        let result = parser
            .detect_and_parse_reasoning("<|channel>thought\nlet me think<channel|>The answer is 4")
            .unwrap();
        assert_eq!(result.reasoning_text, "let me think");
        assert_eq!(result.normal_text, "The answer is 4");
    }

    #[test]
    fn non_streaming_strips_spurious_bare_label() {
        let mut parser = Gemma4Parser::new();
        let result = parser
            .detect_and_parse_reasoning("thought\nThe answer is 4")
            .unwrap();
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "The answer is 4");
    }

    #[test]
    fn non_streaming_plain_output_untouched() {
        let mut parser = Gemma4Parser::new();
        let result = parser
            .detect_and_parse_reasoning("The answer is 4")
            .unwrap();
        assert_eq!(result.normal_text, "The answer is 4");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn streaming_strips_label_split_across_chunks() {
        let mut parser = Gemma4Parser::new();
        let mut reasoning = String::new();
        let mut normal = String::new();
        for chunk in [
            "<|channel>",
            "thou",
            "ght\nstep one ",
            "and two",
            "<channel|>",
            "answer",
        ] {
            let r = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "step one and two");
        assert_eq!(normal, "answer");
    }

    #[test]
    fn streaming_releases_non_label_reasoning() {
        let mut parser = Gemma4Parser::new();
        let mut reasoning = String::new();
        let mut normal = String::new();
        // Reasoning that never carries the role label must not be swallowed.
        for chunk in ["<|channel>", "no label here", "<channel|>", "done"] {
            let r = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&r.reasoning_text);
            normal.push_str(&r.normal_text);
        }
        assert_eq!(reasoning, "no label here");
        assert_eq!(normal, "done");
    }

    #[test]
    fn streaming_releases_held_prefix_when_block_ends_early() {
        let mut parser = Gemma4Parser::new();
        // A reasoning block shorter than the label ("thou") must be released
        // in the same call that closes the block — the stream may end there
        // and no later call could recover it.
        let r = parser
            .parse_reasoning_streaming_incremental("<|channel>thou<channel|>")
            .unwrap();
        assert_eq!(r.reasoning_text, "thou");
        assert_eq!(r.normal_text, "");

        let r = parser
            .parse_reasoning_streaming_incremental("answer")
            .unwrap();
        assert_eq!(r.reasoning_text, "");
        assert_eq!(r.normal_text, "answer");
    }

    #[test]
    fn requires_special_tokens_is_true() {
        assert!(Gemma4Parser::new().requires_special_tokens());
    }
}
