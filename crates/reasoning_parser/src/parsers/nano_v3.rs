// NanoV3 / Nemotron reasoning parser.
//
// The Nemotron chat template supports an `enable_thinking` toggle that injects
// `<think>\n` in the prefill when ON and `<think></think>` when OFF.
// See: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8?chat_template=default
//
// Uses `always_in_reasoning=false` because the thinking toggle is detected at
// runtime via `ThinkingToggle::DefaultOn` + `think_in_prefill=true`, and
// `mark_reasoning_started()` is called when thinking is effectively ON.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// NanoV3 / Nemotron reasoning parser.
pub struct NanoV3Parser {
    base: BaseReasoningParser,
}

impl NanoV3Parser {
    /// Create a new NanoV3 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("nano_v3".to_string()),
        }
    }
}

impl Default for NanoV3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for NanoV3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.base.detect_and_parse_reasoning(text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.base.parse_reasoning_streaming_incremental(text)
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        self.base.flush()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
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
#[path = "../../tests/unit/parsers/nano_v3/tests.rs"]
mod tests;
