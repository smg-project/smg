//! Hy4 reasoning with optional checkpoint-specific structural-token suffixes.
use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

#[derive(Default)]
pub struct HyV4Parser {
    buffer: String,
    reasoning: bool,
    ended: bool,
    suffix: Option<String>,
}

impl HyV4Parser {
    pub fn new() -> Self {
        Self::default()
    }
    fn drain(&mut self) -> ParserResult {
        let mut out = ParserResult::default();
        while !self.buffer.is_empty() {
            if self.ended {
                out.normal_text.push_str(&std::mem::take(&mut self.buffer));
                break;
            }
            let Some(pos) = self.buffer.find('<') else {
                let text = std::mem::take(&mut self.buffer);
                if self.reasoning {
                    out.reasoning_text.push_str(&text);
                } else {
                    out.normal_text.push_str(&text);
                }
                break;
            };
            if pos > 0 {
                let text: String = self.buffer.drain(..pos).collect();
                if self.reasoning {
                    out.reasoning_text.push_str(&text);
                } else {
                    out.normal_text.push_str(&text);
                }
            }
            let close = self.buffer.starts_with("</");
            let prefix = if close { "</think" } else { "<think" };
            if prefix.starts_with(&self.buffer) {
                break;
            }
            if self.buffer.starts_with(prefix) {
                let tail = &self.buffer[prefix.len()..];
                if tail.starts_with(':') || tail.starts_with('>') {
                    let Some(end) = self.buffer.find('>') else {
                        break;
                    };
                    let suffix = &self.buffer[prefix.len()..end];
                    let valid = suffix.is_empty()
                        || (suffix.starts_with(':')
                            && suffix.len() > 1
                            && !suffix.chars().any(|c| c.is_whitespace() || c == '<'));
                    if valid && self.suffix.as_ref().is_none_or(|s| s == suffix) {
                        self.suffix = Some(suffix.to_owned());
                        self.buffer.drain(..end + 1);
                        self.reasoning = !close;
                        self.ended = close;
                        continue;
                    }
                }
            }
            self.buffer.drain(..1);
            if self.reasoning {
                out.reasoning_text.push('<');
            } else {
                out.normal_text.push('<');
            }
        }
        out
    }
}
impl ReasoningParser for HyV4Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        let mut p = Self::new();
        p.reasoning = self.reasoning;
        let mut out = p.parse_reasoning_streaming_incremental(text)?;
        let tail = p.flush()?;
        out.normal_text.push_str(&tail.normal_text);
        out.reasoning_text.push_str(&tail.reasoning_text);
        Ok(out)
    }
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        if self.buffer.len().saturating_add(text.len()) > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(self.buffer.len() + text.len()));
        }
        self.buffer.push_str(text);
        Ok(self.drain())
    }
    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        let text = std::mem::take(&mut self.buffer);
        Ok(if self.reasoning {
            ParserResult::reasoning(text)
        } else {
            ParserResult::normal(text)
        })
    }
    fn reset(&mut self) {
        *self = Self::new();
    }
    fn model_type(&self) -> &str {
        "hy_v4"
    }
    fn requires_special_tokens(&self) -> bool {
        true
    }
    fn is_in_reasoning(&self) -> bool {
        self.reasoning
    }
    fn mark_reasoning_started(&mut self) {
        self.reasoning = true;
        self.ended = false;
    }
    fn mark_think_start_stripped(&mut self) {}
}
