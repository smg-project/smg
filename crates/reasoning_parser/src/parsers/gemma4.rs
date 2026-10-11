//! Gemma4's native thought channel, preserved through gRPC detokenization.
use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

const START: &str = "<|channel>thought\n";
const END: &str = "<channel|>";
const TOOL_START: &str = "<|tool_call>";

#[derive(Default)]
pub struct Gemma4Parser {
    buffer: String,
    reasoning: bool,
    first_feed_done: bool,
    in_tool: bool,
}

impl Gemma4Parser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gemma4 only prefills an open thought channel after a tool response.
    /// Enabling thinking on a new model turn does not imply an open channel.
    pub fn prompt_ends_in_reasoning(prompt: &str) -> bool {
        prompt.ends_with(START)
    }

    fn drain(&mut self) -> ParserResult {
        let mut out = ParserResult::default();
        if !self.first_feed_done && !self.buffer.is_empty() {
            // A retained thought label identifies a suppressed channel opener.
            // A closing marker alone cannot identify preceding text as reasoning.
            if "thought\n".starts_with(&self.buffer) && self.buffer != "thought\n" {
                return out;
            }
            if self.buffer.starts_with("thought\n") {
                self.buffer.drain(.."thought\n".len());
                self.reasoning = true;
            }
            self.first_feed_done = true;
        }
        loop {
            if self.in_tool {
                // A tool argument can quote channel tokens literally. Keep the
                // native frame opaque to reasoning, including chunk splits.
                let mut quoted = false;
                let mut end = None;
                let mut pos = 0;
                while pos < self.buffer.len() {
                    let tail = &self.buffer[pos..];
                    if tail.starts_with("<|\"|>") {
                        quoted = !quoted;
                        pos += "<|\"|>".len();
                    } else if !quoted && tail.starts_with("<tool_call|>") {
                        end = Some(pos + "<tool_call|>".len());
                        break;
                    } else {
                        pos += tail.chars().next().map_or(0, char::len_utf8);
                    }
                }
                let Some(end) = end else {
                    return out;
                };
                out.normal_text.push_str(&self.buffer[..end]);
                self.buffer.drain(..end);
                self.in_tool = false;
                continue;
            }
            let markers: &[&str] = &[START, END, TOOL_START];
            if let Some((pos, marker)) = markers
                .iter()
                .filter_map(|marker| self.buffer.find(marker).map(|pos| (pos, *marker)))
                .min_by_key(|(pos, _)| *pos)
            {
                if self.reasoning {
                    out.reasoning_text.push_str(&self.buffer[..pos]);
                } else {
                    out.normal_text.push_str(&self.buffer[..pos]);
                }
                self.buffer.drain(..pos);
                self.reasoning = marker == START;
                // A native tool frame closes reasoning but belongs to the tool parser.
                if marker == TOOL_START {
                    self.in_tool = true;
                } else {
                    self.buffer.drain(..marker.len());
                }
                continue;
            }

            // Hold only a possible delimiter suffix; arbitrary chunk splits must
            // not leak structural tokens into visible content or reasoning.
            let held = markers
                .iter()
                .flat_map(|marker| {
                    (1..marker.len()).filter(|&size| self.buffer.ends_with(&marker[..size]))
                })
                .max()
                .unwrap_or(0);
            let consumed = self.buffer.len() - held;
            if self.reasoning {
                out.reasoning_text.push_str(&self.buffer[..consumed]);
            } else {
                out.normal_text.push_str(&self.buffer[..consumed]);
            }
            self.buffer.drain(..consumed);
            return out;
        }
    }
}

impl ReasoningParser for Gemma4Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        let mut parser = Self::new();
        parser.reasoning = self.reasoning;
        let mut out = parser.parse_reasoning_streaming_incremental(text)?;
        let tail = parser.flush()?;
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
        "gemma4"
    }

    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.reasoning
    }

    fn mark_reasoning_started(&mut self) {
        self.reasoning = true;
    }

    fn mark_think_start_stripped(&mut self) {}
}
