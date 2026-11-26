//! Sarashina tool call parser
//!
//! Parses tool calls emitted by SoftBank Sarashina models.
//!
//! # Format
//!
//! Sarashina models emit tool calls as a Python-literal list of dicts
//! (single-quoted), optionally prefixed by a `<|tool_calls|>` marker:
//! ```text
//! <|tool_calls|>[{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}]
//! ```
//!
//! In practice `<|tool_calls|>` is a special token (id 127) that is stripped
//! during detokenization, so the parser usually sees only the bare list
//! `[{'name': ..., 'arguments': {...}}]`. The parser therefore treats the
//! marker as optional and also recognizes the bare leading list.
//!
//! The payload uses Python literal syntax (single quotes, `True`/`False`/`None`),
//! so it is parsed with `rustpython_parser` — reusing the same converter as the
//! pythonic parser — rather than `serde_json`.
//!
//! # Field Mapping
//! - `name` → `name`
//! - `arguments` (a dict) → `arguments` (JSON string)

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;

use crate::{
    errors::ParserResult,
    parsers::{
        helpers,
        pythonic::{expression_to_json, parse_python_expression},
    },
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const TOOL_CALLS_MARKER: &str = "<|tool_calls|>";

/// State machine for Sarashina tool parsing.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ParseState {
    /// Looking for the start of a tool-call block.
    Text,
    /// Past the start, accumulating the Python-literal payload.
    InCalls,
}

/// Sarashina tool call parser.
///
/// Handles `[{'name': ..., 'arguments': {...}}]`, with or without a leading
/// `<|tool_calls|>` marker.
pub struct SarashinaParser {
    /// Current parsing state (streaming only).
    state: ParseState,
    /// Buffer for accumulating chunks across streaming calls.
    buffer: String,
    /// Index of the next tool call to stream.
    stream_tool_index: usize,
}

impl SarashinaParser {
    /// Create a new Sarashina parser.
    pub fn new() -> Self {
        Self {
            state: ParseState::Text,
            buffer: String::new(),
            stream_tool_index: 0,
        }
    }

    /// Whether `text` (already left-trimmed) begins a bare tool-call list,
    /// i.e. a list whose first element is a dict: `[{` (allowing whitespace).
    fn starts_bare_list(trimmed: &str) -> bool {
        let mut chars = trimmed.chars();
        if chars.next() != Some('[') {
            return false;
        }
        matches!(chars.find(|c| !c.is_whitespace()), Some('{'))
    }

    /// Whether `text` (already left-trimmed) could still grow into a bare
    /// tool-call list once more chunks arrive (streaming look-ahead).
    fn maybe_bare_list_prefix(trimmed: &str) -> bool {
        if trimmed.is_empty() || Self::starts_bare_list(trimmed) {
            return true;
        }
        // `[` possibly followed by only whitespace — still waiting for the
        // first `{` to arrive in a later chunk.
        let mut chars = trimmed.chars();
        chars.next() == Some('[') && chars.all(|c| c.is_whitespace())
    }

    /// Parse the Python-literal payload (a list of dicts, or a single dict)
    /// into `ToolCall`s. Returns an empty vec when the payload is not a valid
    /// tool-call structure.
    fn parse_payload(payload: &str) -> Vec<ToolCall> {
        let Ok(expr) = parse_python_expression(payload) else {
            return vec![];
        };
        let Ok(value) = expression_to_json(&expr) else {
            return vec![];
        };

        let items = match value {
            Value::Array(arr) => arr,
            single @ Value::Object(_) => vec![single],
            _ => return vec![],
        };

        let mut calls = Vec::new();
        for item in items {
            let Value::Object(obj) = item else { continue };
            let Some(name) = obj.get("name").and_then(|v| v.as_str()) else {
                continue;
            };
            let arguments = obj
                .get("arguments")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "{}".to_string());
            calls.push(ToolCall {
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            });
        }
        calls
    }

    /// Find the byte offset just past the balanced bracket that starts at
    /// `open_idx`, skipping over string literals. Returns `None` when the
    /// bracket is not yet closed (partial payload during streaming).
    fn find_balanced_end(s: &str, open_idx: usize) -> Option<usize> {
        let bytes = s.as_bytes();
        let open = bytes[open_idx];
        let close = match open {
            b'[' => b']',
            b'{' => b'}',
            _ => return None,
        };

        let mut depth = 0usize;
        let mut in_str: Option<u8> = None;
        let mut escaped = false;

        for (i, &c) in bytes.iter().enumerate().skip(open_idx) {
            if let Some(quote) = in_str {
                if escaped {
                    escaped = false;
                } else if c == b'\\' {
                    escaped = true;
                } else if c == quote {
                    in_str = None;
                }
            } else if c == b'\'' || c == b'"' {
                in_str = Some(c);
            } else if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
        }
        None
    }

    /// Try to extract a complete tool-call block from the head of the buffer
    /// (streaming). Consumes the block on success and returns to `Text` state.
    /// If the balanced block is not a valid tool-call list, the consumed span
    /// is emitted as normal text rather than dropped.
    fn try_extract_calls(&mut self) -> StreamingParseResult {
        let Some(open) = self.buffer.find(['[', '{']) else {
            return StreamingParseResult::default();
        };
        let Some(end) = Self::find_balanced_end(&self.buffer, open) else {
            return StreamingParseResult::default();
        };

        let calls = Self::parse_payload(&self.buffer[open..end]);
        let consumed = self.buffer[..end].to_string();
        self.buffer.drain(..end);
        self.state = ParseState::Text;

        if calls.is_empty() {
            // Balanced brackets but not a tool call — preserve as normal text.
            return StreamingParseResult {
                normal_text: consumed,
                calls: vec![],
            };
        }

        // Names are emitted as produced by the model; validation against the
        // request's tool schemas is left to the caller.
        let mut items = Vec::with_capacity(calls.len());
        for call in calls {
            items.push(ToolCallItem {
                tool_index: self.stream_tool_index,
                name: Some(call.function.name),
                parameters: call.function.arguments,
            });
            self.stream_tool_index += 1;
        }
        StreamingParseResult {
            normal_text: String::new(),
            calls: items,
        }
    }
}

impl Default for SarashinaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for SarashinaParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Payload comes after the marker if present, otherwise the bare text.
        let (prefix, rest, had_marker) = match text.find(TOOL_CALLS_MARKER) {
            Some(i) => (&text[..i], &text[i + TOOL_CALLS_MARKER.len()..], true),
            None => ("", text, false),
        };
        // Text to return when there is no tool call. The marker (a special
        // token) is dropped so it never leaks into user-visible output.
        let no_call = || {
            if had_marker {
                format!("{prefix}{rest}").trim().to_string()
            } else {
                text.trim().to_string()
            }
        };

        let Some(open) = rest.find(['[', '{']) else {
            return Ok((no_call(), vec![]));
        };

        // Without a marker, only treat the payload as a tool call when it is the
        // leading content (nothing but whitespace before the bracket), so a
        // bracket inside ordinary prose is left untouched.
        if !had_marker && !rest[..open].trim().is_empty() {
            return Ok((no_call(), vec![]));
        }

        let Some(end) = Self::find_balanced_end(rest, open) else {
            return Ok((no_call(), vec![]));
        };

        let tool_calls = Self::parse_payload(&rest[open..end]);
        if tool_calls.is_empty() {
            // Not a valid tool-call list — leave the surrounding text untouched.
            return Ok((no_call(), vec![]));
        }

        let mut normal_text = prefix.to_string();
        normal_text.push_str(&rest[end..]);
        Ok((normal_text.trim().to_string(), tool_calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        match self.state {
            ParseState::Text => {
                if let Some(pos) = self.buffer.find(TOOL_CALLS_MARKER) {
                    let text_before = self.buffer[..pos].to_string();
                    self.buffer.drain(..pos + TOOL_CALLS_MARKER.len());
                    self.state = ParseState::InCalls;

                    let mut result = self.try_extract_calls();
                    result.normal_text = format!("{}{}", text_before, result.normal_text);
                    return Ok(result);
                }
                if helpers::ends_with_partial_token(&self.buffer, TOOL_CALLS_MARKER).is_some() {
                    // Possible partial marker at the tail — keep buffering.
                    return Ok(StreamingParseResult::default());
                }

                // Marker-less: the detokenized reply is a bare tool-call list.
                let trimmed = self.buffer.trim_start();
                if Self::starts_bare_list(trimmed) {
                    self.state = ParseState::InCalls;
                    return Ok(self.try_extract_calls());
                }
                if Self::maybe_bare_list_prefix(trimmed) {
                    // Ambiguous so far — wait for more chunks.
                    return Ok(StreamingParseResult::default());
                }

                let normal_text = std::mem::take(&mut self.buffer);
                Ok(StreamingParseResult {
                    normal_text,
                    calls: vec![],
                })
            }
            ParseState::InCalls => Ok(self.try_extract_calls()),
        }
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(TOOL_CALLS_MARKER) || Self::starts_bare_list(text.trim_start())
    }

    fn reset(&mut self) {
        self.state = ParseState::Text;
        self.buffer.clear();
        self.stream_tool_index = 0;
    }
}

#[cfg(test)]
#[path = "../../tests/unit/parsers/sarashina/tests.rs"]
mod tests;
