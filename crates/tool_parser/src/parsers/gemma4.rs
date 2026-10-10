//! Gemma4 native tool calls: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`.
//!
//! Matches vLLM 0.31's Gemma4 argument format. Calls are emitted once their
//! argument object closes, so a backend that stops before the closing special
//! token still produces a complete call. Partial or malformed calls remain text.

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::{Map, Value};

use crate::{
    errors::ParserResult,
    json_format,
    parsers::helpers::coerce_by_schema_type,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const START: &str = "<|tool_call>";
const END: &str = "<tool_call|>";
const STRING: &str = r#"<|"|>"#;
const BOUNDARIES: [&str; 3] = [END, "<turn|>", "<|tool_response>"];

#[derive(Default)]
pub struct Gemma4Parser {
    buffer: String,
    next_tool_index: usize,
    after_call: bool,
}

impl Gemma4Parser {
    pub fn new() -> Self {
        Self::default()
    }

    fn drain_ready(&mut self, tools: &[Tool]) -> StreamingParseResult {
        let mut result = StreamingParseResult::default();
        while !self.buffer.is_empty() {
            if self.after_call {
                let trimmed = self.buffer.trim_start();
                if let Some(marker) = BOUNDARIES.iter().find(|m| trimmed.starts_with(**m)) {
                    let consumed = self.buffer.len() - trimmed.len() + marker.len();
                    self.buffer.drain(..consumed);
                    continue;
                }
                if trimmed.is_empty() || BOUNDARIES.iter().any(|m| m.starts_with(trimmed)) {
                    break;
                }
                self.after_call = false;
            }

            if self.buffer.starts_with(START) {
                match decode_call(&self.buffer) {
                    Ok((name, mut args, consumed)) => {
                        if let Some(tool) = tools.iter().find(|t| t.function.name == name) {
                            coerce_arguments(&mut args, &tool.function.parameters);
                        }
                        result.calls.push(ToolCallItem {
                            tool_index: self.next_tool_index,
                            name: Some(name),
                            parameters: json_format::to_string(&args),
                        });
                        self.next_tool_index += 1;
                        self.buffer.drain(..consumed);
                        self.after_call = true;
                    }
                    Err(_) => {
                        // Retain a rejected frame verbatim; recover at the next
                        // native boundary rather than hiding malformed output.
                        let Some(end) = unquoted_marker(&self.buffer[START.len()..], END) else {
                            break;
                        };
                        let consumed = START.len() + end + END.len();
                        result.normal_text.push_str(&self.buffer[..consumed]);
                        self.buffer.drain(..consumed);
                    }
                }
            } else {
                let consumed = self.buffer.find(START).unwrap_or_else(|| {
                    let partial =
                        super::helpers::ends_with_partial_token(&self.buffer, START).unwrap_or(0);
                    self.buffer.len() - partial
                });
                if consumed == 0 {
                    break;
                }
                result.normal_text.push_str(&self.buffer[..consumed]);
                self.buffer.drain(..consumed);
            }
        }
        result
    }

    fn parse_output(output: &str, tools: &[Tool]) -> (String, Vec<ToolCall>) {
        let mut parser = Self {
            buffer: output.to_owned(),
            ..Self::default()
        };
        let result = parser.drain_ready(tools);
        let mut text = result.normal_text;
        text.push_str(&parser.take_unstreamed_normal_text());
        let calls = result
            .calls
            .into_iter()
            .map(|call| ToolCall {
                function: FunctionCall {
                    name: call.name.unwrap_or_default(),
                    arguments: call.parameters,
                },
            })
            .collect();
        (text, calls)
    }
}

#[async_trait]
impl ToolParser for Gemma4Parser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_output(output, &[]))
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_output(output, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        Ok(self.drain_ready(tools))
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(START)
    }

    fn take_unstreamed_normal_text(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug)]
enum DecodeError {
    Incomplete,
    Malformed,
}

fn decode_call(input: &str) -> Result<(String, Value, usize), DecodeError> {
    let body = input[START.len()..].trim_start();
    let prefix = if body.starts_with("call:") {
        "call:"
    } else if body.starts_with(':') {
        ":"
    } else if "call:".starts_with(body) || body.is_empty() {
        return Err(DecodeError::Incomplete);
    } else {
        return Err(DecodeError::Malformed);
    };
    let rest = &body[prefix.len()..];
    let Some(open) = rest.find(['{', '(']) else {
        return Err(if rest.contains(END) {
            DecodeError::Malformed
        } else {
            DecodeError::Incomplete
        });
    };
    let name = rest[..open].trim();
    if name.is_empty() || name.chars().any(char::is_whitespace) || name.contains('<') {
        return Err(DecodeError::Malformed);
    }
    let close = if rest.as_bytes()[open] == b'{' {
        '}'
    } else {
        ')'
    };
    let args_start = input.len() - rest.len() + open + 1;
    let mut reader = ArgumentReader {
        input: &input[args_start..],
        pos: 0,
    };
    let args = reader.object(close, 0)?;
    Ok((name.to_owned(), args, args_start + reader.pos))
}

fn unquoted_marker(input: &str, marker: &str) -> Option<usize> {
    let mut quoted = false;
    let mut pos = 0;
    while pos < input.len() {
        let tail = &input[pos..];
        if tail.starts_with(STRING) {
            quoted = !quoted;
            pos += STRING.len();
        } else if !quoted && tail.starts_with(marker) {
            return Some(pos);
        } else {
            pos += tail.chars().next()?.len_utf8();
        }
    }
    None
}

/// The native format quotes strings with a special token, not JSON escaping.
struct ArgumentReader<'a> {
    input: &'a str,
    pos: usize,
}

impl ArgumentReader<'_> {
    fn remaining(&self) -> &str {
        &self.input[self.pos..]
    }

    fn skip_separators(&mut self) {
        let tail = self.remaining();
        let trimmed = tail.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
        self.pos += tail.len() - trimmed.len();
    }

    fn string(&mut self) -> Result<String, DecodeError> {
        self.pos += STRING.len();
        let Some(end) = self.remaining().find(STRING) else {
            return Err(DecodeError::Incomplete);
        };
        let value = self.remaining()[..end].to_owned();
        self.pos += end + STRING.len();
        Ok(value)
    }

    fn object(&mut self, close: char, depth: usize) -> Result<Value, DecodeError> {
        let mut map = Map::new();
        loop {
            self.skip_separators();
            if self.remaining().starts_with(close) {
                self.pos += close.len_utf8();
                return Ok(Value::Object(map));
            }
            let key = if self.remaining().starts_with(STRING) {
                self.string()?
            } else {
                let Some(end) = self.remaining().find(':') else {
                    return Err(DecodeError::Incomplete);
                };
                let key = self.remaining()[..end].trim().to_owned();
                self.pos += end;
                key
            };
            let whitespace = self.remaining().len() - self.remaining().trim_start().len();
            self.pos += whitespace;
            if key.is_empty() || key.contains(['{', '}', '[', ']', '(', ')', '<']) {
                return Err(DecodeError::Malformed);
            }
            if !self.remaining().starts_with(':') {
                return Err(DecodeError::Malformed);
            }
            self.pos += 1;
            map.insert(key, self.value(close, depth + 1)?);
        }
    }

    fn value(&mut self, close: char, depth: usize) -> Result<Value, DecodeError> {
        if depth > 128 {
            return Err(DecodeError::Malformed);
        }
        let whitespace = self.remaining().len() - self.remaining().trim_start().len();
        self.pos += whitespace;
        if self.remaining().starts_with(STRING) {
            return self.string().map(Value::String);
        }
        let Some(first) = self.remaining().chars().next() else {
            return Err(DecodeError::Incomplete);
        };
        if first == '{' {
            self.pos += 1;
            return self.object('}', depth);
        }
        if first == '[' {
            self.pos += 1;
            let mut values = Vec::new();
            loop {
                self.skip_separators();
                if self.remaining().starts_with(']') {
                    self.pos += 1;
                    return Ok(Value::Array(values));
                }
                values.push(self.value(']', depth + 1)?);
            }
        }
        let Some(end) = self.remaining().find([',', '}', ']', ')']) else {
            return Err(DecodeError::Incomplete);
        };
        let text = self.remaining()[..end].trim();
        let delimiter = self.remaining().as_bytes()[end] as char;
        if text.is_empty() || (delimiter != ',' && delimiter != close) || text.contains('<') {
            return Err(DecodeError::Malformed);
        }
        let value = Value::String(text.to_owned());
        self.pos += end;
        Ok(value)
    }
}

fn coerce_arguments(value: &mut Value, schema: &Value) {
    if let Value::String(text) = value {
        let declared_type = schema.get("type").and_then(Value::as_str);
        // Native quoted strings are already decoded; do not unwrap JSON-looking
        // string contents. The existing helper accepts raw JSON string literals.
        let converted = match declared_type {
            Some("string") => None,
            Some("null") if text.eq_ignore_ascii_case("null") => Some(Value::Null),
            Some("boolean") if text.trim() == "1" => Some(Value::Bool(true)),
            Some("boolean") if text.trim() == "0" => Some(Value::Bool(false)),
            _ => coerce_by_schema_type(text, declared_type),
        };
        if let Some(converted) = converted {
            *value = converted;
        }
    }
    match value {
        Value::Object(values) => {
            if let Some(properties) = schema.get("properties") {
                for (name, value) in values {
                    if let Some(property) = properties.get(name) {
                        coerce_arguments(value, property);
                    }
                }
            }
        }
        Value::Array(values) => {
            if let Some(items) = schema.get("items") {
                for value in values {
                    coerce_arguments(value, items);
                }
            }
        }
        _ => {}
    }
}
