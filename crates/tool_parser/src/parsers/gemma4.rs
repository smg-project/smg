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
                match decode_call(&self.buffer, tools) {
                    Some((name, args, consumed)) => {
                        result.calls.push(ToolCallItem {
                            tool_index: self.next_tool_index,
                            name: Some(name),
                            parameters: json_format::to_string(&args),
                        });
                        self.next_tool_index += 1;
                        self.buffer.drain(..consumed);
                        self.after_call = true;
                    }
                    None => {
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

fn decode_call(input: &str, tools: &[Tool]) -> Option<(String, Value, usize)> {
    let body = input[START.len()..].trim_start();
    let prefix = if body.starts_with("call:") {
        "call:"
    } else if body.starts_with(':') {
        ":"
    } else {
        return None;
    };
    let rest = &body[prefix.len()..];
    let open = rest.find(['{', '('])?;
    let name = rest[..open].trim();
    if name.is_empty() || name.chars().any(char::is_whitespace) || name.contains('<') {
        return None;
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
    let schema = tools
        .iter()
        .find(|tool| tool.function.name == name)
        .map(|tool| &tool.function.parameters)
        .unwrap_or(&Value::Null);
    let args = reader.object(close, 0, schema)?;
    Some((name.to_owned(), args, args_start + reader.pos))
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

    fn string(&mut self) -> Option<String> {
        self.pos += STRING.len();
        let end = self.remaining().find(STRING)?;
        let value = self.remaining()[..end].to_owned();
        self.pos += end + STRING.len();
        Some(value)
    }

    fn object(&mut self, close: char, depth: usize, schema: &Value) -> Option<Value> {
        let mut map = Map::new();
        loop {
            self.skip_separators();
            if self.remaining().starts_with(close) {
                self.pos += close.len_utf8();
                return Some(Value::Object(map));
            }
            let key = if self.remaining().starts_with(STRING) {
                self.string()?
            } else {
                let end = self.remaining().find(':')?;
                let key = self.remaining()[..end].trim().to_owned();
                self.pos += end;
                key
            };
            let whitespace = self.remaining().len() - self.remaining().trim_start().len();
            self.pos += whitespace;
            if key.is_empty() || key.contains(['{', '}', '[', ']', '(', ')', '<']) {
                return None;
            }
            if !self.remaining().starts_with(':') {
                return None;
            }
            self.pos += 1;
            let property = schema
                .get("properties")
                .and_then(|properties| properties.get(&key))
                .unwrap_or(&Value::Null);
            map.insert(key, self.value(close, depth + 1, property)?);
        }
    }

    fn value(&mut self, close: char, depth: usize, schema: &Value) -> Option<Value> {
        if depth > 128 {
            return None;
        }
        let whitespace = self.remaining().len() - self.remaining().trim_start().len();
        self.pos += whitespace;
        if self.remaining().starts_with(STRING) {
            return self.string().map(Value::String);
        }
        let first = self.remaining().chars().next()?;
        if first == '{' {
            self.pos += 1;
            return self.object('}', depth, schema);
        }
        if first == '[' {
            self.pos += 1;
            let mut values = Vec::new();
            loop {
                self.skip_separators();
                if self.remaining().starts_with(']') {
                    self.pos += 1;
                    return Some(Value::Array(values));
                }
                values.push(self.value(
                    ']',
                    depth + 1,
                    schema.get("items").unwrap_or(&Value::Null),
                )?);
            }
        }
        let end = self.remaining().find([',', '}', ']', ')'])?;
        let text = self.remaining()[..end].trim();
        let delimiter = self.remaining().as_bytes()[end] as char;
        if text.is_empty() || (delimiter != ',' && delimiter != close) || text.contains('<') {
            return None;
        }
        // Coerce only bare leaves. Native quoted strings must stay strings even
        // when the model emits one for a numeric schema.
        let value = coerce_bare_argument(text, schema);
        self.pos += end;
        Some(value)
    }
}

fn coerce_bare_argument(text: &str, schema: &Value) -> Value {
    let branches = if schema.get("type").is_some() {
        std::slice::from_ref(schema)
    } else {
        schema
            .get("anyOf")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_else(|| std::slice::from_ref(schema))
    };
    for branch in branches {
        let Some(declared) = branch.get("type") else {
            continue;
        };
        let types = declared
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_else(|| std::slice::from_ref(declared));
        for kind in types {
            let converted = match kind.as_str() {
                Some("string") => None,
                Some("null") if text.eq_ignore_ascii_case("null") => Some(Value::Null),
                Some("boolean") if text == "1" => Some(Value::Bool(true)),
                Some("boolean") if text == "0" => Some(Value::Bool(false)),
                Some(kind @ ("integer" | "number" | "boolean")) => {
                    coerce_by_schema_type(text, Some(kind))
                }
                _ => None,
            };
            if let Some(converted) = converted {
                return converted;
            }
        }
    }
    Value::String(text.to_owned())
}
