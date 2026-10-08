//! Hunyuan v4 tagged tool calls. Checkpoint suffixes are learned from the
//! outer marker rather than hard-coded (e.g. `:6124c78e`).
use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};

use crate::{
    errors::{ParserError, ParserResult},
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

mod streaming;

const LIMIT: usize = 4 * 1024 * 1024;
const START: &str = "<tool_calls";

#[derive(Default)]
pub struct HyV4Parser {
    buffer: String,
    suffix: Option<String>,
    index: usize,
    opening: String,
    stream: streaming::StreamParser,
}

impl HyV4Parser {
    pub fn new() -> Self {
        Self::default()
    }

    fn call(body: &str, suffix: &str, tools: &[Tool]) -> Option<ToolCall> {
        let key_start = format!("<arg_key{suffix}>");
        let key_end = format!("</arg_key{suffix}>");
        let val_start = format!("<arg_value{suffix}>");
        let val_end = format!("</arg_value{suffix}>");
        let (name, mut rest) = body
            .find(&key_start)
            .map_or_else(|| (body.trim(), ""), |i| (body[..i].trim(), &body[i..]));
        if name.is_empty() || name.contains(['<', '>']) {
            return None;
        }
        let props = tools
            .iter()
            .find(|t| t.function.name == name)
            .and_then(|t| t.function.parameters.get("properties"));
        let mut args = Map::new();
        while !rest.trim().is_empty() {
            rest = rest.trim_start().strip_prefix(&key_start)?;
            let (key, tail) = rest.split_once(&key_end)?;
            let key = key.trim();
            if key.is_empty() || args.contains_key(key) {
                return None;
            }
            rest = tail.trim_start().strip_prefix(&val_start)?;
            let (value, tail) = rest.split_once(&val_end)?;
            args.insert(
                key.to_owned(),
                coerce(value, props.and_then(|p| p.get(key))),
            );
            rest = tail;
        }
        Some(ToolCall {
            function: FunctionCall {
                name: name.to_owned(),
                arguments: Value::Object(args).to_string(),
            },
        })
    }

    fn drain(&mut self, tools: &[Tool]) -> StreamingParseResult {
        let mut out = StreamingParseResult::default();
        loop {
            let Some(suffix) = self.suffix.clone() else {
                if let Some(i) = self.buffer.find(START) {
                    out.normal_text.push_str(&self.buffer[..i]);
                    self.buffer.drain(..i);
                    let Some(end) = self.buffer.find('>') else {
                        break;
                    };
                    let suffix = &self.buffer[START.len()..end];
                    if !suffix.is_empty()
                        && (!suffix.starts_with(':')
                            || suffix.len() == 1
                            || suffix.chars().any(|c| c.is_whitespace() || c == '<'))
                    {
                        out.normal_text.push('<');
                        self.buffer.drain(..1);
                        continue;
                    }
                    self.suffix = Some(suffix.to_owned());
                    self.opening = self.buffer[..end + 1].to_owned();
                    self.buffer.drain(..end + 1);
                    continue;
                }
                let keep = (1..START.len())
                    .rev()
                    .find(|&n| self.buffer.ends_with(&START[..n]))
                    .unwrap_or(0);
                let end = self.buffer.len() - keep;
                out.normal_text.push_str(&self.buffer[..end]);
                self.buffer.drain(..end);
                break;
            };
            let close = format!("</tool_calls{suffix}>");
            let begin = format!("<tool_call{suffix}>");
            let end = format!("</tool_call{suffix}>");
            let leading = self.buffer.len() - self.buffer.trim_start().len();
            if self.buffer[leading..].starts_with(&close) {
                self.buffer.drain(..leading + close.len());
                self.opening.clear();
                self.suffix = None;
                continue;
            }
            if !self.buffer[leading..].starts_with(&begin) {
                if begin.starts_with(&self.buffer[leading..])
                    || close.starts_with(&self.buffer[leading..])
                {
                    break;
                }
                out.normal_text.push_str(&std::mem::take(&mut self.opening));
                out.normal_text.push_str(&self.buffer);
                self.buffer.clear();
                self.suffix = None;
                break;
            }
            let body_start = leading + begin.len();
            // A closing-call literal inside an argument value is data, not framing.
            let value_begin = format!("<arg_value{suffix}>");
            let value_end = format!("</arg_value{suffix}>");
            let mut cursor = body_start;
            let end_index = loop {
                let Some(rel) = self.buffer[cursor..].find(&end) else {
                    break None;
                };
                let candidate = cursor + rel;
                if let Some(v) = self.buffer[cursor..candidate].find(&value_begin) {
                    let value_pos = cursor + v + value_begin.len();
                    let Some(vend) = self.buffer[value_pos..].find(&value_end) else {
                        break None;
                    };
                    cursor = value_pos + vend + value_end.len();
                } else {
                    break Some(candidate);
                }
            };
            let Some(i) = end_index else {
                break;
            };
            if let Some(call) = Self::call(&self.buffer[body_start..i], &suffix, tools) {
                self.opening.clear();
                out.calls.push(ToolCallItem {
                    tool_index: self.index,
                    name: Some(call.function.name),
                    parameters: call.function.arguments,
                });
                self.index += 1;
            } else {
                out.normal_text.push_str(&std::mem::take(&mut self.opening));
                out.normal_text.push_str(&self.buffer[..i + end.len()]);
                self.suffix = None;
            }
            self.buffer.drain(..i + end.len());
        }
        out
    }

    /// Constrain a `<tool_calls>` block containing one or more calls. Arguments
    /// use xgrammar's `glm_xml` style: raw strings and schema-typed JSON values,
    /// with required/optional properties handled by xgrammar as for GLM-4.7.
    /// Uses suffix-less markers; this builder has no tokenizer information
    /// from which to select checkpoint-specific suffixes.
    pub fn build_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let calls: Vec<Value> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| {
                json!({
                    "type": "tag",
                    "begin": format!("<tool_call>{}", tool.function.name),
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool.function.parameters,
                        "style": "glm_xml",
                    },
                    "end": "</tool_call>",
                })
            })
            .collect();
        // An empty `or` is unsatisfiable; with no named tool, admit only an
        // empty block, which parses to no calls.
        let content = if calls.is_empty() {
            json!({ "type": "const_string", "value": "" })
        } else {
            json!({ "type": "plus", "content": { "type": "or", "elements": calls } })
        };

        json!({
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_calls>"],
                "tags": [{
                    "begin": "<tool_calls>",
                    "content": content,
                    "end": "</tool_calls>",
                }],
                "at_least_one": at_least_one,
            }
        })
    }

    /// Close prefilled reasoning before the forced call. Prefix exclusions also
    /// prevent checkpoint-suffixed control markers from appearing in reasoning.
    pub fn reasoning_prefix() -> Value {
        json!({
            "type": "tag",
            "begin": "",
            "content": {
                "type": "any_text",
                "excludes": [
                    "<think", "</think",
                    "<tool_call", "</tool_call",
                    "<arg_key", "</arg_key", "<arg_value", "</arg_value",
                ],
            },
            "end": "</think>",
        })
    }
}

fn types(schema: Option<&Value>) -> Vec<&str> {
    let Some(s) = schema else {
        return vec!["string"];
    };
    if let Some(t) = s.get("type") {
        if let Some(t) = t.as_str() {
            return vec![t];
        }
        if let Some(ts) = t.as_array() {
            return ts.iter().filter_map(Value::as_str).collect();
        }
    }
    if let Some(v) = s
        .get("anyOf")
        .or_else(|| s.get("oneOf"))
        .and_then(Value::as_array)
    {
        return v.iter().flat_map(|s| types(Some(s))).collect();
    }
    vec!["string"]
}

fn coerce(raw: &str, schema: Option<&Value>) -> Value {
    let ts = types(schema);
    if ts.contains(&"boolean") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "true" => return Value::Bool(true),
            "false" => return Value::Bool(false),
            _ => {}
        }
    }
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        let matches = ts.iter().any(|t| match *t {
            "integer" | "int" => v.is_i64() || v.is_u64(),
            "number" | "float" | "double" => v.is_number(),
            "object" | "dict" | "map" => v.is_object(),
            "array" | "list" => v.is_array(),
            "null" => v.is_null(),
            _ => false,
        });
        if matches {
            return v;
        }
    }
    // Strings remain literal: preserve whitespace, quotes and XML-like text.
    Value::String(raw.to_owned())
}

#[async_trait]
impl ToolParser for HyV4Parser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_with_tools(output, &[]).await
    }
    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        let mut p = Self::new();
        if output.len() > LIMIT {
            return Err(ParserError::ParsingFailed(
                "Hy4 tool buffer exceeds 4 MiB".into(),
            ));
        }
        p.buffer.push_str(output);
        let mut result = p.drain(tools);
        result
            .normal_text
            .push_str(&format!("{}{}", p.opening, p.buffer));
        Ok((
            result.normal_text,
            result
                .calls
                .into_iter()
                .map(|c| ToolCall {
                    function: FunctionCall {
                        name: c.name.unwrap_or_default(),
                        arguments: c.parameters,
                    },
                })
                .collect(),
        ))
    }
    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.stream.parse(chunk, tools)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(START)
    }
    fn take_unstreamed_normal_text(&mut self) -> String {
        self.stream.flush_text()
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}
