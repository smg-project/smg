//! Incremental Hy4 tool parsing. Only ambiguous typed values and delimiter
//! prefixes are held back; emitted JSON fragments are never revised.
use std::collections::HashSet;

use openai_protocol::common::Tool;
use serde_json::Value;

use super::{coerce, types, LIMIT, START};
use crate::{
    errors::{ParserError, ParserResult},
    types::{StreamingParseResult, ToolCallItem},
};

#[derive(Default)]
enum State {
    #[default]
    Text,
    Group,
    Name,
    Argument,
    Key,
    ValueStart,
    Value,
}

#[derive(Default)]
pub(super) struct StreamParser {
    state: State,
    buffer: String,
    suffix: String,
    prospective: String,
    name: String,
    key: String,
    schema: Option<Value>,
    string_value: bool,
    keys: HashSet<String>,
    index: usize,
    announced: bool,
    group_has_calls: bool,
    failed: bool,
    /// Byte offset already searched for the current buffered end marker.
    scan_from: usize,
}

impl StreamParser {
    fn tag(&self, name: &str) -> String {
        format!("<{name}{}>", self.suffix)
    }

    fn emit(&self, out: &mut StreamingParseResult, name: Option<String>, parameters: String) {
        if name.is_none() && parameters.is_empty() {
            return;
        }
        if let Some(last) = out.calls.last_mut().filter(|c| c.tool_index == self.index) {
            last.parameters.push_str(&parameters);
        } else {
            out.calls.push(ToolCallItem {
                tool_index: self.index,
                name,
                parameters,
            });
        }
    }

    fn consume(&mut self, n: usize) {
        self.buffer.drain(..n);
        self.scan_from = 0;
    }
    fn whitespace(&mut self) {
        let n = self.buffer.len() - self.buffer.trim_start().len();
        if !self.announced {
            self.prospective.push_str(&self.buffer[..n]);
        }
        self.consume(n);
    }
    fn invalid<T>(&mut self, message: &str) -> ParserResult<T> {
        self.failed = true;
        Err(ParserError::ParsingFailed(format!("Hy4 stream: {message}")))
    }

    pub(super) fn parse(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        if self.failed {
            return self.invalid("reset required after malformed input");
        }
        if self
            .buffer
            .len()
            .saturating_add(self.prospective.len())
            .saturating_add(chunk.len())
            > LIMIT
        {
            return self.invalid("buffer exceeds 4 MiB");
        }
        self.buffer.push_str(chunk);
        let mut out = StreamingParseResult::default();
        loop {
            match self.state {
                State::Text => {
                    let Some(i) = self.buffer.find(START) else {
                        let keep = partial_suffix(&self.buffer, START);
                        let n = self.buffer.len() - keep;
                        out.normal_text.push_str(&self.buffer[..n]);
                        self.consume(n);
                        break;
                    };
                    out.normal_text.push_str(&self.buffer[..i]);
                    self.consume(i);
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
                        self.consume(1);
                        continue;
                    }
                    self.suffix = suffix.to_owned();
                    self.group_has_calls = false;
                    self.prospective = self.buffer[..end + 1].to_owned();
                    self.consume(end + 1);
                    self.state = State::Group;
                }
                State::Group => {
                    self.whitespace();
                    let begin = self.tag("tool_call");
                    let end = self.tag("/tool_calls");
                    if self.buffer.starts_with(&end) {
                        self.consume(end.len());
                        self.prospective.clear();
                        self.state = State::Text;
                    } else if self.buffer.starts_with(&begin) {
                        self.prospective.push_str(&begin);
                        self.consume(begin.len());
                        self.state = State::Name;
                    } else if begin.starts_with(&self.buffer) || end.starts_with(&self.buffer) {
                        break;
                    } else {
                        out.normal_text
                            .push_str(&std::mem::take(&mut self.prospective));
                        self.state = State::Text;
                    }
                }
                State::Name => {
                    let Some(i) = self.buffer.find('<') else {
                        break;
                    };
                    let rest = &self.buffer[i..];
                    let key = self.tag("arg_key");
                    let end = self.tag("/tool_call");
                    if !rest.starts_with(&key) && !rest.starts_with(&end) {
                        if key.starts_with(rest) || end.starts_with(rest) {
                            break;
                        }
                        if self.group_has_calls {
                            return self.invalid("expected argument or call end after tool name");
                        }
                        out.normal_text
                            .push_str(&std::mem::take(&mut self.prospective));
                        self.state = State::Text;
                        self.scan_from = 0;
                        continue;
                    }
                    let name = self.buffer[..i].trim();
                    if name.is_empty() || name.contains('>') {
                        if self.group_has_calls {
                            return self.invalid("empty tool name");
                        }
                        out.normal_text
                            .push_str(&std::mem::take(&mut self.prospective));
                        self.state = State::Text;
                        self.scan_from = 0;
                        continue;
                    }
                    self.name = name.to_owned();
                    self.consume(i);
                    self.keys.clear();
                    self.announced = true;
                    self.prospective.clear();
                    self.emit(&mut out, Some(self.name.clone()), "{".into());
                    self.state = State::Argument;
                }
                State::Argument => {
                    self.whitespace();
                    let key = self.tag("arg_key");
                    let end = self.tag("/tool_call");
                    if self.buffer.starts_with(&end) {
                        self.consume(end.len());
                        self.emit(&mut out, None, "}".into());
                        self.index += 1;
                        self.group_has_calls = true;
                        self.announced = false;
                        self.state = State::Group;
                    } else if self.buffer.starts_with(&key) {
                        self.consume(key.len());
                        self.state = State::Key;
                    } else if key.starts_with(&self.buffer) || end.starts_with(&self.buffer) {
                        break;
                    } else {
                        return self.invalid("expected argument or call end");
                    }
                }
                State::Key => {
                    let end = self.tag("/arg_key");
                    let from = self.scan_from.min(self.buffer.len());
                    let Some(i) = self.buffer[from..].find(&end).map(|i| from + i) else {
                        self.scan_from = resume_offset(&self.buffer, end.len());
                        break;
                    };
                    self.key = self.buffer[..i].trim().to_owned();
                    if self.key.is_empty() || self.keys.contains(&self.key) {
                        return self.invalid("empty or duplicate argument key");
                    }
                    self.consume(i + end.len());
                    self.schema = tools
                        .iter()
                        .find(|t| t.function.name == self.name)
                        .and_then(|t| t.function.parameters.get("properties"))
                        .and_then(|p| p.get(&self.key))
                        .cloned();
                    let ts = types(self.schema.as_ref());
                    // A union can resolve to a non-string once complete. Do not
                    // prematurely quote or stream such a value.
                    self.string_value = !ts.is_empty() && ts.iter().all(|t| *t == "string");
                    self.state = State::ValueStart;
                }
                State::ValueStart => {
                    self.whitespace();
                    let begin = self.tag("arg_value");
                    if !self.buffer.starts_with(&begin) {
                        if begin.starts_with(&self.buffer) {
                            break;
                        }
                        return self.invalid("expected argument value");
                    }
                    self.consume(begin.len());
                    let comma = if self.keys.is_empty() { "" } else { "," };
                    let quote = if self.string_value { "\"" } else { "" };
                    self.emit(
                        &mut out,
                        None,
                        format!("{comma}{}:{quote}", serde_json::to_string(&self.key)?),
                    );
                    self.keys.insert(self.key.clone());
                    self.state = State::Value;
                }
                State::Value => {
                    let end = self.tag("/arg_value");
                    let from = self.scan_from.min(self.buffer.len());
                    if let Some(i) = self.buffer[from..].find(&end).map(|i| from + i) {
                        let value = if self.string_value {
                            format!("{}\"", escape(&self.buffer[..i])?)
                        } else {
                            coerce(&self.buffer[..i], self.schema.as_ref()).to_string()
                        };
                        self.emit(&mut out, None, value);
                        self.consume(i + end.len());
                        self.state = State::Argument;
                    } else {
                        if self.string_value {
                            let n = self.buffer.len() - partial_suffix(&self.buffer, &end);
                            self.emit(&mut out, None, escape(&self.buffer[..n])?);
                            self.consume(n);
                        } else {
                            self.scan_from = resume_offset(&self.buffer, end.len());
                        }
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    pub(super) fn flush_text(&mut self) -> String {
        // An announced but truncated call stays incomplete. Do not fabricate
        // braces or leak its remaining framing as assistant content.
        if self.announced
            || self.failed
            || (self.group_has_calls
                && matches!(self.state, State::Group)
                && self.tag("/tool_calls").starts_with(self.buffer.trim()))
        {
            self.buffer.clear();
            self.prospective.clear();
            return String::new();
        }
        let mut text = std::mem::take(&mut self.prospective);
        text.push_str(&std::mem::take(&mut self.buffer));
        text
    }
}

fn resume_offset(text: &str, marker_len: usize) -> usize {
    let mut offset = text.len().saturating_sub(marker_len.saturating_sub(1));
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn partial_suffix(text: &str, marker: &str) -> usize {
    marker
        .char_indices()
        .map(|(i, _)| i)
        .filter(|&i| i > 0)
        .rev()
        .find(|&n| text.ends_with(&marker[..n]))
        .unwrap_or(0)
}

fn escape(text: &str) -> ParserResult<String> {
    let encoded = serde_json::to_string(text)?;
    Ok(encoded[1..encoded.len() - 1].to_owned())
}
