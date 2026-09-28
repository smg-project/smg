//! Tool-call parser for tokenizer-declared response templates.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};
use smg_response_template::{consume, find_close, CompiledTemplate, Field, Opener};

use crate::{
    errors::ParserResult,
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const TOOL_CALLS: &[Field] = &[Field::ToolCalls];

/// Parses the `tool_calls` field of a response template.
///
/// The input is the normal text of the template reasoning parser, where tool
/// blocks keep their framing. The block opener names the function and each
/// argument tag holds one key and value. Streaming emits the name once the
/// opener is complete and each argument once its tag is complete. Values are
/// converted by their declared JSON-schema type; `string` and undeclared
/// values stay as written. Calls to undeclared functions are dropped.
pub struct TemplateToolParser {
    template: Arc<CompiledTemplate>,
    buffer: String,
    /// Bytes at the start of `buffer` kept only as look-behind context.
    context: usize,
    /// Bytes after `context` already searched for closes in the open block.
    searched: usize,
    in_tool_call: bool,
    /// The open block calls an undeclared function and is dropped.
    discard: bool,
    prev_tool_call_arr: Vec<Value>,
    current_tool_id: i32,
    current_tool_name_sent: bool,
    streamed_args_for_tool: Vec<String>,
}

impl TemplateToolParser {
    pub fn new(template: Arc<CompiledTemplate>) -> Self {
        Self {
            template,
            buffer: String::new(),
            context: 0,
            searched: 0,
            in_tool_call: false,
            discard: false,
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Drop `buffer[..end]`, keeping look-behind context.
    fn advance(&mut self, end: usize) {
        self.context = consume(&mut self.buffer, end);
        self.searched = 0;
    }

    /// The function named by the opener at `start` of `hay`.
    fn name(&self, hay: &str, start: usize) -> Option<String> {
        let captures = self
            .template
            .opener_captures(Field::ToolCalls, hay, start)?;
        Some(captures.name("name")?.as_str().to_string())
    }

    /// The arguments in `body`, in order. The first of duplicate keys wins.
    fn arguments(&self, body: &str, types: &[(String, String)]) -> Vec<(String, Value)> {
        let tag = self.template.tag();
        let mut arguments: Vec<(String, Value)> = Vec::new();
        for captures in tag.regex.captures_iter(body) {
            let (Some(key), Some(raw)) = (captures.name("key"), captures.name("value")) else {
                continue;
            };
            let key = key.as_str();
            if arguments.iter().any(|(seen, _)| seen == key) {
                continue;
            }
            let raw = if tag.strip {
                raw.as_str().trim()
            } else {
                raw.as_str()
            };
            let declared = types.iter().find(|(name, _)| name == key);
            let value = match declared.map(|(_, ty)| ty.as_str()) {
                None | Some("string") => None,
                ty => helpers::coerce_by_schema_type(raw.trim(), ty),
            };
            arguments.push((key.to_string(), value.unwrap_or_else(|| raw.into())));
        }
        arguments
    }

    fn parse_all(&self, text: &str, tools: Option<&[Tool]>) -> (String, Vec<ToolCall>) {
        let mut normal_text = String::new();
        let mut calls = Vec::new();
        let mut pos = 0;
        while let Some(Opener::Complete { start, end, .. }) =
            self.template.scan(text, pos, TOOL_CALLS, true)
        {
            normal_text.push_str(&text[pos..start]);
            let body = &text[end..];
            let (body_end, block_end) = find_close(body, self.template.closes(Field::ToolCalls))
                .unwrap_or((body.len(), body.len()));
            pos = end + block_end;
            let Some(name) = self
                .name(text, start)
                .filter(|name| is_declared(tools, name))
            else {
                tracing::warn!("dropping a response-template tool call to an undeclared function");
                continue;
            };
            let types = tools.map_or_else(Vec::new, |tools| param_types(tools, &name));
            let arguments: Map<String, Value> = self
                .arguments(&body[..body_end], &types)
                .into_iter()
                .collect();
            calls.push(ToolCall {
                function: FunctionCall {
                    name,
                    arguments: Value::Object(arguments).to_string(),
                },
            });
        }
        normal_text.push_str(&text[pos..]);
        (normal_text, calls)
    }

    fn start_call(
        &mut self,
        name: Option<String>,
        tools: &[Tool],
        result: &mut StreamingParseResult,
    ) {
        self.in_tool_call = true;
        let Some(name) = name.filter(|name| is_declared(Some(tools), name)) else {
            tracing::warn!("dropping a response-template tool call to an undeclared function");
            self.discard = true;
            return;
        };
        self.current_tool_id = self.current_tool_id.max(0);
        helpers::ensure_capacity(
            self.current_tool_id,
            &mut self.prev_tool_call_arr,
            &mut self.streamed_args_for_tool,
        );
        let index = self.current_tool_id as usize;
        self.prev_tool_call_arr[index] = json!({"name": name, "arguments": {}});
        self.current_tool_name_sent = true;
        result.calls.push(ToolCallItem {
            tool_index: index,
            name: Some(name),
            parameters: String::new(),
        });
    }

    /// Stream the arguments of complete tags in `body`, which ends a block
    /// or the last complete tag.
    fn stream_arguments(
        &mut self,
        body_end: usize,
        tools: &[Tool],
        result: &mut StreamingParseResult,
    ) {
        let index = self.current_tool_id as usize;
        let name = self.prev_tool_call_arr[index]["name"]
            .as_str()
            .unwrap_or_default();
        let types = param_types(tools, name);
        let body = &self.buffer[self.context..self.context + body_end];
        for (key, value) in self.arguments(body, &types) {
            let arguments = &mut self.prev_tool_call_arr[index]["arguments"];
            if arguments.get(&key).is_some() {
                continue;
            }
            let streamed = &mut self.streamed_args_for_tool[index];
            let fragment = format!(
                "{}{}: {value}",
                if streamed.is_empty() { "{" } else { ", " },
                Value::from(key.as_str())
            );
            streamed.push_str(&fragment);
            arguments[key] = value;
            result.calls.push(ToolCallItem {
                tool_index: index,
                name: None,
                parameters: fragment,
            });
        }
    }

    fn finish_call(&mut self, result: &mut StreamingParseResult) {
        if !self.discard {
            let index = self.current_tool_id as usize;
            let streamed = &mut self.streamed_args_for_tool[index];
            let closing = if streamed.is_empty() { "{}" } else { "}" };
            streamed.push_str(closing);
            result.calls.push(ToolCallItem {
                tool_index: index,
                name: None,
                parameters: closing.to_string(),
            });
            self.current_tool_id += 1;
        }
        self.in_tool_call = false;
        self.discard = false;
        self.current_tool_name_sent = false;
    }
}

fn is_declared(tools: Option<&[Tool]>, name: &str) -> bool {
    tools.is_none_or(|tools| tools.iter().any(|tool| tool.function.name == name))
}

fn param_types(tools: &[Tool], name: &str) -> Vec<(String, String)> {
    helpers::param_types_for_function(tools, name)
        .into_iter()
        .collect()
}

#[async_trait]
impl ToolParser for TemplateToolParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(self.parse_all(output, None))
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(self.parse_all(output, Some(tools)))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let mut result = StreamingParseResult::default();
        loop {
            if !self.in_tool_call {
                let found = self
                    .template
                    .scan(&self.buffer, self.context, TOOL_CALLS, false);
                let text_end = match found {
                    Some(Opener::Complete { start, .. } | Opener::Pending { start }) => start,
                    None => self.buffer.len(),
                };
                result
                    .normal_text
                    .push_str(&self.buffer[self.context..text_end]);
                let Some(Opener::Complete { start, end, .. }) = found else {
                    self.advance(text_end);
                    return Ok(result);
                };
                let name = self.name(&self.buffer, start);
                self.advance(end);
                self.start_call(name, tools, &mut result);
                continue;
            }
            // Only look at bytes not searched yet, plus room for a literal
            // that started before them.
            let text = &self.buffer[self.context..];
            let tag_close = self.template.tag().close.as_str();
            let closes = self.template.closes(Field::ToolCalls);
            let longest = closes
                .iter()
                .map(String::len)
                .chain([tag_close.len()])
                .max();
            let mut from = self.searched.saturating_sub(longest.unwrap_or(1) - 1);
            while !text.is_char_boundary(from) {
                from -= 1;
            }
            let close =
                find_close(&text[from..], closes).map(|(start, end)| (from + start, from + end));
            let body_end = match close {
                Some((start, _)) => start,
                None => text[from..]
                    .rfind(tag_close)
                    .map_or(0, |at| from + at + tag_close.len()),
            };
            self.searched = text.len();
            if !self.discard && body_end > 0 {
                self.stream_arguments(body_end, tools, &mut result);
            }
            let Some((_, end)) = close else {
                let searched = self.searched - body_end;
                self.advance(self.context + body_end);
                self.searched = searched;
                return Ok(result);
            };
            self.advance(self.context + end);
            self.finish_call(&mut result);
        }
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        matches!(
            self.template.scan(text, 0, TOOL_CALLS, true),
            Some(Opener::Complete { .. })
        )
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        // As in QwenXmlParser: arguments are streamed as spaced JSON, so close
        // the root object only when exactly the parsed arguments are open.
        let tool_index = self.prev_tool_call_arr.len().checked_sub(1)?;
        let actual = self.streamed_args_for_tool.get(tool_index)?;
        let expected = self.prev_tool_call_arr[tool_index].get("arguments")?;
        if !actual.is_empty()
            && serde_json::from_str::<Value>(&format!("{actual}}}"))
                .ok()
                .as_ref()
                == Some(expected)
        {
            return Some(vec![ToolCallItem {
                tool_index,
                name: None,
                parameters: "}".to_string(),
            }]);
        }
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn take_unstreamed_normal_text(&mut self) -> String {
        let held = std::mem::take(&mut self.buffer);
        let context = std::mem::replace(&mut self.context, 0);
        if self.in_tool_call {
            // An unfinished tag; the arguments before it are recovered by
            // get_unstreamed_tool_args.
            return String::new();
        }
        // Held text is a possible opener. Drop it only if it is one.
        match self.template.scan(&held, context, TOOL_CALLS, true) {
            Some(Opener::Complete { start, .. }) => held[context..start].to_string(),
            _ => held[context..].to_string(),
        }
    }

    fn reset(&mut self) {
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        self.context = 0;
        self.searched = 0;
        self.in_tool_call = false;
        self.discard = false;
    }
}
