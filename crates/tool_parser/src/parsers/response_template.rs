//! Tool parser for a tokenizer's response template.

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;
use smg_response_template::{
    adapter::{CallItem, ResponseParserState},
    ResponseTemplate,
};

use crate::{
    errors::{ParserError, ParserResult},
    json_format,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Reads the `tool_calls` field of a response template with the transformers
/// parser of a [`ResponseParserState`]. When the reasoning parser of the same
/// output feeds the state, the input is its content and the calls are the ones
/// it closed; otherwise this parser feeds the state and reasoning text stays in
/// the content. A call is reported whole, name and arguments, when its region
/// closes. When the template's opener captures the name of every call, a
/// stream reports the name once the region is open and the arguments when it
/// closes, before any output that followed the close. Without an attached
/// state the parser has no prompt, as transformers with `prefix=""`.
pub struct TemplateToolParser {
    template: ResponseTemplate,
    state: Option<ResponseParserState>,
    /// Calls reported so far.
    reported: usize,
    /// The index of the call reported by name, whose arguments are to come.
    named: Option<usize>,
    /// Calls closed at the end of the output.
    unstreamed: Vec<ToolCallItem>,
}

impl TemplateToolParser {
    pub fn new(template: ResponseTemplate) -> Self {
        Self {
            template,
            state: None,
            reported: 0,
            named: None,
            unstreamed: Vec::new(),
        }
    }

    fn new_state(&self, tools: &[Tool]) -> ResponseParserState {
        let tools: Vec<Value> = tools
            .iter()
            .filter_map(|tool| serde_json::to_value(tool).ok())
            .collect();
        ResponseParserState::new(&self.template, "", &tools, false)
    }

    /// Report a new call, the next index.
    fn next(&mut self, name: String, parameters: String) -> ToolCallItem {
        self.reported += 1;
        ToolCallItem {
            tool_index: self.reported - 1,
            name: Some(name),
            parameters,
        }
    }

    fn items(&mut self, items: Vec<CallItem>) -> Vec<ToolCallItem> {
        let mut reported = Vec::new();
        for item in items {
            match item {
                CallItem::Call(call) => {
                    reported.push(self.next(call.name, arguments(call.arguments)));
                }
                CallItem::Name(name) => {
                    let item = self.next(name, String::new());
                    self.named = Some(item.tool_index);
                    reported.push(item);
                }
                CallItem::Arguments(value) => {
                    let parameters = arguments(value);
                    if let Some(tool_index) = self.named.take().filter(|_| !parameters.is_empty()) {
                        reported.push(ToolCallItem {
                            tool_index,
                            name: None,
                            parameters,
                        });
                    }
                }
            }
        }
        reported
    }
}

/// `arguments` as a JSON string: kept when the template made it one.
fn arguments(value: Value) -> String {
    match value {
        Value::String(text) => text,
        value => json_format::to_string(&value),
    }
}

fn failed(error: smg_response_template::ParseError) -> ParserError {
    ParserError::ParsingFailed(format!("response template: {error}"))
}

#[async_trait]
impl ToolParser for TemplateToolParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_with_tools(output, &[]).await
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        let state = self.state.clone().unwrap_or_else(|| self.new_state(tools));
        let (mut text, mut calls) = state.tools(Some(output)).map_err(failed)?;
        let (rest, more) = state.tools(None).map_err(failed)?;
        text.push_str(&rest);
        calls.extend(more);
        let calls = calls
            .into_iter()
            .map(|call| ToolCall {
                function: FunctionCall {
                    name: call.name,
                    arguments: arguments(call.arguments),
                },
            })
            .collect();
        Ok((text, calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        let state = match &self.state {
            Some(state) => state.clone(),
            None => self.state.insert(self.new_state(tools)).clone(),
        };
        let (normal_text, items) = state.tool_items(Some(chunk)).unwrap_or_else(|error| {
            tracing::warn!("response template: {error}; the rest of the output is not parsed");
            (chunk.to_owned(), Vec::new())
        });
        Ok(StreamingParseResult {
            normal_text,
            calls: self.items(items),
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        let state = self.new_state(&[]);
        [Some(text), None]
            .into_iter()
            .any(|text| state.tools(text).is_ok_and(|(_, calls)| !calls.is_empty()))
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        (!self.unstreamed.is_empty()).then(|| self.unstreamed.clone())
    }

    /// Ends the output: the rest of the content, and the calls it closed for
    /// `get_unstreamed_tool_args`.
    fn take_unstreamed_normal_text(&mut self) -> String {
        let Some(state) = self.state.clone() else {
            return String::new();
        };
        let (text, items) = state.tool_items(None).unwrap_or_else(|error| {
            tracing::warn!("response template: {error}");
            (String::new(), Vec::new())
        });
        // An error after deferred output, which has no next chunk to wait for.
        if let Some(error) = state.take_error() {
            tracing::warn!("response template: {error}");
        }
        self.unstreamed = self.items(items);
        text
    }

    fn reset(&mut self) {
        self.state = None;
        self.reported = 0;
        self.named = None;
        self.unstreamed.clear();
    }

    fn attach_response_parser_state(&mut self, state: ResponseParserState) {
        self.state = Some(state);
    }
}
