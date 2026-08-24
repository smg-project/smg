use std::collections::HashSet;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;

use crate::{
    errors::ParserResult,
    partial_json::PartialJson,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const TOOL_CALL_JSON_START: &str = "<|content_invoke_tool_json|>";
const TOOL_CALL_TEXT_START: &str = "<|content_invoke_tool_text|>";
const END_MESSAGE: &str = "<|end_message|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";

const STREAM_CONTROL_TOKENS: [&str; 7] = [
    TOOL_CALL_JSON_START,
    TOOL_CALL_TEXT_START,
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    MODEL_END_SAMPLING,
];

const HEADER_CONTROL_TOKENS: [&str; 7] = [
    TOOL_CALL_JSON_START,
    TOOL_CALL_TEXT_START,
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    MODEL_END_SAMPLING,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingState {
    Text,
    MessageHeader,
    ToolCallJson,
    ToolCallEnd,
    DiscardToolCall,
}

/// Parser for Inkling's TML tool-call format.
///
/// A call is emitted as:
/// `<|message_model|>name<|content_invoke_tool_json|>{"name":"...",`
/// `"args":{...}}<|end_message|>`.
/// TML response framing tokens around ordinary assistant text are removed before
/// that text is returned to the OpenAI-compatible API. Headerless text-mode
/// invocations cannot be represented losslessly by the OpenAI function-call
/// shape, so their frame is safely suppressed instead of becoming answer text.
pub struct InklingParser {
    partial_json: PartialJson,
    buffer: String,
    state: StreamingState,
    current_tool_index: usize,
    current_tool_name: Option<String>,
}

impl InklingParser {
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
            buffer: String::new(),
            state: StreamingState::Text,
            current_tool_index: 0,
            current_tool_name: None,
        }
    }

    /// Strings that only ever open Inkling's native tool-call syntax; banning
    /// them makes tool calls unreachable when `tool_choice` is `"none"`. Both
    /// invocation modes are covered: JSON-arguments and TML text-mode.
    pub const TOOL_CALL_BAN_STRINGS: &'static [&'static str] =
        &[TOOL_CALL_JSON_START, TOOL_CALL_TEXT_START];

    /// Build an xgrammar structural tag that constrains the JSON arguments for
    /// each declared tool while retaining Inkling's native TML framing.
    pub fn build_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let tags = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| {
                let name = serde_json::to_string(&tool.function.name).unwrap_or_default();
                serde_json::json!({
                    "begin": format!(
                        "{TOOL_CALL_JSON_START}{{\"name\":{name},\"args\":"
                    ),
                    "content": {
                        "type": "json_schema",
                        "json_schema": &tool.function.parameters,
                    },
                    "end": format!("}}{END_MESSAGE}"),
                })
            })
            .collect::<Vec<_>>();

        serde_json::json!({
            "format": {
                "type": "triggered_tags",
                "triggers": [TOOL_CALL_JSON_START],
                "tags": tags,
                "at_least_one": at_least_one,
            }
        })
    }

    fn clean_normal_text(text: &str) -> String {
        let controls = [
            MESSAGE_MODEL,
            CONTENT_TEXT,
            CONTENT_THINKING,
            END_MESSAGE,
            MODEL_END_SAMPLING,
        ];
        let mut output = String::new();
        let mut cursor = 0;
        let mut in_header = false;

        while cursor < text.len() {
            let remaining = &text[cursor..];
            let Some((start, token)) = find_earliest_token(remaining, &controls) else {
                if !in_header {
                    output.push_str(remaining);
                }
                break;
            };

            if !in_header {
                output.push_str(&remaining[..start]);
            }
            cursor += start + token.len();
            in_header = token == MESSAGE_MODEL;
        }

        output
    }

    fn parse_complete_impl(
        text: &str,
        allowed_tools: Option<&HashSet<&str>>,
    ) -> (String, Vec<ToolCall>) {
        let tool_markers = [TOOL_CALL_JSON_START, TOOL_CALL_TEXT_START];
        let mut normal_text = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0;

        while let Some((relative_start, marker)) =
            find_earliest_token(&text[cursor..], &tool_markers)
        {
            let marker_start = cursor + relative_start;
            normal_text.push_str(&Self::clean_normal_text(&text[cursor..marker_start]));
            let payload_start = marker_start + marker.len();

            if marker == TOOL_CALL_TEXT_START {
                tracing::warn!(
                    "Ignoring TML text-mode tool invocation; OpenAI tool calls require structured JSON"
                );
                cursor = after_discarded_tool_frame(text, payload_start);
                continue;
            }

            let payload = &text[payload_start..];
            let whitespace = payload.len() - payload.trim_start().len();
            let json_start = payload_start + whitespace;
            let Some(json_len) = complete_json_object_len(&text[json_start..]) else {
                tracing::warn!("Ignoring malformed TML JSON tool invocation");
                cursor = after_discarded_tool_frame(text, payload_start);
                continue;
            };
            let json_end = json_start + json_len;
            let after_json = &text[json_end..];
            let whitespace = after_json.len() - after_json.trim_start().len();
            let end_start = json_end + whitespace;

            let Some(call) = parse_tool_call_json(&text[json_start..json_end], allowed_tools)
            else {
                // Malformed and undefined calls are protocol data. Suppress the
                // whole frame, then resume at the next typed message.
                cursor = after_discarded_tool_frame(text, json_end);
                continue;
            };
            calls.push(call);

            if text[end_start..].starts_with(END_MESSAGE) {
                cursor = end_start + END_MESSAGE.len();
            } else if text[end_start..].starts_with(MODEL_END_SAMPLING) {
                cursor = end_start + MODEL_END_SAMPLING.len();
            } else {
                // Match the streaming parser's permissive ToolCallEnd state:
                // after a complete valid JSON object, a missing end marker does
                // not turn subsequent bytes into tool payload.
                cursor = end_start;
            }
        }

        normal_text.push_str(&Self::clean_normal_text(&text[cursor..]));
        (normal_text, calls)
    }

    fn process_text(&mut self, result: &mut StreamingParseResult) -> bool {
        let state_markers = [MESSAGE_MODEL, TOOL_CALL_JSON_START, TOOL_CALL_TEXT_START];
        if let Some((start, marker)) = find_earliest_token(&self.buffer, &state_markers) {
            let normal_text = Self::clean_normal_text(&self.buffer[..start]);
            result.normal_text.push_str(&normal_text);
            self.buffer.drain(..start + marker.len());
            self.state = if marker == MESSAGE_MODEL {
                StreamingState::MessageHeader
            } else if marker == TOOL_CALL_JSON_START {
                StreamingState::ToolCallJson
            } else {
                tracing::warn!(
                    "Ignoring TML text-mode tool invocation; OpenAI tool calls require structured JSON"
                );
                StreamingState::DiscardToolCall
            };
            return true;
        }

        let keep = longest_partial_token_suffix(&self.buffer, &STREAM_CONTROL_TOKENS);
        let safe_len = self.buffer.len() - keep;
        if safe_len > 0 {
            let safe_text = self.buffer[..safe_len].to_string();
            self.buffer.drain(..safe_len);
            result
                .normal_text
                .push_str(&Self::clean_normal_text(&safe_text));
        }
        false
    }

    fn process_message_header(&mut self) -> bool {
        if let Some((start, token)) = find_earliest_token(&self.buffer, &HEADER_CONTROL_TOKENS) {
            // Everything before the content-type token is an author, tool
            // name, or channel header. It is metadata and must not be emitted
            // as assistant content.
            self.buffer.drain(..start + token.len());
            self.state = if token == MESSAGE_MODEL {
                StreamingState::MessageHeader
            } else if token == TOOL_CALL_JSON_START {
                StreamingState::ToolCallJson
            } else if token == TOOL_CALL_TEXT_START {
                tracing::warn!(
                    "Ignoring TML text-mode tool invocation; OpenAI tool calls require structured JSON"
                );
                StreamingState::DiscardToolCall
            } else {
                StreamingState::Text
            };
            return true;
        }

        // Header bytes are never user-visible. Drop author/channel bytes while
        // retaining only a possible control-token prefix across chunk splits.
        let keep = longest_partial_token_suffix(&self.buffer, &HEADER_CONTROL_TOKENS);
        let safe_len = self.buffer.len() - keep;
        if safe_len > 0 {
            self.buffer.drain(..safe_len);
        }
        false
    }

    fn process_tool_call(
        &mut self,
        allowed_tools: &HashSet<&str>,
        result: &mut StreamingParseResult,
    ) -> bool {
        let whitespace = self.buffer.len() - self.buffer.trim_start().len();
        if whitespace > 0 {
            self.buffer.drain(..whitespace);
        }
        if self.buffer.is_empty() {
            return false;
        }

        if !self.buffer.starts_with('{') {
            self.current_tool_name = None;
            self.state = StreamingState::DiscardToolCall;
            return true;
        }

        if self.current_tool_name.is_none() {
            if let Ok((Value::Object(object), _)) =
                self.partial_json.parse_value(&self.buffer, false)
            {
                if let Some(name) = object.get("name").and_then(Value::as_str) {
                    if allowed_tools.contains(name) {
                        self.current_tool_name = Some(name.to_string());
                        result.calls.push(ToolCallItem {
                            tool_index: self.current_tool_index,
                            name: Some(name.to_string()),
                            parameters: String::new(),
                        });
                    } else {
                        tracing::debug!("Inkling attempted to call undefined tool: {}", name);
                        self.state = StreamingState::DiscardToolCall;
                        return true;
                    }
                }
            }
        }

        let Some(json_len) = complete_json_object_len(&self.buffer) else {
            return false;
        };
        let json = &self.buffer[..json_len];
        let Some(call) = parse_tool_call_json(json, Some(allowed_tools)) else {
            self.current_tool_name = None;
            self.state = StreamingState::DiscardToolCall;
            return true;
        };

        // A complete object may expose the name and arguments in the same
        // chunk. Emit the name first in that case, as required by stream APIs.
        if self.current_tool_name.is_none() {
            result.calls.push(ToolCallItem {
                tool_index: self.current_tool_index,
                name: Some(call.function.name.clone()),
                parameters: String::new(),
            });
        }
        result.calls.push(ToolCallItem {
            tool_index: self.current_tool_index,
            name: None,
            parameters: call.function.arguments,
        });

        self.buffer.drain(..json_len);
        self.current_tool_index += 1;
        self.current_tool_name = None;
        self.state = StreamingState::ToolCallEnd;
        true
    }

    fn process_tool_call_end(&mut self) -> bool {
        let whitespace = self.buffer.len() - self.buffer.trim_start().len();
        if whitespace > 0 {
            self.buffer.drain(..whitespace);
        }
        if self.buffer.is_empty() {
            return false;
        }
        if self.buffer.starts_with(END_MESSAGE) {
            self.buffer.drain(..END_MESSAGE.len());
            self.state = StreamingState::Text;
            return true;
        }
        if END_MESSAGE.starts_with(&self.buffer) {
            return false;
        }

        // Be permissive when a backend omits the end token after an otherwise
        // complete call. The remaining bytes still need normal text handling.
        self.state = StreamingState::Text;
        true
    }

    fn discard_tool_call(&mut self) -> bool {
        let end_tokens = [END_MESSAGE, MODEL_END_SAMPLING];
        if let Some((end, token)) = find_earliest_token(&self.buffer, &end_tokens) {
            self.buffer.drain(..end + token.len());
            self.current_tool_name = None;
            self.state = StreamingState::Text;
            return true;
        }

        let keep = longest_partial_token_suffix(&self.buffer, &end_tokens);
        let safe_len = self.buffer.len() - keep;
        if safe_len > 0 {
            self.buffer.drain(..safe_len);
        }
        false
    }
}

impl Default for InklingParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for InklingParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_complete_impl(text, None))
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        let allowed_tools = tools
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<HashSet<_>>();
        Ok(Self::parse_complete_impl(text, Some(&allowed_tools)))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let allowed_tools = tools
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<HashSet<_>>();
        let mut result = StreamingParseResult::default();

        loop {
            let progressed = match self.state {
                StreamingState::Text => self.process_text(&mut result),
                StreamingState::MessageHeader => self.process_message_header(),
                StreamingState::ToolCallJson => self.process_tool_call(&allowed_tools, &mut result),
                StreamingState::ToolCallEnd => self.process_tool_call_end(),
                StreamingState::DiscardToolCall => self.discard_tool_call(),
            };
            if !progressed {
                break;
            }
        }

        Ok(result)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(TOOL_CALL_JSON_START) || text.contains(TOOL_CALL_TEXT_START)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.state = StreamingState::Text;
        self.current_tool_index = 0;
        self.current_tool_name = None;
    }
}

fn parse_tool_call_json(json: &str, allowed_tools: Option<&HashSet<&str>>) -> Option<ToolCall> {
    let value = serde_json::from_str::<Value>(json).ok()?;
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?;
    let args = object.get("args")?.as_object()?;

    if allowed_tools.is_some_and(|tools| !tools.contains(name)) {
        tracing::debug!("Inkling attempted to call undefined tool: {}", name);
        return None;
    }

    Some(ToolCall {
        function: FunctionCall {
            name: name.to_string(),
            arguments: serde_json::to_string(args).ok()?,
        },
    })
}

/// Find the earliest complete protocol token, preferring the token list order
/// when two entries start at the same byte offset.
fn find_earliest_token<'a>(text: &str, tokens: &'a [&'a str]) -> Option<(usize, &'a str)> {
    tokens
        .iter()
        .filter_map(|token| text.find(token).map(|start| (start, *token)))
        .min_by_key(|(start, _)| *start)
}

/// Return the first byte after a discarded tool frame. Both end markers match
/// the streaming parser's `DiscardToolCall` recovery behavior. If a malformed
/// frame is unterminated, all remaining bytes belong to that protocol frame.
fn after_discarded_tool_frame(text: &str, payload_start: usize) -> usize {
    let end_tokens = [END_MESSAGE, MODEL_END_SAMPLING];
    find_earliest_token(&text[payload_start..], &end_tokens)
        .map_or(text.len(), |(relative_end, token)| {
            payload_start + relative_end + token.len()
        })
}

/// Return the byte length of one complete top-level JSON object. Braces inside
/// quoted strings are ignored, and trailing data is left for the caller.
fn complete_json_object_len(text: &str) -> Option<usize> {
    if !text.starts_with('{') {
        return None;
    }

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, byte) in text.bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn longest_partial_token_suffix(buffer: &str, tokens: &[&str]) -> usize {
    tokens
        .iter()
        .flat_map(|token| {
            token
                .char_indices()
                .skip(1)
                .map(move |(index, _)| &token[..index])
        })
        .filter(|prefix| buffer.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}
