use std::collections::HashMap;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Qwen XML format parser for tool calls
///
/// Handles the Qwen XML specific XML format:
/// `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`
///
/// Features:
/// - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
/// - XML-style function declaration: `<function=name>`
/// - XML-style parameters: `<parameter=key>value</parameter>`
/// - String-typed parameter values stream as they are generated; other types
///   arrive whole once their value is complete (coercion needs all of it)
///
/// Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8?chat_template=default
pub struct QwenXmlParser {
    /// Regex for extracting tool calls in parse_complete
    extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,

    /// XML format streaming state
    in_tool_call: bool,
    current_function_name: String,
    current_parameters: serde_json::Map<String, Value>,

    /// Precompiled regex patterns for XML format parsing
    xml_function_pattern: Regex,
    xml_param_pattern: Regex,
    xml_param_open_pattern: Regex,

    /// The string parameter whose value is being streamed, if any
    open_parameter: Option<OpenParameter>,
}

/// Closing tags a parameter value runs into. A suffix of the buffer that is a
/// prefix of one of them is held back until the next chunk decides.
const CLOSING_TAGS: [&str; 3] = ["</parameter>", "</function>", "</tool_call>"];

/// A string-typed `<parameter>` whose `</parameter>` has not arrived yet.
///
/// Its value is streamed as it is generated, JSON-escaped and trimmed like
/// the complete parse, instead of being held until the parameter closes.
struct OpenParameter {
    key: String,
    /// Offset in the buffer of the first byte after the opening tag.
    value_start: usize,
    /// Offset in the buffer up to which the value has been streamed. `None`
    /// until the first non-whitespace byte: the `"key": "` prefix goes out
    /// with it, so a parameter without a value yet is never invented.
    streamed_end: Option<usize>,
    /// The escaped value streamed so far; only the remainder goes out when
    /// the parameter closes.
    sent: String,
    /// A JSON string literal (`"..."`), which the complete parse unwraps: it
    /// is left to arrive whole.
    literal: bool,
}

/// The body of `text` as a JSON string, without the surrounding quotes.
fn json_string_body(text: &str) -> String {
    let quoted = serde_json::to_string(text).unwrap_or_default();
    quoted
        .strip_prefix('"')
        .and_then(|body| body.strip_suffix('"'))
        .unwrap_or_default()
        .to_string()
}

/// Parse a raw parameter value, similar to Python's `_safe_val`.
///
/// Argument values are treated **literally** — no HTML-entity decoding. This
/// matches Qwen's own official API (DashScope), verified for both Qwen3-Coder
/// and Qwen3.5: a tool argument whose value contains `&amp;`, `&lt;`, `&#39;`
/// is returned with those entities intact. The Qwen XML tool format is not
/// HTML-escaped on render either (the chat template emits values via
/// `| tojson | safe` / `| string`), so parsing must not unescape it. vLLM's
/// `Qwen3CoderToolParser`, SGLang's `qwen3_coder_detector`, and Qwen-Agent all
/// agree, passing argument values through verbatim.
///
/// 1. Try to parse as JSON (numbers, booleans, null, objects, arrays)
/// 2. Fall back to string if JSON parsing fails
fn safe_val(raw: &str) -> Value {
    let trimmed = raw.trim();

    // Try JSON parsing first
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v;
    }

    // Handle Python-style literals (True, False, None)
    match trimmed {
        "True" => return Value::Bool(true),
        "False" => return Value::Bool(false),
        "None" => return Value::Null,
        _ => {}
    }

    // Fall back to string
    Value::String(trimmed.to_string())
}

/// Coerce an XML parameter value by its declared schema type, falling back to
/// [`safe_val`] inference when the type is unknown.
///
/// Values are treated literally; see [`safe_val`] for why the format is not
/// HTML-unescaped.
fn coerce_value(raw: &str, declared_type: Option<&str>) -> Value {
    let trimmed = raw.trim();
    helpers::coerce_by_schema_type(trimmed, declared_type).unwrap_or_else(|| safe_val(raw))
}

impl QwenXmlParser {
    /// Create a new Qwen XML parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        // Support XML format: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
        let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>";
        let extractor = Regex::new(pattern).expect("Valid regex pattern");

        // Precompile XML format regex patterns for performance
        let xml_function_pattern =
            Regex::new(r"<function=([^>]+)>").expect("Valid XML function pattern");
        let xml_param_pattern = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
            .expect("Valid XML parameter pattern");
        let xml_param_open_pattern =
            Regex::new(r"<parameter=([^>]+)>").expect("Valid XML parameter start pattern");

        Self {
            extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            tool_call_start_token: "<tool_call>",
            tool_call_end_token: "</tool_call>",
            in_tool_call: false,
            current_function_name: String::new(),
            current_parameters: serde_json::Map::new(),
            xml_function_pattern,
            xml_param_pattern,
            xml_param_open_pattern,
            open_parameter: None,
        }
    }

    /// Parse XML format tool call: <function=name><parameter=key>value</parameter></function>
    fn parse_xml_format(&self, content: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        let function_captures = self
            .xml_function_pattern
            .captures(content)
            .ok_or_else(|| ParserError::ParsingFailed("No function name found".to_string()))?;

        let function_name = function_captures
            .get(1)
            .ok_or_else(|| ParserError::ParsingFailed("Function name capture failed".to_string()))?
            .as_str()
            .trim()
            .to_string();

        if function_name.is_empty() {
            return Ok(None);
        }

        let param_types = helpers::param_types_for_function(tools, &function_name);
        let mut parameters = serde_json::Map::new();

        for cap in self.xml_param_pattern.captures_iter(content) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let json_value = coerce_value(value, param_types.get(&key).map(String::as_str));
                parameters.insert(key, json_value);
            }
        }

        let arguments = serde_json::to_string(&parameters)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(Some(ToolCall {
            function: FunctionCall {
                name: function_name,
                arguments,
            },
        }))
    }

    /// Parse and stream complete parameters from buffer
    /// Returns tool call items to emit (similar to Python's _parse_and_stream_parameters)
    fn parse_and_stream_parameters(
        &mut self,
        tools: &[Tool],
        parameter_end: usize,
    ) -> Vec<ToolCallItem> {
        let mut calls: Vec<ToolCallItem> = vec![];
        let param_types = helpers::param_types_for_function(tools, &self.current_function_name);

        // Leave parameters from subsequent coalesced calls for their own iteration.
        let mut new_params = serde_json::Map::new();
        for cap in self
            .xml_param_pattern
            .captures_iter(&self.buffer[..parameter_end])
        {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let json_value = coerce_value(value, param_types.get(&key).map(String::as_str));
                new_params.insert(key, json_value);
            }
        }

        // Calculate parameter diff and stream updates
        if new_params != self.current_parameters {
            let current_args = &mut self.streamed_args_for_tool[self.current_tool_id as usize];

            if self.current_parameters.is_empty() {
                // First parameter(s) - build JSON fragment (without closing brace)
                let mut items = Vec::new();
                for (key, value) in &new_params {
                    let key_json =
                        serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    items.push(format!("{key_json}: {value_json}"));
                }
                let json_fragment = format!("{{{}", items.join(", "));

                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: None,
                    parameters: json_fragment.clone(),
                });
                *current_args = json_fragment;
            } else {
                // Additional parameters - add them incrementally
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut continuation_parts = Vec::new();
                    for key in new_keys {
                        if let Some(value) = new_params.get(key) {
                            let key_json =
                                serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                            let value_json = serde_json::to_string(value).unwrap_or_default();
                            continuation_parts.push(format!("{key_json}: {value_json}"));
                        }
                    }

                    let json_fragment = format!(", {}", continuation_parts.join(", "));

                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });
                    current_args.push_str(&json_fragment);
                }
            }

            // Update current state
            self.current_parameters.clone_from(&new_params);
            if let Some(tool_obj) =
                self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut()
            {
                tool_obj.insert("arguments".to_string(), Value::Object(new_params));
            }
        }

        calls
    }

    /// Close the string parameter streamed so far once its `</parameter>` has
    /// arrived: its remainder goes out before the buffer's complete parameters
    /// are parsed, which records it so that it is not sent a second time.
    fn finish_closed_parameter(
        &mut self,
        tools: &[Tool],
        parameter_end: usize,
    ) -> Vec<ToolCallItem> {
        let closed_at = self.open_parameter.as_ref().and_then(|open| {
            self.buffer[open.value_start..parameter_end]
                .find("</parameter>")
                .map(|close| open.value_start + close)
        });
        let Some(value_end) = closed_at else {
            return Vec::new();
        };
        let param_types = helpers::param_types_for_function(tools, &self.current_function_name);
        self.finish_open_parameter(&param_types, value_end)
    }

    /// Stream the value of the parameter still open at the end of the buffer
    /// when its declared type is `string`. Other types need the complete value
    /// for coercion and keep arriving whole from `parse_and_stream_parameters`,
    /// which runs first: the open parameter follows every complete one.
    fn stream_open_parameter(&mut self, tools: &[Tool], parameter_end: usize) -> Vec<ToolCallItem> {
        let mut calls = Vec::new();
        let param_types = helpers::param_types_for_function(tools, &self.current_function_name);

        // The last opening tag without a closing tag is the parameter now open.
        let region = &self.buffer[..parameter_end];
        let Some(open_tag) = self.xml_param_open_pattern.captures_iter(region).last() else {
            return calls;
        };
        let value_start = open_tag.get(0).map_or(parameter_end, |m| m.end());
        let value_region = &region[value_start..];
        if value_region.contains("</parameter>") {
            return calls;
        }
        let key = open_tag.get(1).map_or("", |m| m.as_str()).trim();
        if param_types.get(key).map(String::as_str) != Some("string") {
            return calls;
        }
        if self
            .open_parameter
            .as_ref()
            .is_none_or(|open| open.key != key || open.value_start != value_start)
        {
            self.open_parameter = Some(OpenParameter {
                key: key.to_string(),
                value_start,
                streamed_end: None,
                sent: String::new(),
                literal: false,
            });
        }

        // The value runs to a closing tag (a `</function>` without the
        // `</parameter>` ends it too); a suffix that may begin one is held back.
        let region_end = match value_region.find("</function>") {
            Some(end) => value_start + end,
            None => {
                let held = CLOSING_TAGS
                    .iter()
                    .filter_map(|tag| helpers::ends_with_partial_token(value_region, tag))
                    .max()
                    .unwrap_or(0);
                parameter_end - held
            }
        };

        let tool_index = self.current_tool_id as usize;
        let Some(open) = self.open_parameter.as_mut() else {
            return calls;
        };
        if open.literal {
            return calls;
        }
        let from = match open.streamed_end {
            Some(end) => end,
            None => {
                let raw = &self.buffer[value_start..region_end];
                let from = value_start + (raw.len() - raw.trim_start().len());
                if from >= region_end {
                    return calls;
                }
                if self.buffer[from..].starts_with('"') {
                    open.literal = true;
                    return calls;
                }
                from
            }
        };
        if region_end <= from {
            return calls;
        }
        let fresh = self.buffer[from..region_end].trim_end();
        if fresh.is_empty() {
            return calls;
        }
        let body = json_string_body(fresh);
        let mut delta = String::new();
        if open.streamed_end.is_none() {
            let key_json =
                serde_json::to_string(&open.key).unwrap_or_else(|_| format!("\"{}\"", open.key));
            delta.push_str(if self.streamed_args_for_tool[tool_index].is_empty() {
                "{"
            } else {
                ", "
            });
            delta.push_str(&key_json);
            delta.push_str(": \"");
        }
        delta.push_str(&body);
        open.sent.push_str(&body);
        open.streamed_end = Some(from + fresh.len());
        self.streamed_args_for_tool[tool_index].push_str(&delta);
        calls.push(ToolCallItem {
            tool_index,
            name: None,
            parameters: delta,
        });
        calls
    }

    /// Close the streamed parameter whose value ends at `value_end`: hand out
    /// what the complete value still lacks and the closing quote, and record
    /// the parameter so `parse_and_stream_parameters` does not send it again.
    fn finish_open_parameter(
        &mut self,
        param_types: &HashMap<String, String>,
        value_end: usize,
    ) -> Vec<ToolCallItem> {
        let Some(open) = self.open_parameter.take() else {
            return Vec::new();
        };
        if open.streamed_end.is_none() {
            return Vec::new();
        }
        let tool_index = self.current_tool_id as usize;
        let value = coerce_value(
            &self.buffer[open.value_start..value_end],
            param_types.get(&open.key).map(String::as_str),
        );
        let full = serde_json::to_string(&value).unwrap_or_default();
        let body = full
            .strip_prefix('"')
            .and_then(|body| body.strip_suffix('"'))
            .unwrap_or_default();
        // What was streamed cannot be retracted: should the complete value
        // not extend it, the quote alone closes the string.
        let mut tail = body
            .strip_prefix(open.sent.as_str())
            .unwrap_or_default()
            .to_string();
        tail.push('"');
        self.streamed_args_for_tool[tool_index].push_str(&tail);
        if let Some(arguments) = self.prev_tool_call_arr[tool_index]
            .get_mut("arguments")
            .and_then(Value::as_object_mut)
        {
            arguments.insert(open.key.clone(), value.clone());
        }
        self.current_parameters.insert(open.key, value);
        vec![ToolCallItem {
            tool_index,
            name: None,
            parameters: tail,
        }]
    }

    /// Shared non-streaming parse, schema-aware when `tools` are provided.
    fn parse_complete_inner(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Qwen XML format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where the first tool call begins
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find(self.tool_call_start_token)
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        // Extract tool calls
        let mut parsed = Vec::new();
        for captures in self.extractor.captures_iter(text) {
            if let Some(content_str) = captures.get(1) {
                let content = content_str.as_str().trim();

                match self.parse_xml_format(content, tools) {
                    Ok(Some(tool)) => parsed.push(tool),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!("Failed to parse XML tool call: {:?}", e);
                        continue;
                    }
                }
            }
        }

        // If no tools were successfully parsed despite having markers, return entire text
        if parsed.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, parsed))
    }

    /// Reset streaming state for next tool call
    fn reset_streaming_state(&mut self) {
        self.in_tool_call = false;
        self.current_tool_name_sent = false;
        self.current_function_name.clear();
        self.current_parameters.clear();
        self.open_parameter = None;
    }
}

impl Default for QwenXmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for QwenXmlParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, &[])
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, tools)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        let mut normal_text = String::new();
        let mut calls: Vec<ToolCallItem> = vec![];

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            // If we're not in a tool call and don't see a start token, return normal text
            if !self.in_tool_call && !self.buffer.contains(self.tool_call_start_token) {
                // Check for partial start token
                if helpers::ends_with_partial_token(&self.buffer, self.tool_call_start_token)
                    .is_none()
                {
                    normal_text.push_str(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Look for tool call start
            if !self.in_tool_call {
                if let Some(s) = self.buffer.find(self.tool_call_start_token) {
                    normal_text.push_str(&self.buffer[..s]);
                    self.buffer = self.buffer[s + self.tool_call_start_token.len()..].to_string();
                    self.in_tool_call = true;
                    self.current_tool_name_sent = false;
                    self.current_function_name.clear();
                    self.current_parameters.clear();
                    continue;
                } else {
                    break;
                }
            }

            // We're in a tool call, try to parse function name if not sent yet
            if !self.current_tool_name_sent {
                if let Some(captures) = self.xml_function_pattern.captures(&self.buffer) {
                    if let Some(name_match) = captures.get(1) {
                        let function_name = name_match.as_str().trim().to_string();

                        // Validate function name
                        if tool_indices.contains_key(&function_name) {
                            self.current_function_name.clone_from(&function_name);
                            self.current_tool_name_sent = true;

                            // Initialize tool call tracking
                            if self.current_tool_id == -1 {
                                self.current_tool_id = 0;
                            }

                            // Ensure tracking arrays are large enough
                            helpers::ensure_capacity(
                                self.current_tool_id,
                                &mut self.prev_tool_call_arr,
                                &mut self.streamed_args_for_tool,
                            );

                            // Store tool call info
                            self.prev_tool_call_arr[self.current_tool_id as usize] = serde_json::json!({
                                "name": function_name,
                                "arguments": {}
                            });

                            // Send tool name
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: Some(function_name),
                                parameters: String::new(),
                            });

                            // Remove processed function declaration from buffer
                            // Safe: captures.get(0) always returns Some (group 0 is the entire match)
                            self.buffer =
                                self.buffer[captures.get(0).map_or(0, |m| m.end())..].to_string();
                            continue;
                        } else {
                            // Invalid function name, reset state
                            tracing::warn!("Invalid function name: {}", function_name);
                            self.reset_streaming_state();
                            normal_text.push_str(&self.buffer);
                            self.buffer.clear();
                            break;
                        }
                    }
                } else {
                    // Function name not complete yet, wait for more text
                    break;
                }
            }

            // Parse parameters: string values as they stream, the rest once complete
            if self.current_tool_name_sent {
                let end_pos = self.buffer.find(self.tool_call_end_token);
                let parameter_end = end_pos.unwrap_or(self.buffer.len());
                // In buffer order: the streamed parameter that just closed,
                // the complete parameters, then the one still open.
                calls.extend(self.finish_closed_parameter(tools, parameter_end));
                calls.extend(self.parse_and_stream_parameters(tools, parameter_end));
                calls.extend(self.stream_open_parameter(tools, parameter_end));

                // Check if tool call is complete
                if let Some(end_pos) = end_pos {
                    // Parameter fragments leave the root open; braces in values are data.
                    // A string still streaming when the call closes is closed with it.
                    let unclosed_string = self
                        .open_parameter
                        .take()
                        .is_some_and(|open| open.streamed_end.is_some());
                    let current_args =
                        &mut self.streamed_args_for_tool[self.current_tool_id as usize];
                    let mut closing = String::new();
                    if unclosed_string {
                        closing.push('"');
                    }
                    closing.push_str(if current_args.is_empty() { "{}" } else { "}" });
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: closing.clone(),
                    });
                    current_args.push_str(&closing);

                    // Complete the tool call
                    self.buffer =
                        self.buffer[end_pos + self.tool_call_end_token.len()..].to_string();
                    self.reset_streaming_state();
                    self.current_tool_id += 1;
                    continue;
                } else {
                    // Tool call not complete yet, wait for more text
                    break;
                }
            }

            break;
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.tool_call_start_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        let tool_index = self.prev_tool_call_arr.len().checked_sub(1)?;
        // A string value the stream ended in is closed with what arrived.
        if self
            .open_parameter
            .as_ref()
            .is_some_and(|open| open.streamed_end.is_some())
        {
            return Some(vec![ToolCallItem {
                tool_index,
                name: None,
                parameters: "\"}".to_string(),
            }]);
        }
        let actual = self.streamed_args_for_tool.get(tool_index)?;
        let expected = self.prev_tool_call_arr[tool_index].get("arguments")?;
        // XML parameters use spaced JSON, unlike the generic compact-prefix recovery.
        // Append only the root brace, and only for exactly the already parsed values.
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

    fn reset(&mut self) {
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        self.reset_streaming_state();
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;

    use super::*;

    #[test]
    fn test_safe_val_json() {
        assert_eq!(safe_val("42"), Value::Number(42.into()));
        assert_eq!(safe_val("1.5"), serde_json::json!(1.5));
        assert_eq!(safe_val("true"), Value::Bool(true));
        assert_eq!(safe_val("false"), Value::Bool(false));
        assert_eq!(safe_val("null"), Value::Null);
        assert_eq!(
            safe_val(r#"{"key": "value"}"#),
            serde_json::json!({"key": "value"})
        );
        assert_eq!(safe_val(r"[1, 2, 3]"), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_safe_val_python_literals() {
        assert_eq!(safe_val("True"), Value::Bool(true));
        assert_eq!(safe_val("False"), Value::Bool(false));
        assert_eq!(safe_val("None"), Value::Null);
    }

    #[test]
    fn test_safe_val_string_fallback() {
        assert_eq!(
            safe_val("hello world"),
            Value::String("hello world".to_string())
        );
        assert_eq!(safe_val("  spaces  "), Value::String("spaces".to_string()));
    }

    // Values are treated literally: entity-like substrings must NOT be decoded
    // (parity with Qwen's official API, which returns them intact).
    #[test]
    fn test_safe_val_preserves_html_entities() {
        assert_eq!(
            safe_val("&lt;div&gt;"),
            Value::String("&lt;div&gt;".to_string())
        );
        assert_eq!(
            safe_val("Tom &amp; Jerry"),
            Value::String("Tom &amp; Jerry".to_string())
        );
        // Numeric/hex entities are likewise left untouched.
        assert_eq!(safe_val("it&#39;s"), Value::String("it&#39;s".to_string()));
        assert_eq!(safe_val("&#x3C;"), Value::String("&#x3C;".to_string()));
    }

    fn tool_with_props(props: Value) -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": props}),
                strict: None,
                extra: Default::default(),
            },
        }]
    }

    // String-typed params stay strings even when they look numeric/bool/array/object.
    #[tokio::test]
    async fn test_schema_aware_coercion_keeps_strings() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "flag": {"type": "string"},
            "coords": {"type": "string"},
            "cfg": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>\n<function=f>\n\
            <parameter=limit>4</parameter>\n\
            <parameter=flag>true</parameter>\n\
            <parameter=coords>[60,30]</parameter>\n\
            <parameter=cfg>{\"a\": 1}</parameter>\n\
            <parameter=count>5</parameter>\n\
            </function>\n</tool_call>";
        let (_, calls) = QwenXmlParser::new()
            .parse_complete_with_tools(text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["flag"], Value::String("true".to_string()));
        assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
        assert_eq!(args["cfg"], Value::String("{\"a\": 1}".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }

    // The streaming path threads `tools` separately, so cover it too.
    #[tokio::test]
    async fn test_streaming_schema_aware_coercion() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>\n<function=f>\n\
            <parameter=limit>4</parameter>\n\
            <parameter=count>5</parameter>\n\
            </function>\n</tool_call>";
        let result = QwenXmlParser::new()
            .parse_incremental(text, &tools)
            .await
            .unwrap();
        let args: String = result.calls.iter().map(|c| c.parameters.as_str()).collect();
        assert!(
            args.contains(r#""limit": "4""#),
            "string param must stay string: {args}"
        );
        assert!(
            args.contains(r#""count": 5"#),
            "int param must coerce: {args}"
        );
    }

    // Golden conformance test (regression guard for #1888): a tool argument
    // whose value contains HTML entities must round-trip UNCHANGED, matching
    // Qwen's official API (verified on Qwen3-Coder and Qwen3.5). Covers both the
    // schema-typed `string` path and the schema-less inference fallback.
    #[tokio::test]
    async fn test_arg_values_with_entities_roundtrip_unchanged() {
        let literal = "<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s";

        // Function name matches `tool_with_props` (`f`) so the schema-typed
        // branch below actually resolves the `content` param's type.
        let text = format!(
            "<tool_call>\n<function=f>\n\
             <parameter=content>\n{literal}\n</parameter>\n\
             </function>\n</tool_call>"
        );

        // Schema-less inference path (`safe_val`): not valid JSON -> stays a
        // literal string with entities intact.
        let (_, calls) = QwenXmlParser::new().parse_complete(&text).await.unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], Value::String(literal.to_string()));

        // Schema-typed `string` path (`coerce_value` -> `coerce_by_schema_type`).
        let tools = tool_with_props(serde_json::json!({
            "content": {"type": "string"},
        }));
        let (_, calls) = QwenXmlParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], Value::String(literal.to_string()));
    }

    fn bash_tool() -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "bash".to_string(),
                description: None,
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"}
                    }
                }),
                strict: None,
                extra: Default::default(),
            },
        }]
    }

    /// Feed `chunks` and collect the argument deltas (name-less items with
    /// parameters) together with the index of the chunk that produced each.
    async fn argument_deltas(
        parser: &mut QwenXmlParser,
        chunks: &[&str],
        tools: &[Tool],
    ) -> Vec<(usize, String)> {
        let mut deltas = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            for call in parser.parse_incremental(chunk, tools).await.unwrap().calls {
                if call.name.is_none() && !call.parameters.is_empty() {
                    deltas.push((index, call.parameters));
                }
            }
        }
        deltas
    }

    // The value of a string parameter goes out as it is generated (smg-lab#27):
    // the first argument bytes follow the name within a few tokens instead of
    // the whole value arriving once `</parameter>` closes it.
    #[tokio::test]
    async fn test_string_values_stream_before_the_parameter_closes() {
        let tools = bash_tool();
        let chunks = [
            "<tool_call>",
            "\n",
            "<function=",
            "bash",
            ">\n",
            "<parameter=",
            "command",
            ">\n",
            "ls",
            " -",
            "la",
            " /",
            "tmp",
            "\n",
            "</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ];
        let close_at = chunks.iter().position(|c| *c == "</parameter>").unwrap();
        let mut parser = QwenXmlParser::new();
        let deltas = argument_deltas(&mut parser, &chunks, &tools).await;

        assert!(deltas.len() > 2, "arguments arrived as {deltas:?}");
        assert_eq!(deltas[0], (8, "{\"command\": \"ls".to_string()));
        let before_close = deltas.iter().filter(|(at, _)| *at < close_at).count();
        assert_eq!(before_close, 5, "{deltas:?}");
        let joined: String = deltas.iter().map(|(_, d)| d.as_str()).collect();
        assert_eq!(joined, "{\"command\": \"ls -la /tmp\"}");
        assert!(parser.get_unstreamed_tool_args().is_none());
        assert!(parser.take_unstreamed_normal_text().is_empty());
    }

    // Whatever the chunking, the streamed arguments re-assemble to exactly
    // the complete parse: escaped like JSON, trimmed like the XML value.
    #[tokio::test]
    async fn test_streamed_string_values_match_the_complete_parse_at_every_split() {
        let tools = bash_tool();
        let command = "<parameter=command>\n  echo \"a\\b\"\t<tag> \n\n</parameter>\n";
        let timeout = "<parameter=timeout>30</parameter>\n";
        // Both orders: a chunk may carry a complete parameter and the start of
        // the string one.
        for text in [
            format!("<tool_call>\n<function=bash>\n{command}{timeout}</function>\n</tool_call>"),
            format!("<tool_call>\n<function=bash>\n{timeout}{command}</function>\n</tool_call>"),
        ] {
            let (_, complete) = QwenXmlParser::new()
                .parse_complete_with_tools(&text, &tools)
                .await
                .unwrap();
            let expected: Value = serde_json::from_str(&complete[0].function.arguments).unwrap();
            assert_eq!(expected["command"], "echo \"a\\b\"\t<tag>");
            assert_eq!(expected["timeout"], 30);

            let mut feeds: Vec<Vec<String>> = vec![text.chars().map(|c| c.to_string()).collect()];
            for (split, _) in text.char_indices().skip(1) {
                feeds.push(vec![text[..split].to_string(), text[split..].to_string()]);
            }
            for chunks in feeds {
                let mut parser = QwenXmlParser::new();
                let mut args = String::new();
                for chunk in &chunks {
                    for call in parser.parse_incremental(chunk, &tools).await.unwrap().calls {
                        args.push_str(&call.parameters);
                    }
                }
                assert!(parser.get_unstreamed_tool_args().is_none(), "{chunks:?}");
                assert_eq!(
                    serde_json::from_str::<Value>(&args).ok().as_ref(),
                    Some(&expected),
                    "args {args}; chunks {chunks:?}"
                );
            }
        }
    }

    // Coercion needs the complete value: non-string parameters and JSON
    // string literals (which the complete parse unwraps) still arrive whole.
    #[tokio::test]
    async fn test_values_that_need_the_complete_text_arrive_whole() {
        let tools = bash_tool();
        for (text, expected) in [
            (
                "<tool_call><function=bash><parameter=timeout>30</parameter></function></tool_call>",
                ["{\"timeout\": 30", "}"],
            ),
            (
                "<tool_call><function=bash><parameter=command>\"ls\"</parameter></function></tool_call>",
                ["{\"command\": \"ls\"", "}"],
            ),
        ] {
            let chunks: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let mut parser = QwenXmlParser::new();
            let deltas: Vec<String> = argument_deltas(&mut parser, &chunk_refs, &tools)
                .await
                .into_iter()
                .map(|(_, delta)| delta)
                .collect();
            assert_eq!(deltas, expected);
        }
    }

    // A value the stream ends in is closed with what arrived, as the engine's
    // own streaming parser does; a parameter without any value is not invented.
    #[tokio::test]
    async fn test_end_of_stream_closes_a_partially_streamed_value() {
        let tools = bash_tool();
        let mut parser = QwenXmlParser::new();
        let deltas = argument_deltas(
            &mut parser,
            &[
                "<tool_call><function=bash><parameter=command>ls -l",
                "a /tm",
            ],
            &tools,
        )
        .await;
        let mut args: String = deltas.into_iter().map(|(_, d)| d).collect();
        assert_eq!(args, "{\"command\": \"ls -la /tm");
        for item in parser.get_unstreamed_tool_args().unwrap() {
            args.push_str(&item.parameters);
        }
        assert_eq!(args, "{\"command\": \"ls -la /tm\"}");
        assert!(parser.take_unstreamed_normal_text().is_empty());

        let mut parser = QwenXmlParser::new();
        let deltas = argument_deltas(
            &mut parser,
            &["<tool_call><function=bash><parameter=command>\n"],
            &tools,
        )
        .await;
        assert!(deltas.is_empty(), "{deltas:?}");
        let pending: String = parser
            .get_unstreamed_tool_args()
            .unwrap()
            .into_iter()
            .map(|item| item.parameters)
            .collect();
        assert_eq!(pending, "{}");
    }
}
