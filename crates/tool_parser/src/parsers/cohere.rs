//! Cohere Command model tool call parser
//!
//! Parses tool calls from `<|START_ACTION|>...<|END_ACTION|>` blocks.
//! Supports both CMD3 and CMD4 formats.
//!
//! # Format
//!
//! Cohere models output tool calls in the following format:
//! ```text
//! <|START_RESPONSE|>Let me help with that.<|END_RESPONSE|>
//! <|START_ACTION|>
//! {"tool_name": "search", "parameters": {"query": "rust programming"}}
//! <|END_ACTION|>
//! ```
//!
//! Or for multiple tool calls:
//! ```text
//! <|START_ACTION|>
//! [
//!   {"tool_name": "search", "parameters": {"query": "rust"}},
//!   {"tool_name": "get_weather", "parameters": {"city": "Paris"}}
//! ]
//! <|END_ACTION|>
//! ```
//!
//! # Field Mapping
//! - `tool_name` → `name`
//! - `parameters` → `arguments`

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;

use crate::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    partial_json::PartialJson,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const START_ACTION: &str = "<|START_ACTION|>";
const END_ACTION: &str = "<|END_ACTION|>";
const START_RESPONSE: &str = "<|START_RESPONSE|>";
const END_RESPONSE: &str = "<|END_RESPONSE|>";
const START_TEXT: &str = "<|START_TEXT|>";
const END_TEXT: &str = "<|END_TEXT|>";

/// State machine for Cohere tool parsing
#[derive(Debug, Clone, Copy, PartialEq)]
enum ParseState {
    /// Looking for START_ACTION marker
    Text,
    /// Inside an action block, parsing JSON
    InAction,
}

/// Cohere Command model tool call parser
///
/// Handles the Cohere-specific format:
/// `<|START_ACTION|>{"tool_name": "func", "parameters": {...}}<|END_ACTION|>`
pub struct CohereParser {
    /// Current parsing state
    state: ParseState,

    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,

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
}

impl CohereParser {
    /// Create a new Cohere parser
    pub fn new() -> Self {
        Self {
            state: ParseState::Text,
            partial_json: PartialJson::default(),
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Clean text by removing response markers
    fn clean_text(text: &str) -> String {
        text.replace(START_RESPONSE, "")
            .replace(END_RESPONSE, "")
            .replace(START_TEXT, "")
            .replace(END_TEXT, "")
    }

    /// Convert a Cohere tool call JSON object to our ToolCall format
    fn convert_tool_call(json_str: &str) -> ParserResult<Vec<ToolCall>> {
        let value: Value = serde_json::from_str(json_str.trim())
            .map_err(|e| ParserError::ParsingFailed(format!("Invalid JSON: {e}")))?;

        let tools = match value {
            Value::Array(arr) => arr,
            single => vec![single],
        };

        tools
            .into_iter()
            .filter_map(|tool| {
                // Cohere uses "tool_name" instead of "name"
                let name = tool
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .or_else(|| tool.get("name").and_then(|v| v.as_str()))?;

                // Cohere uses "parameters" instead of "arguments"
                let parameters = tool
                    .get("parameters")
                    .or_else(|| tool.get("arguments"))
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".to_string());

                Some(Ok(ToolCall {
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: parameters,
                    },
                }))
            })
            .collect()
    }

    /// Extract JSON between START_ACTION and END_ACTION (skip markers inside strings).
    fn extract_action_json(text: &str) -> Option<(usize, &str, usize)> {
        let start_idx = text.find(START_ACTION)?;
        let json_start = start_idx + START_ACTION.len();
        let end_offset = Self::find_end_action_outside_strings(&text[json_start..])?;
        let json_str = &text[json_start..json_start + end_offset];
        Some((
            start_idx,
            json_str,
            json_start + end_offset + END_ACTION.len(),
        ))
    }

    /// Offset of END_ACTION in `s`, ignoring occurrences inside JSON strings.
    /// Incomplete / unclosed quotes return None so streaming can wait for more
    /// input (no find/rfind fallback — that breaks mid-string streams and can
    /// swallow a following action).
    fn find_end_action_outside_strings(s: &str) -> Option<usize> {
        let mut in_string = false;
        let mut escape = false;
        let bytes = s.as_bytes();
        let needle = END_ACTION.as_bytes();
        let mut i = 0;
        while i + needle.len() <= bytes.len() {
            if escape {
                escape = false;
                i += 1;
                continue;
            }
            let b = bytes[i];
            if in_string {
                if b == b'\\' {
                    escape = true;
                    i += 1;
                    continue;
                }
                if b == b'"' {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            if b == b'"' {
                in_string = true;
                i += 1;
                continue;
            }
            if Some(&b) == needle.first() && bytes[i..].starts_with(needle) {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

impl Default for CohereParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for CohereParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Cohere format
        if !self.has_tool_markers(text) {
            let cleaned = Self::clean_text(text);
            return Ok((cleaned.trim().to_string(), vec![]));
        }

        let mut normal_text = String::new();
        let mut tool_calls = Vec::new();
        let mut remaining = text;

        while let Some((start_idx, json_str, end_idx)) = Self::extract_action_json(remaining) {
            // Text before action
            normal_text.push_str(&remaining[..start_idx]);

            // Parse tool calls from this action block
            match Self::convert_tool_call(json_str) {
                Ok(calls) => tool_calls.extend(calls),
                Err(e) => {
                    tracing::debug!("Failed to parse Cohere tool call: {}", e);
                }
            }

            remaining = &remaining[end_idx..];
        }

        // Append any remaining text after last action block
        normal_text.push_str(remaining);

        // Clean up response markers
        let cleaned_text = Self::clean_text(&normal_text);

        Ok((cleaned_text.trim().to_string(), tool_calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        match self.state {
            ParseState::Text => {
                // Check for START_ACTION marker
                let start_pos = self.buffer.find(START_ACTION);
                if let Some(pos) = start_pos {
                    // Emit text before the action as normal text
                    let text_before = Self::clean_text(&self.buffer[..pos]);

                    // Switch to InAction state and keep only content after START_ACTION
                    self.state = ParseState::InAction;
                    self.buffer.drain(..pos + START_ACTION.len());

                    return Ok(StreamingParseResult {
                        normal_text: text_before,
                        calls: vec![],
                    });
                }

                // Check for partial START_ACTION
                if helpers::ends_with_partial_token(&self.buffer, START_ACTION).is_some() {
                    // Keep buffering
                    return Ok(StreamingParseResult::default());
                }

                // No action starting, emit cleaned text
                let cleaned = Self::clean_text(&self.buffer);
                self.buffer.clear();
                Ok(StreamingParseResult {
                    normal_text: cleaned,
                    calls: vec![],
                })
            }

            ParseState::InAction => {
                // Check if we have END_ACTION
                if let Some(pos) = Self::find_end_action_outside_strings(&self.buffer) {
                    // We have complete JSON - extract it before modifying buffer
                    let json_content = self.buffer[..pos].to_string();

                    // Build tool indices
                    let tool_indices = helpers::get_tool_indices(tools);

                    // Create a temporary buffer for the helper (it expects to manage buffer state)
                    let mut temp_buffer = String::new();

                    // Use helper for streaming - pass JSON directly with offset 0
                    let result = helpers::handle_json_tool_streaming(
                        &json_content,
                        0,
                        &mut self.partial_json,
                        &tool_indices,
                        &mut temp_buffer,
                        &mut self.current_tool_id,
                        &mut self.current_tool_name_sent,
                        &mut self.streamed_args_for_tool,
                        &mut self.prev_tool_call_arr,
                    )?;

                    // Move past END_ACTION and switch back to Text state
                    self.buffer.drain(..pos + END_ACTION.len());
                    self.state = ParseState::Text;

                    return Ok(result);
                }

                // Partial JSON - buffer and wait for END_ACTION
                // Unlike formats without end markers, we can't stream partial JSON safely
                // because we don't know if the JSON is complete until we see END_ACTION
                Ok(StreamingParseResult::default())
            }
        }
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(START_ACTION) || text.contains(END_ACTION)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn take_unstreamed_normal_text(&mut self) -> String {
        // Covers both a partial START_ACTION held in Text state and an action
        // block whose END_ACTION never arrived (truncated stream). Text held
        // in the Text state still carries response markers, which every other
        // emission path in this parser strips, so clean it the same way rather
        // than leaking control tokens into client-visible content.
        let text = helpers::take_unstreamed_normal_text(&mut self.buffer, self.current_tool_id);
        if text.is_empty() {
            text
        } else {
            Self::clean_text(&text)
        }
    }

    fn reset(&mut self) {
        self.state = ParseState::Text;
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_single_tool_call() {
        let parser = CohereParser::new();
        let input = r#"<|START_RESPONSE|>Let me search for that.<|END_RESPONSE|>
<|START_ACTION|>
{"tool_name": "search", "parameters": {"query": "rust programming"}}
<|END_ACTION|>"#;

        let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(normal_text, "Let me search for that.");
        assert_eq!(tools[0].function.name, "search");

        let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
    }

    #[tokio::test]
    async fn test_multiple_tool_calls_array() {
        let parser = CohereParser::new();
        let input = r#"<|START_ACTION|>
[
  {"tool_name": "search", "parameters": {"query": "rust"}},
  {"tool_name": "get_weather", "parameters": {"city": "Paris"}}
]
<|END_ACTION|>"#;

        let (_, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, "search");
        assert_eq!(tools[1].function.name, "get_weather");
    }

    #[tokio::test]
    async fn test_no_tool_calls() {
        let parser = CohereParser::new();
        let input = "<|START_RESPONSE|>Hello, how can I help?<|END_RESPONSE|>";

        let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 0);
        assert_eq!(normal_text, "Hello, how can I help?");
    }

    #[tokio::test]
    async fn test_has_tool_markers() {
        let parser = CohereParser::new();

        assert!(parser.has_tool_markers("<|START_ACTION|>"));
        assert!(parser.has_tool_markers("<|END_ACTION|>"));
        assert!(parser.has_tool_markers("Some text <|START_ACTION|> more"));
        assert!(!parser.has_tool_markers("Just plain text"));
        assert!(!parser.has_tool_markers("[TOOL_CALLS]")); // Mistral format
    }

    #[tokio::test]
    async fn test_empty_parameters() {
        let parser = CohereParser::new();
        let input = r#"<|START_ACTION|>{"tool_name": "ping"}<|END_ACTION|>"#;

        let (_, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "ping");
        assert_eq!(tools[0].function.arguments, "{}");
    }

    #[tokio::test]
    async fn test_nested_json() {
        let parser = CohereParser::new();
        let input = r#"<|START_ACTION|>
{"tool_name": "process", "parameters": {"config": {"nested": {"value": [1, 2, 3]}}}}
<|END_ACTION|>"#;

        let (_, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);

        let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
        assert_eq!(
            args["config"]["nested"]["value"],
            serde_json::json!([1, 2, 3])
        );
    }

    #[tokio::test]
    async fn test_text_markers_cleaned() {
        let parser = CohereParser::new();
        let input = r#"<|START_TEXT|>Some intro<|END_TEXT|>
<|START_ACTION|>{"tool_name": "test", "parameters": {}}<|END_ACTION|>
<|START_TEXT|>Conclusion<|END_TEXT|>"#;

        let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert!(normal_text.contains("Some intro"));
        assert!(normal_text.contains("Conclusion"));
        assert!(!normal_text.contains("<|START_TEXT|>"));
        assert!(!normal_text.contains("<|END_TEXT|>"));
    }

    #[tokio::test]
    async fn test_malformed_json() {
        let parser = CohereParser::new();
        let input = r#"<|START_ACTION|>{"tool_name": invalid}<|END_ACTION|>"#;

        let (_, tools) = parser.parse_complete(input).await.unwrap();
        // Should gracefully handle malformed JSON
        assert_eq!(tools.len(), 0);
    }
}
