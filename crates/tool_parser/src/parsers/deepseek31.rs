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

/// DeepSeek V3.1 format parser for tool calls
///
/// Handles the DeepSeek V3.1 format:
/// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json_args}<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
///
/// Differences from V3:
/// - No `function` type prefix before `<｜tool▁sep｜>`
/// - No markdown code block wrapping around JSON arguments
/// - Function name directly after `<｜tool▁call▁begin｜>`
/// - Raw JSON arguments directly after `<｜tool▁sep｜>`
///
/// Reference: <https://huggingface.co/deepseek-ai/DeepSeek-V3.1>
pub struct DeepSeek31Parser {
    /// Regex for extracting complete tool call blocks
    tool_call_extractor: Regex,
    /// Regex for extracting function name and arguments from a complete block
    func_detail_extractor: Regex,
    /// Regex for matching partial tool calls during streaming
    partial_tool_call_regex: Regex,
    /// Regex for removing completed tool calls from buffer
    tool_call_end_pattern: Regex,

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

impl DeepSeek31Parser {
    /// Create a new DeepSeek V3.1 parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        let tool_call_extractor = Regex::new(r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>")
            .expect("Valid regex pattern");

        let func_detail_extractor =
            Regex::new(r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>")
                .expect("Valid regex pattern");

        let partial_tool_call_regex =
            Regex::new(r"(?s)<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)")
                .expect("Valid regex pattern");

        let tool_call_end_pattern = Regex::new(r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>")
            .expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            partial_tool_call_regex,
            tool_call_end_pattern,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
        }
    }

    /// Parse a single complete tool call block
    fn parse_tool_call(&self, block: &str) -> ParserResult<ToolCall> {
        let captures = self.func_detail_extractor.captures(block).ok_or_else(|| {
            ParserError::ParsingFailed("Failed to match tool call pattern".to_string())
        })?;

        let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();
        if func_name.is_empty() {
            return Err(ParserError::ParsingFailed(
                "Empty function name".to_string(),
            ));
        }

        let json_args = captures.get(2).map_or("{}", |m| m.as_str()).trim();

        let value = serde_json::from_str::<Value>(json_args)
            .map_err(|e| ParserError::ParsingFailed(format!("Invalid JSON: {e}")))?;

        let args = if value.is_object() {
            value
        } else {
            serde_json::json!({ "value": value })
        };

        let arguments =
            serde_json::to_string(&args).map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(ToolCall {
            function: FunctionCall {
                name: func_name.to_string(),
                arguments,
            },
        })
    }
}

impl Default for DeepSeek31Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DeepSeek31Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let idx = text
            .find("<｜tool▁calls▁begin｜>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        let mut tools = Vec::new();
        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str()) {
                Ok(tool) => tools.push(tool),
                Err(e) => {
                    tracing::debug!("Failed to parse tool call: {}", e);
                    continue;
                }
            }
        }

        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        let has_tool_call =
            self.has_tool_markers(&current_text) || current_text.contains("<｜tool▁call▁begin｜>");

        if !has_tool_call {
            let mut normal_text = std::mem::take(&mut self.buffer);
            for end_token in [
                "<｜tool▁calls▁end｜>",
                "<｜tool▁call▁end｜>",
                "<｜end▁of▁sentence｜>",
            ] {
                normal_text = normal_text.replace(end_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        let tool_indices = helpers::get_tool_indices(tools);
        let mut calls: Vec<ToolCallItem> = Vec::new();

        if let Some(captures) = self.partial_tool_call_regex.captures(&current_text) {
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();
            let func_args_raw = captures.get(2).map_or("", |m| m.as_str()).trim();

            if !tool_indices.contains_key(func_name) {
                tracing::debug!("Invalid tool name '{}' - skipping", func_name);
                helpers::reset_current_tool_state(
                    &mut self.buffer,
                    &mut self.current_tool_name_sent,
                    &mut self.streamed_args_for_tool,
                    &self.prev_tool_call_arr,
                );
                return Ok(StreamingParseResult::default());
            }

            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            if self.current_tool_name_sent {
                let tool_id = self.current_tool_id as usize;
                let last_sent = self
                    .streamed_args_for_tool
                    .get(tool_id)
                    .map(|s| s.as_str())
                    .unwrap_or("");

                // Strip end markers that may arrive in the same chunk as the final
                // JSON bytes. The partial_tool_call_regex group 2 greedily captures
                // everything after <｜tool▁sep｜>, including any trailing end tokens.
                // Use an iterative loop so stacked markers in any order are all removed.
                const END_MARKERS: [&str; 3] = [
                    "<｜end▁of▁sentence｜>",
                    "<｜tool▁calls▁end｜>",
                    "<｜tool▁call▁end｜>",
                ];
                let mut func_args_clean = func_args_raw.trim_end();
                loop {
                    let before = func_args_clean;
                    for marker in END_MARKERS {
                        func_args_clean = func_args_clean.trim_end_matches(marker).trim_end();
                    }
                    if func_args_clean == before {
                        break;
                    }
                }

                let argument_diff = func_args_clean
                    .strip_prefix(last_sent)
                    .unwrap_or(func_args_clean);

                if !argument_diff.is_empty() {
                    calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: argument_diff.to_string(),
                    });
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(argument_diff);
                    }
                }

                if helpers::is_complete_json(func_args_clean) {
                    if let Ok(parsed_args) = serde_json::from_str::<Value>(func_args_clean) {
                        let tool_id = self.current_tool_id as usize;
                        if tool_id < self.prev_tool_call_arr.len() {
                            if let Some(obj) = self.prev_tool_call_arr[tool_id].as_object_mut() {
                                obj.insert("arguments".to_string(), parsed_args);
                            }
                        }
                    }

                    if let Some(mat) = self.tool_call_end_pattern.find(&current_text) {
                        self.buffer = current_text[mat.end()..].to_string();
                    } else {
                        self.buffer.clear();
                    }

                    let result = StreamingParseResult {
                        normal_text: String::new(),
                        calls,
                    };

                    self.current_tool_id += 1;
                    self.current_tool_name_sent = false;
                    return Ok(result);
                }
            } else {
                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(func_name.to_string()),
                    parameters: String::new(),
                });
                self.current_tool_name_sent = true;

                let tool_id = self.current_tool_id as usize;
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": {},
                });
            }
        }

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool▁calls▁begin｜>")
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
    }
}
