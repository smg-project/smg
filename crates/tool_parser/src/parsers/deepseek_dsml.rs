use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::ParserResult,
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// DeepSeek DSML format parser for tool calls (V3.2 and V4).
///
/// Both variants share an identical invoke/parameter grammar and streaming
/// state machine. They differ only in the outer block-name token:
/// - V3.2: `function_calls`
/// - V4:   `tool_calls`
///
/// ```text
/// <｜DSML｜{block_name}>
/// <｜DSML｜invoke name="func">
/// <｜DSML｜parameter name="key" string="true">value</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜{block_name}>
/// ```
///
/// Also supports direct JSON inside invoke blocks as a fallback format.
///
/// References:
/// - <https://huggingface.co/deepseek-ai/DeepSeek-V3.2>
/// - <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
pub struct DeepSeekDsmlParser {
    /// Outer block-name token: `function_calls` (V3.2) or `tool_calls` (V4).
    block_name: &'static str,
    /// Cached `<｜DSML｜{block_name}>` for marker-scan hot paths.
    block_open: String,
    /// Cached `</｜DSML｜{block_name}>` for streaming cleanup.
    block_close: String,

    /// Regex for extracting full outer-block content
    tool_call_complete_regex: Regex,
    /// Regex for extracting complete invoke blocks (name + body)
    invoke_complete_regex: Regex,
    /// Regex for extracting complete parameter tags (name, string attr, value)
    parameter_complete_regex: Regex,
    /// Regex for matching partial parameter tag during streaming (no closing tag)
    partial_parameter_regex: Regex,
    /// Regex for matching invoke blocks (complete or partial, for streaming)
    invoke_regex: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,
    /// Stores complete tool call info for each tool being parsed
    prev_tool_call_arr: Vec<Value>,
    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,
    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,
    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,
    /// Calls whose invoke and argument object have both closed successfully.
    completed_tool_call_count: usize,
}

/// Full DSML closing tags for suffix-based stripping during streaming.
const DSML_PARAMETER_END_TAG: &str = "</｜DSML｜parameter>";
const DSML_INVOKE_END_TAG: &str = "</｜DSML｜invoke>";
const DSML_OPEN_SENTINEL: &str = "<｜DSML｜";
const DSML_CLOSE_SENTINEL: &str = "</｜DSML｜";

/// DeepSeek end-of-sentence marker. Some engines emit this as raw text at the
/// end of a truncated turn; it must never bleed into tool-call argument bytes.
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";

/// Strip a trailing partial DSML closing tag from a string.
///
/// If the string ends with a prefix of `closing_tag` (e.g. `"Tokyo</｜DSML｜para"`
/// ends with a prefix of `"</｜DSML｜parameter>"`), that trailing portion is removed.
/// Unlike character-set stripping, this only removes text that actually starts
/// the specified closing tag, so legitimate value bytes are preserved.
fn strip_dsml_trailing(s: &str, closing_tag: &str) -> String {
    for (idx, _) in s.char_indices() {
        if closing_tag.starts_with(&s[idx..]) {
            return s[..idx].to_string();
        }
    }
    s.to_string()
}

impl DeepSeekDsmlParser {
    /// Create a DeepSeek V3.2 parser (outer block token `function_calls`).
    pub fn v32() -> Self {
        Self::new("function_calls")
    }

    /// Create a DeepSeek V4 parser (outer block token `tool_calls`).
    pub fn v4() -> Self {
        Self::new("tool_calls")
    }

    /// Which block name this instance parses (`function_calls` or `tool_calls`).
    pub fn block_name(&self) -> &'static str {
        self.block_name
    }

    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    fn new(block_name: &'static str) -> Self {
        // `regex::escape` is not needed — both known block names are
        // `[a-z_]`-only. If a future variant introduces regex metacharacters,
        // wrap with `regex::escape(block_name)` before interpolation.
        let tool_call_complete_regex = Regex::new(&format!(
            r"(?s)<｜DSML｜{block_name}>(.*?)</｜DSML｜{block_name}>"
        ))
        .expect("Valid regex pattern");

        let invoke_complete_regex =
            Regex::new(r#"(?s)<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>"#)
                .expect("Valid regex pattern");

        let parameter_complete_regex = Regex::new(
            r#"(?s)<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>"#,
        )
        .expect("Valid regex pattern");

        let partial_parameter_regex = Regex::new(
            r#"(?s)<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*)$"#,
        )
        .expect("Valid regex pattern");

        // `[^"]*` (not `+`) so a malformed `name=""` still matches and can be
        // advanced past by the empty/invalid-name handling in `parse_incremental`.
        // Without this, a bad `name=""` invoke would stall the buffer forever
        // and suppress every subsequent delta in the same stream.
        let invoke_regex =
            Regex::new(r#"(?s)<｜DSML｜invoke\s+name="([^"]*)"\s*>(.*?)(</｜DSML｜invoke>|$)"#)
                .expect("Valid regex pattern");

        Self {
            block_name,
            block_open: format!("<｜DSML｜{block_name}>"),
            block_close: format!("</｜DSML｜{block_name}>"),
            tool_call_complete_regex,
            invoke_complete_regex,
            parameter_complete_regex,
            partial_parameter_regex,
            invoke_regex,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            completed_tool_call_count: 0,
        }
    }

    /// Parse DSML parameters from invoke content into a JSON string.
    ///
    /// Supports two formats:
    /// 1. Direct JSON: content starts with `{` — returned as-is
    /// 2. XML parameters: `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`
    ///
    /// When `allow_partial` is true (streaming), also matches open parameter tags
    /// and strips trailing DSML fragments.
    fn parse_parameters_from_dsml(&self, invoke_content: &str, allow_partial: bool) -> String {
        let trimmed = invoke_content.trim();

        // Direct JSON path
        if trimmed.starts_with('{') {
            if allow_partial {
                // `strip_dsml_trailing` handles partial `</｜DSML｜invoke>` prefixes
                // but can't match the EOS sentinel (different prefix). Strip it
                // unconditionally so a truncated turn doesn't leak EOS into args.
                return strip_dsml_trailing(trimmed, DSML_INVOKE_END_TAG).replace(EOS_TOKEN, "");
            } else if trimmed.ends_with('}') {
                return trimmed.to_string();
            }
        }

        // XML parameter path
        let mut params = serde_json::Map::new();

        for cap in self.parameter_complete_regex.captures_iter(invoke_content) {
            let name = cap.get(1).map_or("", |m| m.as_str());
            let is_string = cap.get(2).map_or("true", |m| m.as_str());
            // Strip any stray EOS marker — should never legitimately appear
            // inside a closed parameter, but defend against malformed output.
            let value = cap.get(3).map_or("", |m| m.as_str()).replace(EOS_TOKEN, "");

            let json_value = if is_string == "true" {
                Value::String(value.to_string())
            } else {
                serde_json::from_str(value.trim())
                    .unwrap_or_else(|_| Value::String(value.to_string()))
            };

            params.insert(name.to_string(), json_value);
        }

        // Partial parameter matching for streaming
        // Following SGLang: strip DSML fragments from remaining content BEFORE
        // running the partial regex, so the regex captures a clean value.
        if allow_partial {
            // Find where the last complete parameter match ended
            let last_match_end = self
                .parameter_complete_regex
                .find_iter(invoke_content)
                .last()
                .map(|m| m.end())
                .unwrap_or(0);

            let remaining = &invoke_content[last_match_end..];
            let cleaned = strip_dsml_trailing(remaining, DSML_PARAMETER_END_TAG);

            if let Some(cap) = self.partial_parameter_regex.captures(&cleaned) {
                let name = cap.get(1).map_or("", |m| m.as_str());
                let is_string = cap.get(2).map_or("true", |m| m.as_str());
                // Strip EOS before trimming — `strip_dsml_trailing` above only
                // handles `</｜DSML｜parameter>` prefixes, so a truncated turn
                // with `value<EOS>` would otherwise stream EOS as arg bytes.
                let value = cap.get(3).map_or("", |m| m.as_str()).replace(EOS_TOKEN, "");
                let value = value.trim();

                // Only add if we have actual content and this param isn't already complete
                if !value.is_empty() && !params.contains_key(name) {
                    let json_value = if is_string == "true" {
                        Value::String(value.to_string())
                    } else {
                        serde_json::from_str(value)
                            .unwrap_or_else(|_| Value::String(value.to_string()))
                    };
                    params.insert(name.to_string(), json_value);
                }
            }
        }

        serde_json::to_string(&Value::Object(params)).unwrap_or_else(|_| "{}".to_string())
    }

    fn arguments_are_valid_object(content: &str, arguments: &str) -> bool {
        let candidate = if content.trim().starts_with('{') {
            content.trim()
        } else {
            arguments
        };
        matches!(serde_json::from_str(candidate), Ok(Value::Object(_)))
    }

    /// Parse a single complete invoke block into a ToolCall.
    fn parse_invoke(&self, name: &str, content: &str) -> Option<ToolCall> {
        let arguments = self.parse_parameters_from_dsml(content, false);
        if !Self::arguments_are_valid_object(content, &arguments) {
            return None;
        }

        Some(ToolCall {
            function: FunctionCall {
                name: name.trim().to_string(),
                arguments,
            },
        })
    }
}

// Intentionally no `Default` impl — callers must pick `v32()` or `v4()`.

#[async_trait]
impl ToolParser for DeepSeekDsmlParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let Some(idx) = text.find(self.block_open.as_str()) else {
            let marker_start = [DSML_OPEN_SENTINEL, DSML_CLOSE_SENTINEL]
                .into_iter()
                .filter_map(|marker| text.find(marker))
                .min();
            let Some(marker_start) = marker_start else {
                return Ok((text.to_string(), Vec::new()));
            };
            return Ok((text[..marker_start].trim_end().to_string(), Vec::new()));
        };
        let normal_text = text[..idx].trim_end().to_string();

        let mut tools = Vec::new();
        let mut last_block_end = None;

        for fc_cap in self.tool_call_complete_regex.captures_iter(text) {
            last_block_end = fc_cap.get(0).map(|outer| outer.end());
            let fc_content = fc_cap.get(1).map_or("", |m| m.as_str());

            for inv_cap in self.invoke_complete_regex.captures_iter(fc_content) {
                let func_name = inv_cap.get(1).map_or("", |m| m.as_str());
                let invoke_content = inv_cap.get(2).map_or("", |m| m.as_str());

                if let Some(tool) = self.parse_invoke(func_name, invoke_content) {
                    tools.push(tool);
                }
            }
        }

        let mut normal_text = normal_text;
        if let Some(end) = last_block_end {
            normal_text.push_str(&text[end..].replace(EOS_TOKEN, ""));
        }

        Ok((normal_text, tools))
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        let (normal_text, mut calls) = self.parse_complete(output).await?;
        let tool_indices = helpers::get_tool_indices(tools);
        calls.retain(|call| tool_indices.contains_key(call.function.name.as_str()));
        Ok((normal_text, calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        // Check for DSML markers or partial DSML prefixes.
        //
        // `<｜DSML｜` is a single BPE token in DeepSeek's tokenizer (id 128793),
        // so real streams deliver it atomically. We flag the stream as DSML as
        // soon as the sentinel appears anywhere in the buffer — we don't wait
        // for a complete outer `<｜DSML｜{function,tool}_calls>` opener, because
        // live backends chunk the opener into per-token pieces after the
        // sentinel (e.g. `<｜DSML｜` + `tool` + `_c` + `all` + `s` + `>`).
        // Without this, chunk 2 of a live stream would flush the buffer on the
        // passthrough path and lose the sentinel, turning every subsequent
        // chunk into plain text. (See regression test
        // `test_deepseek_dsml_v4_streaming_bpe_chunked_opener`.)
        let has_dsml =
            current_text.contains(DSML_OPEN_SENTINEL) || current_text.contains(DSML_CLOSE_SENTINEL);
        let has_partial_prefix =
            helpers::ends_with_partial_token(&current_text, DSML_OPEN_SENTINEL).is_some()
                || helpers::ends_with_partial_token(&current_text, DSML_CLOSE_SENTINEL).is_some();

        if !has_dsml && !has_partial_prefix {
            // DeepSeek V4 emits blank lines immediately before a DSML tool-call
            // block. Hold whitespace until the next token tells us whether it
            // belongs to normal text or is only block framing.
            if current_text.trim().is_empty() {
                return Ok(StreamingParseResult::default());
            }
            let mut normal_text = std::mem::take(&mut self.buffer);
            for end_token in [
                self.block_close.as_str(),
                DSML_INVOKE_END_TAG,
                DSML_PARAMETER_END_TAG,
                EOS_TOKEN,
            ] {
                normal_text = normal_text.replace(end_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // If we have partial prefix but no actual DSML content, buffer and wait
        if !has_dsml && has_partial_prefix {
            return Ok(StreamingParseResult::default());
        }

        let mut normal_text = String::new();
        if let Some(start) = self.buffer.find(self.block_open.as_str()) {
            normal_text = self.buffer[..start].trim_end().to_string();
            self.buffer = self.buffer[start..].to_string();
        }

        let tool_indices = helpers::get_tool_indices(tools);
        let mut all_calls: Vec<ToolCallItem> = Vec::new();

        // Process invoke blocks in a loop (handles multiple complete invokes in buffer)
        loop {
            let buf_snapshot = self.buffer.clone();
            let invoke_match = self.invoke_regex.captures(&buf_snapshot);

            let captures = match invoke_match {
                Some(c) => c,
                None => break,
            };

            let func_name = captures
                .get(1)
                .map_or(String::new(), |m| m.as_str().trim().to_string());
            let invoke_content = captures
                .get(2)
                .map_or(String::new(), |m| m.as_str().to_string());
            let is_complete = captures
                .get(3)
                .is_some_and(|m| m.as_str().contains("</｜DSML｜invoke>"));
            let match_end = captures.get(0).map(|m| m.end());
            drop(captures);

            // Skip if tool name is absent or not in the provided tools list.
            // Empty names reach this branch because `invoke_regex` allows
            // `name=""` (quantifier `*` not `+`); the loosened regex + this
            // guard together ensure a malformed `name=""` block is advanced
            // past instead of trapping the buffer forever.
            let name_invalid =
                func_name.is_empty() || !tool_indices.contains_key(func_name.as_str());
            if name_invalid {
                tracing::debug!("Invalid tool name '{}' - skipping", func_name);
                if is_complete {
                    // Complete invalid invoke — advance buffer past it and try next
                    if let Some(end) = match_end {
                        self.buffer = self.buffer[end..].to_string();
                    }
                    continue;
                } else {
                    // Incomplete invalid invoke — reset state and wait for more data
                    // Return any calls already collected from previous complete invokes
                    helpers::reset_current_tool_state(
                        &mut self.buffer,
                        &mut self.current_tool_name_sent,
                        &mut self.streamed_args_for_tool,
                        &self.prev_tool_call_arr,
                    );
                    return Ok(StreamingParseResult {
                        normal_text,
                        calls: all_calls,
                    });
                }
            }

            let current_args = self.parse_parameters_from_dsml(&invoke_content, !is_complete);
            if is_complete && !Self::arguments_are_valid_object(&invoke_content, &current_args) {
                tracing::debug!("Invalid arguments for tool '{}' - skipping", func_name);
                if let Some(end) = match_end {
                    self.buffer = self.buffer[end..].to_string();
                }
                if self.current_tool_name_sent {
                    self.current_tool_id += 1;
                }
                self.current_tool_name_sent = false;
                continue;
            }

            // Initialize state on first tool
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

            // Emit tool name if not sent
            if !self.current_tool_name_sent && !func_name.is_empty() {
                all_calls.push(ToolCallItem {
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

            let tool_id = self.current_tool_id as usize;

            // Compute diff against what we've already sent
            let sent_len = self
                .streamed_args_for_tool
                .get(tool_id)
                .map(|s| s.len())
                .unwrap_or(0);

            let prev_args = if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id]
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            };

            let argument_diff = if is_complete {
                if sent_len < current_args.len() {
                    Some(current_args[sent_len..].to_string())
                } else {
                    Some(String::new())
                }
            } else if let Some(prev) = &prev_args {
                if current_args == *prev {
                    None
                } else {
                    let prefix = helpers::find_common_prefix(prev, &current_args);
                    if prefix.len() > sent_len {
                        Some(prefix[sent_len..].to_string())
                    } else {
                        None
                    }
                }
            } else if sent_len < current_args.len() && current_args != "{}" {
                // First partial chunk — no prev_args yet, emit from sent_len.
                // Skip empty "{}" to avoid corrupting the delta stream when the
                // buffer ends right after <invoke> with no parameter content yet.
                Some(current_args[sent_len..].to_string())
            } else {
                None
            };

            if let Some(diff) = argument_diff {
                if !diff.is_empty() {
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(&diff);
                    }
                    all_calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: diff,
                    });
                }
            }

            // Update prev state
            if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id] = serde_json::json!({
                    "name": func_name,
                    "arguments": current_args,
                });
            }

            // If invoke is complete, advance to next tool
            if is_complete {
                if let Some(end) = match_end {
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    self.buffer.clear();
                }
                self.completed_tool_call_count += 1;
                self.current_tool_id += 1;
                self.current_tool_name_sent = false;
                continue;
            } else {
                break;
            }
        }

        // Once the outer close tag is complete, discard all DSML framing and
        // release only genuine trailing assistant text. In live V4 streams the
        // closing sentinel is one token followed by the block-name pieces, so
        // treating only the opening sentinel as DSML leaks the close tag.
        if let Some(start) = self.buffer.find(self.block_close.as_str()) {
            let tail = self.buffer[start + self.block_close.len()..].replace(EOS_TOKEN, "");
            self.buffer.clear();
            normal_text.push_str(&tail);
        }

        Ok(StreamingParseResult {
            normal_text,
            calls: all_calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(DSML_OPEN_SENTINEL) || text.contains(DSML_CLOSE_SENTINEL)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn completed_tool_call_count(&self) -> Option<usize> {
        Some(self.completed_tool_call_count)
    }

    fn finish_incremental(&mut self) -> StreamingParseResult {
        let marker_start = [DSML_OPEN_SENTINEL, DSML_CLOSE_SENTINEL]
            .into_iter()
            .filter_map(|marker| self.buffer.find(marker))
            .min();
        let normal_text = match marker_start {
            Some(start) => self.buffer[..start].trim_end().to_string(),
            None => std::mem::take(&mut self.buffer),
        };
        self.buffer.clear();
        StreamingParseResult {
            normal_text,
            calls: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
        self.completed_tool_call_count = 0;
    }
}
