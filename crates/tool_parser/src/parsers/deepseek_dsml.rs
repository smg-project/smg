use std::collections::HashMap;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::{json, Value};

use crate::{
    errors::{ParserError, ParserResult},
    json_format,
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// The DSML tag dialect a [`DeepSeekDsmlParser`] speaks.
///
/// V3.2 and V4 share one grammar and differ only in the outer block name.
/// V4.1 puts a leading space inside every tag (`<｜DSML｜ calls>`,
/// `<｜DSML｜ invoke ...>`, `<｜DSML｜ parameter ...>`) and follows the
/// reference parser's rules rather than V4's (see [`DsmlDialect::is_v41`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsmlDialect {
    V32,
    V4,
    V41,
}

impl DsmlDialect {
    /// The outer block tag name (rendered as `<｜DSML｜{block}>`).
    pub const fn block(self) -> &'static str {
        match self {
            Self::V32 => "function_calls",
            Self::V4 => "tool_calls",
            Self::V41 => " calls",
        }
    }

    /// The invoke tag name (rendered as `<｜DSML｜{invoke} name="...">`).
    pub const fn invoke(self) -> &'static str {
        match self {
            Self::V41 => " invoke",
            Self::V32 | Self::V4 => "invoke",
        }
    }

    /// The parameter tag name (rendered as `<｜DSML｜{parameter} name="..." string="...">`).
    pub const fn parameter(self) -> &'static str {
        match self {
            Self::V41 => " parameter",
            Self::V32 | Self::V4 => "parameter",
        }
    }

    /// V4.1 parsing rules, from the reference parser (`encoding.py`), vLLM
    /// Python and SGLang: exactly the `"\n\n"` separator before the block is
    /// stripped from the content; `arguments` are serialised like Python's
    /// `json.dumps`; an invoke outside a block is accepted; text after the
    /// tool section is dropped; unknown tool names are forwarded unfiltered.
    const fn is_v41(self) -> bool {
        matches!(self, Self::V41)
    }
}

/// DeepSeek DSML format parser for tool calls (V3.2, V4 and V4.1).
///
/// All dialects share the invoke/parameter grammar and the streaming state
/// machine; the tag literals and regexes are built from the [`DsmlDialect`].
///
/// ```text
/// <｜DSML｜{block}>
/// <｜DSML｜{invoke} name="func">
/// <｜DSML｜{parameter} name="key" string="true">value</｜DSML｜{parameter}>
/// </｜DSML｜{invoke}>
/// </｜DSML｜{block}>
/// ```
///
/// Also supports direct JSON inside invoke blocks as a fallback format.
///
/// References:
/// - <https://huggingface.co/deepseek-ai/DeepSeek-V3.2>
/// - <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
/// - <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>
pub struct DeepSeekDsmlParser {
    dialect: DsmlDialect,
    /// Cached `<｜DSML｜{block}>` for marker-scan hot paths.
    block_open: String,
    /// Cached `</｜DSML｜{block}>` for streaming cleanup.
    block_close: String,
    /// Cached `<｜DSML｜{invoke} name="`: an invoke opener, for V4.1's
    /// block-less invokes and streaming holdback.
    invoke_open_prefix: String,
    /// Cached `</｜DSML｜{invoke}>` for suffix-based stripping during streaming.
    invoke_end_tag: String,
    /// Cached `</｜DSML｜{parameter}>` for suffix-based stripping during streaming.
    parameter_end_tag: String,

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
    /// V4.1 streaming: a tool marker has been seen, so everything from it on
    /// belongs to the tool section (text after it is never content again).
    in_tool_section: bool,
}

/// The DSML sentinel every tag starts with (`<` + this + tag name).
const DSML_TOKEN: &str = "｜DSML｜";

/// DeepSeek end-of-sentence marker. Some engines emit this as raw text at the
/// end of a truncated turn; it must never bleed into tool-call argument bytes.
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";

/// The blank line the V4.1 renderer and model put between content and a tool
/// block; the reference parser strips exactly this from the content.
const V41_BLOCK_SEPARATOR: &str = "\n\n";

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

/// Length in bytes of the longest suffix of `text` that is a proper prefix of
/// one of `patterns` — the part a streaming parser must hold back because the
/// next chunk could complete it.
fn longest_partial_suffix(text: &str, patterns: &[&str]) -> usize {
    let max_len = patterns.iter().map(|p| p.len()).max().unwrap_or(0);
    let earliest_start = text.len().saturating_sub(max_len.saturating_sub(1));
    text.char_indices()
        .map(|(start, _)| start)
        .filter(|start| *start >= earliest_start)
        .find(|start| {
            let suffix = &text[*start..];
            patterns
                .iter()
                .any(|pattern| pattern.len() > suffix.len() && pattern.starts_with(suffix))
        })
        .map_or(0, |start| text.len() - start)
}

impl DeepSeekDsmlParser {
    /// Create a DeepSeek V3.2 parser (outer block token `function_calls`).
    pub fn v32() -> Self {
        Self::new(DsmlDialect::V32)
    }

    /// Create a DeepSeek V4 parser (outer block token `tool_calls`).
    pub fn v4() -> Self {
        Self::new(DsmlDialect::V4)
    }

    /// Create a DeepSeek V4.1 parser (spaced tags, reference parsing rules).
    pub fn v41() -> Self {
        Self::new(DsmlDialect::V41)
    }

    /// Which dialect this instance parses.
    pub fn dialect(&self) -> DsmlDialect {
        self.dialect
    }

    /// Which block name this instance parses (`function_calls`, `tool_calls`
    /// or V4.1's ` calls`).
    pub fn block_name(&self) -> &'static str {
        self.dialect.block()
    }

    #[expect(
        clippy::expect_used,
        reason = "regex patterns are built from compile-time tag literals"
    )]
    fn new(dialect: DsmlDialect) -> Self {
        let dsml = DSML_TOKEN;
        let block = regex::escape(dialect.block());
        let inv = regex::escape(dialect.invoke());
        let par = regex::escape(dialect.parameter());

        let tool_call_complete_regex =
            Regex::new(&format!(r"(?s)<{dsml}{block}>(.*?)</{dsml}{block}>"))
                .expect("Valid regex pattern");

        let invoke_complete_regex = Regex::new(&format!(
            r#"(?s)<{dsml}{inv}\s+name="([^"]+)"\s*>(.*?)</{dsml}{inv}>"#
        ))
        .expect("Valid regex pattern");

        let parameter_complete_regex = Regex::new(&format!(
            r#"(?s)<{dsml}{par}\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</{dsml}{par}>"#
        ))
        .expect("Valid regex pattern");

        let partial_parameter_regex = Regex::new(&format!(
            r#"(?s)<{dsml}{par}\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*)$"#
        ))
        .expect("Valid regex pattern");

        // `[^"]*` (not `+`) so a malformed `name=""` still matches and can be
        // advanced past by the empty/invalid-name handling in `stream_invokes`.
        // Without this, a bad `name=""` invoke would stall the buffer forever
        // and suppress every subsequent delta in the same stream.
        let invoke_regex = Regex::new(&format!(
            r#"(?s)<{dsml}{inv}\s+name="([^"]*)"\s*>(.*?)(</{dsml}{inv}>|$)"#
        ))
        .expect("Valid regex pattern");

        Self {
            dialect,
            block_open: format!("<{dsml}{}>", dialect.block()),
            block_close: format!("</{dsml}{}>", dialect.block()),
            invoke_open_prefix: format!("<{dsml}{} name=\"", dialect.invoke()),
            invoke_end_tag: format!("</{dsml}{}>", dialect.invoke()),
            parameter_end_tag: format!("</{dsml}{}>", dialect.parameter()),
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
            in_tool_section: false,
        }
    }

    /// Serialise an arguments object the way this dialect's reference does:
    /// Python `json.dumps` spacing for V4.1, compact JSON for V3.2/V4.
    fn dump_arguments(&self, params: serde_json::Map<String, Value>) -> String {
        let value = Value::Object(params);
        if self.dialect.is_v41() {
            json_format::to_string(&value)
        } else {
            serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
        }
    }

    /// Byte offset of the first tool marker in `text`: the block opener, or
    /// for V4.1 also a block-less invoke opener.
    fn first_tool_marker(&self, text: &str) -> Option<usize> {
        let block = text.find(self.block_open.as_str());
        if !self.dialect.is_v41() {
            return block;
        }
        let invoke = text.find(self.invoke_open_prefix.as_str());
        match (block, invoke) {
            (Some(b), Some(i)) => Some(b.min(i)),
            (b, i) => b.or(i),
        }
    }

    /// V4.1 content before a tool marker: the reference strips exactly the
    /// blank-line separator the model emits before the block.
    fn v41_content(text: &str) -> String {
        text.strip_suffix(V41_BLOCK_SEPARATOR)
            .unwrap_or(text)
            .replace(EOS_TOKEN, "")
    }

    /// Parse DSML parameters from invoke content into a JSON string.
    ///
    /// Supports two formats:
    /// 1. Direct JSON: content starts with `{` — returned as-is
    /// 2. XML parameters: `<｜DSML｜{parameter} name="k" string="true|false">v</｜DSML｜{parameter}>`
    ///
    /// When `allow_partial` is true (streaming), also matches open parameter tags
    /// and strips trailing DSML fragments.
    fn parse_parameters_from_dsml(&self, invoke_content: &str, allow_partial: bool) -> String {
        let trimmed = invoke_content.trim();

        // Direct JSON path (also the shape SGLang's grammar emits under a
        // structural-tag constraint). The text is the arguments object and is
        // passed through as written — a compact body stays compact rather
        // than being re-spaced like DSML parameters — so streaming deltas stay
        // consistent (SGLang does the same).
        if trimmed.starts_with('{') {
            if allow_partial {
                // `strip_dsml_trailing` handles partial `</｜DSML｜invoke>` prefixes
                // but can't match the EOS sentinel (different prefix). Strip it
                // unconditionally so a truncated turn doesn't leak EOS into args.
                return strip_dsml_trailing(trimmed, &self.invoke_end_tag).replace(EOS_TOKEN, "");
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

            // `string="true"` is a JSON string as written; `string="false"` is
            // parsed JSON, and JSON that does not parse stays the raw string.
            // No schema-driven coercion (the reference parser has none).
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
            let cleaned = strip_dsml_trailing(remaining, &self.parameter_end_tag);

            if let Some(cap) = self.partial_parameter_regex.captures(&cleaned) {
                let name = cap.get(1).map_or("", |m| m.as_str());
                let is_string = cap.get(2).map_or("true", |m| m.as_str());
                // Strip EOS before trimming — `strip_dsml_trailing` above only
                // handles `</｜DSML｜parameter>` prefixes, so a truncated turn
                // with `value<EOS>` would otherwise stream EOS as arg bytes.
                let value = cap.get(3).map_or("", |m| m.as_str()).replace(EOS_TOKEN, "");

                // Only add if we have actual content and this param isn't already
                // complete. A `string="true"` value is taken raw, exactly as the
                // complete path takes it: trimming a partial value would make the
                // streamed bytes diverge from the finished value and the delta
                // offsets slice it mid-character. A `string="false"` value is
                // held back until its text parses as JSON: emitting an unfinished
                // `[1, true, nul` as a raw string would start the delta stream
                // with a quote the finished array never has, corrupting every
                // later delta.
                if !value.is_empty() && !params.contains_key(name) {
                    let json_value = if is_string == "true" {
                        Some(Value::String(value.clone()))
                    } else {
                        serde_json::from_str(value.trim()).ok()
                    };
                    if let Some(json_value) = json_value {
                        params.insert(name.to_string(), json_value);
                    }
                }
            }
        }

        self.dump_arguments(params)
    }

    /// Parse a single complete invoke block into a ToolCall
    fn parse_invoke(&self, name: &str, content: &str) -> ToolCall {
        let arguments = self.parse_parameters_from_dsml(content, false);

        ToolCall {
            function: FunctionCall {
                name: name.trim().to_string(),
                arguments,
            },
        }
    }

    /// V4.1 one-shot parse: content is everything before the first tool
    /// marker minus the blank-line separator; every complete invoke after it
    /// is a call (inside a block or not); text after the section is dropped.
    fn parse_complete_v41(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        let idx = self
            .first_tool_marker(text)
            .ok_or_else(|| ParserError::ParsingFailed("DSML marker not found".to_string()))?;
        let normal_text = Self::v41_content(&text[..idx]);
        let tools = self
            .invoke_complete_regex
            .captures_iter(&text[idx..])
            .map(|cap| {
                let name = cap.get(1).map_or("", |m| m.as_str());
                let body = cap.get(2).map_or("", |m| m.as_str());
                self.parse_invoke(name, body)
            })
            .collect();
        Ok((normal_text, tools))
    }

    /// V4.1 streaming. Before the tool section, content is flushed as it
    /// arrives, holding back only a suffix that could still become the
    /// separator plus a tool opener (so `"\n\n"` right before a block is
    /// stripped exactly as in the one-shot path; a blank line at the very end
    /// of a stream with no tool call is held and never flushed). From the
    /// first tool marker on, everything belongs to the tool section: invokes
    /// stream as deltas and any other text is dropped.
    fn parse_incremental_v41(&mut self) -> StreamingParseResult {
        let mut normal_text = String::new();
        if !self.in_tool_section {
            match self.first_tool_marker(&self.buffer) {
                Some(idx) => {
                    normal_text = Self::v41_content(&self.buffer[..idx]);
                    self.buffer.drain(..idx);
                    self.in_tool_section = true;
                }
                None => {
                    let with_separator_block = format!("{V41_BLOCK_SEPARATOR}{}", self.block_open);
                    let with_separator_invoke =
                        format!("{V41_BLOCK_SEPARATOR}{}", self.invoke_open_prefix);
                    let keep = longest_partial_suffix(
                        &self.buffer,
                        &[
                            with_separator_block.as_str(),
                            with_separator_invoke.as_str(),
                            self.block_open.as_str(),
                            self.invoke_open_prefix.as_str(),
                        ],
                    );
                    let emit_end = self.buffer.len() - keep;
                    normal_text = self.buffer[..emit_end].replace(EOS_TOKEN, "");
                    self.buffer.drain(..emit_end);
                    return StreamingParseResult {
                        normal_text,
                        calls: vec![],
                    };
                }
            }
        }
        let calls = self.stream_invokes(None);
        StreamingParseResult { normal_text, calls }
    }

    /// Stream every invoke currently in the buffer as name/argument deltas.
    /// `allowed_names` filters unknown tool names (V3.2/V4); `None` forwards
    /// every named invoke (V4.1, like the reference engines).
    fn stream_invokes(
        &mut self,
        allowed_names: Option<&HashMap<String, usize>>,
    ) -> Vec<ToolCallItem> {
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
                .is_some_and(|m| m.as_str().contains(self.invoke_end_tag.as_str()));
            let match_end = captures.get(0).map(|m| m.end());
            drop(captures);

            // Skip if tool name is absent or (when filtering) not in the
            // provided tools list. Empty names reach this branch because
            // `invoke_regex` allows `name=""` (quantifier `*` not `+`); the
            // loosened regex + this guard together ensure a malformed
            // `name=""` block is advanced past instead of trapping the buffer.
            let name_invalid = func_name.is_empty()
                || allowed_names.is_some_and(|names| !names.contains_key(func_name.as_str()));
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
                    return all_calls;
                }
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
                self.prev_tool_call_arr[tool_id] = json!({
                    "name": func_name,
                    "arguments": {},
                });
            }

            // Parse current arguments (partial or complete)
            let current_args = self.parse_parameters_from_dsml(&invoke_content, !is_complete);
            let tool_id = self.current_tool_id as usize;

            // Compute diff against what we've already sent
            let sent_len = self
                .streamed_args_for_tool
                .get(tool_id)
                .map(|s| s.len())
                .unwrap_or(0);

            // The snapshot the previous pass streamed against; before any
            // pass it is the empty object, so the first partial snapshot is
            // diffed like every later one and only the prefix two consecutive
            // snapshots agree on is streamed. Emitting a first snapshot whole
            // would send its closing brace and a still-growing value (a
            // `string="false"` number, a `string="true"` body) ahead of bytes
            // that later snapshots change, and the completion delta would then
            // append a tail that no longer lines up with what was sent.
            let prev_args = self
                .prev_tool_call_arr
                .get(tool_id)
                .and_then(|prev| prev.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_string();

            let argument_diff = if is_complete {
                if sent_len < current_args.len() {
                    Some(current_args.get(sent_len..).unwrap_or_default().to_string())
                } else {
                    Some(String::new())
                }
            } else if current_args == prev_args {
                None
            } else {
                let prefix = helpers::find_common_prefix(&prev_args, &current_args);
                if prefix.len() > sent_len {
                    Some(prefix.get(sent_len..).unwrap_or_default().to_string())
                } else {
                    None
                }
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
                self.prev_tool_call_arr[tool_id] = json!({
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
                self.current_tool_id += 1;
                self.current_tool_name_sent = false;
                continue;
            } else {
                break;
            }
        }

        all_calls
    }

    /// The V4.1 tool-call grammar as an xgrammar structural tag, mirroring
    /// vLLM's `deepseek_v41` builder for a forced tool choice
    /// (`tool_choice: required` or a named function): a blank line, the
    /// `<｜DSML｜ calls>` block, at least one invoke of a listed tool, and the
    /// block close. Each parameter constrains the DSML syntax only — a
    /// `string="true"` body is any text without a closing tag, a
    /// `string="false"` body is any JSON value; parameter names and schemas
    /// are not lowered (vLLM leaves that as a TODO too). `at_least_one` is
    /// wired to `tool_choice` by the registry; like the Kimi builders,
    /// `stop_after_first` is left unset so a single listed tool may still be
    /// called more than once.
    pub fn build_v41_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let dialect = DsmlDialect::V41;
        let calls_start = format!("<{DSML_TOKEN}{}>", dialect.block());
        let calls_end = format!("</{DSML_TOKEN}{}>", dialect.block());
        let invoke_end = format!("</{DSML_TOKEN}{}>", dialect.invoke());
        let parameter_end = format!("</{DSML_TOKEN}{}>", dialect.parameter());

        let parameter = json!({
            "type": "tag",
            "begin": format!("<{DSML_TOKEN}{} name=\"", dialect.parameter()),
            "content": {
                "type": "sequence",
                "elements": [
                    { "type": "regex", "pattern": "[^\"]+" },
                    { "type": "const_string", "value": "\" string=\"" },
                    {
                        "type": "or",
                        "elements": [
                            {
                                "type": "sequence",
                                "elements": [
                                    { "type": "const_string", "value": "true\">" },
                                    {
                                        "type": "any_text",
                                        "excludes": [parameter_end, invoke_end, calls_end],
                                    },
                                ],
                            },
                            {
                                "type": "sequence",
                                "elements": [
                                    { "type": "const_string", "value": "false\">" },
                                    { "type": "json_schema", "json_schema": true },
                                ],
                            },
                        ],
                    },
                ],
            },
            "end": format!("{parameter_end}\n"),
        });

        let invokes: Vec<Value> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| {
                json!({
                    "type": "tag",
                    "begin": format!(
                        "<{DSML_TOKEN}{} name=\"{}\">\n",
                        dialect.invoke(),
                        tool.function.name
                    ),
                    "content": { "type": "star", "content": parameter },
                    "end": format!("{invoke_end}\n"),
                })
            })
            .collect();

        json!({
            "format": {
                "type": "sequence",
                "elements": [
                    { "type": "const_string", "value": format!("{V41_BLOCK_SEPARATOR}{calls_start}\n") },
                    {
                        "type": "tags_with_separator",
                        "tags": invokes,
                        "separator": "",
                        "at_least_one": at_least_one,
                    },
                    { "type": "const_string", "value": calls_end },
                ],
            }
        })
    }
}

// Intentionally no `Default` impl — callers must pick `v32()`, `v4()` or `v41()`.

#[async_trait]
impl ToolParser for DeepSeekDsmlParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        if self.dialect.is_v41() {
            return self.parse_complete_v41(text);
        }

        let idx = text
            .find(self.block_open.as_str())
            .ok_or_else(|| ParserError::ParsingFailed("DSML marker not found".to_string()))?;
        let normal_text = text[..idx].trim_end().to_string();

        let mut tools = Vec::new();

        for fc_cap in self.tool_call_complete_regex.captures_iter(text) {
            let fc_content = fc_cap.get(1).map_or("", |m| m.as_str());

            for inv_cap in self.invoke_complete_regex.captures_iter(fc_content) {
                let func_name = inv_cap.get(1).map_or("", |m| m.as_str());
                let invoke_content = inv_cap.get(2).map_or("", |m| m.as_str());

                tools.push(self.parse_invoke(func_name, invoke_content));
            }
        }

        if tools.is_empty() {
            return Ok((normal_text, vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        if self.dialect.is_v41() {
            return Ok(self.parse_incremental_v41());
        }

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
        let has_dsml = current_text.contains("<｜DSML｜");
        let has_partial_prefix = current_text.ends_with('<')
            || current_text.ends_with("<｜")
            || current_text.ends_with("</")
            || current_text.ends_with("</｜");

        if !has_dsml && !has_partial_prefix {
            let mut normal_text = std::mem::take(&mut self.buffer);
            for end_token in [
                self.block_close.as_str(),
                self.invoke_end_tag.as_str(),
                self.parameter_end_tag.as_str(),
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

        let tool_indices = helpers::get_tool_indices(tools);
        let calls = self.stream_invokes(Some(&tool_indices));

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        self.first_tool_marker(text).is_some()
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        // The tracked snapshot is the exact argument string the deltas were
        // diffed against, so it is compared verbatim; the shared helper would
        // re-serialise it as a quoted JSON string and never match.
        let tool_index = self.prev_tool_call_arr.len().checked_sub(1)?;
        let snapshot = self.prev_tool_call_arr[tool_index]
            .get("arguments")?
            .as_str()?;
        let streamed = self.streamed_args_for_tool.get(tool_index)?;
        let remaining = snapshot.strip_prefix(streamed.as_str())?;
        (!remaining.is_empty()).then(|| {
            vec![ToolCallItem {
                tool_index,
                name: None,
                parameters: remaining.to_string(),
            }]
        })
    }

    fn take_unstreamed_normal_text(&mut self) -> String {
        // V4.1 holds back a suffix that could still become the separator plus
        // a tool opener; V3.2/V4 hold `<`/`</` prefixes and, once the DSML
        // sentinel has arrived, everything from it on. At end of stream the
        // held text is content when no tool syntax ever started, and is
        // dropped otherwise: a truncated opener or invoke is not content (the
        // remaining arguments of a truncated invoke come from
        // `get_unstreamed_tool_args`).
        if self.in_tool_section || self.buffer.contains("<｜DSML｜") {
            self.buffer.clear();
            return String::new();
        }
        helpers::take_unstreamed_normal_text(&mut self.buffer, self.current_tool_id)
            .replace(EOS_TOKEN, "")
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
        self.in_tool_section = false;
    }
}
