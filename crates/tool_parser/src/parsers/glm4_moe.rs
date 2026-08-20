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

/// GLM-4 MoE format parser for tool calls
///
/// Handles both GLM-4 MoE and GLM-4.7 MoE formats:
/// - GLM-4: `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
/// - GLM-4.7: `<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>`
///
/// Features:
/// - XML-style tags for tool calls
/// - Key-value pairs for arguments
/// - Support for multiple sequential tool calls
pub struct Glm4MoeParser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
    /// Regex for extracting argument key-value pairs
    arg_extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
}

impl Glm4MoeParser {
    /// Create a new generic GLM MoE parser with a custom func_detail_extractor pattern
    ///
    /// # Arguments
    /// - `func_detail_pattern`: Regex pattern for extracting function name and arguments
    ///   - For GLM-4: `r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>"`
    ///   - For GLM-4.7: `r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>"`
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub(crate) fn new(func_detail_pattern: &str) -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<tool_call>.*?</tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        let arg_pattern = r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>";
        let arg_extractor = Regex::new(arg_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            arg_extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            bot_token: "<tool_call>",
            eot_token: "</tool_call>",
        }
    }

    /// Create a new GLM-4.5/4.6 MoE parser (with newline-based format)
    pub fn glm45() -> Self {
        Self::new(r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>")
    }

    /// Create a new GLM-4.7 MoE parser (with whitespace-based format)
    pub fn glm47() -> Self {
        Self::new(r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>")
    }

    /// Parse arguments, coercing each value by its declared schema type and
    /// falling back to [`infer_value`] when the type is unknown.
    fn parse_arguments(
        &self,
        args_text: &str,
        param_types: &HashMap<String, String>,
    ) -> ParserResult<serde_json::Map<String, Value>> {
        let mut arguments = serde_json::Map::new();
        let mut consumed = 0;

        for capture in self.arg_extractor.captures_iter(args_text) {
            let full_match = capture.get(0).ok_or_else(|| {
                ParserError::ParsingFailed("malformed GLM argument block".to_string())
            })?;
            if !args_text[consumed..full_match.start()].trim().is_empty() {
                return Err(ParserError::ParsingFailed(
                    "malformed GLM argument tags".to_string(),
                ));
            }

            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            if key.is_empty() {
                return Err(ParserError::ParsingFailed(
                    "GLM argument key must not be empty".to_string(),
                ));
            }
            let raw_value = capture.get(2).map_or("", |m| m.as_str());
            let declared_type = param_types.get(key).map(String::as_str);
            let value_str = if declared_type == Some("string") {
                raw_value
            } else {
                raw_value.trim()
            };

            let value = helpers::coerce_by_schema_type(value_str, declared_type)
                .unwrap_or_else(|| infer_value(value_str));

            arguments.insert(key.to_string(), value);
            consumed = full_match.end();
        }

        if !args_text[consumed..].trim().is_empty() {
            return Err(ParserError::ParsingFailed(
                "malformed GLM argument tags".to_string(),
            ));
        }

        Ok(arguments)
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function name
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get arguments text
            let args_text = captures.get(2).map_or("", |m| m.as_str());

            let param_types = helpers::param_types_for_function(tools, func_name);
            let arguments = self.parse_arguments(args_text, &param_types)?;

            let arguments_str = serde_json::to_string(&arguments)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: func_name.to_string(),
                    arguments: arguments_str,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse all tool calls from text (shared logic for complete and incremental parsing)
    fn parse_tool_calls_from_text(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<Vec<ToolCall>> {
        let mut parsed = Vec::new();
        let tool_indices = helpers::get_tool_indices(tools);

        for mat in self.tool_call_extractor.find_iter(text) {
            let Some(tool) = self.parse_tool_call(mat.as_str(), tools)? else {
                return Err(ParserError::ParsingFailed(
                    "malformed GLM tool call".to_string(),
                ));
            };
            if !tools.is_empty() && !tool_indices.contains_key(&tool.function.name) {
                return Err(ParserError::InvalidToolName(tool.function.name));
            }
            parsed.push(tool);
        }

        Ok(parsed)
    }
}

impl Glm4MoeParser {
    const STRUCTURAL_MARKERS: [&str; 6] = [
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
    ];

    fn partial_marker_suffix_len(text: &str, marker: &str) -> usize {
        (1..marker.len())
            .rev()
            .find(|&len| text.ends_with(&marker[..len]))
            .unwrap_or(0)
    }

    fn partial_structural_marker_suffix_len(text: &str) -> usize {
        Self::STRUCTURAL_MARKERS
            .iter()
            .map(|marker| Self::partial_marker_suffix_len(text, marker))
            .max()
            .unwrap_or(0)
    }

    fn contains_structural_fragment(text: &str) -> bool {
        ["<tool_", "</tool_", "<arg_", "</arg_"]
            .iter()
            .any(|prefix| text.contains(prefix))
    }

    fn has_post_call_structural_residue(&self, text: &str) -> bool {
        let mut saw_call = false;
        let mut previous_end = 0;

        for block in self.tool_call_extractor.find_iter(text) {
            if saw_call && Self::contains_structural_fragment(&text[previous_end..block.start()]) {
                return true;
            }
            saw_call = true;
            previous_end = block.end();
        }

        saw_call
            && (Self::contains_structural_fragment(&text[previous_end..])
                || Self::partial_structural_marker_suffix_len(&text[previous_end..]) > 0)
    }

    fn record_tool_call(&mut self, tool_call: ToolCall) -> ToolCallItem {
        if self.current_tool_id == -1 {
            self.current_tool_id = 0;
            self.prev_tool_call_arr.clear();
            self.streamed_args_for_tool.clear();
        }

        let tool_id = self.current_tool_id as usize;
        helpers::ensure_capacity(
            self.current_tool_id,
            &mut self.prev_tool_call_arr,
            &mut self.streamed_args_for_tool,
        );

        if let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) {
            self.prev_tool_call_arr[tool_id] = serde_json::json!({
                "name": tool_call.function.name,
                "arguments": args,
            });
        }
        self.streamed_args_for_tool[tool_id].clone_from(&tool_call.function.arguments);
        self.current_tool_id += 1;

        ToolCallItem {
            tool_index: tool_id,
            name: Some(tool_call.function.name),
            parameters: tool_call.function.arguments,
        }
    }

    fn drain_incremental(
        &mut self,
        tools: &[Tool],
        end_of_input: bool,
    ) -> ParserResult<StreamingParseResult> {
        let mut normal_text = String::new();
        let mut calls = Vec::new();
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            if let Some(start) = self.buffer.find(self.bot_token) {
                if start > 0 {
                    if self.current_tool_id == -1 {
                        normal_text.push_str(&self.buffer[..start]);
                    }
                    self.buffer.drain(..start);
                    continue;
                }

                let Some(end_pos) = self.buffer.find(self.eot_token) else {
                    break;
                };
                let block_end = end_pos + self.eot_token.len();
                let block = self.buffer[..block_end].to_string();

                let Some(tool_call) = self.parse_tool_call(&block, tools)? else {
                    return Err(ParserError::ParsingFailed(
                        "malformed GLM tool call".to_string(),
                    ));
                };
                // Only validate tool names against a schema when tools are
                // provided; no-schema callers (inference-only, native markup
                // decoding) pass an empty slice and should accept any name.
                if !tools.is_empty() && !tool_indices.contains_key(&tool_call.function.name) {
                    return Err(ParserError::InvalidToolName(tool_call.function.name));
                }
                self.buffer.drain(..block_end);
                calls.push(self.record_tool_call(tool_call));
                continue;
            }

            if self.current_tool_id != -1 && Self::contains_structural_fragment(&self.buffer) {
                return Err(ParserError::ParsingFailed(
                    "unexpected GLM structure after tool call".to_string(),
                ));
            }

            // If bot_token is already in the buffer (but eot_token hasn't arrived),
            // we must hold everything from that index so the tool-call marker isn't
            // drained and emitted as normal text.
            let held_len = if let Some(start) = self.buffer.find(self.bot_token) {
                self.buffer.len() - start
            } else if self.current_tool_id == -1 {
                Self::partial_marker_suffix_len(&self.buffer, self.bot_token)
            } else {
                Self::partial_structural_marker_suffix_len(&self.buffer)
            };
            let emit_len = self.buffer.len() - held_len;
            if self.current_tool_id == -1 {
                normal_text.push_str(&self.buffer[..emit_len]);
            }
            self.buffer.drain(..emit_len);
            break;
        }

        if end_of_input && !self.buffer.is_empty() {
            // At EOF, release short ambiguous prefixes (≤ 2 bytes) as normal
            // text rather than failing with Incomplete. A lone "<" or "</" in
            // ordinary output should not trigger an upstream error.
            if self.current_tool_id == -1
                && self.buffer.len() <= 2
                && !Self::contains_structural_fragment(&self.buffer)
            {
                normal_text.push_str(&self.buffer);
                self.buffer.clear();
            } else {
                return Err(ParserError::Incomplete);
            }
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    /// Shared non-streaming parse, schema-aware when `tools` are provided.
    fn parse_complete_inner(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        let has_complete_marker = self.has_tool_markers(text);
        // Short ambiguous prefixes ("<", "</", "<t") are too common in
        // ordinary text to treat as incomplete at EOF; only reject meaningful
        // structural prefixes (3+ bytes, e.g. "<to", "</t", "<ar"). This
        // mirrors the streaming EOF path, which releases <=2-byte buffers.
        let has_partial_marker = Self::partial_structural_marker_suffix_len(text) > 2;
        if has_partial_marker {
            return Err(ParserError::Incomplete);
        }
        // A standalone unmatched closing marker (`eot_token`) with no matching
        // opening `bot_token` is malformed structured output. Without this
        // guard `has_complete_marker` would be false and the raw GLM markup
        // would leak to clients as ordinary assistant text, bypassing the
        // parse-error/502 path. The streaming path already rejects this via
        // its post-call structural-residue checks.
        if !has_complete_marker && text.contains(self.eot_token) {
            return Err(ParserError::ParsingFailed(
                "unmatched GLM closing marker without opening marker".to_string(),
            ));
        }
        if !has_complete_marker {
            return Ok((text.to_string(), vec![]));
        }
        // Strip extracted tool-call blocks so orphan marker checks are not
        // tripped by marker text that happens to appear inside argument values
        // (e.g. a search query containing the literal "<tool_call>").
        let mut stripped = String::with_capacity(text.len());
        let mut last_end = 0;
        for m in self.tool_call_extractor.find_iter(text) {
            stripped.push_str(&text[last_end..m.start()]);
            last_end = m.end();
        }
        stripped.push_str(&text[last_end..]);

        if stripped.contains(self.bot_token) {
            return Err(ParserError::Incomplete);
        }
        if stripped.contains(self.eot_token) || self.has_post_call_structural_residue(text)
        {
            return Err(ParserError::ParsingFailed(
                "unexpected GLM structure outside tool call".to_string(),
            ));
        }

        // Find where tool calls begin
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find("<tool_call>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        let parsed = self.parse_tool_calls_from_text(text, tools)?;

        // Structured output must never silently fall back to raw marker text.
        if parsed.is_empty() {
            return Err(ParserError::ParsingFailed(
                "malformed or incomplete GLM tool call".to_string(),
            ));
        }

        Ok((normal_text, parsed))
    }
}

/// Infer a JSON value from raw text when the schema type is unknown: JSON
/// (numbers/bools/null/objects/arrays), then Python-style literals, then string.
fn infer_value(value_str: &str) -> Value {
    if let Ok(json_val) = serde_json::from_str::<Value>(value_str) {
        return json_val;
    }
    match value_str {
        "true" | "True" => Value::Bool(true),
        "false" | "False" => Value::Bool(false),
        "null" | "None" => Value::Null,
        _ => {
            if let Ok(num) = value_str.parse::<i64>() {
                Value::Number(num.into())
            } else if let Ok(num) = value_str.parse::<f64>() {
                serde_json::Number::from_f64(num)
                    .map_or_else(|| Value::String(value_str.to_string()), Value::Number)
            } else {
                Value::String(value_str.to_string())
            }
        }
    }
}

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::glm45()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
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
        self.drain_incremental(tools, false)
    }

    async fn finalize(&mut self, tools: &[Tool]) -> ParserResult<StreamingParseResult> {
        self.drain_incremental(tools, true)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.bot_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;

    use super::*;

    fn tool_with_props(props: Value) -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": props}),
                strict: None,
            },
        }]
    }

    // String-typed params stay strings even when they look numeric/bool/array.
    #[tokio::test]
    async fn test_schema_aware_coercion_keeps_strings() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "flag": {"type": "string"},
            "coords": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n\
            <arg_key>coords</arg_key>\n<arg_value>[60,30]</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
        let (_, calls) = Glm4MoeParser::glm45()
            .parse_complete_with_tools(text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["flag"], Value::String("true".to_string()));
        assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }

    // The streaming path threads `tools` separately, so cover it too.
    #[tokio::test]
    async fn test_streaming_schema_aware_coercion() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
        let result = Glm4MoeParser::glm45()
            .parse_incremental(text, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }

    #[tokio::test]
    async fn test_non_streaming_rejects_incomplete_or_malformed_markers() {
        let parser = Glm4MoeParser::glm47();

        for text in [
            "plain text <tool_cal",
            "<tool_call>f<arg_key>query</arg_key>",
            "<tool_call></tool_call>",
        ] {
            assert!(
                parser.parse_complete(text).await.is_err(),
                "structured marker must not fall back to raw text: {text}"
            );
        }
    }

    #[tokio::test]
    async fn test_non_streaming_rejects_standalone_unmatched_closing_marker() {
        let parser = Glm4MoeParser::glm47();
        // Build the closing marker via `format!` so the literal tag cannot be
        // accidentally stripped from the source. A standalone `</tool_call>`
        // with no matching `<tool_call>` is malformed structured output and
        // must surface as a parse error rather than leaking as ordinary text.
        let eot = format!("<{}tool_call>", "/");

        for text in [
            eot.clone(),
            format!("some preamble{eot}"),
            format!("lorem ipsum {eot} trailing"),
        ] {
            assert!(
                parser.parse_complete(&text).await.is_err(),
                "a standalone unmatched GLM closing marker must not leak as ordinary text: {text}"
            );
        }
    }

    // Short (<=2 byte) ambiguous prefixes like "</" or "<t" are common in
    // ordinary text and must not be rejected as Incomplete by the complete
    // path, mirroring the streaming EOF exemption. Longer prefixes (3+ bytes)
    // are still treated as incomplete.
    #[tokio::test]
    async fn test_non_streaming_accepts_short_ambiguous_prefixes() {
        let parser = Glm4MoeParser::glm47();

        for text in ["some output </", "more output <t", "trailing <"] {
            let (normal, calls) = parser
                .parse_complete(text)
                .await
                .expect("short ambiguous prefix must not be Incomplete");
            assert!(calls.is_empty(), "no tool calls expected for: {text}");
            assert_eq!(normal, text, "text must pass through unchanged: {text}");
        }

        // 3+ byte prefixes are still meaningful and must be rejected.
        for text in ["output <to", "output </t", "output <ar"] {
            assert!(
                parser.parse_complete(text).await.is_err(),
                "meaningful structural prefix must be Incomplete: {text}"
            );
        }
    }

    #[tokio::test]
    async fn test_non_streaming_rejects_incomplete_second_call_after_valid_call() {
        let parser = Glm4MoeParser::glm47();
        let valid = "<tool_call>f</tool_call>";

        for trailing in [
            "<tool_cal",
            "<tool_call>f<arg_key>query</arg_key><arg_value>unfinished",
            "<tool_call></tool_call>",
        ] {
            let text = format!("{valid}{trailing}");
            assert!(
                parser.parse_complete(&text).await.is_err(),
                "a valid first call must not hide an incomplete second call: {text}"
            );
        }
    }

    #[tokio::test]
    async fn test_non_streaming_rejects_unknown_tool_name() {
        let tools = tool_with_props(serde_json::json!({}));
        let text = "<tool_call>missing</tool_call>";

        let error = Glm4MoeParser::glm47()
            .parse_complete_with_tools(text, &tools)
            .await
            .unwrap_err();

        assert!(matches!(error, ParserError::InvalidToolName(name) if name == "missing"));
    }

    #[tokio::test]
    async fn test_streaming_rejects_unknown_tool_name() {
        let tools = tool_with_props(serde_json::json!({}));
        let text = "<tool_call>missing</tool_call>";
        let mut parser = Glm4MoeParser::glm47();

        let error = parser.parse_incremental(text, &tools).await.unwrap_err();

        assert!(matches!(error, ParserError::InvalidToolName(name) if name == "missing"));
        assert!(matches!(
            parser.parse_incremental("", &tools).await.unwrap_err(),
            ParserError::InvalidToolName(name) if name == "missing"
        ));
    }

    #[tokio::test]
    async fn test_rejects_malformed_argument_tags() {
        let tools = tool_with_props(serde_json::json!({"query": {"type": "string"}}));

        for text in [
            "<tool_call>f<arg_key>query</arg_key><arg_value>people</arg_val></tool_call>",
            "<tool_call>f<arg_key>query</arg_key><arg_value>people</tool_call>",
            "<tool_call>f<arg_key>query</arg_key>people</tool_call>",
            "<tool_call>f<arg_value>people</arg_value></tool_call>",
        ] {
            assert!(
                Glm4MoeParser::glm47()
                    .parse_complete_with_tools(text, &tools)
                    .await
                    .is_err(),
                "malformed argument tags must fail in complete parsing: {text}"
            );

            let mut parser = Glm4MoeParser::glm47();
            assert!(
                parser.parse_incremental(text, &tools).await.is_err(),
                "malformed argument tags must fail in incremental parsing: {text}"
            );
        }
    }

    #[tokio::test]
    async fn test_rejects_structural_residue_after_valid_call() {
        let tools = tool_with_props(serde_json::json!({"query": {"type": "string"}}));
        let valid = "<tool_call>f</tool_call>";

        for residue in [
            "</tool_call>",
            "</tool_cal",
            "<arg_key>query</arg_key>",
            "<arg_value>people</arg_value>",
        ] {
            let text = format!("{valid}{residue}");
            assert!(
                Glm4MoeParser::glm47()
                    .parse_complete_with_tools(&text, &tools)
                    .await
                    .is_err(),
                "complete parsing must reject post-call structural residue: {text}"
            );
        }
    }

    #[tokio::test]
    async fn test_streaming_rejects_unmatched_end_marker_at_every_split() {
        let tools = tool_with_props(serde_json::json!({}));
        let valid = "<tool_call>f</tool_call>";
        let residue = "</tool_call>";

        for split in 0..=residue.len() {
            let mut parser = Glm4MoeParser::glm47();
            let first = format!("{valid}{}", &residue[..split]);
            let mut failed = parser.parse_incremental(&first, &tools).await.is_err();
            if !failed {
                failed = parser
                    .parse_incremental(&residue[split..], &tools)
                    .await
                    .is_err();
            }
            if !failed {
                failed = parser.finalize(&tools).await.is_err();
            }

            assert!(
                failed,
                "unmatched end marker must fail at split {split}: {first:?} + {:?}",
                &residue[split..]
            );
        }
    }
}
