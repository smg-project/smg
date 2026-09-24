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

    /// The xgrammar structural tag for the GLM-4.7 / GLM-5 tool-call format,
    /// mirroring xgrammar's built-in `glm_4_7` tag (the one vLLM applies for
    /// `required` and named tool choices). Each call is `<tool_call>{name}`,
    /// the arguments rendered from the tool's JSON schema in xgrammar's
    /// `glm_xml` style (`<arg_key>k</arg_key><arg_value>v</arg_value>`), then
    /// `</tool_call>`; free text is allowed around the calls and `<tool_call>`
    /// is the trigger. `at_least_one` forces a call, for `required` and named
    /// choices.
    ///
    /// Wire shape: `{"type": "json_schema", "json_schema": …, "style":
    /// "glm_xml"}` is xgrammar's `JSONSchemaFormat` (python/xgrammar/
    /// structural_tag.py), present since the 0.2 line that also ships the
    /// built-in `glm_4_7` tag; TokenSpeed pins 0.2.3, where the tag emitted
    /// here compiles with `Grammar.from_structural_tag` and accepts the
    /// model's real call syntax. The schema is passed through as written,
    /// like the JSON-schema builders (mistral, kimik2, inkling): a tool with
    /// no parameters, whether the field is absent or an explicit `null`,
    /// arrives as `{}` (`Function::parameters` normalises both) and renders
    /// as a call with an empty argument body.
    ///
    /// Thinking prompts: GLM-4.7 (`enable_thinking`, on by default) and GLM-5
    /// (always) end the generation prompt with `<think>`, so a call forced
    /// from the first token would land inside the thinking block, where the
    /// reasoning parser swallows it. This builder is thinking-agnostic; when
    /// the gateway reports that the prompt ends inside `<think>`, the
    /// registry wraps the tag in [`Self::reasoning_prefix`] (see
    /// `ParserRegistry::generate_tool_constraint`).
    pub fn build_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let tags: Vec<Value> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| {
                serde_json::json!({
                    "type": "tag",
                    "begin": format!("<tool_call>{}", tool.function.name),
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool.function.parameters,
                        "style": "glm_xml",
                    },
                    "end": "</tool_call>",
                })
            })
            .collect();
        serde_json::json!({
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": tags,
                "at_least_one": at_least_one,
            }
        })
    }

    /// The reasoning block that precedes a forced call on a thinking prompt:
    /// xgrammar's built-in `glm_4_7` prefix (`reasoning=True`) — free text
    /// closed by `</think>`, with the think and tool-call control tokens
    /// excluded so the model can neither open a call nor nest a block inside
    /// its reasoning. The registry emits `sequence[prefix, calls]`, so the
    /// first `<tool_call>` follows `</think>` directly, as in the built-in.
    pub fn reasoning_prefix() -> Value {
        serde_json::json!({
            "type": "tag",
            "begin": "",
            "content": {
                "type": "any_text",
                "excludes": [
                    "<think>", "</think>",
                    "<tool_call>", "</tool_call>",
                    "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
                ],
            },
            "end": "</think>",
        })
    }

    /// Parse arguments, coercing each value by its declared schema type and
    /// falling back to [`infer_value`] when the type is unknown.
    fn parse_arguments(
        &self,
        args_text: &str,
        param_types: &HashMap<String, String>,
    ) -> serde_json::Map<String, Value> {
        let mut arguments = serde_json::Map::new();

        for capture in self.arg_extractor.captures_iter(args_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

            let value =
                helpers::coerce_by_schema_type(value_str, param_types.get(key).map(String::as_str))
                    .unwrap_or_else(|| infer_value(value_str));

            arguments.insert(key.to_string(), value);
        }

        arguments
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function name
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get arguments text
            let args_text = captures.get(2).map_or("", |m| m.as_str());

            let param_types = helpers::param_types_for_function(tools, func_name);
            let arguments = self.parse_arguments(args_text, &param_types);

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
    fn parse_tool_calls_from_text(&self, text: &str, tools: &[Tool]) -> Vec<ToolCall> {
        let mut parsed = Vec::new();

        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str(), tools) {
                Ok(Some(tool)) => parsed.push(tool),
                Ok(None) => continue,
                Err(e) => {
                    tracing::debug!("Failed to parse tool call: {}", e);
                    continue;
                }
            }
        }

        parsed
    }
}

impl Glm4MoeParser {
    /// Shared non-streaming parse, schema-aware when `tools` are provided.
    fn parse_complete_inner(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find("<tool_call>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        let parsed = self.parse_tool_calls_from_text(text, tools);

        // If no tools were successfully parsed despite having markers, return entire text as fallback
        if parsed.is_empty() {
            return Ok((text.to_string(), vec![]));
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
        // Python logic: Wait for complete tool call, then parse it all at once
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have bot_token
        let start = current_text.find(self.bot_token);
        if start.is_none() {
            self.buffer.clear();
            // If we're in the middle of streaming (current_tool_id > 0), don't return text
            let normal_text = if self.current_tool_id > 0 {
                String::new()
            } else {
                current_text.clone()
            };
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Check if we have eot_token (end of tool call)
        let end = current_text.find(self.eot_token);
        if let Some(end_pos) = end {
            // We have a complete tool call!

            // Initialize state if this is the first tool call
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            // Ensure we have enough entries in our tracking arrays
            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            // Parse the complete block using shared helper
            let block_end = end_pos + self.eot_token.len();
            let parsed_tools = self.parse_tool_calls_from_text(&current_text[..block_end], tools);

            // Extract normal text before tool calls
            let idx = current_text.find(self.bot_token);
            let normal_text = if let Some(pos) = idx {
                current_text[..pos].trim().to_string()
            } else {
                String::new()
            };

            // Build tool indices for validation
            let tool_indices = helpers::get_tool_indices(tools);

            let mut calls = Vec::new();

            if !parsed_tools.is_empty() {
                // Take the first tool and convert to ToolCallItem
                let tool_call = &parsed_tools[0];
                let tool_id = self.current_tool_id as usize;

                // Validate tool name
                if !tool_indices.contains_key(&tool_call.function.name) {
                    // Invalid tool name - skip this tool, preserve indexing for next tool
                    tracing::debug!("Invalid tool name '{}' - skipping", tool_call.function.name);
                    helpers::reset_current_tool_state(
                        &mut self.buffer,
                        &mut false, // glm45_moe/glm47_moe doesn't track name_sent per tool
                        &mut self.streamed_args_for_tool,
                        &self.prev_tool_call_arr,
                    );
                    return Ok(StreamingParseResult::default());
                }

                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: Some(tool_call.function.name.clone()),
                    parameters: tool_call.function.arguments.clone(),
                });

                // Store in tracking arrays
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }

                // Parse parameters as JSON and store
                if let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) {
                    self.prev_tool_call_arr[tool_id] = serde_json::json!({
                        "name": tool_call.function.name,
                        "arguments": args,
                    });
                }

                if self.streamed_args_for_tool.len() <= tool_id {
                    self.streamed_args_for_tool
                        .resize_with(tool_id + 1, String::new);
                }
                self.streamed_args_for_tool[tool_id].clone_from(&tool_call.function.arguments);

                self.current_tool_id += 1;
            }

            // Remove processed portion from buffer
            self.buffer = current_text[block_end..].to_string();
            return Ok(StreamingParseResult { normal_text, calls });
        }

        // No complete tool call yet - return normal text before start token
        // Safe: start.is_none() case was handled above (early return)
        let Some(start_pos) = start else {
            return Ok(StreamingParseResult::default());
        };
        let normal_text = current_text[..start_pos].to_string();
        self.buffer = current_text[start_pos..].to_string();

        Ok(StreamingParseResult {
            normal_text,
            calls: vec![],
        })
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

    fn tool(name: &str, strict: Option<bool>, parameters: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters,
                strict,
            },
        }
    }

    /// The tag follows xgrammar's built-in `glm_4_7` shape: one `<tool_call>`
    /// trigger, a `glm_xml`-styled schema per tool passed through as written
    /// (a no-argument tool's `{}` included), and `at_least_one` forcing a call.
    #[test]
    fn structural_tag_mirrors_the_xgrammar_glm_4_7_model() {
        let schema =
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let tools = vec![
            tool("get_weather", None, schema.clone()),
            tool("lookup", Some(false), schema.clone()),
            tool("ping", None, serde_json::json!({})),
            tool("", None, schema.clone()),
        ];
        let tag = Glm4MoeParser::build_structural_tag(&tools, true);
        let format = &tag["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(format["triggers"], serde_json::json!(["<tool_call>"]));
        assert_eq!(format["at_least_one"], true);
        let tags = format["tags"].as_array().unwrap();
        assert_eq!(tags.len(), 3, "a nameless tool has no tag");
        assert_eq!(tags[0]["begin"], "<tool_call>get_weather");
        assert_eq!(tags[0]["end"], "</tool_call>");
        assert_eq!(tags[0]["content"]["type"], "json_schema");
        assert_eq!(tags[0]["content"]["style"], "glm_xml");
        assert_eq!(tags[0]["content"]["json_schema"], schema);
        assert_eq!(
            tags[1]["content"]["json_schema"], schema,
            "strict is not a constraint switch here, as in the other builders"
        );
        assert_eq!(
            tags[2]["content"]["json_schema"],
            serde_json::json!({}),
            "a no-argument tool keeps its empty schema"
        );

        let tag = Glm4MoeParser::build_structural_tag(&tools, false);
        assert_eq!(tag["format"]["at_least_one"], false);
    }

    #[test]
    fn reasoning_prefix_mirrors_the_xgrammar_glm_4_7_reasoning_block() {
        let prefix = Glm4MoeParser::reasoning_prefix();
        assert_eq!(prefix["type"], "tag");
        assert_eq!(prefix["begin"], "");
        assert_eq!(prefix["end"], "</think>");
        assert_eq!(prefix["content"]["type"], "any_text");
        let excludes = prefix["content"]["excludes"].as_array().unwrap();
        for token in [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        ] {
            assert!(
                excludes.contains(&Value::String(token.to_string())),
                "{token} must not appear inside the reasoning block"
            );
        }
    }

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
}
