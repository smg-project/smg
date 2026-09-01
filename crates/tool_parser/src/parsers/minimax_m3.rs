use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::{Map, Value};

use crate::{
    errors::ParserResult,
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Namespace marker MiniMax M3 prepends before each structural tag.
const NAMESPACE: &str = "]<]minimax[>[";
/// Opening marker for a tool-call block.
const TOOL_CALL_START: &str = "]<]minimax[>[<tool_call>";
/// Closing marker for a tool-call block.
const TOOL_CALL_END: &str = "]<]minimax[>[</tool_call>";
/// Opening marker for an `<invoke ...>` element (attributes follow before `>`).
const INVOKE_START: &str = "]<]minimax[>[<invoke";
/// Closing marker for an `<invoke>` element.
const INVOKE_END: &str = "]<]minimax[>[</invoke>";
/// Opening marker prefix for a parameter element (`<name>`).
const ELEMENT_START: &str = "]<]minimax[>[<";
/// Opening marker prefix for a parameter closing tag (`</name>`).
const ELEMENT_END_START: &str = "]<]minimax[>[</";
/// Reserved field name used to preserve mixed text within an object element.
const MIXED_TEXT_FIELD: &str = "$text";

/// MiniMax M3 format parser for tool calls.
///
/// Handles the MiniMax M3 specific framing, where the namespace marker
/// `]<]minimax[>[` is prepended before every structural tag:
///
/// ```text
/// ]<]minimax[>[<tool_call>
/// ]<]minimax[>[<invoke name="func">
/// ]<]minimax[>[<key>value]<]minimax[>[</key>
/// ]<]minimax[>[</invoke>
/// ]<]minimax[>[</tool_call>
/// ```
///
/// Differences from MiniMax M2:
/// - Each structural tag is prefixed with the namespace marker `]<]minimax[>[`.
/// - The start token is `]<]minimax[>[<tool_call>` (not `<minimax:tool_call>`).
/// - A single tool-call block may contain multiple `<invoke>` tags.
/// - Parameters are expressed with parameter-name XML tags and may nest
///   recursively to form objects and arrays.
///
/// Reference: vLLM `MinimaxM3ToolParser` (`tool_call_start_token =
/// "]<]minimax[>[<tool_call>"`).
pub struct MinimaxM3Parser {
    // Streaming state
    buffer: String,
    prev_tool_call_arr: Vec<Value>,
    current_tool_id: i32,
    streamed_args_for_tool: Vec<String>,
    in_tool_call: bool,
}

/// A parsed parameter value: either leaf text or nested child elements.
enum ParamValue {
    Text(String),
    Elements(Vec<(String, ParamValue)>),
}

impl MinimaxM3Parser {
    /// Create a new MiniMax M3 parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            in_tool_call: false,
        }
    }

    /// Parse a leaf value from text, coercing by declared schema type when known,
    /// otherwise inferring the JSON type (number/bool/null) and defaulting to string.
    fn coerce_leaf(text: &str, declared_type: Option<&str>) -> Value {
        // An empty container renders as an element with no children and no
        // text, which reaches the leaf path indistinguishable from an empty
        // string. `coerce_by_schema_type` cannot parse "" as JSON, so without
        // this the value would be inferred as `""` and fail the tool's schema.
        if text.trim().is_empty() {
            match declared_type {
                Some("array") => return Value::Array(Vec::new()),
                Some("object") => return Value::Object(Map::new()),
                _ => {}
            }
        }

        if let Some(value) = helpers::coerce_by_schema_type(text, declared_type) {
            return value;
        }
        Self::infer_value(text)
    }

    /// Infer a JSON value from a raw text leaf (no schema available).
    fn infer_value(text: &str) -> Value {
        match text {
            "true" | "True" => return Value::Bool(true),
            "false" | "False" => return Value::Bool(false),
            "null" | "None" => return Value::Null,
            _ => {}
        }

        if let Ok(num) = text.parse::<i64>() {
            return Value::Number(num.into());
        }
        if let Ok(num) = text.parse::<f64>() {
            if let Some(n) = serde_json::Number::from_f64(num) {
                return Value::Number(n);
            }
        }

        Value::String(text.to_string())
    }

    /// Length of the longest suffix of `buffer` that is a proper prefix of `token`.
    ///
    /// Unlike [`helpers::ends_with_partial_token`] (which returns the shortest such
    /// match), this prefers the longest match. The M3 start token contains repeated
    /// `]` characters, so a shortest-match would mis-align and leak the marker's
    /// leading bytes as normal text.
    fn longest_partial_suffix(buffer: &str, token: &str) -> Option<usize> {
        if buffer.is_empty() || token.is_empty() {
            return None;
        }
        token
            .char_indices()
            .skip(1)
            .map(|(i, _)| i)
            .filter(|&i| buffer.ends_with(&token[..i]))
            .max()
    }

    /// Decode common XML entities.
    fn decode_xml_entities(text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&")
    }

    /// Extract the `name="..."` attribute value from an invoke header (the text
    /// between `]<]minimax[>[<invoke` and the closing `>`).
    fn parse_invoke_name(header: &str) -> Option<String> {
        let idx = header.find("name")?;
        let after = &header[idx + "name".len()..];
        let after = after.trim_start();
        let after = after.strip_prefix('=')?.trim_start();
        if let Some(rest) = after.strip_prefix('"') {
            let end = rest.find('"')?;
            Some(rest[..end].trim().to_string())
        } else if let Some(rest) = after.strip_prefix('\'') {
            let end = rest.find('\'')?;
            Some(rest[..end].trim().to_string())
        } else {
            let end = after
                .find(|c: char| c.is_whitespace())
                .unwrap_or(after.len());
            let name = after[..end].trim();
            if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            }
        }
    }

    /// Parse the body of one element (recursively), returning its value and the
    /// number of bytes consumed up to and including the matching close tag.
    /// `name` is the element name whose close tag terminates this body.
    fn parse_element_body(input: &str, name: &str) -> Option<(ParamValue, usize)> {
        let close_tag = format!("{ELEMENT_END_START}{name}>");
        let mut pos = 0;
        let mut text = String::new();
        let mut children: Vec<(String, ParamValue)> = Vec::new();

        loop {
            // Accumulate any text up to the next namespace marker; a body
            // with no further marker is an unterminated element.
            let rest = &input[pos..];
            let text_chunk = &rest[..rest.find(NAMESPACE)?];
            text.push_str(text_chunk);
            pos += text_chunk.len();

            let rest = &input[pos..];
            if rest.starts_with(&close_tag) {
                pos += close_tag.len();
                break;
            }
            if rest.starts_with(ELEMENT_START) {
                // Child element.
                let (child_name, child_value, consumed) = Self::parse_element(rest)?;
                children.push((child_name, child_value));
                pos += consumed;
                continue;
            }
            // A namespace marker that is neither the expected close nor a child
            // start: malformed body.
            return None;
        }

        let value = if children.is_empty() {
            ParamValue::Text(text)
        } else {
            if !text.trim().is_empty() {
                Self::push_mixed_text(&mut children, text);
            }
            ParamValue::Elements(children)
        };
        Some((value, pos))
    }

    /// Parse a complete element starting at `input` (which must begin with
    /// `]<]minimax[>[<name>`). Returns `(name, value, consumed_bytes)`.
    fn parse_element(input: &str) -> Option<(String, ParamValue, usize)> {
        let after_start = input.strip_prefix(ELEMENT_START)?;
        let gt = after_start.find('>')?;
        let name = after_start[..gt].trim().to_string();
        if name.is_empty() || name.starts_with('/') {
            return None;
        }
        let body_start = ELEMENT_START.len() + gt + 1;
        let (value, body_consumed) = Self::parse_element_body(&input[body_start..], &name)?;
        Some((name, value, body_start + body_consumed))
    }

    /// Preserve mixed text content under a reserved object field, avoiding a
    /// collision with an existing child name by prefixing `$`.
    fn push_mixed_text(children: &mut Vec<(String, ParamValue)>, text: String) {
        let mut field = MIXED_TEXT_FIELD.to_string();
        while children.iter().any(|(name, _)| *name == field) {
            field.insert(0, '$');
        }
        children.push((field, ParamValue::Text(text)));
    }

    /// The scalar `type` declared by a schema node, when it has one.
    fn schema_type(schema: Option<&Value>) -> Option<&str> {
        schema?.get("type")?.as_str()
    }

    /// Convert a parsed parameter value into a JSON value, coercing leaves by
    /// the schema node they sit under: array elements descend into `items`,
    /// object members into their `properties` entry.
    fn value_to_json(value: ParamValue, schema: Option<&Value>) -> Value {
        match value {
            ParamValue::Text(text) => {
                let decoded = Self::decode_xml_entities(&text);
                Self::coerce_leaf(&decoded, Self::schema_type(schema))
            }
            ParamValue::Elements(children) => {
                // Repeated child names under an `array` schema (or with duplicate
                // keys) collapse to an array; otherwise build an object.
                if Self::schema_type(schema) == Some("array") {
                    let items_schema = schema.and_then(|s| s.get("items"));
                    let items = children
                        .into_iter()
                        .map(|(_, v)| Self::value_to_json(v, items_schema))
                        .collect();
                    return Value::Array(items);
                }

                let properties = schema
                    .and_then(|s| s.get("properties"))
                    .and_then(Value::as_object);
                let mut map: Map<String, Value> = Map::new();
                for (name, child) in children {
                    let child_schema = properties.and_then(|p| p.get(&name));
                    let child_json = Self::value_to_json(child, child_schema);
                    match map.get_mut(&name) {
                        Some(Value::Array(arr)) => arr.push(child_json),
                        Some(existing) => {
                            let prev = existing.take();
                            *existing = Value::Array(vec![prev, child_json]);
                        }
                        None => {
                            map.insert(name, child_json);
                        }
                    }
                }
                Value::Object(map)
            }
        }
    }

    /// Parse all parameter elements inside a complete invoke body into a JSON
    /// object. `params_schema` is the function's `parameters` JSON schema.
    /// `None` when an element starts but fails to parse: emitting the
    /// arguments collected so far would run the tool with silently missing
    /// parameters.
    fn parse_invoke_params(body: &str, params_schema: Option<&Value>) -> Option<Value> {
        let properties = params_schema
            .and_then(|s| s.get("properties"))
            .and_then(Value::as_object);
        let mut map: Map<String, Value> = Map::new();
        let mut pos = 0;

        loop {
            let rest = &body[pos..];
            let trimmed = rest.trim_start();
            let trim_len = rest.len() - trimmed.len();
            if trimmed.is_empty() {
                break;
            }
            if !trimmed.starts_with(ELEMENT_START) {
                // Tolerant: ordinary text at a parameter boundary ends the params.
                break;
            }
            pos += trim_len;
            let (name, value, consumed) = Self::parse_element(&body[pos..])?;
            pos += consumed;
            let json = Self::value_to_json(value, properties.and_then(|p| p.get(&name)));
            match map.get_mut(&name) {
                Some(Value::Array(arr)) => arr.push(json),
                Some(existing) => {
                    let prev = existing.take();
                    *existing = Value::Array(vec![prev, json]);
                }
                None => {
                    map.insert(name, json);
                }
            }
        }

        Some(Value::Object(map))
    }

    /// Parse a single invoke block (the text between `]<]minimax[>[<invoke` and
    /// `]<]minimax[>[</invoke>`) into a tool call.
    fn parse_invoke(block: &str, tools: &[Tool]) -> Option<ToolCall> {
        // `block` begins at the namespace marker for `<invoke`.
        let after_invoke = block.strip_prefix(INVOKE_START)?;
        let gt = after_invoke.find('>')?;
        let header = &after_invoke[..gt];
        let name = Self::parse_invoke_name(header)?;
        let body = &after_invoke[gt + 1..];

        let params_schema = tools
            .iter()
            .find(|t| t.function.name == name)
            .map(|t| &t.function.parameters);
        let arguments = Self::parse_invoke_params(body, params_schema)?;
        let arguments_str = serde_json::to_string(&arguments).ok()?;

        Some(ToolCall {
            function: FunctionCall {
                name,
                arguments: arguments_str,
            },
        })
    }

    /// Parse all complete tool-call blocks in `text`.
    /// Returns the tool calls and the byte position of the first tool-call block.
    fn parse_tool_calls(text: &str, tools: &[Tool]) -> (Vec<ToolCall>, Option<usize>) {
        let mut calls = Vec::new();
        let mut first_pos = None;
        let mut search_from = 0;

        while let Some(rel_start) = text[search_from..].find(TOOL_CALL_START) {
            let block_start = search_from + rel_start;
            let inner_start = block_start + TOOL_CALL_START.len();
            let Some(rel_end) = text[inner_start..].find(TOOL_CALL_END) else {
                break;
            };
            let inner_end = inner_start + rel_end;
            let inner = &text[inner_start..inner_end];

            // Extract each invoke block within the tool-call wrapper.
            let mut invoke_from = 0;
            while let Some(rel_inv) = inner[invoke_from..].find(INVOKE_START) {
                let inv_start = invoke_from + rel_inv;
                let Some(rel_inv_end) = inner[inv_start..].find(INVOKE_END) else {
                    break;
                };
                let inv_end = inv_start + rel_inv_end;
                let inv_block = &inner[inv_start..inv_end];
                if let Some(call) = Self::parse_invoke(inv_block, tools) {
                    if first_pos.is_none() {
                        first_pos = Some(block_start);
                    }
                    calls.push(call);
                }
                invoke_from = inv_end + INVOKE_END.len();
            }

            search_from = inner_end + TOOL_CALL_END.len();
        }

        (calls, first_pos)
    }

    /// Shared non-streaming parse. `tools` empty means infer types from text.
    fn parse_complete_inner(text: &str, tools: &[Tool]) -> (String, Vec<ToolCall>) {
        if !text.contains(TOOL_CALL_START) {
            return (text.to_string(), vec![]);
        }
        let (calls, first_pos) = Self::parse_tool_calls(text, tools);
        if calls.is_empty() {
            return (text.to_string(), vec![]);
        }
        let normal_text = match first_pos {
            Some(pos) => text[..pos].to_string(),
            None => text.to_string(),
        };
        (normal_text, calls)
    }
}

impl Default for MinimaxM3Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MinimaxM3Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_complete_inner(text, &[]))
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        Ok(Self::parse_complete_inner(text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let mut normal_text = String::new();
        let mut calls = Vec::new();

        loop {
            // Outside a tool call: emit normal text until a start token appears.
            if !self.in_tool_call {
                if let Some(start) = self.buffer.find(TOOL_CALL_START) {
                    normal_text.push_str(&self.buffer[..start]);
                    self.buffer = self.buffer[start..].to_string();
                    self.in_tool_call = true;
                    continue;
                }

                // No start token: flush text, holding back a potential partial token.
                if let Some(partial_len) =
                    Self::longest_partial_suffix(&self.buffer, TOOL_CALL_START)
                {
                    let end = self.buffer.len() - partial_len;
                    normal_text.push_str(&self.buffer[..end]);
                    self.buffer = self.buffer[end..].to_string();
                } else {
                    normal_text.push_str(&self.buffer);
                    self.buffer.clear();
                }
                break;
            }

            // Inside a tool call: wait for the complete end token before emitting.
            let Some(end_rel) = self.buffer.find(TOOL_CALL_END) else {
                break;
            };
            let block_end = end_rel + TOOL_CALL_END.len();
            let block = self.buffer[..block_end].to_string();
            self.buffer = self.buffer[block_end..].to_string();
            self.in_tool_call = false;

            let (block_calls, _) = Self::parse_tool_calls(&block, tools);
            if block_calls.is_empty() {
                // No invoke parsed: match the non-streaming path, which
                // returns malformed blocks as text rather than dropping them.
                normal_text.push_str(&block);
                continue;
            }
            for call in block_calls {
                if self.current_tool_id == -1 {
                    self.current_tool_id = 0;
                } else {
                    self.current_tool_id += 1;
                }
                let tool_id = self.current_tool_id as usize;
                helpers::ensure_capacity(
                    self.current_tool_id,
                    &mut self.prev_tool_call_arr,
                    &mut self.streamed_args_for_tool,
                );

                let args = call.function.arguments.clone();
                if tool_id < self.streamed_args_for_tool.len() {
                    self.streamed_args_for_tool[tool_id].clone_from(&args);
                }
                let parsed_args: Value =
                    serde_json::from_str(&args).unwrap_or_else(|_| Value::Object(Map::new()));
                if tool_id < self.prev_tool_call_arr.len() {
                    self.prev_tool_call_arr[tool_id] = serde_json::json!({
                        "name": call.function.name,
                        "arguments": parsed_args,
                    });
                }

                // Emit name then full arguments for this completed invoke.
                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: Some(call.function.name),
                    parameters: String::new(),
                });
                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: None,
                    parameters: args,
                });
            }
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(TOOL_CALL_START)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
        self.in_tool_call = false;
    }
}
