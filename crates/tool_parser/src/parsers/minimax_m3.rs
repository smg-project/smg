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
    wrapper_prefix_held: bool,
    wrapper_scan_pos: usize,
    current_function_name: Option<String>,
    current_parameters: Map<String, Value>,
    parameter_emitted: bool,
    active_element: Option<StreamingElement>,
    discard_invoke_body: bool,
    invoke_aborted: bool,
}

/// A parsed parameter value: either leaf text or nested child elements.
enum ParamValue {
    Text(String),
    Elements(Vec<(String, ParamValue)>),
}

/// One open parameter element in the incremental XML-like decoder.
struct ElementFrame {
    name: String,
    text: String,
    children: Vec<(String, ParamValue)>,
}

/// Stack for the top-level parameter currently being decoded. Text already
/// known not to contain a namespace marker is consumed into the stack so long
/// values are not rescanned after every chunk.
struct StreamingElement {
    stack: Vec<ElementFrame>,
}

enum ElementProgress {
    Incomplete,
    Complete(String, ParamValue),
    Malformed,
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
            wrapper_prefix_held: false,
            wrapper_scan_pos: 0,
            current_function_name: None,
            current_parameters: Map::new(),
            parameter_emitted: false,
            active_element: None,
            discard_invoke_body: false,
            invoke_aborted: false,
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

    fn is_partial_token(buffer: &str, token: &str) -> bool {
        !buffer.is_empty() && buffer.len() < token.len() && token.starts_with(buffer)
    }

    /// Begin decoding a parameter opening tag at the front of the buffer.
    /// None means the header is split across chunks.
    fn start_streaming_element(buffer: &mut String) -> Result<Option<StreamingElement>, ()> {
        let Some(after_start) = buffer.strip_prefix(ELEMENT_START) else {
            return Err(());
        };
        let Some(gt) = after_start.find('>') else {
            return Ok(None);
        };
        let name = after_start[..gt].trim();
        if name.is_empty() || name.starts_with('/') {
            return Err(());
        }

        let consumed = ELEMENT_START.len() + gt + 1;
        let name = name.to_string();
        buffer.drain(..consumed);
        Ok(Some(StreamingElement {
            stack: vec![ElementFrame {
                name,
                text: String::new(),
                children: Vec::new(),
            }],
        }))
    }

    /// Advance an active parameter without revisiting bytes consumed from prior
    /// chunks. Structural tags are interpreted only once their closing angle
    /// bracket is available, making every split within a namespace or tag safe.
    fn advance_streaming_element(
        buffer: &mut String,
        element: &mut StreamingElement,
    ) -> ElementProgress {
        loop {
            let Some(namespace_pos) = buffer.find(NAMESPACE) else {
                let held = Self::longest_partial_suffix(buffer, NAMESPACE).unwrap_or(0);
                let safe_end = buffer.len() - held;
                if safe_end > 0 {
                    let text: String = buffer.drain(..safe_end).collect();
                    if let Some(frame) = element.stack.last_mut() {
                        frame.text.push_str(&text);
                    }
                }
                return ElementProgress::Incomplete;
            };

            if namespace_pos > 0 {
                let text: String = buffer.drain(..namespace_pos).collect();
                if let Some(frame) = element.stack.last_mut() {
                    frame.text.push_str(&text);
                }
            }

            let after_namespace = &buffer[NAMESPACE.len()..];
            if after_namespace.is_empty() {
                return ElementProgress::Incomplete;
            }
            let Some(after_lt) = after_namespace.strip_prefix('<') else {
                return ElementProgress::Malformed;
            };
            let Some(gt) = after_lt.find('>') else {
                return ElementProgress::Incomplete;
            };
            let tag = after_lt[..gt].trim();
            if tag.is_empty() {
                return ElementProgress::Malformed;
            }
            let consumed = NAMESPACE.len() + 1 + gt + 1;

            if let Some(close_name) = tag.strip_prefix('/') {
                let close_name = close_name.trim();
                if element.stack.last().map(|frame| frame.name.as_str()) != Some(close_name) {
                    return ElementProgress::Malformed;
                }
                buffer.drain(..consumed);
                let Some(mut frame) = element.stack.pop() else {
                    return ElementProgress::Malformed;
                };
                let value = if frame.children.is_empty() {
                    ParamValue::Text(frame.text)
                } else {
                    if !frame.text.trim().is_empty() {
                        Self::push_mixed_text(&mut frame.children, frame.text);
                    }
                    ParamValue::Elements(frame.children)
                };

                if let Some(parent) = element.stack.last_mut() {
                    parent.children.push((frame.name, value));
                    continue;
                }
                return ElementProgress::Complete(frame.name, value);
            }

            let name = tag.to_string();
            buffer.drain(..consumed);
            element.stack.push(ElementFrame {
                name,
                text: String::new(),
                children: Vec::new(),
            });
        }
    }

    fn tool_schema<'a>(tools: &'a [Tool], name: &str) -> Option<&'a Value> {
        tools
            .iter()
            .find(|tool| tool.function.name == name)
            .map(|tool| &tool.function.parameters)
    }

    fn property_schema<'a>(schema: Option<&'a Value>, name: &str) -> Option<&'a Value> {
        schema?
            .get("properties")
            .and_then(Value::as_object)?
            .get(name)
    }

    /// Serialize one complete top-level parameter. Repeated names emit a later
    /// duplicate member with the same aggregate value as the complete parser;
    /// JSON consumers use that last member.
    fn emit_parameter(&mut self, name: String, value: Value) -> Option<ToolCallItem> {
        match self.current_parameters.get_mut(&name) {
            Some(Value::Array(values)) => {
                values.push(value);
            }
            Some(existing) => {
                let first = existing.take();
                *existing = Value::Array(vec![first, value]);
            }
            None => {
                self.current_parameters.insert(name.clone(), value);
            }
        }

        let key = serde_json::to_string(&name).ok()?;
        let value = serde_json::to_string(self.current_parameters.get(&name)?).ok()?;
        let separator = if self.parameter_emitted { "," } else { "{" };
        let fragment = format!("{separator}{key}:{value}");
        self.parameter_emitted = true;

        let tool_index = self.current_tool_id as usize;
        self.streamed_args_for_tool[tool_index].push_str(&fragment);
        Some(ToolCallItem {
            tool_index,
            name: None,
            parameters: fragment,
        })
    }

    fn finish_streaming_invoke(&mut self) -> ToolCallItem {
        let tool_index = self.current_tool_id as usize;
        let fragment = if self.parameter_emitted { "}" } else { "{}" }.to_string();
        self.streamed_args_for_tool[tool_index].push_str(&fragment);
        self.prev_tool_call_arr[tool_index] = serde_json::json!({
            "name": self.current_function_name.clone(),
            "arguments": self.current_parameters.clone(),
        });
        self.abandon_streaming_invoke();
        ToolCallItem {
            tool_index,
            name: None,
            parameters: fragment,
        }
    }

    fn abandon_streaming_invoke(&mut self) {
        self.current_function_name = None;
        self.current_parameters.clear();
        self.parameter_emitted = false;
        self.active_element = None;
        self.discard_invoke_body = false;
        self.invoke_aborted = false;
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
                    self.buffer.drain(..start);
                    self.in_tool_call = true;
                    self.wrapper_prefix_held = true;
                    self.wrapper_scan_pos = TOOL_CALL_START.len();
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

            // Once an invoke header is known, consume each top-level parameter
            // incrementally and emit it as soon as the whole value is valid.
            if self.current_function_name.is_some() {
                if self.active_element.is_some() {
                    let progress = match self.active_element.as_mut() {
                        Some(element) => Self::advance_streaming_element(&mut self.buffer, element),
                        None => ElementProgress::Incomplete,
                    };
                    match progress {
                        ElementProgress::Incomplete => break,
                        ElementProgress::Malformed => {
                            // Previously emitted bytes cannot be retracted. Leave
                            // the JSON object open so this call is not executable.
                            self.active_element = None;
                            self.discard_invoke_body = true;
                            self.invoke_aborted = true;
                            continue;
                        }
                        ElementProgress::Complete(name, value) => {
                            self.active_element = None;
                            let schema = Self::tool_schema(
                                tools,
                                self.current_function_name.as_deref().unwrap_or_default(),
                            );
                            let value =
                                Self::value_to_json(value, Self::property_schema(schema, &name));
                            if let Some(call) = self.emit_parameter(name, value) {
                                calls.push(call);
                            }
                            continue;
                        }
                    }
                }

                if self.discard_invoke_body {
                    let invoke_end = self.buffer.find(INVOKE_END);
                    let wrapper_end = self.buffer.find(TOOL_CALL_END);
                    match (invoke_end, wrapper_end) {
                        (Some(invoke), Some(wrapper)) if wrapper < invoke => {
                            self.buffer.drain(..wrapper + TOOL_CALL_END.len());
                            self.abandon_streaming_invoke();
                            self.in_tool_call = false;
                            self.wrapper_prefix_held = false;
                            self.wrapper_scan_pos = 0;
                        }
                        (Some(invoke), _) => {
                            self.buffer.drain(..invoke + INVOKE_END.len());
                            if self.invoke_aborted {
                                self.abandon_streaming_invoke();
                            } else {
                                calls.push(self.finish_streaming_invoke());
                            }
                        }
                        (None, Some(wrapper)) => {
                            self.buffer.drain(..wrapper + TOOL_CALL_END.len());
                            self.abandon_streaming_invoke();
                            self.in_tool_call = false;
                            self.wrapper_prefix_held = false;
                            self.wrapper_scan_pos = 0;
                        }
                        (None, None) => {
                            let invoke_held =
                                Self::longest_partial_suffix(&self.buffer, INVOKE_END).unwrap_or(0);
                            let wrapper_held =
                                Self::longest_partial_suffix(&self.buffer, TOOL_CALL_END)
                                    .unwrap_or(0);
                            let held = invoke_held.max(wrapper_held);
                            let discard = self.buffer.len() - held;
                            self.buffer.drain(..discard);
                            break;
                        }
                    }
                    continue;
                }

                let whitespace = self.buffer.len() - self.buffer.trim_start().len();
                let candidate = &self.buffer[whitespace..];
                if candidate.starts_with(INVOKE_END) {
                    self.buffer.drain(..whitespace + INVOKE_END.len());
                    calls.push(self.finish_streaming_invoke());
                    continue;
                }
                if candidate.starts_with(TOOL_CALL_END) {
                    self.buffer.drain(..whitespace + TOOL_CALL_END.len());
                    self.abandon_streaming_invoke();
                    self.in_tool_call = false;
                    self.wrapper_prefix_held = false;
                    self.wrapper_scan_pos = 0;
                    continue;
                }
                if candidate.is_empty()
                    || Self::is_partial_token(candidate, INVOKE_END)
                    || Self::is_partial_token(candidate, ELEMENT_START)
                    || Self::is_partial_token(candidate, TOOL_CALL_END)
                {
                    break;
                }
                if candidate.starts_with(ELEMENT_START) {
                    self.buffer.drain(..whitespace);
                    match Self::start_streaming_element(&mut self.buffer) {
                        Ok(Some(element)) => {
                            self.active_element = Some(element);
                            continue;
                        }
                        Ok(None) => break,
                        Err(()) => {
                            self.discard_invoke_body = true;
                            self.invoke_aborted = true;
                            continue;
                        }
                    }
                }

                // Ordinary text at a parameter boundary terminates argument
                // parsing on the complete path. Consume through the invoke end.
                self.discard_invoke_body = true;
                continue;
            }

            // Retain the wrapper prefix until the first valid header is known.
            // This preserves the existing all-text fallback for a malformed
            // complete wrapper while still avoiding repeated scans.
            if self.wrapper_prefix_held {
                let search = &self.buffer[self.wrapper_scan_pos..];
                let invoke = search
                    .find(INVOKE_START)
                    .map(|position| self.wrapper_scan_pos + position);
                let wrapper = search
                    .find(TOOL_CALL_END)
                    .map(|position| self.wrapper_scan_pos + position);

                if matches!((invoke, wrapper), (None, Some(_)))
                    || matches!((invoke, wrapper), (Some(a), Some(b)) if b < a)
                {
                    let end = wrapper.unwrap_or_default() + TOOL_CALL_END.len();
                    normal_text.push_str(&self.buffer[..end]);
                    self.buffer.drain(..end);
                    self.in_tool_call = false;
                    self.wrapper_prefix_held = false;
                    self.wrapper_scan_pos = 0;
                    continue;
                }

                let Some(invoke_pos) = invoke else {
                    let invoke_held =
                        Self::longest_partial_suffix(search, INVOKE_START).unwrap_or(0);
                    let wrapper_held =
                        Self::longest_partial_suffix(search, TOOL_CALL_END).unwrap_or(0);
                    self.wrapper_scan_pos = self.buffer.len() - invoke_held.max(wrapper_held);
                    break;
                };
                let after_invoke = &self.buffer[invoke_pos + INVOKE_START.len()..];
                let Some(gt) = after_invoke.find('>') else {
                    self.wrapper_scan_pos = invoke_pos;
                    break;
                };
                let malformed_header = after_invoke
                    .find(NAMESPACE)
                    .is_some_and(|namespace| namespace < gt);
                let name = if malformed_header {
                    None
                } else {
                    Self::parse_invoke_name(&after_invoke[..gt])
                };
                let Some(name) = name else {
                    let tail = &self.buffer[invoke_pos..];
                    let invoke_end = tail.find(INVOKE_END).map(|position| invoke_pos + position);
                    let wrapper_end = tail
                        .find(TOOL_CALL_END)
                        .map(|position| invoke_pos + position);
                    match (invoke_end, wrapper_end) {
                        (Some(invoke), Some(wrapper)) if wrapper < invoke => {
                            let end = wrapper + TOOL_CALL_END.len();
                            normal_text.push_str(&self.buffer[..end]);
                            self.buffer.drain(..end);
                            self.in_tool_call = false;
                            self.wrapper_prefix_held = false;
                            self.wrapper_scan_pos = 0;
                        }
                        (Some(invoke), _) => {
                            self.wrapper_scan_pos = invoke + INVOKE_END.len();
                        }
                        (None, Some(wrapper)) => {
                            let end = wrapper + TOOL_CALL_END.len();
                            normal_text.push_str(&self.buffer[..end]);
                            self.buffer.drain(..end);
                            self.in_tool_call = false;
                            self.wrapper_prefix_held = false;
                            self.wrapper_scan_pos = 0;
                        }
                        (None, None) => {
                            self.wrapper_scan_pos = invoke_pos;
                            break;
                        }
                    }
                    continue;
                };

                let header_end = invoke_pos + INVOKE_START.len() + gt + 1;
                self.buffer.drain(..header_end);
                self.wrapper_prefix_held = false;
                self.wrapper_scan_pos = 0;
                if self.current_tool_id == -1 {
                    self.current_tool_id = 0;
                } else {
                    self.current_tool_id += 1;
                }
                helpers::ensure_capacity(
                    self.current_tool_id,
                    &mut self.prev_tool_call_arr,
                    &mut self.streamed_args_for_tool,
                );
                self.current_function_name = Some(name.clone());
                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(name),
                    parameters: String::new(),
                });
                continue;
            }

            // Between announced invokes, skip separators and retain only a
            // possible partial next-invoke or wrapper-close marker.
            let whitespace = self.buffer.len() - self.buffer.trim_start().len();
            let candidate = &self.buffer[whitespace..];
            if candidate.starts_with(TOOL_CALL_END) {
                self.buffer.drain(..whitespace + TOOL_CALL_END.len());
                self.in_tool_call = false;
                self.wrapper_scan_pos = 0;
                continue;
            }
            if candidate.is_empty()
                || Self::is_partial_token(candidate, INVOKE_START)
                || Self::is_partial_token(candidate, TOOL_CALL_END)
            {
                break;
            }

            if let Some(after_invoke) = candidate.strip_prefix(INVOKE_START) {
                let Some(gt) = after_invoke.find('>') else {
                    break;
                };
                let malformed_header = after_invoke
                    .find(NAMESPACE)
                    .is_some_and(|namespace| namespace < gt);
                let name = if malformed_header {
                    None
                } else {
                    Self::parse_invoke_name(&after_invoke[..gt])
                };
                let Some(name) = name else {
                    let invoke_end = candidate.find(INVOKE_END);
                    let wrapper_end = candidate.find(TOOL_CALL_END);
                    match (invoke_end, wrapper_end) {
                        (Some(invoke), Some(wrapper)) if wrapper < invoke => {
                            self.buffer
                                .drain(..whitespace + wrapper + TOOL_CALL_END.len());
                            self.in_tool_call = false;
                        }
                        (Some(invoke), _) => {
                            self.buffer.drain(..whitespace + invoke + INVOKE_END.len());
                        }
                        (None, Some(wrapper)) => {
                            self.buffer
                                .drain(..whitespace + wrapper + TOOL_CALL_END.len());
                            self.in_tool_call = false;
                        }
                        (None, None) => break,
                    }
                    continue;
                };

                let header_end = whitespace + INVOKE_START.len() + gt + 1;
                self.buffer.drain(..header_end);
                if self.current_tool_id == -1 {
                    self.current_tool_id = 0;
                } else {
                    self.current_tool_id += 1;
                }
                helpers::ensure_capacity(
                    self.current_tool_id,
                    &mut self.prev_tool_call_arr,
                    &mut self.streamed_args_for_tool,
                );
                self.current_function_name = Some(name.clone());
                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: Some(name),
                    parameters: String::new(),
                });
                continue;
            }

            let next_invoke = candidate.find(INVOKE_START);
            let next_end = candidate.find(TOOL_CALL_END);
            if let Some(next) = match (next_invoke, next_end) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            } {
                self.buffer.drain(..whitespace + next);
                continue;
            }
            let invoke_held = Self::longest_partial_suffix(candidate, INVOKE_START).unwrap_or(0);
            let end_held = Self::longest_partial_suffix(candidate, TOOL_CALL_END).unwrap_or(0);
            let held = invoke_held.max(end_held);
            let discard = self.buffer.len() - held;
            self.buffer.drain(..discard);
            break;
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
        self.wrapper_prefix_held = false;
        self.wrapper_scan_pos = 0;
        self.abandon_streaming_invoke();
    }
}
