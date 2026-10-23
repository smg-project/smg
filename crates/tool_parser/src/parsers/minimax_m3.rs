use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};

use crate::{
    errors::ParserResult,
    factory::GrammarStart,
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
/// Opening marker of the model's thought.
const THINK_START: &str = "<mm:think>";
/// Closing marker of the model's thought.
const THINK_END: &str = "</mm:think>";
/// The element tag of a list's items, as the model writes an array.
const ITEM: &str = "item";
/// The markers that end a call wherever they stand; no argument value may hold one.
const CALL_EXITS: [&str; 3] = ["</invoke>", "<invoke name=\"", "</tool_call>"];
/// How deep the structural tag follows a schema's nesting; below that, a value is any text.
const DEPTH: usize = 16;
/// How many elements one invoke's grammar may hold; past that, the values still to write are
/// free text. An acyclic schema unfolds as k^d elements for k properties over d levels, which the
/// depth cap alone does not bound; ordinary tool schemas stay far below this. The budget is each
/// invoke's own: a tag for many tools holds up to this many elements per tool. The budget is spent
/// depth first, in the schema's order, but an object's later properties each keep one element
/// while an earlier one is descended into, so what runs short loses its structure from the inside
/// out: the nested objects first, the top-level arguments last. The budget counts elements, not
/// bytes: a value's own spelling (an enum's words, a number's schema) is counted by
/// [`LEAF_BYTES`], and a property typed `null`, written as nothing, costs none.
const NODES: usize = 1024;
/// How many bytes the leaves of one invoke may spell: an enum's words, a number's schema with the
/// definitions its pointers reach. Past that, a string with an enum is text as it is and a number
/// is free text, so a value that is large, or reused through references, cannot multiply the
/// grammar: a 5,000-word enum reached by a thousand number leaves is written while it fits, not a
/// thousand times. The budget is each invoke's own, as [`NODES`] is, and is spent in the schema's
/// order.
const LEAF_BYTES: usize = 256 * 1024;
/// How many bytes the walks behind one invoke's number leaves may size, the refused walks counted
/// too. A walk stops at what the leaves may still spell, but a leaf that is refused spends none of
/// that, so a thousand distinct leaves at a definition just past the budget would each walk it to
/// the end; past this allowance a number leaf with pointers is free text without a walk. A leaf
/// without pointers walks nothing but itself, which the request already holds, and the allowance
/// neither bounds nor charges it.
const WALK_BYTES: usize = 4 * LEAF_BYTES;
/// What an enum's word costs beyond its length: the alternative that spells it.
const WORD_FRAMING: usize = 36;

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
/// Reference: the serving engine's `MinimaxM3ToolParser` (`tool_call_start_token =
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

    /// Whether a buffered tool-call wrapper can still become a valid invoke.
    ///
    /// Whitespace is allowed between the wrapper and the invoke marker. Once
    /// the first non-whitespace bytes diverge from that marker, later input
    /// cannot turn the candidate into a tool call.
    fn could_start_invoke(buffer: &str) -> bool {
        let Some(after_wrapper) = buffer.strip_prefix(TOOL_CALL_START) else {
            return false;
        };
        let candidate = after_wrapper.trim_start();
        if candidate.is_empty() || INVOKE_START.starts_with(candidate) {
            return true;
        }

        let Some(after_invoke) = candidate.strip_prefix(INVOKE_START) else {
            return false;
        };
        after_invoke.is_empty()
            || after_invoke
                .chars()
                .next()
                .is_some_and(|c| c.is_whitespace() || c == '>')
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
    /// An empty schema-declared container may omit its close immediately before
    /// its parent closes; in that case the parent marker remains unconsumed.
    fn parse_element_body(
        input: &str,
        name: &str,
        schema: Option<&Value>,
        parent_name: Option<&str>,
    ) -> Option<(ParamValue, usize)> {
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
            if text.trim().is_empty()
                && children.is_empty()
                && matches!(Self::schema_type(schema), Some("array" | "object"))
                && parent_name.is_some_and(|parent| {
                    rest.starts_with(&format!("{ELEMENT_END_START}{parent}>"))
                })
            {
                // The parent consumes its own close; the empty child's close
                // was omitted by the model.
                break;
            }
            if rest.starts_with(ELEMENT_START) {
                // Child element.
                let (child_name, child_value, consumed) =
                    Self::parse_element(rest, schema, Some(name))?;
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
    fn parse_element(
        input: &str,
        parent_schema: Option<&Value>,
        parent_name: Option<&str>,
    ) -> Option<(String, ParamValue, usize)> {
        let after_start = input.strip_prefix(ELEMENT_START)?;
        let gt = after_start.find('>')?;
        let name = after_start[..gt].trim().to_string();
        if name.is_empty() || name.starts_with('/') {
            return None;
        }
        let body_start = ELEMENT_START.len() + gt + 1;
        let parent_schema = Self::container_schema(parent_schema).or(parent_schema);
        let schema = if Self::schema_type(parent_schema) == Some("array") {
            parent_schema.and_then(|schema| schema.get("items"))
        } else {
            Self::property_schema(parent_schema, &name)
        };
        let schema = Self::container_schema(schema).or(schema);
        let (value, body_consumed) =
            Self::parse_element_body(&input[body_start..], &name, schema, parent_name)?;
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

    /// Resolve a direct container or a transparent composition wrapper.
    /// Nullable unions must have exactly one container branch; scalar/unknown
    /// alternatives are ambiguous. Only singleton allOf wrappers are unwrapped:
    /// choosing one of several conjuncts would discard the others' constraints.
    fn container_schema(schema: Option<&Value>) -> Option<&Value> {
        let schema = schema?;
        if schema.get("type").is_some() {
            return matches!(Self::schema_type(Some(schema)), Some("array" | "object"))
                .then_some(schema);
        }

        let mut compositions = ["oneOf", "anyOf", "allOf"]
            .into_iter()
            .filter_map(|key| schema.get(key).map(|branches| (key, branches)));
        let (keyword, branches) = compositions.next()?;
        if compositions.next().is_some() {
            return None;
        }
        // Unwrapping must not drop sibling validation or structural keywords.
        if schema.as_object()?.keys().any(|key| {
            key != keyword
                && !matches!(
                    key.as_str(),
                    "title"
                        | "description"
                        | "default"
                        | "examples"
                        | "deprecated"
                        | "readOnly"
                        | "writeOnly"
                        | "$comment"
                )
        }) {
            return None;
        }
        let branches = branches.as_array()?;
        if keyword == "allOf" && branches.len() != 1 {
            return None;
        }
        let mut container = None;
        for branch in branches {
            if keyword != "allOf" && Self::schema_type(Some(branch)) == Some("null") {
                continue;
            }
            let resolved = Self::container_schema(Some(branch))?;
            if container.replace(resolved).is_some() {
                return None;
            }
        }
        container
    }

    /// Find a named property in this schema or a nested composition branch.
    fn property_schema<'a>(schema: Option<&'a Value>, name: &str) -> Option<&'a Value> {
        let schema = schema?;
        if let Some(property) = schema.get("properties").and_then(|value| value.get(name)) {
            return Some(property);
        }

        for keyword in ["oneOf", "anyOf", "allOf"] {
            for branch in schema
                .get(keyword)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(property) = Self::property_schema(Some(branch), name) {
                    return Some(property);
                }
            }
        }

        None
    }

    /// Convert a parsed parameter value into a JSON value, coercing leaves by
    /// the schema node they sit under: array elements descend into `items`,
    /// object members into their `properties` entry.
    fn value_to_json(value: ParamValue, schema: Option<&Value>) -> Value {
        let schema = Self::container_schema(schema).or(schema);
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

                let mut map: Map<String, Value> = Map::new();
                for (name, child) in children {
                    let child_schema = Self::property_schema(schema, &name);
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
            let (name, value, consumed) = Self::parse_element(&body[pos..], params_schema, None)?;
            pos += consumed;
            let json = Self::value_to_json(value, Self::property_schema(params_schema, &name));
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

    /// Constrain the forced calls in the model's own syntax: one block,
    /// `]<]minimax[>[<tool_call>` and a newline, one invoke or more, the
    /// separator and `</tool_call>`; an invoke is the separator,
    /// `<invoke name="NAME">`, one element per schema property in the
    /// schema's order (optional unless the schema requires it), the
    /// separator, `</invoke>` and a newline; an element is the separator and
    /// `<key>`, the value, the separator and `</key>`. A value follows the
    /// type its schema pins: a string as text free of the separator, a string
    /// `enum` as one of its values, a number as the JSON its schema accepts, a
    /// boolean as one of the words the parser reads, an object as nested
    /// elements, an array as `<item>` elements, `null` as nothing (the
    /// template leaves such an argument out), anything untyped as free text
    /// up to the closing tag. A nameless tool is skipped; with no named tool
    /// the block is empty, which parses to no calls. The separator stays in
    /// the trigger: this parser reads the block only with it.
    ///
    /// The forced block is written at once, so the tag opens with what the
    /// model writes before a call when the template leaves the thought to it
    /// (its default): the thought block, or the bare `</mm:think>` the model
    /// writes for an empty one, or nothing. A prompt that ends inside the
    /// thought gets the registry's [`Self::reasoning_prefix`] in front as
    /// well (see `ParserRegistry::generate_tool_constraint`).
    pub fn build_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let invokes: Vec<Value> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| Elements::of(&tool.function.parameters).invoke(&tool.function.name))
            .collect();
        // An empty list of invokes is unsatisfiable; with no named tool, admit
        // only an empty block.
        let invokes = if invokes.is_empty() {
            json!({ "type": "const_string", "value": "" })
        } else {
            json!({
                "type": "tags_with_separator",
                "tags": invokes,
                "separator": "",
                "at_least_one": true,
            })
        };
        json!({
            "format": {
                "type": "sequence",
                "elements": [
                    Self::before_the_block(),
                    {
                        "type": "triggered_tags",
                        "triggers": [TOOL_CALL_START],
                        "tags": [{
                            "type": "tag",
                            "begin": format!("{TOOL_CALL_START}\n"),
                            "content": invokes,
                            "end": TOOL_CALL_END,
                        }],
                        "at_least_one": at_least_one,
                    },
                ],
            }
        })
    }

    /// What the model writes before the block when the template leaves the
    /// thought to it: its thought between `<mm:think>` and `</mm:think>`,
    /// free of the markers and the separator, or the bare close it writes
    /// for an empty thought, or nothing at all.
    fn before_the_block() -> Value {
        json!({
            "type": "optional",
            "content": {
                "type": "or",
                "elements": [
                    {
                        "type": "tag",
                        "begin": THINK_START,
                        "content": {
                            "type": "any_text",
                            "excludes": [THINK_START, THINK_END, NAMESPACE],
                        },
                        "end": THINK_END,
                    },
                    { "type": "const_string", "value": THINK_END },
                ],
            },
        })
    }

    /// What precedes a forced call on a prompt that ends inside the thought,
    /// by where the engine starts the grammar; the registry emits
    /// `sequence[prefix, calls]`. Where the engine applies the grammar only
    /// after its reasoning parser has seen the model's own `</mm:think>`,
    /// nothing precedes the calls. Where it applies the grammar from the
    /// first token, the prefix is the thought, free of the think markers and
    /// the separator, closed by `</mm:think>`, so the call follows the
    /// thought. Where the worker does not say, the prefix is free text that
    /// writes no separator, so no call marker, with nothing owed at its end:
    /// the thought and its close on the one kind of engine, nothing on the
    /// other.
    pub fn reasoning_prefix(start: GrammarStart) -> Option<Value> {
        match start {
            GrammarStart::AfterReasoning => None,
            GrammarStart::Unknown => Some(json!({
                "type": "any_text",
                "excludes": [NAMESPACE],
            })),
            GrammarStart::FirstToken => Some(json!({
                "type": "tag",
                "begin": "",
                "content": {
                    "type": "any_text",
                    "excludes": [THINK_START, THINK_END, NAMESPACE],
                },
                "end": THINK_END,
            })),
        }
    }
}

/// The elements of one invoke, written from the tool's `parameters` schema.
///
/// Local references (`#/$defs/Name`, `#/definitions/Name`) are followed, the pointer's escapes
/// undone; a bare `#` is not, since it would name the fragment's own document once the engine
/// compiles a fragment alone; a reference back into a schema already being written, directly or
/// through another definition, is written as free text where it recurs, so a schema that refers
/// to itself stays bounded, the depth cap the outer bound. The elements of one invoke are counted
/// against [`NODES`]: once they are spent, the values still to write (the rest of an object, a
/// list) are free text, so a wide fan-out stays bounded too; before a property is descended into,
/// one element is held for each of the properties after it that writes one, so an early object
/// cannot take the elements its siblings need for their own tags. The leaves' spellings are
/// counted against [`LEAF_BYTES`]: an enum's words, and a number's schema with the definitions
/// its pointers reach (those and no other, each written once per fragment); once they are spent,
/// an enum is text as it is and a number free text. The walks that size a number's definitions,
/// refused ones too, are counted against [`WALK_BYTES`]; once it is spent, a number with pointers
/// is free text without a walk.
///
/// The `symphony` crate's `grammar/xml.rs` writes the same model's block from its table, and
/// spells the same output for a schema within the budgets; it follows the references and the
/// depth cap the same way and has the same element and byte budgets, and takes the same walk
/// allowance in its follow-up.
struct Elements<'a> {
    root: &'a Value,
    /// The pointers of the references on the path being written, outermost first.
    path: Vec<&'a str>,
    /// The elements this invoke may still write.
    nodes: usize,
    /// How many of them are held for the properties after the one being written, at every level
    /// of the descent together.
    reserved: usize,
    /// The bytes the invoke's leaves may still spell.
    leaf_bytes: usize,
    /// The bytes the walks behind its number leaves may still size, refused walks counted too.
    walk_bytes: usize,
    /// The serialized size of each root definition a fragment reached, by the entry's address,
    /// worked out once per invoke.
    entry_bytes: HashMap<usize, usize>,
    /// What each number leaf's fragment costs and reaches, by the leaf schema's address: `None`
    /// for a leaf the budget or the walks' allowance refused, which they refuse again, both only
    /// shrinking.
    reached_by_leaf: HashMap<usize, Option<(usize, Vec<Reached<'a>>)>>,
    /// What each enum costs, by its word list's address.
    enum_bytes: HashMap<usize, usize>,
}

/// A root definition a fragment reaches: its block, its name and its entry.
type Reached<'a> = (&'static str, &'a str, &'a Value);

/// How a value's schema is written: as free text, at the depth cap or behind a reference that
/// recurs; or by its type, its references followed and on the path, `pushed` of them, which the
/// writer unwinds once the value is written. A schema typed `null` is written as nothing, and is
/// neither ([`Elements::how`]).
enum Writes<'a> {
    FreeText,
    Typed {
        schema: &'a Value,
        pushed: usize,
        depth: usize,
    },
}

/// What writing a property gives: nothing, for a property typed `null`; no element, once the
/// invoke's budget is spent, when the caller writes the rest as free text; or the element.
enum Element {
    Nothing,
    Spent,
    Written(Value),
}

impl<'a> Elements<'a> {
    fn of(root: &'a Value) -> Self {
        Self {
            root,
            path: Vec::new(),
            nodes: NODES,
            reserved: 0,
            leaf_bytes: LEAF_BYTES,
            walk_bytes: WALK_BYTES,
            entry_bytes: HashMap::new(),
            reached_by_leaf: HashMap::new(),
            enum_bytes: HashMap::new(),
        }
    }

    /// `cost` bytes off the leaves' budget; `false` when they do not fit, when the caller writes
    /// the value's fallback and the budget stays for a smaller leaf.
    fn takes_bytes(&mut self, cost: usize) -> bool {
        if cost > self.leaf_bytes {
            return false;
        }
        self.leaf_bytes -= cost;
        true
    }

    /// Whether the budget has an element left beyond the ones held for later siblings.
    fn has_an_element(&self) -> bool {
        self.nodes > self.reserved
    }

    /// One element off the budget, which [`Self::has_an_element`] said was there.
    fn take_an_element(&mut self) {
        self.nodes = self.nodes.saturating_sub(1);
    }

    /// How `schema` is written at `depth`, or `None` for a schema typed `null`, written as
    /// nothing: the one place that decides, for the count of an object's properties that write
    /// and for the writing itself.
    fn how(&mut self, schema: &'a Value, depth: usize) -> Option<Writes<'a>> {
        let Some(depth) = depth.checked_sub(1) else {
            return Some(Writes::FreeText);
        };
        let Some((schema, pushed)) = self.follow(schema) else {
            return Some(Writes::FreeText);
        };
        if MinimaxM3Parser::schema_type(Some(schema)) == Some("null") {
            self.unwind(pushed);
            return None;
        }
        Some(Writes::Typed {
            schema,
            pushed,
            depth,
        })
    }

    /// Whether a property's schema writes an element: everything but `null`, as [`Self::how`]
    /// has it.
    fn writes_an_element(&mut self, schema: &'a Value, depth: usize) -> bool {
        match self.how(schema, depth) {
            None => false,
            Some(writes) => {
                self.leaves(writes);
                true
            }
        }
    }

    /// A value classified and not written: its references, if [`Self::how`] followed any, off
    /// the path again.
    fn leaves(&mut self, writes: Writes<'a>) {
        if let Writes::Typed { pushed, .. } = writes {
            self.unwind(pushed);
        }
    }

    /// One invoke: the separator and the opener with the tool's name, the
    /// elements the schema shapes, the separator, `</invoke>` and a newline.
    fn invoke(mut self, name: &str) -> Value {
        let content = match self.follow(self.root) {
            Some((schema, pushed)) => {
                let content = self.members(schema, "</invoke>", DEPTH);
                self.unwind(pushed);
                content
            }
            None => Self::free_text("</invoke>"),
        };
        json!({
            "type": "tag",
            "begin": format!("{INVOKE_START} name=\"{name}\">"),
            "content": content,
            "end": format!("{INVOKE_END}\n"),
        })
    }

    /// The schema a local pointer names: an entry under `$defs` or `definitions`, the pointer's
    /// escapes (`~1` for `/`, `~0` for `~`) undone; nothing for a bare `#`, a pointer elsewhere,
    /// or a missing target.
    fn definition(&self, pointer: &str) -> Option<&'a Value> {
        let mut segments = pointer.strip_prefix("#/")?.split('/');
        let block = match segments.next()? {
            block @ ("$defs" | "definitions") => block,
            _ => return None,
        };
        let mut node = self.root.get(block)?;
        for segment in segments {
            node = node.get(segment.replace("~1", "/").replace("~0", "~"))?;
        }
        Some(node)
    }

    /// `schema` with its references followed, and how many of them the path took on (the caller
    /// unwinds them once the schema is written); `None` when a reference re-enters a schema on
    /// the path, or the hops exceed the depth cap: that node is written as free text. A pointer
    /// that names nothing stops the following, and the schema stays what it is: untyped.
    fn follow(&mut self, schema: &'a Value) -> Option<(&'a Value, usize)> {
        let mut schema = schema;
        let mut pushed = 0;
        while let Some(pointer) = schema.get("$ref").and_then(Value::as_str) {
            let Some(target) = self.definition(pointer) else {
                break;
            };
            if self.path.contains(&pointer) || pushed == DEPTH {
                self.unwind(pushed);
                return None;
            }
            self.path.push(pointer);
            pushed += 1;
            schema = target;
        }
        Some((schema, pushed))
    }

    /// The path without its last `pushed` references.
    fn unwind(&mut self, pushed: usize) {
        let keep = self.path.len().saturating_sub(pushed);
        self.path.truncate(keep);
    }

    /// An object's elements: one per property in the schema's order, each
    /// optional unless the schema requires it, a property typed `null` left
    /// out at no cost to the budget; free text up to `close` for a schema
    /// without properties, and for the properties left once the invoke's
    /// element budget is spent. While a property's value is written, one
    /// element is held for each property after it that writes one, so each
    /// still gets its own tag.
    fn members(&mut self, schema: &'a Value, close: &str, depth: usize) -> Value {
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty());
        let Some(properties) = properties else {
            return Self::free_text(close);
        };
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        // The properties that write, counted first: while one is written, one element is held
        // for each of them after it.
        let mut later: usize = 0;
        for schema in properties.values() {
            if self.writes_an_element(schema, depth) {
                later += 1;
            }
        }
        let mut elements: Vec<Value> = Vec::new();
        for (key, schema) in properties {
            match self.element(key, schema, depth, later.saturating_sub(1)) {
                Element::Nothing => {}
                Element::Spent => {
                    elements.push(Self::free_text(close));
                    break;
                }
                Element::Written(element) => {
                    later = later.saturating_sub(1);
                    elements.push(if required.contains(&key.as_str()) {
                        element
                    } else {
                        json!({ "type": "optional", "content": element })
                    });
                }
            }
        }
        if elements.is_empty() {
            return json!({ "type": "const_string", "value": "" });
        }
        json!({ "type": "sequence", "elements": elements })
    }

    /// One element: the separator and `<key>`, the value, the separator and
    /// `</key>`, one element off the budget, with `later` held for the
    /// properties after it while its value is written; nothing for a property
    /// typed `null`, at no cost and whatever the budget; no element once the
    /// budget is spent.
    fn element(&mut self, key: &str, schema: &'a Value, depth: usize, later: usize) -> Element {
        let Some(writes) = self.how(schema, depth) else {
            return Element::Nothing;
        };
        if !self.has_an_element() {
            self.leaves(writes);
            return Element::Spent;
        }
        self.take_an_element();
        self.reserved += later;
        let value = self.value(&format!("</{key}>"), writes);
        self.reserved -= later;
        Element::Written(json!({
            "type": "sequence",
            "elements": [
                { "type": "const_string", "value": format!("{ELEMENT_START}{key}>") },
                value,
                { "type": "const_string", "value": format!("{ELEMENT_END_START}{key}>") },
            ],
        }))
    }

    /// The value between an element's tags, as [`Self::how`] says it is
    /// written: free text, or by the type its schema pins.
    fn value(&mut self, close: &str, writes: Writes<'a>) -> Value {
        let Writes::Typed {
            schema,
            pushed,
            depth,
        } = writes
        else {
            return Self::free_text(close);
        };
        let value = match MinimaxM3Parser::schema_type(Some(schema)) {
            Some("boolean") => Self::one_of(["true", "false", "True", "False"]),
            Some("object") => self.members(schema, close, depth),
            Some("array") => match schema.get("items") {
                Some(items) if self.has_an_element() => {
                    self.take_an_element();
                    self.items(items, depth)
                }
                _ => Self::free_text(close),
            },
            Some("string") => match schema.get("enum").and_then(Value::as_array) {
                Some(values) if !values.is_empty() && values.iter().all(Value::is_string) => {
                    let key = std::ptr::from_ref(values) as usize;
                    let cost = *self.enum_bytes.entry(key).or_insert_with(|| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(|word| word.len() + WORD_FRAMING)
                            .sum()
                    });
                    if self.takes_bytes(cost) {
                        Self::one_of(values.iter().filter_map(Value::as_str))
                    } else {
                        Self::text(close)
                    }
                }
                _ => Self::text(close),
            },
            Some("integer" | "number") => self.fragment(schema, close),
            _ => Self::free_text(close),
        };
        self.unwind(pushed);
        value
    }

    /// A list's items: any number of `<item>` elements, each a value of the
    /// items' type, an item typed `null` an empty element.
    fn items(&mut self, items: &'a Value, depth: usize) -> Value {
        let open = format!("{ELEMENT_START}{ITEM}>");
        let close = format!("{ELEMENT_END_START}{ITEM}>");
        let item = match self.how(items, depth) {
            Some(writes) => {
                let value = self.value(&format!("</{ITEM}>"), writes);
                json!({
                    "type": "sequence",
                    "elements": [
                        { "type": "const_string", "value": open },
                        value,
                        { "type": "const_string", "value": close },
                    ],
                })
            }
            None => json!({ "type": "const_string", "value": format!("{open}{close}") }),
        };
        json!({ "type": "star", "content": item })
    }

    /// A number's schema as the engine gets it: with the root's definitions its pointers reach
    /// beside it, those and no other (the entry a pointer names by its first segment, and the
    /// entries those reach), so each pointer resolves once the engine compiles the fragment
    /// alone; a definition the schema declares itself stays. Free text up to `close` once the
    /// leaves' byte budget cannot take the fragment, definitions included, or the walks' allowance
    /// ([`WALK_BYTES`]) is spent before the fragment's definitions are sized.
    fn fragment(&mut self, schema: &'a Value, close: &str) -> Value {
        let Some((cost, reached)) = self.reached_within(schema) else {
            return Self::free_text(close);
        };
        if !self.takes_bytes(cost) {
            return Self::free_text(close);
        }
        let mut fragment = schema.clone();
        if let Some(object) = fragment.as_object_mut() {
            for (block, name, entry) in reached {
                let slot = object.entry(block.to_string()).or_insert_with(|| json!({}));
                if let Some(entries) = slot.as_object_mut() {
                    entries
                        .entry(name.to_string())
                        .or_insert_with(|| entry.clone());
                }
            }
        }
        json!({ "type": "json_schema", "json_schema": fragment })
    }

    /// What a number leaf's fragment costs, with the root's definitions its pointers reach,
    /// directly or through one another (each as its block, `$defs` or `definitions`, its name,
    /// the first segment after the block, since a pointer deeper into an entry needs the whole
    /// entry, and its entry), when that fits what the leaves may still spell; `None` as soon as
    /// the running count passes it, so a leaf the budget refuses costs the walk up to the budget
    /// and no more. Worked out once per leaf schema per invoke: a leaf refused once is refused
    /// again, the budget only shrinking. The walks, refused ones too, spend [`WALK_BYTES`] at the
    /// pace of the bytes they size; past it a leaf with pointers is refused without a walk, while
    /// a leaf without pointers, which walks nothing but itself, is neither bound nor charged.
    fn reached_within(&mut self, schema: &'a Value) -> Option<(usize, Vec<Reached<'a>>)> {
        let key = std::ptr::from_ref(schema) as usize;
        if let Some(known) = self.reached_by_leaf.get(&key) {
            return match known {
                Some((cost, reached)) if *cost <= self.leaf_bytes => Some((*cost, reached.clone())),
                _ => None,
            };
        }
        // A leaf without pointers walks nothing but itself, which the request already holds: the
        // allowance is for the definitions a leaf reaches, and neither bounds nor charges one.
        let walks = !local_pointers(schema).is_empty();
        let limit = if walks {
            self.leaf_bytes.min(self.walk_bytes)
        } else {
            self.leaf_bytes
        };
        let found = self.walk_reached(schema, limit);
        if walks {
            self.walk_bytes = self
                .walk_bytes
                .saturating_sub(found.as_ref().map_or(limit, |(cost, _)| *cost));
        }
        self.reached_by_leaf.insert(key, found.clone());
        found
    }

    /// The walk behind [`Self::reached_within`], within `limit`.
    fn walk_reached(
        &mut self,
        schema: &'a Value,
        limit: usize,
    ) -> Option<(usize, Vec<Reached<'a>>)> {
        let mut cost = json_bytes(schema);
        if cost > limit {
            return None;
        }
        let mut reached: Vec<Reached<'a>> = Vec::new();
        let mut seen: HashSet<(&'static str, &'a str)> = HashSet::new();
        let mut pending: Vec<&'a Value> = vec![schema];
        while let Some(node) = pending.pop() {
            for pointer in local_pointers(node) {
                let Some((block, name, entry)) = self.entry_of(pointer) else {
                    continue;
                };
                if !seen.insert((block, name)) {
                    continue;
                }
                cost += self.entry_bytes(entry);
                if cost > limit {
                    return None;
                }
                reached.push((block, name, entry));
                pending.push(entry);
            }
        }
        Some((cost, reached))
    }

    /// The block, the name and the entry a local pointer names, the name's escapes undone (`~1`
    /// for `/`, `~0` for `~`); nothing for a bare `#`, a pointer elsewhere or a missing entry.
    fn entry_of(&self, pointer: &str) -> Option<(&'static str, &'a str, &'a Value)> {
        let mut segments = pointer.strip_prefix("#/")?.split('/');
        let block = match segments.next()? {
            "$defs" => "$defs",
            "definitions" => "definitions",
            _ => return None,
        };
        let entries = self.root.get(block)?.as_object()?;
        let name = segments.next()?.replace("~1", "/").replace("~0", "~");
        let (key, entry) = entries.get_key_value(name.as_str())?;
        Some((block, key.as_str(), entry))
    }

    /// The serialized size of a definition's entry, worked out once per invoke.
    fn entry_bytes(&mut self, entry: &'a Value) -> usize {
        let key = std::ptr::from_ref(entry) as usize;
        if let Some(bytes) = self.entry_bytes.get(&key) {
            return *bytes;
        }
        let bytes = json_bytes(entry);
        self.entry_bytes.insert(key, bytes);
        bytes
    }

    /// One of the words.
    fn one_of(words: impl IntoIterator<Item = &'a str>) -> Value {
        let words: Vec<Value> = words
            .into_iter()
            .map(|word| json!({ "type": "const_string", "value": word }))
            .collect();
        json!({ "type": "or", "elements": words })
    }

    /// A string as it is: text that holds neither the separator, nor `close`,
    /// nor a marker that ends the call.
    fn text(close: &str) -> Value {
        Self::excluding(&[NAMESPACE, close])
    }

    /// A value the schema does not shape: text that holds neither `close` nor
    /// a marker that ends the call, separators and nested tags included,
    /// which the parser reads as it infers.
    fn free_text(close: &str) -> Value {
        Self::excluding(&[close])
    }

    /// Any text that holds none of `first` and none of the markers that end
    /// the call, each listed once.
    fn excluding(first: &[&str]) -> Value {
        let mut excludes: Vec<&str> = Vec::new();
        for text in first.iter().chain(CALL_EXITS.iter()) {
            if !excludes.contains(text) {
                excludes.push(text);
            }
        }
        json!({ "type": "any_text", "excludes": excludes })
    }
}

/// The local pointers in `schema`, wherever a `$ref`, `$dynamicRef` or `$recursiveRef` stands.
fn local_pointers(schema: &Value) -> Vec<&str> {
    fn collect<'v>(schema: &'v Value, pointers: &mut Vec<&'v str>) {
        match schema {
            Value::Object(object) => {
                for (key, value) in object {
                    if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef") {
                        if let Some(pointer) =
                            value.as_str().filter(|pointer| pointer.starts_with('#'))
                        {
                            pointers.push(pointer);
                        }
                    }
                    collect(value, pointers);
                }
            }
            Value::Array(items) => items.iter().for_each(|item| collect(item, pointers)),
            _ => {}
        }
    }
    let mut pointers = Vec::new();
    collect(schema, &mut pointers);
    pointers
}

/// About how many bytes `value` takes serialized, the escapes counted: what a leaf costs the
/// invoke's byte budget.
fn json_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(flag) => {
            if *flag {
                4
            } else {
                5
            }
        }
        Value::Number(number) => number.to_string().len(),
        Value::String(text) => {
            text.len()
                + 2
                + text
                    .bytes()
                    .filter(|byte| matches!(byte, b'"' | b'\\') || *byte < 0x20)
                    .count()
        }
        Value::Array(items) => {
            2 + items.iter().map(json_bytes).sum::<usize>() + items.len().saturating_sub(1)
        }
        Value::Object(object) => {
            2 + object
                .iter()
                .map(|(key, value)| key.len() + 3 + json_bytes(value))
                .sum::<usize>()
                + object.len().saturating_sub(1)
        }
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
                if !Self::could_start_invoke(&self.buffer) {
                    // Release the false wrapper, then resume the normal-text
                    // scan so a later marker (including a partial one) is still
                    // recognized rather than flushed as ordinary content.
                    normal_text.push_str(TOOL_CALL_START);
                    self.buffer.drain(..TOOL_CALL_START.len());
                    self.in_tool_call = false;
                    continue;
                }
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

    fn take_unstreamed_normal_text(&mut self) -> String {
        // Completed blocks are removed from `buffer`, so anything left here is
        // an independent, incomplete candidate and must be returned verbatim.
        // Leave the parser ready to process ordinary text if it is reused.
        self.in_tool_call = false;
        std::mem::take(&mut self.buffer)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
        self.in_tool_call = false;
    }
}
