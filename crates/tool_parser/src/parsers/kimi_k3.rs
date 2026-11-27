//! Kimi-K3 (XTML) tool-call parser
//!
//! Ports the Kimi-K3 reference tool-call parser: turns the generated XTML
//! `response` and `tools` channels back into plain `content` and `ToolCall`s.
//!
//! # Format
//!
//! K3 assistant tool calls live in a nested `tools` channel, emitted after a
//! sibling `response` channel that is unwrapped into content:
//! ```text
//! <|open|>response<|sep|> CONTENT <|close|>response<|sep|>
//! <|open|>tools<|sep|>
//!   <|open|>call tool="NAME" index="N"<|sep|>
//!     <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|>
//!   <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! ```
//! `<|open|>`, `<|close|>`, `<|sep|>` are literal strings in the detokenized
//! text (not special tokens from this parser's point of view). Markers
//! tolerate optional whitespace within them (e.g. `<|open|> tools <|sep|>`)
//! as defense in depth; this is a no-op on clean input. Bodies use
//! non-greedy matching so each block stops at its own first closing marker.
//!
//! # Argument decoding
//! - `type="string"` -> the value is raw text, used as-is (no unescaping).
//! - any other type -> the value is JSON-decoded; on JSON error, falls back
//!   to the raw string rather than erroring mid-stream.
//!
//! # Attribute escaping
//! Attribute values (`tool=`, `index=`, `key=`, `type=`) are escaped on the
//! encode side (`&` -> `&amp;`, `"` -> `&quot;`); decoding reverses that in
//! order (`&quot;` -> `"` then `&amp;` -> `&`).
//!
//! # Tool-call id
//! SMG's [`ToolCall`] has no `id` field (one is assigned later by
//! `model_gateway`), so this parser only produces `name` + `arguments` (+
//! content). Streaming [`ToolCallItem::tool_index`] is a per-message
//! zero-based ordinal assigned in emission order, independent of the XTML
//! `index` attribute. Deriving it from the ordinal — rather than trusting the
//! model-supplied `index` — keeps streamed indices (and the ids later built
//! from them) unique and monotonic even when a model emits duplicate, sparse,
//! or out-of-order `index` values, and matches the non-streaming path, which
//! also indexes tool calls by their emission order.
//!
//! Known limitation (inherited from the reference): because string argument
//! and response bodies are emitted raw, a value that literally contains
//! `<|close|>argument<|sep|>` or `<|close|>response<|sep|>` is
//! indistinguishable from a real closing marker.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::{json, Value};

use crate::{
    errors::ParserResult,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";

/// Channel closings that may precede the tools section, one per shape the
/// generation prompt can leave open: nothing, `think` (thinking mode), or
/// `response` (non-thinking mode). See [`KimiK3Parser::build_structural_tag`].
const TOOLS_SECTION_PREFIXES: [&str; 3] = ["", THINK_CLOSE, RESPONSE_CLOSE];

/// Bare XTML control markers, used to assemble structural-tag literals.
const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";

/// A decoded `<|open|>call ...<|sep|>...<|close|>call<|sep|>` block.
///
/// The XTML `index` attribute is deliberately not captured: streamed tool
/// indices are assigned by emission order (see the module-level `Tool-call id`
/// docs), so a model-supplied `index` never influences the parser output.
struct DecodedCall {
    name: String,
    /// Compact JSON object string.
    arguments: String,
}

/// Kimi-K3 XTML tool-call parser.
///
/// Handles both non-streaming (`parse_complete`) and streaming
/// (`parse_incremental`) extraction of the `tools` channel, and unwraps the
/// sibling `response` channel into `content`.
pub struct KimiK3Parser {
    /// Matches `<|open|>tools<|sep|>` (tolerant of inner whitespace).
    tools_open_re: Regex,
    /// Matches `<|close|>tools<|sep|>`.
    tools_close_re: Regex,
    /// Matches `<|open|>response<|sep|>`.
    response_open_re: Regex,
    /// Matches `<|close|>response<|sep|>`.
    response_close_re: Regex,
    /// Matches a stray `<|close|>message<|sep|>` (stripped from content).
    message_close_re: Regex,
    /// Matches one `call` block, capturing `attrs` and `body`.
    call_re: Regex,
    /// Matches one `argument` block, capturing `attrs` and `val`.
    arg_re: Regex,
    /// Matches one `key="value"` attribute pair.
    attr_re: Regex,
    /// Matches a complete, unwrapped `response` channel, capturing `c`.
    response_re: Regex,

    /// Accumulates every chunk seen so far (plays the role of the streaming
    /// `current_text`, which the engine rebuilds from scratch each call).
    buffer: String,
    /// Byte offset into `buffer` up to which response content has already
    /// been emitted.
    sent_content_idx: usize,
    /// Number of tool calls already emitted.
    sent_tool_call_count: usize,
}

impl KimiK3Parser {
    /// Build an xgrammar structural tag that constrains Kimi-K3 XTML tool
    /// calls to the declared `tools`.
    ///
    /// The grammar mirrors the encoder (`kimi_k3_xtml.rs`): a tool call is
    ///
    /// ```text
    /// <|open|>tools<|sep|>
    ///   <|open|>call tool="NAME" index="N"<|sep|>
    ///     <|open|>argument key="K" type="T"<|sep|>VALUE<|close|>argument<|sep|>
    ///   <|close|>call<|sep|>
    /// <|close|>tools<|sep|>
    /// ```
    ///
    /// A `triggered_tags` format dispatches to the tools section, then forces
    /// the framing above. All K3 tool calls live in a single section, so its
    /// `content` is a `plus` of call blocks (native parallel calls) and a lone
    /// section per prefix suffices — unlike K2, which repeats a tag per call.
    ///
    /// # Closing the prefilled channel first
    ///
    /// `at_least_one` makes xgrammar mask every token that is not a trigger
    /// prefix, so the trigger set decides what may precede the tools section.
    /// K3's generation prompt leaves a channel open (`<|open|>think<|sep|>` in
    /// thinking mode, `<|open|>response<|sep|>` otherwise), and that channel has
    /// to close before a sibling `tools` section can start. Triggering on
    /// `<|open|>tools<|sep|>` alone would forbid the closing marker, forcing a
    /// tools section nested inside a channel the model can never close — output
    /// the reasoning parser then swallows whole, leaving `tool_calls` empty.
    ///
    /// So emit one tag per prefix in `TOOLS_SECTION_PREFIXES` and register each
    /// closing marker as a trigger, mirroring the Harmony builder, which likewise
    /// pairs one tag per channel-entry shape with a trigger per prefix.
    ///
    /// Within a call, arguments are
    /// constrained per the JSON schema (Strict mode): properties are pinned in
    /// the schema's declared order (the crate enables `serde_json`'s
    /// `preserve_order`), each `required` property mandatory and each optional
    /// property individually skippable in its declared slot; every `type=`
    /// attribute and value is pinned to the property's type. This declared
    /// order is a *generation* constraint only — `decode_call` parses arguments
    /// order-insensitively, so a model constrained here still round-trips even
    /// though the grammar admits just one ordering. Schemas without usable
    /// `properties` fall back to a permissive skeleton so the grammar is never
    /// infeasible.
    ///
    /// `at_least_one` is wired to `tool_choice`: `true` for `"required"` (a
    /// tool call must be emitted), `false` for `"auto"`. Mirroring K2,
    /// `stop_after_first` is left unset so trailing tokens after the section
    /// (`<|close|>message<|sep|><|end_of_msg|>`) remain valid.
    pub fn build_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let call_formats: Vec<Value> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| build_call_format(&tool.function.name, tool.function.parameters_schema()))
            .collect();

        let call_choice = if call_formats.is_empty() {
            permissive_call_format()
        } else {
            json!({ "type": "or", "elements": call_formats })
        };

        let content = json!({ "type": "plus", "content": call_choice });

        // One tools section per prefill shape; only `begin` differs.
        let tags: Vec<Value> = TOOLS_SECTION_PREFIXES
            .iter()
            .map(|prefix| {
                json!({
                    "begin": format!("{prefix}{TOOLS_OPEN}"),
                    "content": content,
                    "end": TOOLS_CLOSE,
                })
            })
            .collect();

        json!({
            "format": {
                "type": "triggered_tags",
                "triggers": [TOOLS_OPEN, THINK_CLOSE, RESPONSE_CLOSE],
                "tags": tags,
                "at_least_one": at_least_one,
            }
        })
    }

    /// Create a new Kimi-K3 parser.
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        Self {
            tools_open_re: Regex::new(r"<\|open\|>\s*tools\s*<\|sep\|>")
                .expect("valid regex"),
            tools_close_re: Regex::new(r"<\|close\|>\s*tools\s*<\|sep\|>")
                .expect("valid regex"),
            response_open_re: Regex::new(r"<\|open\|>\s*response\s*<\|sep\|>")
                .expect("valid regex"),
            response_close_re: Regex::new(r"<\|close\|>\s*response\s*<\|sep\|>")
                .expect("valid regex"),
            message_close_re: Regex::new(r"<\|close\|>\s*message\s*<\|sep\|>")
                .expect("valid regex"),
            call_re: Regex::new(
                r"(?s)<\|open\|>\s*call\s+(?P<attrs>.*?)<\|sep\|>(?P<body>.*?)<\|close\|>\s*call\s*<\|sep\|>",
            )
            .expect("valid regex"),
            arg_re: Regex::new(
                r"(?s)<\|open\|>\s*argument\s+(?P<attrs>.*?)<\|sep\|>(?P<val>.*?)<\|close\|>\s*argument\s*<\|sep\|>",
            )
            .expect("valid regex"),
            attr_re: Regex::new(r#"(?P<k>\w+)="(?P<v>[^"]*)""#).expect("valid regex"),
            response_re: Regex::new(
                r"(?s)<\|open\|>\s*response\s*<\|sep\|>(?P<c>.*?)<\|close\|>\s*response\s*<\|sep\|>",
            )
            .expect("valid regex"),
            buffer: String::new(),
            sent_content_idx: 0,
            sent_tool_call_count: 0,
        }
    }

    /// Parse the `key="value"` attribute pairs in `s`, unescaping each value
    /// (`&quot;` -> `"` then `&amp;` -> `&`, the reverse of the encode order).
    fn attrs(&self, s: &str) -> HashMap<String, String> {
        self.attr_re
            .captures_iter(s)
            .map(|m| {
                let key = m.name("k").map_or("", |g| g.as_str()).to_string();
                let value = m
                    .name("v")
                    .map_or("", |g| g.as_str())
                    .replace("&quot;", "\"")
                    .replace("&amp;", "&");
                (key, value)
            })
            .collect()
    }

    /// Decode one `call` block (its `attrs` segment and `body`) into a
    /// [`DecodedCall`]. Each argument is re-typed per its `type=` tag:
    /// strings pass through raw, everything else is JSON-decoded (falling
    /// back to the raw string on malformed JSON, so a partial stream never
    /// errors). Returns `None` when no tool name is present.
    fn decode_call(&self, attrs: &str, body: &str) -> Option<DecodedCall> {
        let call_attrs = self.attrs(attrs);
        let tool_name = call_attrs.get("tool").cloned().unwrap_or_default();
        if tool_name.is_empty() {
            return None;
        }

        let mut arguments = serde_json::Map::new();
        for arg_match in self.arg_re.captures_iter(body) {
            let arg_attrs_text = arg_match.name("attrs").map_or("", |g| g.as_str());
            let arg_attrs = self.attrs(arg_attrs_text);
            let key = arg_attrs.get("key").cloned().unwrap_or_default();
            let arg_type = arg_attrs.get("type").map_or("string", String::as_str);
            let raw_value = arg_match.name("val").map_or("", |g| g.as_str());

            let value = if arg_type == "string" {
                Value::String(raw_value.to_string())
            } else {
                serde_json::from_str::<Value>(raw_value)
                    .unwrap_or_else(|_| Value::String(raw_value.to_string()))
            };
            arguments.insert(key, value);
        }

        let arguments_str = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());

        Some(DecodedCall {
            name: tool_name,
            arguments: arguments_str,
        })
    }

    /// Strip XTML response/message markers from generated response text.
    ///
    /// In chat serving, `<|open|>response<|sep|>` is often part of the
    /// prompt generation prefix, so the model output may only contain the
    /// body plus `<|close|>response<|sep|>`. Handles both that
    /// consumed-prefix shape and a complete
    /// `<|open|>response<|sep|>...<|close|>response<|sep|>` wrapper.
    fn strip_response_content(&self, text: &str) -> Option<String> {
        let stripped = if let Some(m_open) = self.response_open_re.find(text) {
            if let Some(m_close) = self.response_close_re.find_at(text, m_open.end()) {
                text[m_open.end()..m_close.start()].to_string()
            } else {
                text[m_open.end()..].to_string()
            }
        } else {
            self.response_close_re.replace_all(text, "").into_owned()
        };
        let stripped = self
            .message_close_re
            .replace_all(&stripped, "")
            .into_owned();
        if stripped.is_empty() {
            None
        } else {
            Some(stripped)
        }
    }

    /// Compute non-streaming content: prefer the unwrapped `response`
    /// channel; else fall back to stripping markers from `before` (the text
    /// preceding the `tools` channel, or the whole output when there is no
    /// `tools` channel at all).
    fn content(&self, model_output: &str, before: &str) -> Option<String> {
        if let Some(m) = self.response_re.captures(model_output) {
            let c = m.name("c").map_or("", |g| g.as_str());
            return if c.is_empty() {
                None
            } else {
                Some(c.to_string())
            };
        }
        self.strip_response_content(before)
    }

    /// Compute the streaming-safe slice of response content, updating
    /// `sent_content_idx`. This is what keeps split markers from leaking:
    /// content is only released up to the start of the next recognized
    /// marker (or up to the point where a partial marker might still be
    /// growing at the tail of `current_text`).
    fn extract_response_content(&mut self, current_text: &str) -> Option<String> {
        let m_open = self.response_open_re.find(current_text);
        let body_start = m_open.map_or(0, |m| m.end());

        let tools_start = self
            .tools_open_re
            .find_at(current_text, body_start)
            .map(|m| m.start());
        let response_end = self
            .response_close_re
            .find_at(current_text, body_start)
            .map(|m| m.start());

        let sendable_idx = match (tools_start, response_end) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => {
                let overlap = partial_tag_overlap(current_text, RESPONSE_OPEN)
                    .max(partial_tag_overlap(current_text, RESPONSE_CLOSE))
                    .max(partial_tag_overlap(current_text, TOOLS_OPEN));
                current_text.len() - overlap
            }
        };

        if sendable_idx <= body_start {
            return None;
        }
        if self.sent_content_idx < body_start {
            self.sent_content_idx = body_start;
        }
        if sendable_idx <= self.sent_content_idx {
            return None;
        }

        let content = current_text[self.sent_content_idx..sendable_idx].to_string();
        self.sent_content_idx = sendable_idx;
        if content.is_empty() {
            None
        } else {
            Some(content)
        }
    }

    /// Decode every complete `call` block in `section` (a slice starting
    /// just past the `tools`-open marker), skipping blocks with no tool
    /// name.
    fn decode_calls_in_section(&self, section: &str) -> Vec<DecodedCall> {
        let mut calls = Vec::new();
        for m in self.call_re.captures_iter(section) {
            let attrs_text = m.name("attrs").map_or("", |g| g.as_str());
            let body = m.name("body").map_or("", |g| g.as_str());
            if let Some(decoded) = self.decode_call(attrs_text, body) {
                calls.push(decoded);
            }
        }
        calls
    }
}

// ---------------------------------------------------------------------------
// Structural-tag construction (guided decoding)
// ---------------------------------------------------------------------------

/// How one argument value is constrained in the structural tag.
enum ArgShape {
    /// The `type=` attribute and value grammar are pinned from the schema.
    Fixed {
        type_attr: &'static str,
        value: Value,
    },
    /// The schema is ambiguous (union / missing / `anyOf`); allow any
    /// well-formed `type=` attribute and any raw value up to the marker.
    Permissive,
}

/// Escape an attribute value the same way the encoder does (`&` -> `&amp;`,
/// then `"` -> `&quot;`), so grammar literals match the emitted bytes.
fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

/// Build the `sequence` grammar for a single `<|open|>call ...<|close|>call`
/// block for the tool `name` with the given JSON-schema `params`.
fn build_call_format(name: &str, params: &Value) -> Value {
    let esc_name = escape_attr(name);
    json!({
        "type": "sequence",
        "elements": [
            { "type": "const_string", "value": format!("{OPEN}call tool=\"{esc_name}\" index=\"") },
            { "type": "regex", "pattern": "[1-9][0-9]*" },
            { "type": "const_string", "value": format!("\"{SEP}") },
            build_args_format(params),
            { "type": "const_string", "value": format!("{CLOSE}call{SEP}") },
        ]
    })
}

/// Build the argument-region grammar for a tool's parameter schema (Strict
/// mode): one slot per property in the schema's declared order, `required`
/// properties mandatory and the rest wrapped in `optional` (individually
/// skippable, in place). Falls back to a permissive skeleton when the schema
/// has no usable `properties`.
fn build_args_format(params: &Value) -> Value {
    let Some(properties) = params.get("properties").and_then(Value::as_object) else {
        return permissive_args_format();
    };
    if properties.is_empty() {
        return permissive_args_format();
    }

    let required: HashSet<&str> = params
        .get("required")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let defs = RootDefs::from_params(params);

    let elements: Vec<Value> = properties
        .iter()
        .map(|(key, subschema)| {
            let argument = build_argument_format(key, subschema, defs);
            if required.contains(key.as_str()) {
                argument
            } else {
                json!({ "type": "optional", "content": argument })
            }
        })
        .collect();

    json!({ "type": "sequence", "elements": elements })
}

/// Build the grammar for one `<|open|>argument ...<|close|>argument` block,
/// pinning `key=`, `type=`, and the value grammar from the property schema.
fn build_argument_format(key: &str, subschema: &Value, defs: RootDefs<'_>) -> Value {
    let esc_key = escape_attr(key);
    match classify_arg(subschema, defs) {
        ArgShape::Fixed { type_attr, value } => json!({
            "type": "sequence",
            "elements": [
                { "type": "const_string", "value": format!("{OPEN}argument key=\"{esc_key}\" type=\"{type_attr}\"{SEP}") },
                value,
                { "type": "const_string", "value": format!("{CLOSE}argument{SEP}") },
            ]
        }),
        ArgShape::Permissive => json!({
            "type": "sequence",
            "elements": [
                { "type": "const_string", "value": format!("{OPEN}argument key=\"{esc_key}\" type=\"") },
                { "type": "any_text", "excludes": ["\"", "<|"] },
                { "type": "const_string", "value": format!("\"{SEP}") },
                { "type": "any_text", "excludes": [CLOSE, OPEN, SEP] },
                { "type": "const_string", "value": format!("{CLOSE}argument{SEP}") },
            ]
        }),
    }
}

/// The definition blocks that sit at the root of a tool's parameter schema,
/// plus the root schema itself.
///
/// Each argument's subschema is compiled as its own standalone grammar
/// document, but a `$ref` such as `#/$defs/Node` is a pointer from the *root*
/// of the document it appears in. Without the definition block travelling
/// alongside the subschema the pointer dangles, and the grammar compiler
/// rejects the whole tool call rather than just that argument.
#[derive(Clone, Copy, Default)]
struct RootDefs<'a> {
    root: Option<&'a Value>,
    defs: Option<&'a Value>,
    definitions: Option<&'a Value>,
}

impl<'a> RootDefs<'a> {
    fn from_params(params: &'a Value) -> Self {
        Self {
            root: Some(params),
            defs: params.get("$defs"),
            definitions: params.get("definitions"),
        }
    }

    fn is_empty(self) -> bool {
        self.defs.is_none() && self.definitions.is_none()
    }

    fn block(self, name: &str) -> Option<&'a Value> {
        match name {
            "$defs" => self.defs,
            "definitions" => self.definitions,
            _ => None,
        }
    }

    /// Resolve a local JSON pointer against the parameter schema.
    ///
    /// `#` addresses the root schema itself, which a root-recursive property
    /// legitimately names.
    fn resolve_pointer(self, pointer: &str) -> Option<&'a Value> {
        if pointer == "#" || pointer == "#/" {
            return self.root;
        }
        let rest = pointer.strip_prefix("#/")?;
        let mut segments = rest.split('/');
        let mut node = self.block(segments.next()?)?;
        for segment in segments {
            // JSON-pointer escapes, innermost first per RFC 6901.
            let key = segment.replace("~1", "/").replace("~0", "~");
            node = node.get(&key)?;
        }
        Some(node)
    }

    /// The schema a subschema's own `$ref` names, if it has one.
    ///
    /// Only the pointer is followed — sibling keywords stay with the property
    /// and take precedence, since in 2020-12 a `$ref` applies *alongside* them
    /// rather than replacing them. Following a single hop also keeps a `$ref`
    /// cycle from looping here; a recursive definition still compiles because
    /// the block below travels with the schema and the compiler expands it.
    fn resolve_ref(self, subschema: &'a Value) -> Option<&'a Value> {
        let pointer = subschema.get("$ref")?.as_str()?;
        self.resolve_pointer(pointer)
    }

    /// Whether every local pointer inside `schema` names something that exists.
    ///
    /// One dangling pointer makes the compiler reject the entire tool call, so
    /// an argument that contains one is better left permissive.
    fn local_refs_resolve(self, schema: &Value) -> bool {
        match schema {
            Value::Object(map) => {
                for (key, value) in map {
                    if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef") {
                        if let Some(pointer) = value.as_str() {
                            if pointer.starts_with('#') && self.resolve_pointer(pointer).is_none() {
                                return false;
                            }
                        }
                    }
                }
                map.values().all(|value| self.local_refs_resolve(value))
            }
            Value::Array(items) => items.iter().all(|item| self.local_refs_resolve(item)),
            _ => true,
        }
    }

    /// Return `subschema` with the definition blocks merged in, so any local
    /// pointer inside it still resolves once it is compiled on its own.
    ///
    /// A subschema that declares its own block keeps every entry in it; the
    /// root's entries fill in the names it does not define, so a property with
    /// a partial block does not lose access to the rest of the tool's
    /// definitions. Only schemas that actually reference something pay the
    /// extra bytes.
    fn attach(self, subschema: &Value) -> Value {
        if self.is_empty() || !contains_local_ref(subschema) {
            return subschema.clone();
        }
        let mut out = subschema.clone();
        let Some(object) = out.as_object_mut() else {
            return subschema.clone();
        };
        for (name, block) in [("$defs", self.defs), ("definitions", self.definitions)] {
            let Some(Value::Object(root_entries)) = block else {
                continue;
            };
            match object.get_mut(name) {
                Some(Value::Object(own)) => {
                    for (key, value) in root_entries {
                        own.entry(key.clone()).or_insert_with(|| value.clone());
                    }
                }
                Some(_) => {}
                None => {
                    object.insert(name.to_string(), Value::Object(root_entries.clone()));
                }
            }
        }
        out
    }
}

/// Whether `schema` contains a reference pointing inside its own document.
///
/// Covers `$ref` plus the `$dynamicRef` / `$recursiveRef` spellings, so a
/// schema that uses one of the less common forms still gets the definition
/// block attached rather than silently compiling against a dangling pointer.
fn contains_local_ref(schema: &Value) -> bool {
    match schema {
        Value::Object(map) => {
            let refs_locally = map.iter().any(|(key, value)| {
                matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef")
                    && value.as_str().is_some_and(|t| t.starts_with('#'))
            });
            refs_locally || map.values().any(contains_local_ref)
        }
        Value::Array(items) => items.iter().any(contains_local_ref),
        _ => false,
    }
}

/// Map a property schema to its XTML `type=` attribute and value grammar.
/// K3 emits string bodies verbatim (no JSON quoting) while every other type is
/// compact JSON, and it derives `type=` from the *value*, so `integer` becomes
/// `number`.
fn classify_arg(subschema: &Value, defs: RootDefs<'_>) -> ArgShape {
    // A pointer naming something the schema does not contain would make the
    // compiler reject the whole tool call, so keep such an argument permissive
    // rather than emitting a grammar that cannot compile.
    if !defs.local_refs_resolve(subschema) {
        return ArgShape::Permissive;
    }

    // A `$ref` contributes the keywords its target declares, but a keyword on
    // the property itself wins: in 2020-12 `$ref` applies *alongside* its
    // siblings rather than replacing them. The subschema is still embedded
    // whole further down, so the pointer and every sibling reach the compiler.
    let target = defs.resolve_ref(subschema);
    let keyword = |name: &str| {
        subschema
            .get(name)
            .or_else(|| target.and_then(|schema| schema.get(name)))
    };

    // A string `enum` constrains the raw body to one of the literals.
    if let Some(values) = keyword("enum").and_then(Value::as_array) {
        if !values.is_empty() && values.iter().all(Value::is_string) {
            let options: Vec<Value> = values
                .iter()
                .filter_map(Value::as_str)
                .map(|v| json!({ "type": "const_string", "value": v }))
                .collect();
            return ArgShape::Fixed {
                type_attr: "string",
                value: json!({ "type": "or", "elements": options }),
            };
        }
    }

    match keyword("type").and_then(Value::as_str) {
        // Verbatim body -> json_schema (which expects quotes) cannot be used.
        Some("string") => ArgShape::Fixed {
            type_attr: "string",
            value: json!({ "type": "any_text", "excludes": [CLOSE, OPEN, SEP] }),
        },
        Some("integer") | Some("number") => ArgShape::Fixed {
            type_attr: "number",
            value: json!({ "type": "json_schema", "json_schema": defs.attach(subschema) }),
        },
        Some("boolean") => ArgShape::Fixed {
            type_attr: "boolean",
            value: json!({ "type": "json_schema", "json_schema": { "type": "boolean" } }),
        },
        Some("object") => ArgShape::Fixed {
            type_attr: "object",
            value: json!({ "type": "json_schema", "json_schema": defs.attach(subschema) }),
        },
        Some("array") => ArgShape::Fixed {
            type_attr: "array",
            value: json!({ "type": "json_schema", "json_schema": defs.attach(subschema) }),
        },
        Some("null") => ArgShape::Fixed {
            type_attr: "null",
            value: json!({ "type": "const_string", "value": "null" }),
        },
        _ => ArgShape::Permissive,
    }
}

/// Permissive argument region: zero or more structurally-valid argument blocks
/// with unconstrained keys/types/values. Used when the schema is unusable.
fn permissive_args_format() -> Value {
    json!({ "type": "star", "content": permissive_argument_format() })
}

/// One structurally-valid argument block with unconstrained key/type/value.
///
/// The two exclusion conventions below are deliberate, not an oversight:
/// quote-delimited attribute tokens (`key=`, `type=`) exclude the `<|` marker
/// *prefix*, which blocks any partial or full control marker inside a short
/// identifier; the free-text value is marker-delimited and may legitimately
/// contain a lone `<` or `|`, so it excludes only the complete `OPEN`/`CLOSE`/
/// `SEP` markers to avoid over-restricting real content.
fn permissive_argument_format() -> Value {
    json!({
        "type": "sequence",
        "elements": [
            { "type": "const_string", "value": format!("{OPEN}argument key=\"") },
            { "type": "any_text", "excludes": ["\"", "<|"] },
            { "type": "const_string", "value": "\" type=\"" },
            { "type": "any_text", "excludes": ["\"", "<|"] },
            { "type": "const_string", "value": format!("\"{SEP}") },
            { "type": "any_text", "excludes": [CLOSE, OPEN, SEP] },
            { "type": "const_string", "value": format!("{CLOSE}argument{SEP}") },
        ]
    })
}

/// Permissive call block used only when no declared tool name is usable.
fn permissive_call_format() -> Value {
    json!({
        "type": "sequence",
        "elements": [
            { "type": "const_string", "value": format!("{OPEN}call tool=\"") },
            { "type": "any_text", "excludes": ["\"", "<|"] },
            { "type": "const_string", "value": "\" index=\"" },
            { "type": "regex", "pattern": "[1-9][0-9]*" },
            { "type": "const_string", "value": format!("\"{SEP}") },
            permissive_args_format(),
            { "type": "const_string", "value": format!("{CLOSE}call{SEP}") },
        ]
    })
}

/// Length of the longest prefix of `tag` that `text` ends with (up to
/// `tag.len() - 1`, since a full match would have been caught by the marker
/// regexes already). Used to hold back a possibly-still-growing partial
/// marker at the tail of the streamed text instead of leaking it as content.
fn partial_tag_overlap(text: &str, tag: &str) -> usize {
    let max_len = text.len().min(tag.len().saturating_sub(1));
    for n in (1..=max_len).rev() {
        if text.ends_with(&tag[..n]) {
            return n;
        }
    }
    0
}

impl Default for KimiK3Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for KimiK3Parser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        let Some(m_open) = self.tools_open_re.find(output) else {
            let content = self.content(output, output);
            return Ok((content.unwrap_or_default(), vec![]));
        };

        let before = &output[..m_open.start()];
        let start = m_open.end();
        let section_end = self
            .tools_close_re
            .find_at(output, start)
            .map_or(output.len(), |m| m.start());
        let section = &output[start..section_end];

        let calls = self
            .decode_calls_in_section(section)
            .into_iter()
            .map(|decoded| ToolCall {
                function: FunctionCall {
                    name: decoded.name,
                    arguments: decoded.arguments,
                },
            })
            .collect();

        let content = self.content(output, before);
        Ok((content.unwrap_or_default(), calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = self.buffer.clone();

        let content = self.extract_response_content(&current_text);

        let Some(m_tools) = self.tools_open_re.find(&current_text) else {
            return Ok(StreamingParseResult {
                normal_text: content.unwrap_or_default(),
                calls: vec![],
            });
        };

        let section = &current_text[m_tools.end()..];
        let decoded_calls = self.decode_calls_in_section(section);

        if decoded_calls.len() <= self.sent_tool_call_count {
            return Ok(StreamingParseResult {
                normal_text: content.unwrap_or_default(),
                calls: vec![],
            });
        }

        let calls = decoded_calls
            .iter()
            .skip(self.sent_tool_call_count)
            .enumerate()
            .map(|(i, decoded)| ToolCallItem {
                // Per-message zero-based ordinal in emission order; the XTML
                // `index` attribute is deliberately ignored (see module docs).
                tool_index: self.sent_tool_call_count + i,
                name: Some(decoded.name.clone()),
                parameters: decoded.arguments.clone(),
            })
            .collect();
        self.sent_tool_call_count = decoded_calls.len();

        Ok(StreamingParseResult {
            normal_text: content.unwrap_or_default(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        self.tools_open_re.is_match(text)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sent_content_idx = 0;
        self.sent_tool_call_count = 0;
    }
}

#[cfg(test)]
#[path = "../../tests/unit/parsers/kimi_k3/tests.rs"]
mod tests;
