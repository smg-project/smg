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
    /// Strings that only ever open K3's native tool-call syntax; banning them
    /// makes tool calls unreachable when `tool_choice` is `"none"`. ONLY the
    /// tools-section opener qualifies: the structural-tag triggers also list
    /// `<|close|>think<|sep|>` / `<|close|>response<|sep|>` (as tag-begin
    /// prefixes), but those close sections ordinary generation must emit, so
    /// excluding them would corrupt normal output.
    pub const TOOL_CALL_BAN_STRINGS: &'static [&'static str] = &[TOOLS_OPEN];

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
    /// So emit one tag per prefix in [`TOOLS_SECTION_PREFIXES`] and register each
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
            .map(|tool| build_call_format(&tool.function.name, &tool.function.parameters))
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

    let elements: Vec<Value> = properties
        .iter()
        .map(|(key, subschema)| {
            let argument = build_argument_format(key, subschema);
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
fn build_argument_format(key: &str, subschema: &Value) -> Value {
    let esc_key = escape_attr(key);
    match classify_arg(subschema) {
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

/// Map a property schema to its XTML `type=` attribute and value grammar.
/// K3 emits string bodies verbatim (no JSON quoting) while every other type is
/// compact JSON, and it derives `type=` from the *value*, so `integer` becomes
/// `number`.
fn classify_arg(subschema: &Value) -> ArgShape {
    // A string `enum` constrains the raw body to one of the literals.
    if let Some(values) = subschema.get("enum").and_then(Value::as_array) {
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

    match subschema.get("type").and_then(Value::as_str) {
        // Verbatim body -> json_schema (which expects quotes) cannot be used.
        Some("string") => ArgShape::Fixed {
            type_attr: "string",
            value: json!({ "type": "any_text", "excludes": [CLOSE, OPEN, SEP] }),
        },
        Some("integer") | Some("number") => ArgShape::Fixed {
            type_attr: "number",
            value: json!({ "type": "json_schema", "json_schema": subschema }),
        },
        Some("boolean") => ArgShape::Fixed {
            type_attr: "boolean",
            value: json!({ "type": "json_schema", "json_schema": { "type": "boolean" } }),
        },
        Some("object") => ArgShape::Fixed {
            type_attr: "object",
            value: json!({ "type": "json_schema", "json_schema": subschema }),
        },
        Some("array") => ArgShape::Fixed {
            type_attr: "array",
            value: json!({ "type": "json_schema", "json_schema": subschema }),
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
mod tests {
    use serde_json::Value;

    use super::*;

    fn _arg(key: &str, typ: &str, value: &str) -> String {
        format!(r#"{OPEN}argument key="{key}" type="{typ}"{SEP}{value}{CLOSE}argument{SEP}"#)
    }

    fn _call(tool: &str, index: i32, args: &[String]) -> String {
        let body: String = args.concat();
        format!(r#"{OPEN}call tool="{tool}" index="{index}"{SEP}{body}{CLOSE}call{SEP}"#)
    }

    fn _response(content: &str) -> String {
        format!("{OPEN}response{SEP}{content}{CLOSE}response{SEP}")
    }

    fn _tools(calls: &[String]) -> String {
        let body: String = calls.concat();
        format!("{OPEN}tools{SEP}{body}{CLOSE}tools{SEP}")
    }

    #[tokio::test]
    async fn test_parse_complete_response_and_typed_arguments() {
        let parser = KimiK3Parser::new();
        let input = _response("answer")
            + &_tools(&[_call(
                "calc",
                1,
                &[
                    _arg("x", "number", "1"),
                    _arg("flag", "boolean", "true"),
                    _arg("text", "string", "raw"),
                ],
            )]);

        let (content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(content, "answer");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args,
            serde_json::json!({"x": 1, "flag": true, "text": "raw"})
        );
    }

    #[tokio::test]
    async fn test_parse_complete_unescapes_attributes() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[_call(
            "a&amp;b&quot;c",
            1,
            &[_arg("k&amp;q", "string", "v")],
        )]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "a&b\"c");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"k&q": "v"}));
    }

    #[tokio::test]
    async fn test_parse_complete_allows_less_than_in_attributes() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[_call("calc<beta", 1, &[_arg("foo<bar", "string", "raw")])]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc<beta");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"foo<bar": "raw"}));
    }

    #[tokio::test]
    async fn test_parse_complete_no_tools_channel() {
        let parser = KimiK3Parser::new();
        let input = _response("hi");

        let (content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(content, "hi");
        assert_eq!(calls.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_complete_whitespace_degraded_markers() {
        let parser = KimiK3Parser::new();
        let input = format!("{OPEN} response {SEP}answer{CLOSE} response {SEP}");

        let (content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(content, "answer");
        assert_eq!(calls.len(), 0);
    }

    #[tokio::test]
    async fn test_parse_complete_malformed_json_argument_falls_back_to_raw() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[_call("calc", 1, &[_arg("x", "number", "not-json")])]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"x": "not-json"}));
    }

    #[tokio::test]
    async fn test_parse_complete_multiple_tool_calls() {
        let parser = KimiK3Parser::new();
        let input = _tools(&[
            _call("first", 1, &[_arg("a", "number", "1")]),
            _call("second", 2, &[_arg("b", "number", "2")]),
        ]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "first");
        assert_eq!(calls[1].function.name, "second");
    }

    #[tokio::test]
    async fn test_decode_call_without_index_attribute() {
        let parser = KimiK3Parser::new();
        let input = format!(
            r#"{OPEN}tools{SEP}{OPEN}call tool="calc"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}"#
        );

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "calc");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn test_has_tool_markers() {
        let parser = KimiK3Parser::new();
        assert!(parser.has_tool_markers(&_tools(&[])));
        assert!(!parser.has_tool_markers(&_response("hi")));
        assert!(!parser.has_tool_markers("plain text"));
    }

    #[tokio::test]
    async fn test_streaming_split_markers_do_not_leak() {
        let mut parser = KimiK3Parser::new();
        let hi_chunk = format!("{SEP}Hi");
        let call_open_chunk = format!(r#"{OPEN}call tool="calc" index="1"{SEP}"#);
        let arg_chunk = _arg("x", "number", "1");
        let close_call_chunk = format!("{CLOSE}call");

        let chunks: [&str; 10] = [
            OPEN,
            "response",
            hi_chunk.as_str(),
            OPEN,
            "tools",
            SEP,
            call_open_chunk.as_str(),
            arg_chunk.as_str(),
            close_call_chunk.as_str(),
            SEP,
        ];

        let mut content = String::new();
        let mut calls = Vec::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &[]).await.unwrap();
            content.push_str(&result.normal_text);
            calls.extend(result.calls);
        }

        assert_eq!(content, "Hi");
        assert!(!content.contains(OPEN));
        assert!(!content.contains(SEP));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("calc"));
        let args: Value = serde_json::from_str(&calls[0].parameters).unwrap();
        assert_eq!(args, serde_json::json!({"x": 1}));
    }

    #[tokio::test]
    async fn test_streaming_consumed_response_prefix_no_call_keeps_content() {
        let mut parser = KimiK3Parser::new();
        let close_response_chunk = format!("response{SEP}");
        let chunks: [&str; 4] = ["O", "K", CLOSE, close_response_chunk.as_str()];

        let mut content = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &[]).await.unwrap();
            assert!(!result.normal_text.contains(CLOSE));
            content.push_str(&result.normal_text);
        }
        assert_eq!(content, "OK");
    }

    #[tokio::test]
    async fn test_streaming_multiple_tool_calls_increment_tool_index() {
        let mut parser = KimiK3Parser::new();
        let input = _tools(&[
            _call("first", 1, &[_arg("a", "number", "1")]),
            _call("second", 2, &[_arg("b", "number", "2")]),
        ]);

        let result = parser.parse_incremental(&input, &[]).await.unwrap();
        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
    }

    #[tokio::test]
    async fn test_streaming_duplicate_xtml_index_uses_ordinal() {
        // A model that emits colliding, sparse, or out-of-order `index`
        // attributes must still yield unique, monotonic streamed indices: the
        // parser assigns them by emission order and ignores the attribute.
        let mut parser = KimiK3Parser::new();
        let input = _tools(&[
            _call("first", 7, &[_arg("a", "number", "1")]),
            _call("second", 7, &[_arg("b", "number", "2")]),
            _call("third", 3, &[_arg("c", "number", "3")]),
        ]);

        let result = parser.parse_incremental(&input, &[]).await.unwrap();
        assert_eq!(result.calls.len(), 3);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
        assert_eq!(result.calls[2].tool_index, 2);
    }

    #[tokio::test]
    async fn test_reset_clears_streaming_state() {
        let mut parser = KimiK3Parser::new();

        let primed = format!("{OPEN}response{SEP}Hello{CLOSE}response{SEP}")
            + &_tools(&[_call("calc", 1, &[_arg("x", "number", "1")])]);
        let primed_result = parser.parse_incremental(&primed, &[]).await.unwrap();
        assert_eq!(primed_result.calls.len(), 1);

        parser.reset();

        let fresh_content = format!("{OPEN}response{SEP}Fresh{CLOSE}response{SEP}");
        let content_result = parser.parse_incremental(&fresh_content, &[]).await.unwrap();
        assert_eq!(content_result.normal_text, "Fresh");

        let fresh_call = _tools(&[_call("calc", 1, &[_arg("x", "number", "2")])]);
        let call_result = parser.parse_incremental(&fresh_call, &[]).await.unwrap();
        assert_eq!(call_result.calls.len(), 1);
        assert_eq!(call_result.calls[0].name.as_deref(), Some("calc"));
    }

    #[test]
    fn test_factory_resolves_moonshot_kimi_k3() {
        let factory = crate::factory::ParserFactory::new();
        assert_eq!(
            factory
                .registry()
                .resolve_model_to_parser("moonshotai/Kimi-K3"),
            Some("kimi_k3".to_string())
        );
    }

    #[test]
    fn test_factory_resolves_underscore_kimi_k3() {
        let factory = crate::factory::ParserFactory::new();
        for id in ["kimi_k3", "Kimi_K3", "moonshotai/Kimi_K3"] {
            assert_eq!(
                factory.registry().resolve_model_to_parser(id),
                Some("kimi_k3".to_string()),
                "expected `{id}` to resolve to the kimi_k3 parser"
            );
        }
    }

    #[test]
    fn test_factory_registers_kimi_k3_structural_tag() {
        let factory = crate::factory::ParserFactory::new();
        assert!(factory.registry().has_structural_tag("kimi_k3"));
    }

    fn sample_tools() -> Vec<Tool> {
        serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "units": {"type": "string", "enum": ["c", "f"]}
                    },
                    "required": ["city", "days"]
                }
            }
        }]))
        .unwrap()
    }

    #[test]
    fn test_build_structural_tag_frames_tools_section() {
        let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
        let format = &tag["format"];
        assert_eq!(format["type"], "triggered_tags");
        assert_eq!(format["triggers"][0], TOOLS_OPEN);
        assert_eq!(format["at_least_one"], true);
        // `stop_after_first` is intentionally unset (mirrors K2) so trailing
        // `<|close|>message<|sep|><|end_of_msg|>` tokens stay valid.
        assert!(format.get("stop_after_first").is_none());

        let section = &format["tags"][0];
        assert_eq!(section["begin"], TOOLS_OPEN);
        assert_eq!(section["end"], TOOLS_CLOSE);
        // A tools section is one or more call blocks.
        assert_eq!(section["content"]["type"], "plus");
    }

    #[test]
    fn test_build_structural_tag_allows_closing_prefilled_channel() {
        // Regression: the prompt leaves `think` or `response` open, and
        // `at_least_one` admits only trigger-prefixed tokens. Each closing
        // marker therefore needs both a trigger and a matching tag, or the
        // model can never close the channel it was left in.
        let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
        let format = &tag["format"];

        let triggers: Vec<&str> = format["triggers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        assert!(triggers.contains(&THINK_CLOSE), "got: {triggers:?}");
        assert!(triggers.contains(&RESPONSE_CLOSE), "got: {triggers:?}");
        assert!(triggers.contains(&TOOLS_OPEN), "got: {triggers:?}");

        let begins: Vec<&str> = format["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["begin"].as_str().unwrap())
            .collect();
        assert_eq!(
            begins,
            vec![
                TOOLS_OPEN,
                &format!("{THINK_CLOSE}{TOOLS_OPEN}")[..],
                &format!("{RESPONSE_CLOSE}{TOOLS_OPEN}")[..],
            ]
        );

        // Every variant frames an identical tools section.
        for section in format["tags"].as_array().unwrap() {
            assert_eq!(section["end"], TOOLS_CLOSE);
            assert_eq!(section["content"]["type"], "plus");
        }
    }

    #[tokio::test]
    async fn test_parse_complete_handles_think_close_prefixed_tools_section() {
        // What the grammar now produces under thinking + forced tools: close
        // the prefilled think channel, then open tools.
        let parser = KimiK3Parser::new();
        let input = THINK_CLOSE.to_string()
            + &_tools(&[_call("get_weather", 1, &[_arg("city", "string", "SF")])]);

        let (_content, calls) = parser.parse_complete(&input).await.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"city": "SF"}));
    }

    #[test]
    fn test_build_structural_tag_pins_call_framing_and_tool_name() {
        let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
        let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
        assert_eq!(call["type"], "sequence");
        let elems = call["elements"].as_array().unwrap();
        assert_eq!(
            elems[0]["value"],
            "<|open|>call tool=\"get_weather\" index=\""
        );
        assert_eq!(elems[1]["type"], "regex");
        assert_eq!(elems[2]["value"], "\"<|sep|>");
        assert_eq!(elems[4]["value"], "<|close|>call<|sep|>");
    }

    #[test]
    fn test_build_structural_tag_strict_arguments_by_type() {
        let tag = KimiK3Parser::build_structural_tag(&sample_tools(), true);
        let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
        let args = &call["elements"][3];
        assert_eq!(args["type"], "sequence");
        let arg_elems = args["elements"].as_array().unwrap();
        // city (required, string), days (required, number), units (optional, enum).
        assert_eq!(arg_elems.len(), 3);

        // Required string: bare sequence, verbatim body via any_text.
        assert_eq!(arg_elems[0]["type"], "sequence");
        assert_eq!(
            arg_elems[0]["elements"][0]["value"],
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>"
        );
        assert_eq!(arg_elems[0]["elements"][1]["type"], "any_text");

        // Required integer: rendered as type="number", JSON-schema-constrained value.
        assert_eq!(
            arg_elems[1]["elements"][0]["value"],
            "<|open|>argument key=\"days\" type=\"number\"<|sep|>"
        );
        assert_eq!(arg_elems[1]["elements"][1]["type"], "json_schema");

        // Optional string enum: wrapped in `optional`, value is an `or` of literals.
        assert_eq!(arg_elems[2]["type"], "optional");
        let units = &arg_elems[2]["content"];
        assert_eq!(
            units["elements"][0]["value"],
            "<|open|>argument key=\"units\" type=\"string\"<|sep|>"
        );
        assert_eq!(units["elements"][1]["type"], "or");
        assert_eq!(units["elements"][1]["elements"][0]["value"], "c");
        assert_eq!(units["elements"][1]["elements"][1]["value"], "f");
    }

    #[test]
    fn test_build_structural_tag_escapes_attribute_values() {
        let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "a&b\"c",
                "parameters": {
                    "type": "object",
                    "properties": {"k\"q": {"type": "string"}},
                    "required": ["k\"q"]
                }
            }
        }]))
        .unwrap();
        let tag = KimiK3Parser::build_structural_tag(&tools, true);
        let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
        assert_eq!(
            call["elements"][0]["value"],
            "<|open|>call tool=\"a&amp;b&quot;c\" index=\""
        );
        let arg = &call["elements"][3]["elements"][0];
        assert_eq!(
            arg["elements"][0]["value"],
            "<|open|>argument key=\"k&quot;q\" type=\"string\"<|sep|>"
        );
    }

    #[test]
    fn test_build_structural_tag_falls_back_when_no_properties() {
        let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object"}}
        }]))
        .unwrap();
        let tag = KimiK3Parser::build_structural_tag(&tools, false);
        assert_eq!(tag["format"]["at_least_one"], false);
        let call = &tag["format"]["tags"][0]["content"]["content"]["elements"][0];
        // Argument region degrades to a permissive `star` of argument blocks.
        assert_eq!(call["elements"][3]["type"], "star");
    }
}
