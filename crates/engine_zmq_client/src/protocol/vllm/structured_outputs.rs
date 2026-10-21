// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/structured_outputs.rs.

use std::{borrow::Cow, fmt::Write as _, sync::OnceLock};

use memchr::memmem;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::error::{Error, Result};

/// Structured-output backend selected for EngineCore grammar compilation.
///
/// Python stores this in `StructuredOutputsParams._backend` after request
/// validation and the engine uses it as is. The engine's structured-output
/// manager builds one backend per process, from the first structured request
/// it sees, and compiles every later request on that one whatever its own
/// `_backend` says. A sender that knows the engine's configured backend pins
/// it on every request; otherwise this client stamps the default, xgrammar,
/// on every constraint: a structural tag in the triggered-tags format
/// compiles on xgrammar alone (guidance's parser knows only the legacy
/// structures/triggers shape), so an engine that may receive one must settle
/// on xgrammar whichever constraint reaches it first. xgrammar compiles a
/// JSON schema with features it lacks with those keywords dropped (an answer
/// off the schema, not an error), so a sender that stamps it refuses such a
/// schema before it reaches the engine, as vLLM's frontend does under
/// `backend=xgrammar` ([`has_xgrammar_unsupported_json_features`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StructuredOutputBackend {
    #[default]
    Xgrammar,
    Guidance,
    Outlines,
    LmFormatEnforcer,
}

/// The single structured-output constraint selected for a request.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredOutputConstraint {
    /// JSON schema (as a dict/object or JSON string) constraining the output.
    Json(Value),
    /// Regular expression the output must match.
    Regex(String),
    /// List of allowed output strings (the model must produce one of these).
    Choice(Vec<String>),
    /// Context-free grammar (in EBNF-like notation) the output must conform to.
    Grammar(String),
    /// Output must be valid JSON (free-form, no schema).
    JsonObject,
    /// Structural tag configuration (JSON-encoded string).
    StructuralTag(String),
}

/// Additional structured-output options that do not select the constraint mode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StructuredOutputOptions {
    /// Disable any additional whitespace in guided JSON output.
    pub disable_any_whitespace: bool,
    /// Disable `additionalProperties` in JSON schema output.
    pub disable_additional_properties: bool,
    /// Custom whitespace pattern for guided JSON output.
    pub whitespace_pattern: Option<String>,
}

/// Parameters for configuring structured outputs (guided decoding).
///
/// This is the semantic Rust representation: exactly one constraint mode is
/// always selected. The Python/msgpack product-shaped representation is kept in
/// the private wire type below and used only at serde boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredOutputsParams {
    pub constraint: StructuredOutputConstraint,
    pub options: StructuredOutputOptions,
    /// Structured-output backend, mirroring Python's internal `_backend`: the
    /// one the sender pinned, else the default ([`StructuredOutputBackend`]).
    pub backend: StructuredOutputBackend,
}

impl StructuredOutputsParams {
    pub fn json(json: Value) -> Self {
        Self::from_constraint(StructuredOutputConstraint::Json(json))
    }

    pub fn regex(regex: impl Into<String>) -> Self {
        Self::from_constraint(StructuredOutputConstraint::Regex(regex.into()))
    }

    pub fn choice(choice: Vec<String>) -> Self {
        Self::from_constraint(StructuredOutputConstraint::Choice(choice))
    }

    pub fn grammar(grammar: impl Into<String>) -> Self {
        Self::from_constraint(StructuredOutputConstraint::Grammar(grammar.into()))
    }

    pub fn json_object() -> Self {
        Self::from_constraint(StructuredOutputConstraint::JsonObject)
    }

    pub fn structural_tag(structural_tag: impl Into<String>) -> Self {
        Self::from_constraint(StructuredOutputConstraint::StructuralTag(
            structural_tag.into(),
        ))
    }

    /// The constraint under the default backend. A choice stays a choice
    /// here: the sender pins the engine's backend through
    /// [`with_backend`](Self::with_backend) before it sends, which lowers a
    /// choice to the grammar xgrammar compiles and keeps it for a backend
    /// that serves choices itself (guidance); lowering here would hand that
    /// backend a grammar it does not parse.
    fn from_constraint(constraint: StructuredOutputConstraint) -> Self {
        Self {
            constraint,
            options: StructuredOutputOptions::default(),
            backend: StructuredOutputBackend::default(),
        }
    }

    /// Pin `backend` the way vLLM's frontend stamps `_backend` after
    /// validation: a `choice` headed for xgrammar becomes the equivalent
    /// grammar (`validate_xgrammar_grammar` rewrites it; the engine's xgrammar
    /// backend compiles no choice of its own).
    pub fn with_backend(mut self, backend: StructuredOutputBackend) -> Self {
        if backend == StructuredOutputBackend::Xgrammar {
            if let StructuredOutputConstraint::Choice(choices) = &self.constraint {
                self.constraint = StructuredOutputConstraint::Grammar(choice_as_grammar(choices));
            }
        }
        self.backend = backend;
        self
    }
}

/// vLLM's `choice_as_grammar` (`v1/structured_output/utils.py`): the EBNF the
/// frontend substitutes for a `choice` constraint xgrammar is to compile.
pub fn choice_as_grammar(choices: &[String]) -> String {
    fn escape(choice: &str) -> String {
        let mut out = String::with_capacity(choice.len());
        for ch in choice.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                // The remaining C0 controls and DEL.
                ch if (ch as u32) < 0x20 || ch as u32 == 0x7F => {
                    out.push_str(&format!("\\u{:04x}", ch as u32));
                }
                ch => out.push(ch),
            }
        }
        out
    }
    let alternatives: Vec<String> = choices
        .iter()
        .map(|choice| format!("\"{}\"", escape(choice)))
        .collect();
    format!("root ::= {}", alternatives.join(" | "))
}

/// A JSON-schema feature xgrammar's converter drops instead of refusing, as
/// [`xgrammar_unsupported_json_feature`] finds it: the keyword (or the
/// keywords that conflict; a `format` with its value), and the JSON pointer
/// (RFC 6901) of the schema object that carries it, from the schema the check
/// was given (empty at its root). For a feature inside a structural tag the
/// pointer runs from the tag's root, and `openers` holds the `begin` text of
/// each tag around the schema, outermost first: a call's tag opens with the
/// tool's name in the template's syntax, so they name the tool; empty for a
/// schema outside any tag (a JSON answer's)
/// ([`xgrammar_unsupported_feature_in_structural_tag`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XgrammarUnsupportedFeature {
    pub keyword: Cow<'static, str>,
    pub pointer: String,
    pub openers: Vec<String>,
}

/// vLLM's `has_xgrammar_unsupported_json_features`
/// (`v1/structured_output/backend_xgrammar.py`), keyword for keyword: the
/// schema features xgrammar's converter drops instead of refusing. vLLM's
/// frontend refuses such a schema with a 400 under `backend=xgrammar`
/// (`validate_xgrammar_grammar`); a sender stamping xgrammar does the same. A
/// schema given as a string is parsed as the frontend parses it; one that is
/// not JSON is left to xgrammar, which refuses it at compile.
pub fn has_xgrammar_unsupported_json_features(schema: &Value) -> bool {
    xgrammar_unsupported_json_feature(schema).is_some()
}

/// The first feature of `schema` xgrammar cannot enforce, in the order of
/// vLLM's rules ([`has_xgrammar_unsupported_json_features`]), with where it
/// stands; `None` for a schema xgrammar compiles whole.
pub fn xgrammar_unsupported_json_feature(schema: &Value) -> Option<XgrammarUnsupportedFeature> {
    let parsed: Value;
    let schema = match schema {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(value) => {
                parsed = value;
                &parsed
            }
            Err(_) => return None,
        },
        other => other,
    };
    let mut pointer = String::new();
    let keyword = unsupported_in(schema, &mut pointer)?;
    Some(XgrammarUnsupportedFeature {
        keyword,
        pointer,
        openers: Vec::new(),
    })
}

/// [`xgrammar_unsupported_json_feature`] for a schema as its text: the text
/// is scanned for the rules' keywords first
/// ([`text_names_an_xgrammar_unsupported_keyword`]) and parsed only when it
/// names one, so the common schema costs the scan alone.
pub fn xgrammar_unsupported_feature_in_schema_text(
    schema: &str,
) -> Option<XgrammarUnsupportedFeature> {
    if !text_names_an_xgrammar_unsupported_keyword(schema) {
        return None;
    }
    let schema: Value = serde_json::from_str(schema).ok()?;
    xgrammar_unsupported_json_feature(&schema)
}

/// The first feature xgrammar cannot enforce among the `json_schema` elements
/// of a structural tag (xgrammar compiles each with the converter a
/// JSON-schema constraint goes through, the same keywords dropped), with the
/// pointer from the tag's root
/// (`/format/tags/1/content/json_schema/properties/tags`) and the tags around
/// it by their `begin`; `None` for a tag that carries none, and for one that
/// is not JSON, which xgrammar refuses at compile. The tag's text is scanned
/// for the rules' keywords first
/// ([`text_names_an_xgrammar_unsupported_keyword`]): a tag that names none
/// carries none and is not parsed, so a forced call pays the scan alone for
/// the common tag.
pub fn xgrammar_unsupported_feature_in_structural_tag(
    tag: &str,
) -> Option<XgrammarUnsupportedFeature> {
    if !text_names_an_xgrammar_unsupported_keyword(tag) {
        return None;
    }
    let tag: Value = serde_json::from_str(tag).ok()?;
    let mut pointer = String::new();
    let mut openers = Vec::new();
    let keyword = unsupported_in_tag(&tag, &mut pointer, &mut openers)?;
    Some(XgrammarUnsupportedFeature {
        keyword,
        pointer,
        openers,
    })
}

/// The string formats xgrammar's converter knows; any other is dropped.
const STRING_SUPPORTED_FORMATS: [&str; 14] = [
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
];

/// The rules on one schema object, then on its nested schemas (an object
/// value, or the objects of an array value), as vLLM's check recurses.
/// `pointer` holds the path down to `value` and is cut back on the way up, so
/// it reads the hit's path when one returns.
fn unsupported_in(value: &Value, pointer: &mut String) -> Option<Cow<'static, str>> {
    let Value::Object(obj) = value else {
        return None;
    };
    if let Some(keyword) = unsupported_keyword_of(obj) {
        return Some(keyword);
    }
    for (key, nested) in obj {
        match nested {
            Value::Object(_) => {
                let len = pointer.len();
                push_segment(pointer, key);
                if let Some(keyword) = unsupported_in(nested, pointer) {
                    return Some(keyword);
                }
                pointer.truncate(len);
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    if !item.is_object() {
                        continue;
                    }
                    let len = pointer.len();
                    push_segment(pointer, key);
                    let _ = write!(pointer, "/{index}");
                    if let Some(keyword) = unsupported_in(item, pointer) {
                        return Some(keyword);
                    }
                    pointer.truncate(len);
                }
            }
            _ => {}
        }
    }
    None
}

/// The rule one schema object breaks, as its keyword (a `format` with the
/// value xgrammar does not know): vLLM's checks in their order, or `None`.
fn unsupported_keyword_of(obj: &serde_json::Map<String, Value>) -> Option<Cow<'static, str>> {
    let kind = obj.get("type");
    let has = |wanted: &str| match kind {
        Some(Value::String(one)) => one == wanted,
        Some(Value::Array(items)) => items.iter().any(|item| item.as_str() == Some(wanted)),
        _ => false,
    };
    if (has("integer") || has("number")) && obj.contains_key("multipleOf") {
        return Some("multipleOf".into());
    }
    if has("array") {
        for key in ["uniqueItems", "contains", "minContains", "maxContains"] {
            if obj.contains_key(key) {
                return Some(key.into());
            }
        }
    }
    if has("string") {
        if let Some(format) = obj.get("format") {
            if !format
                .as_str()
                .is_some_and(|format| STRING_SUPPORTED_FORMATS.contains(&format))
            {
                return Some(format!("format: {format}").into());
            }
        }
        // xgrammar drops minLength/maxLength next to a pattern or format.
        if has_pattern_and_length_bounds(obj) {
            return Some("minLength/maxLength beside pattern or format".into());
        }
    }
    if has("object") {
        if obj
            .get("propertyNames")
            .and_then(Value::as_object)
            .is_some_and(has_pattern_and_length_bounds)
        {
            return Some("propertyNames with minLength/maxLength beside pattern or format".into());
        }
        // propertyNames conflicts with the other property keywords.
        if obj.contains_key("propertyNames")
            && (obj.contains_key("properties")
                || obj.contains_key("patternProperties")
                || obj
                    .get("additionalProperties")
                    .is_some_and(Value::is_object)
                || obj
                    .get("unevaluatedProperties")
                    .is_some_and(|value| *value != Value::Bool(true)))
        {
            return Some("propertyNames beside other property keywords".into());
        }
        // Several patternProperties, or one next to properties, conflict.
        if obj
            .get("patternProperties")
            .and_then(Value::as_object)
            .is_some_and(|patterns| obj.contains_key("properties") || patterns.len() > 1)
        {
            return Some("patternProperties beside properties, or more than one".into());
        }
    }
    None
}

fn has_pattern_and_length_bounds(obj: &serde_json::Map<String, Value>) -> bool {
    (obj.contains_key("pattern") || obj.contains_key("format"))
        && (obj.contains_key("minLength") || obj.contains_key("maxLength"))
}

/// [`unsupported_in`] at every `json_schema` element under `value` (an object
/// with `"type": "json_schema"`, its schema under `json_schema`), the tag's
/// other objects and arrays walked for them; `pointer` as there, and
/// `openers` the `begin` of each tag on the way down (an object with one),
/// taken back on the way up, so it reads the tags around the hit when one
/// returns.
fn unsupported_in_tag(
    value: &Value,
    pointer: &mut String,
    openers: &mut Vec<String>,
) -> Option<Cow<'static, str>> {
    match value {
        Value::Object(obj) => {
            if obj.get("type").and_then(Value::as_str) == Some("json_schema") {
                let schema = obj.get("json_schema")?;
                push_segment(pointer, "json_schema");
                return unsupported_in(schema, pointer);
            }
            let opens = obj.get("begin").and_then(Value::as_str);
            if let Some(begin) = opens {
                openers.push(begin.to_string());
            }
            for (key, nested) in obj {
                let len = pointer.len();
                push_segment(pointer, key);
                if let Some(keyword) = unsupported_in_tag(nested, pointer, openers) {
                    return Some(keyword);
                }
                pointer.truncate(len);
            }
            if opens.is_some() {
                openers.pop();
            }
            None
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                let len = pointer.len();
                let _ = write!(pointer, "/{index}");
                if let Some(keyword) = unsupported_in_tag(item, pointer, openers) {
                    return Some(keyword);
                }
                pointer.truncate(len);
            }
            None
        }
        _ => None,
    }
}

/// `key` behind `/` as a JSON pointer segment, `~` and `/` escaped as RFC
/// 6901 has them.
fn push_segment(pointer: &mut String, key: &str) {
    pointer.push('/');
    if !key.contains(['~', '/']) {
        pointer.push_str(key);
        return;
    }
    for ch in key.chars() {
        match ch {
            '~' => pointer.push_str("~0"),
            '/' => pointer.push_str("~1"),
            other => pointer.push(other),
        }
    }
}

/// Whether `text`, a JSON schema or a structural tag as a JSON serializer
/// writes it, names a keyword one of the rules reads: `multipleOf`,
/// `uniqueItems`, `contains` (`minContains`, `maxContains` with it),
/// `minLength`/`maxLength` (a `pattern` matters only beside them),
/// `propertyNames`, `patternProperties`, or a `format` whose value is not an
/// object. A text that names none breaks no rule, so its caller skips the
/// parse; one that does is parsed and checked, a string that ends in
/// `Length` or `Contains` costing the parse and no more. The one exception: a
/// `format` with an object for its value is not looked for, since a
/// structural tag's own `format` key and a property named `format` hold
/// objects, so a string schema with an object for its `format` (no schema
/// writes one; the rule would flag it, as vLLM's does) is left to the engine.
fn text_names_an_xgrammar_unsupported_keyword(text: &str) -> bool {
    static KEYWORDS: OnceLock<Vec<memmem::Finder<'static>>> = OnceLock::new();
    static FORMAT: OnceLock<memmem::Finder<'static>> = OnceLock::new();
    let bytes = text.as_bytes();
    const NAMED: [&[u8]; 7] = [
        b"\"multipleOf\"",
        b"\"uniqueItems\"",
        b"\"contains\"",
        b"Contains\"",
        b"Length\"",
        b"\"propertyNames\"",
        b"\"patternProperties\"",
    ];
    let keywords = KEYWORDS.get_or_init(|| {
        NAMED
            .iter()
            .map(|name| memmem::Finder::new(*name))
            .collect()
    });
    if keywords.iter().any(|keyword| keyword.find(bytes).is_some()) {
        return true;
    }
    let format = FORMAT.get_or_init(|| memmem::Finder::new(b"\"format\""));
    let mut rest = bytes;
    while let Some(at) = format.find(rest) {
        rest = &rest[at + b"\"format\"".len()..];
        let Some(value) = json_whitespace_trimmed(rest).strip_prefix(b":") else {
            continue;
        };
        if !matches!(json_whitespace_trimmed(value).first(), None | Some(b'{')) {
            return true;
        }
    }
    false
}

/// `bytes` past its leading JSON whitespace.
fn json_whitespace_trimmed(bytes: &[u8]) -> &[u8] {
    let blank = bytes
        .iter()
        .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        .count();
    &bytes[blank..]
}

/// `true` when a boolean is `false`; used to drop default-`false` flags from the
/// serialized map to match the sparse `omit_defaults` wire shape. Takes `&bool`
/// because serde's `skip_serializing_if` requires a by-reference predicate.
#[expect(clippy::trivially_copy_pass_by_ref)]
fn is_false(v: &bool) -> bool {
    !*v
}

/// Wire-compatible structured-output payload used by Python engine-core.
///
/// Python models `StructuredOutputsParams` as a product-shaped dataclass with
/// several optional constraint fields, then validates that exactly one of those
/// fields is present. This client exposes [`StructuredOutputsParams`] as an
/// enum-backed domain type instead, while using this private wire type for
/// ser/de.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
struct WireStructuredOutputsParams {
    json: Option<Value>,
    regex: Option<String>,
    choice: Option<Vec<String>>,
    grammar: Option<String>,
    json_object: Option<bool>,
    disable_any_whitespace: bool,
    disable_additional_properties: bool,
    whitespace_pattern: Option<String>,
    structural_tag: Option<String>,
    /// The backend the sender pinned (vLLM's frontend sets it after
    /// validation; the engine reads it as is). Absent falls back to the
    /// default.
    #[serde(default, rename = "_backend")]
    backend: Option<StructuredOutputBackend>,
}

/// Borrowed send-side view of [`WireStructuredOutputsParams`]; keeps the wire
/// shape without cloning the constraint (JSON schemas can be large).
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct WireStructuredOutputsParamsRef<'a> {
    json: Option<&'a Value>,
    regex: Option<&'a str>,
    choice: Option<&'a [String]>,
    grammar: Option<&'a str>,
    json_object: Option<bool>,
    #[serde(skip_serializing_if = "is_false")]
    disable_any_whitespace: bool,
    #[serde(skip_serializing_if = "is_false")]
    disable_additional_properties: bool,
    whitespace_pattern: Option<&'a str>,
    structural_tag: Option<&'a str>,
    #[serde(rename = "_backend")]
    backend: StructuredOutputBackend,
}

impl TryFrom<WireStructuredOutputsParams> for StructuredOutputsParams {
    type Error = Error;

    fn try_from(raw: WireStructuredOutputsParams) -> Result<Self> {
        use StructuredOutputConstraint::*;

        let mut constraint = None;

        macro_rules! insert_constraint {
            ($name:literal, $value:expr) => {
                if let Some(value) = $value {
                    if let Some((existing, _)) = constraint {
                        return Err(Error::InvalidStructuredOutputsParams {
                            message: format!(
                                "multiple structured output constraints specified: {existing}, {}",
                                $name
                            ),
                        });
                    }
                    constraint = Some(($name, value));
                }
            };
        }

        insert_constraint!("json", raw.json.map(Json));
        insert_constraint!("regex", raw.regex.map(Regex));
        insert_constraint!("choice", raw.choice.map(Choice));
        insert_constraint!("grammar", raw.grammar.map(Grammar));
        match raw.json_object {
            Some(true) => {
                insert_constraint!("json_object", Some(JsonObject));
            }
            Some(false) => {
                return Err(Error::InvalidStructuredOutputsParams {
                    message: "structured_outputs.json_object must be true if set; omit structured_outputs to disable structured outputs".to_string(),
                });
            }
            None => {}
        }
        insert_constraint!("structural_tag", raw.structural_tag.map(StructuralTag));

        let constraint_ref =
            constraint
                .map(|(_, c)| c)
                .ok_or_else(|| Error::InvalidStructuredOutputsParams {
                    message: "missing structured output constraint".to_string(),
                })?;
        Ok(Self {
            options: StructuredOutputOptions {
                disable_any_whitespace: raw.disable_any_whitespace,
                disable_additional_properties: raw.disable_additional_properties,
                whitespace_pattern: raw.whitespace_pattern,
            },
            backend: raw.backend.unwrap_or_default(),
            constraint: constraint_ref,
        })
    }
}

impl<'a> From<&'a StructuredOutputsParams> for WireStructuredOutputsParamsRef<'a> {
    fn from(params: &'a StructuredOutputsParams) -> Self {
        let mut raw = Self {
            json: None,
            regex: None,
            choice: None,
            grammar: None,
            json_object: None,
            disable_any_whitespace: params.options.disable_any_whitespace,
            disable_additional_properties: params.options.disable_additional_properties,
            whitespace_pattern: params.options.whitespace_pattern.as_deref(),
            structural_tag: None,
            backend: params.backend,
        };

        match &params.constraint {
            StructuredOutputConstraint::Json(json) => raw.json = Some(json),
            StructuredOutputConstraint::Regex(regex) => raw.regex = Some(regex.as_str()),
            StructuredOutputConstraint::Choice(choice) => raw.choice = Some(choice.as_slice()),
            StructuredOutputConstraint::Grammar(grammar) => raw.grammar = Some(grammar.as_str()),
            StructuredOutputConstraint::JsonObject => raw.json_object = Some(true),
            StructuredOutputConstraint::StructuralTag(structural_tag) => {
                raw.structural_tag = Some(structural_tag.as_str());
            }
        }

        raw
    }
}

impl Serialize for StructuredOutputsParams {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        WireStructuredOutputsParamsRef::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StructuredOutputsParams {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireStructuredOutputsParams::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_outputs_backend_honours_a_pinned_value() {
        // A sender that knows the engine's backend pins it; the engine reads
        // `_backend` as is, so decoding keeps it and re-encoding repeats it.
        let raw = serde_json::json!({
            "json_object": true,
            "_backend": "guidance",
        });
        let params: StructuredOutputsParams = serde_json::from_value(raw).unwrap();
        assert_eq!(params.backend, StructuredOutputBackend::Guidance);
        assert_eq!(params.constraint, StructuredOutputConstraint::JsonObject);
        let value = serde_json::to_value(&params).unwrap();
        assert_eq!(value["_backend"], "guidance");

        // Unpinned, the default applies.
        let raw = serde_json::json!({ "json_object": true });
        let params: StructuredOutputsParams = serde_json::from_value(raw).unwrap();
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
    }

    /// Pinning xgrammar rewrites a choice into the grammar vLLM's frontend
    /// substitutes (the engine's xgrammar backend compiles no choice);
    /// guidance keeps the choice, which it compiles natively.
    #[test]
    fn with_backend_lowers_a_choice_for_xgrammar() {
        let choices = vec![
            "yes".to_string(),
            "no \"way\"\n".to_string(),
            "\u{1}".to_string(),
        ];
        let params = StructuredOutputsParams::choice(choices.clone())
            .with_backend(StructuredOutputBackend::Xgrammar);
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Grammar(
                "root ::= \"yes\" | \"no \\\"way\\\"\\n\" | \"\\u0001\"".to_string()
            )
        );
        let value = serde_json::to_value(&params).unwrap();
        assert!(value.get("choice").is_none());
        assert_eq!(value["_backend"], "xgrammar");

        let params = StructuredOutputsParams::choice(choices.clone())
            .with_backend(StructuredOutputBackend::Guidance);
        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Choice(choices)
        );
        assert_eq!(params.backend, StructuredOutputBackend::Guidance);

        // A Lark grammar passes to xgrammar unchanged (it parses Lark itself).
        let params = StructuredOutputsParams::grammar("start: \"a\" | \"b\"")
            .with_backend(StructuredOutputBackend::Xgrammar);
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Grammar("start: \"a\" | \"b\"".to_string())
        );
    }

    /// The engine keeps the backend of its first structured request for its
    /// life, and the triggered-tags format compiles on xgrammar alone: every
    /// constraint is stamped xgrammar unless the sender pins another, so the
    /// kind that reaches a fresh engine first never decides whether a later
    /// structural tag compiles.
    #[test]
    fn every_constraint_defaults_to_xgrammar() {
        for params in [
            StructuredOutputsParams::json(serde_json::json!({"type": "object"})),
            StructuredOutputsParams::json(serde_json::json!({"type": "integer", "multipleOf": 3})),
            StructuredOutputsParams::json_object(),
            StructuredOutputsParams::regex("[a-z]+"),
            StructuredOutputsParams::grammar("root ::= \"a\""),
            StructuredOutputsParams::choice(vec!["a".into()]),
            StructuredOutputsParams::structural_tag(r#"{"format":{}}"#),
        ] {
            assert_eq!(
                params.backend,
                StructuredOutputBackend::Xgrammar,
                "{params:?}"
            );
            let value = serde_json::to_value(&params).unwrap();
            assert_eq!(value["_backend"], "xgrammar", "{params:?}");
        }
        let value =
            serde_json::to_value(StructuredOutputsParams::structural_tag(r#"{"format":{}}"#))
                .unwrap();
        assert_eq!(value["structural_tag"], r#"{"format":{}}"#);
    }

    /// vLLM's static xgrammar feature check, the one a sender stamping
    /// xgrammar refuses a schema by: the keywords xgrammar would drop, nested
    /// and under a list-valued type as well; a plain object passes, and a
    /// schema string is parsed as the frontend parses it.
    #[test]
    fn xgrammar_unsupported_features_are_detected() {
        let lacks = |schema: Value| has_xgrammar_unsupported_json_features(&schema);
        assert!(!lacks(
            serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}})
        ));
        assert!(lacks(
            serde_json::json!({"type": "integer", "multipleOf": 5})
        ));
        // Nested, and under a list-valued type.
        assert!(lacks(
            serde_json::json!({"type": "object", "properties": {"tags": {
            "type": ["array", "null"], "uniqueItems": true}}})
        ));
        assert!(!lacks(
            serde_json::json!({"type": "string", "format": "email"})
        ));
        assert!(lacks(
            serde_json::json!({"type": "string", "format": "iri"})
        ));
        assert!(lacks(
            serde_json::json!({"type": "string", "pattern": "^a", "maxLength": 3})
        ));
        assert!(lacks(serde_json::json!({"type": "object",
            "propertyNames": {"pattern": "^x"}, "properties": {"x": {}}})));
        assert!(lacks(serde_json::json!({"type": "object",
            "patternProperties": {"^a": {}, "^b": {}}})));
        // A schema string is parsed like the frontend parses it; one that is
        // not JSON is left to xgrammar.
        assert!(lacks(Value::String(
            r#"{"type":"number","multipleOf":0.5}"#.into()
        )));
        assert!(!lacks(Value::String("not json".into())));
    }

    #[test]
    fn structured_outputs_json_roundtrips_through_wire_shape() {
        let mut params = StructuredOutputsParams::json(serde_json::json!({"type": "object"}));
        params.options.disable_any_whitespace = true;
        params.options.whitespace_pattern = Some(" ".to_string());

        let value = serde_json::to_value(&params).unwrap();
        assert_eq!(value["json"], serde_json::json!({"type": "object"}));
        assert_eq!(value["disable_any_whitespace"], true);
        assert_eq!(value["whitespace_pattern"], " ");
        assert!(value.get("disable_additional_properties").is_none());
        assert_eq!(
            serde_json::from_value::<StructuredOutputsParams>(value).unwrap(),
            params
        );
    }

    #[test]
    fn structured_outputs_rejects_missing_constraint() {
        let error =
            serde_json::from_value::<StructuredOutputsParams>(serde_json::json!({})).unwrap_err();

        assert!(error
            .to_string()
            .contains("missing structured output constraint"));
    }

    #[test]
    fn structured_outputs_rejects_multiple_constraints() {
        let error = serde_json::from_value::<StructuredOutputsParams>(serde_json::json!({
            "json": {"type": "object"},
            "regex": ".*",
        }))
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("multiple structured output constraints specified: json, regex"));
    }

    #[test]
    fn structured_outputs_rejects_json_object_false() {
        let error = serde_json::from_value::<StructuredOutputsParams>(serde_json::json!({
            "json_object": false,
        }))
        .unwrap_err();

        assert!(error.to_string().contains("json_object must be true"));
    }

    #[test]
    fn structured_outputs_serializes_through_raw_shape() {
        let params = StructuredOutputsParams {
            constraint: StructuredOutputConstraint::StructuralTag(
                r#"{"type":"structural_tag"}"#.to_string(),
            ),
            options: StructuredOutputOptions::default(),
            backend: StructuredOutputBackend::Xgrammar,
        };

        let value = serde_json::to_value(params).unwrap();

        assert_eq!(value["structural_tag"], r#"{"type":"structural_tag"}"#);
        assert_eq!(value["_backend"], "xgrammar");
        assert!(value.get("json").is_none());
    }

    /// Every rule names its keyword and the place of the schema object that
    /// breaks it, as a JSON pointer from the schema's root: the shapes the
    /// tool corpora carry (an array parameter under `uniqueItems`, a string
    /// with length bounds beside a pattern), nested under an array of
    /// objects, under a list-valued type, a key the pointer escapes, and a
    /// schema given as text.
    #[test]
    fn an_unsupported_feature_is_named_with_its_place() {
        let feature = |schema: Value| xgrammar_unsupported_json_feature(&schema);
        let found = |keyword: &str, pointer: &str| {
            Some(XgrammarUnsupportedFeature {
                keyword: keyword.to_string().into(),
                pointer: pointer.to_string(),
                openers: Vec::new(),
            })
        };
        assert_eq!(
            feature(serde_json::json!({"type": "object", "properties": {
                "days": {"type": "array", "items": {"type": "string", "enum": ["mon", "tue"]},
                         "uniqueItems": true}}, "required": ["days"]})),
            found("uniqueItems", "/properties/days")
        );
        assert_eq!(
            feature(
                serde_json::json!({"type": "object", "properties": {"listings": {
                "type": "array", "items": {"type": "object", "properties": {
                    "amenities": {"type": "array", "items": {"type": "string"},
                                  "uniqueItems": true}}}}}})
            ),
            found(
                "uniqueItems",
                "/properties/listings/items/properties/amenities"
            )
        );
        assert_eq!(
            feature(
                serde_json::json!({"type": "string", "minLength": 1, "maxLength": 100,
                "pattern": "^[a-z]+$", "default": "hello"})
            ),
            found("minLength/maxLength beside pattern or format", "")
        );
        assert_eq!(
            feature(serde_json::json!({"type": "object", "properties": {
                "n": {"type": "integer", "multipleOf": 3}}})),
            found("multipleOf", "/properties/n")
        );
        for key in ["contains", "minContains", "maxContains"] {
            assert_eq!(
                feature(serde_json::json!({"type": "object", "properties": {
                    "xs": {"type": ["array", "null"], key: {"type": "string"}}}})),
                found(key, "/properties/xs"),
                "{key}"
            );
        }
        assert_eq!(
            feature(serde_json::json!({"type": "object", "properties": {
                "data": {"type": "string", "format": "binary"}}})),
            found("format: \"binary\"", "/properties/data")
        );
        assert_eq!(
            feature(serde_json::json!({"type": "object", "properties": {
                "when": {"type": "string", "format": "date-time"}}})),
            None
        );
        assert_eq!(
            feature(serde_json::json!({"type": "object", "propertyNames": {
                "pattern": "^x", "maxLength": 3}})),
            found(
                "propertyNames with minLength/maxLength beside pattern or format",
                ""
            )
        );
        assert_eq!(
            feature(
                serde_json::json!({"type": "object", "propertyNames": {"pattern": "^x"},
                "properties": {"x": {}}})
            ),
            found("propertyNames beside other property keywords", "")
        );
        assert_eq!(
            feature(
                serde_json::json!({"type": "object", "properties": {"m": {"type": "object",
                "patternProperties": {"^a": {}, "^b": {}}}}})
            ),
            found(
                "patternProperties beside properties, or more than one",
                "/properties/m"
            )
        );
        // Objects in an array value are walked with their index; a key with
        // `/` or `~` is escaped in the pointer.
        assert_eq!(
            feature(serde_json::json!({"anyOf": [{"type": "string"},
                {"type": "object", "properties": {"a/b~c": {"type": "integer", "multipleOf": 2}}}]})),
            found("multipleOf", "/anyOf/1/properties/a~1b~0c")
        );
        assert_eq!(
            feature(Value::String(
                r#"{"type":"number","multipleOf":0.5}"#.into()
            )),
            found("multipleOf", "")
        );
        assert_eq!(feature(Value::String("not json".into())), None);
        assert_eq!(
            xgrammar_unsupported_feature_in_schema_text(
                r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"uniqueItems":true}}}"#
            ),
            found("uniqueItems", "/properties/tags")
        );
        assert_eq!(
            xgrammar_unsupported_feature_in_schema_text(
                r#"{"type":"object","properties":{"city":{"type":"string"}}}"#
            ),
            None
        );
    }

    /// A structural tag is checked at its `json_schema` elements, wherever
    /// the format nests them, and nowhere else: the tag's own `format`,
    /// `pattern` (a regex element) and `excludes` keys trip no rule; the
    /// pointer runs from the tag's root, and the tags around the schema are
    /// named by their `begin` (the call's, which carries the tool's name, and
    /// a parameter tag's under it); a tag that is not JSON is left to the
    /// engine.
    #[test]
    fn a_structural_tag_is_checked_at_its_json_schema_elements() {
        let found = |keyword: &str, pointer: &str, openers: &[&str]| {
            Some(XgrammarUnsupportedFeature {
                keyword: keyword.to_string().into(),
                pointer: pointer.to_string(),
                openers: openers.iter().map(|begin| begin.to_string()).collect(),
            })
        };
        let call = |name: &str, parameters: Value| {
            serde_json::json!({
                "begin": format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": "),
                "content": {"type": "json_schema", "json_schema": parameters,
                            "excludes": ["</tool_call>"], "max_whitespace_cnt": 8},
                "end": "}\n</tool_call>",
            })
        };
        let plain =
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let unique = serde_json::json!({"type": "object", "properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": true}}});
        let calls = |tags: Vec<Value>| {
            serde_json::json!({"format": {"type": "triggered_tags", "triggers": ["<tool_call>"],
                "tags": tags, "at_least_one": true}})
            .to_string()
        };
        let opens_tagged = "<tool_call>\n{\"name\": \"tagged\", \"arguments\": ";
        assert_eq!(
            xgrammar_unsupported_feature_in_structural_tag(&calls(vec![
                call("plain", plain.clone()),
                call("tagged", unique.clone()),
            ])),
            found(
                "uniqueItems",
                "/format/tags/1/content/json_schema/properties/tags",
                &[opens_tagged]
            )
        );
        assert_eq!(
            xgrammar_unsupported_feature_in_structural_tag(&calls(vec![call(
                "plain",
                plain.clone()
            )])),
            None
        );
        // Behind a reasoning prefix (a sequence), and a regex element whose
        // `pattern` and a const string are not schemas.
        let wrapped = serde_json::json!({"format": {"type": "sequence", "elements": [
            {"type": "any_text", "excludes": ["<tool_call>"]},
            {"type": "const_string", "value": "</think>"},
            {"type": "regex", "pattern": "[ \n]{0,8}"},
            {"type": "triggered_tags", "triggers": ["<tool_call>"],
             "tags": [call("tagged", unique.clone())], "at_least_one": true},
        ]}})
        .to_string();
        assert_eq!(
            xgrammar_unsupported_feature_in_structural_tag(&wrapped),
            found(
                "uniqueItems",
                "/format/elements/3/tags/0/content/json_schema/properties/tags",
                &[opens_tagged]
            )
        );
        // A call written as a tag per parameter: the call's tag and the
        // parameter's are both named, the call's first; the first tag's
        // schema, which passes, leaves no opener behind.
        let parameter = |name: &str, schema: Value| {
            serde_json::json!({"type": "tag", "begin": format!("<param name=\"{name}\">"),
                "content": {"type": "json_schema", "json_schema": schema}, "end": "</param>\n"})
        };
        let per_parameter = serde_json::json!({"format": {"type": "tags_with_separator",
            "tags": [{"type": "tag", "begin": "<invoke name=\"plan\">\n",
                      "content": {"type": "sequence", "elements": [
                          parameter("city", serde_json::json!({"type": "string"})),
                          parameter("days", serde_json::json!({"type": "array",
                              "items": {"type": "string"}, "uniqueItems": true}))]},
                      "end": "</invoke>\n"}],
            "separator": "", "at_least_one": true}})
        .to_string();
        assert_eq!(
            xgrammar_unsupported_feature_in_structural_tag(&per_parameter),
            found(
                "uniqueItems",
                "/format/tags/0/content/elements/1/content/json_schema",
                &["<invoke name=\"plan\">\n", "<param name=\"days\">"]
            )
        );
        // A JSON answer's schema alone: no tag around it.
        assert_eq!(
            xgrammar_unsupported_feature_in_structural_tag(
                &serde_json::json!({"format": {"type": "json_schema",
                    "json_schema": {"type": "integer", "multipleOf": 3}}})
                .to_string()
            ),
            found("multipleOf", "/format/json_schema", &[])
        );
        assert_eq!(
            xgrammar_unsupported_feature_in_structural_tag("<not json>"),
            None
        );
    }

    /// The keyword scan a constraint's text goes through before its parse:
    /// a text that names none of the rules' keywords is not parsed; one that
    /// does is, a `format` with any value but an object included and the
    /// tag's own `format` object excluded, whatever the spacing; a string
    /// schema with an object for its `format` is the documented exception
    /// the scan does not see.
    #[test]
    fn the_keyword_scan_spares_the_parse_of_a_text_that_names_none() {
        let names = text_names_an_xgrammar_unsupported_keyword;
        assert!(!names(
            r#"{"format":{"type":"triggered_tags","triggers":["<tool_call>"],"tags":[{"begin":"<tool_call>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"q":{"type":"string","pattern":"^a"}}}},"end":"</tool_call>"}]}}"#
        ));
        assert!(!names(
            r#"{"type":"object","properties":{"city":{"type":"string","description":"the city"}}}"#
        ));
        for text in [
            r#"{"type":"integer","multipleOf":3}"#,
            r#"{"type":"array","uniqueItems":true}"#,
            r#"{"type":"array","contains":{}}"#,
            r#"{"type":"array","maxContains":1}"#,
            r#"{"type":"string","pattern":"^a","maxLength":3}"#,
            r#"{"type":"string","minLength":1,"format":"email"}"#,
            r#"{"type":"object","propertyNames":{}}"#,
            r#"{"type":"object","patternProperties":{}}"#,
            r#"{"type":"string","format":"binary"}"#,
            r#"{"type": "string", "format" : "binary"}"#,
            r#"{"format":{"type":"json_schema","json_schema":{"type":"string","format":"iri"}}}"#,
            // A description that ends in a keyword costs the parse, no more.
            r#"{"type":"string","description":"the given maxLength"}"#,
        ] {
            assert!(names(text), "{text}");
        }
        assert!(!names(
            r#"{"format":{"type":"json_schema","json_schema":{"type":"string"}}}"#
        ));
        assert!(!names(
            r#"{"type":"object","properties":{"format":{"type":"string"}}}"#
        ));
        // A keyword is looked for as a quoted key; a word in a description is not one.
        assert!(!names(
            r#"{"type":"string","description":"what the box contains"}"#
        ));
        assert!(names(r#"{"type":"string","format":["email"]}"#));
        assert!(names(r#"{"type":"string","format":null}"#));
        // The exception: an object for a `format` is not seen by the scan.
        assert!(!names(r#"{"type":"string","format":{}}"#));
    }
}
