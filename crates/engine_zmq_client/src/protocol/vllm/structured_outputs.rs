// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/structured_outputs.rs.

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

/// Whether a nested schema value (an object, or an array of objects) fails
/// `check`, the recursion of [`has_xgrammar_unsupported_json_features`].
fn any_nested(obj: &serde_json::Map<String, Value>, check: fn(&Value) -> bool) -> bool {
    obj.values().any(|value| match value {
        Value::Object(_) => check(value),
        Value::Array(items) => items.iter().any(|item| item.is_object() && check(item)),
        _ => false,
    })
}

/// vLLM's `has_xgrammar_unsupported_json_features`
/// (`v1/structured_output/backend_xgrammar.py`), keyword for keyword: the
/// schema features xgrammar's converter drops instead of refusing. vLLM's
/// frontend refuses such a schema with a 400 under `backend=xgrammar`
/// (`validate_xgrammar_grammar`); a sender stamping xgrammar does the same. A
/// schema given as a string is parsed as the frontend parses it; one that is
/// not JSON is left to xgrammar, which refuses it at compile.
pub fn has_xgrammar_unsupported_json_features(schema: &Value) -> bool {
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
    fn schema_types(obj: &serde_json::Map<String, Value>) -> Vec<&str> {
        match obj.get("type") {
            Some(Value::String(one)) => vec![one.as_str()],
            Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        }
    }
    fn has_pattern_and_length_bounds(obj: &serde_json::Map<String, Value>) -> bool {
        (obj.contains_key("pattern") || obj.contains_key("format"))
            && (obj.contains_key("minLength") || obj.contains_key("maxLength"))
    }
    fn check(value: &Value) -> bool {
        let Value::Object(obj) = value else {
            return false;
        };
        let types = schema_types(obj);
        let has = |kind: &str| types.contains(&kind);
        if (has("integer") || has("number")) && obj.contains_key("multipleOf") {
            return true;
        }
        if has("array")
            && ["uniqueItems", "contains", "minContains", "maxContains"]
                .iter()
                .any(|key| obj.contains_key(*key))
        {
            return true;
        }
        if has("string")
            && obj.get("format").is_some_and(|format| {
                !format
                    .as_str()
                    .is_some_and(|format| STRING_SUPPORTED_FORMATS.contains(&format))
            })
        {
            return true;
        }
        // xgrammar drops minLength/maxLength next to a pattern or format.
        if has("string") && has_pattern_and_length_bounds(obj) {
            return true;
        }
        if has("object")
            && obj
                .get("propertyNames")
                .and_then(Value::as_object)
                .is_some_and(has_pattern_and_length_bounds)
        {
            return true;
        }
        // propertyNames conflicts with the other property keywords.
        if has("object")
            && obj.contains_key("propertyNames")
            && (obj.contains_key("properties")
                || obj.contains_key("patternProperties")
                || obj
                    .get("additionalProperties")
                    .is_some_and(Value::is_object)
                || obj
                    .get("unevaluatedProperties")
                    .is_some_and(|value| *value != Value::Bool(true)))
        {
            return true;
        }
        // Several patternProperties, or one next to properties, conflict.
        if has("object")
            && obj
                .get("patternProperties")
                .and_then(Value::as_object)
                .is_some_and(|patterns| obj.contains_key("properties") || patterns.len() > 1)
        {
            return true;
        }
        any_nested(obj, check)
    }
    let parsed: Value;
    let schema = match schema {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(value) => {
                parsed = value;
                &parsed
            }
            Err(_) => return false,
        },
        other => other,
    };
    check(schema)
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
}
