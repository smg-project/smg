// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/structured_outputs.rs.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::error::{Error, Result};

/// Structured-output backend selected for EngineCore grammar compilation.
///
/// Python stores this in `StructuredOutputsParams._backend` after request
/// validation. This client selects the backend per constraint: structural
/// tags require xgrammar (the triggered-tags format is not understood by
/// guidance's legacy structures/triggers parser); everything else lowers to
/// guidance. Peer-supplied `_backend` values are ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StructuredOutputBackend {
    Xgrammar,
    #[default]
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
    /// Structured-output backend, mirroring Python's internal `_backend`.
    ///
    /// Peer-supplied values are ignored during deserialization. This matches the
    /// Python request boundary, where `_backend` is set by validation rather
    /// than accepted as a request-level backend selector.
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

    fn from_constraint(constraint: StructuredOutputConstraint) -> Self {
        // Structural tags use the triggered-tags format that only xgrammar
        // compiles; guidance's parser expects the legacy structures/triggers
        // shape and fails the request at grammar build.
        let backend = match &constraint {
            StructuredOutputConstraint::StructuralTag(_) => StructuredOutputBackend::Xgrammar,
            _ => StructuredOutputBackend::default(),
        };
        Self {
            constraint,
            options: StructuredOutputOptions::default(),
            backend,
        }
    }
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
    #[serde(
        default,
        rename = "_backend",
        deserialize_with = "serde_with::rust::deserialize_ignore_any"
    )]
    backend: StructuredOutputBackend,
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

        Ok(Self {
            constraint: constraint.map(|(_, c)| c).ok_or_else(|| {
                Error::InvalidStructuredOutputsParams {
                    message: "missing structured output constraint".to_string(),
                }
            })?,
            options: StructuredOutputOptions {
                disable_any_whitespace: raw.disable_any_whitespace,
                disable_additional_properties: raw.disable_additional_properties,
                whitespace_pattern: raw.whitespace_pattern,
            },
            backend: raw.backend,
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
    fn structured_outputs_backend_ignores_deserialized_value() {
        let params: StructuredOutputsParams = serde_json::from_value(serde_json::json!({
            "json_object": true,
            "_backend": "xgrammar",
        }))
        .unwrap();

        assert_eq!(params.backend, StructuredOutputBackend::Guidance);
        assert_eq!(params.constraint, StructuredOutputConstraint::JsonObject);

        let value = serde_json::to_value(params).unwrap();
        assert_eq!(value["_backend"], "guidance");
    }

    #[test]
    fn structural_tag_selects_xgrammar_backend() {
        // The triggered-tags format only compiles under xgrammar; guidance's
        // legacy parser fails the request at grammar build.
        let params = StructuredOutputsParams::structural_tag(r#"{"format":{}}"#);
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);

        let value = serde_json::to_value(params).unwrap();
        assert_eq!(value["_backend"], "xgrammar");
        assert_eq!(value["structural_tag"], r#"{"format":{}}"#);
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
