use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `response_template` object of `tokenizer_config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseTemplate {
    /// Values for fields that the response omits, keyed by field name.
    #[serde(default)]
    pub defaults: BTreeMap<String, Value>,
    /// Regex that must match the rendered prompt prefix.
    pub start_anchor_pattern: String,
    /// Field templates keyed by field name.
    pub fields: BTreeMap<String, FieldTemplate>,
}

/// How one output field is delimited and parsed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldTemplate {
    /// Regex that opens the field.
    pub open_pattern: String,
    /// Literal delimiter or delimiters that close the field.
    pub close: ClosePattern,
    /// Content parser name: `text` or `xml-inline`.
    pub content: String,
    /// Content parser arguments, required for `xml-inline`.
    #[serde(default)]
    pub content_args: Option<ContentArgs>,
    /// Whether the field may occur more than once.
    #[serde(default)]
    pub repeats: bool,
    /// Transform applied to each parsed tool call.
    #[serde(default)]
    pub transform: Option<Transform>,
}

/// One literal close delimiter, or several alternatives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClosePattern {
    /// A single close delimiter.
    One(String),
    /// Alternative close delimiters; the earliest match closes the field.
    Many(Vec<String>),
}

impl ClosePattern {
    /// The declared close delimiters.
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

/// Arguments of the `xml-inline` content parser.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentArgs {
    /// Regex for one argument, capturing `key` and `value` in that order.
    pub tag_pattern: String,
    /// How each captured value is parsed.
    pub value_parser: ValueParser,
}

/// Parser for `xml-inline` argument values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueParser {
    /// Value parser name; only `text` is supported.
    pub name: String,
    /// Value parser options.
    #[serde(default)]
    pub args: ValueParserArgs,
}

/// Options of the `text` value parser.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueParserArgs {
    /// Trim surrounding whitespace from each value.
    #[serde(default)]
    pub strip: bool,
}

/// A structural transform. Placeholders that occupy an entire string preserve
/// the value type; placeholders embedded in a string are interpolated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Transform(pub Value);
