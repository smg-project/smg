use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseTemplate {
    #[serde(default)]
    pub defaults: BTreeMap<String, Value>,
    pub start_anchor_pattern: String,
    pub fields: BTreeMap<String, FieldTemplate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldTemplate {
    pub open_pattern: String,
    pub close: ClosePattern,
    pub content: String,
    #[serde(default)]
    pub content_args: Option<ContentArgs>,
    #[serde(default)]
    pub repeats: bool,
    #[serde(default)]
    pub transform: Option<Transform>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClosePattern {
    One(String),
    Many(Vec<String>),
}

impl ClosePattern {
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentArgs {
    pub tag_pattern: String,
    pub value_parser: ValueParser,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueParser {
    pub name: String,
    #[serde(default)]
    pub args: ValueParserArgs,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ValueParserArgs {
    #[serde(default)]
    pub strip: bool,
}

/// A structural transform. Placeholders that occupy an entire string preserve
/// the value type; placeholders embedded in a string are interpolated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Transform(pub Value);
