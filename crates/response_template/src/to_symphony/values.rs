//! The values of a compiled table's calls, read as transformers reads them: one implementation,
//! the crate's own port, behind symphony's value hook.

use serde_json::Value;
use symphony::shape::{Duplicates, Schema, ValueParser, ValueRead};

use crate::{
    content_parsers::{Content, ContentParser},
    error::{PyResult, Taint},
    response_parser::{coerce, coerce_argument, schema_types},
};

/// transformers 5.18.0's reading of a call's values, for a compiled table
/// ([`symphony::Format::reads_values_with`]). It runs the crate's port of transformers, the code
/// [`parse_response`](crate::parse_response) runs, so the compiled table and the template's own
/// reader read a value with one implementation:
///
/// - a tagged value's text reads by the template's `value_parser` (`_sub_parse`): as it is, as
///   `text` (stripped with Python's `str.strip` unless `strip: false`), as `json` (in its dialect,
///   text that is not JSON read as `text` with `allow_non_json`), or as Python's `int`, `float`
///   and `bool` do;
/// - a result that is a string is then cast by the parameter's schema as `_coerce` casts it: to
///   the first type `_schema_types` lists (`type`, a list of types, `anyOf`, `oneOf`, `nullable`;
///   not `$ref`, `allOf`, `enum` or `const`) that the text is, or kept as the text; a result that
///   is a list, where those types hold no array, has each string in it cast so, as
///   `_coerce_tool_calls` does;
/// - a string member of JSON arguments is cast the same way (`_coerce_tool_calls`), so it is held
///   until it closes when its schema lists a type a string can be cast to;
/// - a key written twice keeps its last value, as a Python dict does.
///
/// Where transformers raises (a value its parser cannot read, a schema `_schema_types` cannot
/// read), or holds a value this crate's JSON cannot (NaN, an infinity, an integer outside 64 bits,
/// a lone surrogate), the value is the text, as a JSON string, and the member keeps the model's
/// bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Transformers;

/// The reader a compiled table names.
pub static TRANSFORMERS: Transformers = Transformers;

/// The types a string can be cast to: a member of one of them waits until it is whole.
const CASTS: [&str; 6] = ["integer", "number", "boolean", "null", "object", "array"];

impl ValueRead for Transformers {
    fn read(&self, parser: &ValueParser, text: &str, schema: Option<Schema<'_>>) -> String {
        let mut taint = None;
        let Ok(mut value) = sub_parse(parser, text, &mut taint) else {
            return string(text);
        };
        if let Some(types) = types(schema) {
            if coerce_argument(&mut value, &types, &mut taint).is_err() {
                return string(text);
            }
        }
        if taint.is_some() {
            return string(text);
        }
        serde_json::to_string(&value).unwrap_or_else(|_| string(text))
    }

    fn streams(&self, _parser: &ValueParser, schema: Option<Schema<'_>>) -> bool {
        types(schema).is_none_or(|types| !types.iter().any(|t| CASTS.contains(&t.as_str())))
    }

    fn member(&self, raw: &str, schema: Option<Schema<'_>>) -> Option<String> {
        let text: String = serde_json::from_str(raw).ok()?;
        let types = types(schema)?;
        let mut taint: Taint = None;
        let value = coerce(&text, &types, &mut taint).ok()?;
        if taint.is_some() || value == Value::String(text) {
            return None;
        }
        serde_json::to_string(&value).ok()
    }

    fn duplicates(&self) -> Duplicates {
        Duplicates::Last
    }
}

/// The types `_schema_types` reads in the parameter's schema; `None` when there are none, or the
/// request declares no such parameter, or `_schema_types` raises.
fn types(schema: Option<Schema<'_>>) -> Option<Vec<String>> {
    schema_types(schema?.property)
        .ok()
        .filter(|types| !types.is_empty())
}

/// transformers' `_sub_parse` of a tagged value's text with the template's `value_parser`.
fn sub_parse(parser: &ValueParser, text: &str, taint: &mut Taint) -> PyResult<Value> {
    let content = match parser {
        ValueParser::Raw => return Ok(Value::String(text.to_owned())),
        ValueParser::Text { strip } => {
            Content::value(ContentParser::Text, *strip, false, false, Vec::new())
        }
        ValueParser::Json {
            dialect,
            allow_non_json,
            strip,
        } => Content::value(
            ContentParser::Json,
            *strip,
            dialect.bare_keys,
            *allow_non_json,
            dialect
                .string_delims
                .iter()
                .map(|(open, close)| (open.to_string(), close.to_string()))
                .collect(),
        ),
        ValueParser::Int => Content::value(ContentParser::Int, true, false, false, Vec::new()),
        ValueParser::Float => Content::value(ContentParser::Float, true, false, false, Vec::new()),
        ValueParser::Bool { strip } => {
            Content::value(ContentParser::Bool, *strip, false, false, Vec::new())
        }
    };
    content.parse(text, taint)
}

/// `text` as a JSON string.
fn string(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}
