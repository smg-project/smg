//! Serialize a [`serde_json::Value`] like Python's `json.dumps(value, ensure_ascii=False)`:
//! spaced `", "` / `": "` separators, single line, raw UTF-8.
//!
//! serde_json only ships compact (`{"a":1}`) and pretty (multi-line) formatters,
//! neither of which matches `json.dumps`. The DeepSeek V3.2/V4 prompt encoders
//! embed tool schemas and argument values as JSON, and the engine's reference encoder
//! uses `json.dumps`, so compact output would shift the model off its training
//! distribution. This adds just the spacing serde_json lacks.

use std::io;

use serde::Serialize;
use serde_json::{
    ser::{Formatter, Serializer},
    Value,
};

/// `serde_json` formatter that adds Python `json.dumps` default separator spacing.
struct PythonDefaultFormatter;

impl Formatter for PythonDefaultFormatter {
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
}

/// Serialize `value` like `json.dumps(value, ensure_ascii=False)`.
pub(crate) fn to_string(value: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PythonDefaultFormatter);
    if value.serialize(&mut ser).is_err() {
        return "null".to_string();
    }
    String::from_utf8(buf).unwrap_or_else(|_| "null".to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn matches_python_json_dumps() {
        // json.dumps({...}, ensure_ascii=False): spaced separators, raw unicode.
        let v = json!({"name": "get_weather", "args": [1, 2, {"a": true}], "city": "广州"});
        assert_eq!(
            to_string(&v),
            r#"{"name": "get_weather", "args": [1, 2, {"a": true}], "city": "广州"}"#
        );
    }

    #[test]
    fn empty_containers() {
        assert_eq!(to_string(&json!({})), "{}");
        assert_eq!(to_string(&json!([])), "[]");
    }

    #[test]
    fn separators_inside_content_are_left_untouched() {
        // `,` / `:` inside keys and values must not be spaced — only the
        // structural separators are. Each matches json.dumps(ensure_ascii=False).
        assert_eq!(to_string(&json!({"a": "x, y: z"})), r#"{"a": "x, y: z"}"#);
        assert_eq!(to_string(&json!({"k:1": "v,2"})), r#"{"k:1": "v,2"}"#);
        // A value that *is* a separator string.
        assert_eq!(
            to_string(&json!({"sep": ", ", "kv": ": "})),
            r#"{"sep": ", ", "kv": ": "}"#
        );
        // A value that is itself a JSON-looking string (stays escaped, unspaced).
        assert_eq!(
            to_string(&json!({"s": "{\"inner\": 1, \"b\": [2, 3]}"})),
            r#"{"s": "{\"inner\": 1, \"b\": [2, 3]}"}"#
        );
        assert_eq!(
            to_string(&json!(["a,b", "c:d", "e, f: g"])),
            r#"["a,b", "c:d", "e, f: g"]"#
        );
    }
}
