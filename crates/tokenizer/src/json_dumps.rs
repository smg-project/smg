//! Serialize a [`serde_json::Value`] like Python's `json.dumps(value, ensure_ascii=False)`:
//! spaced `", "` / `": "` separators, single line, raw UTF-8.
//!
//! serde_json only ships compact (`{"a":1}`) and pretty (multi-line) formatters,
//! neither of which matches `json.dumps`. The DeepSeek and GLM-5.3 prompt encoders
//! and the DeepSeek V3.1 tools block embed tool schemas and argument values as JSON,
//! and their references (the engine's encoder or the template's `tojson`) write them
//! as `json.dumps` does, so compact output would shift the model off its training
//! distribution. The spacing is the chat template's `tojson` formatter with no
//! indent, Python's default separators and `ensure_ascii = false`, so the prompt
//! encoders and the template filter render a value the same way.

use serde_json::Value;

use crate::chat_template::{serialize_with_python_json, JsonSeparators};

/// Serialize `value` like `json.dumps(value, ensure_ascii=False)`. A `Value`
/// serializes into memory without an error; the fallback keeps the signature
/// infallible for the prompt encoders' string builders.
pub(crate) fn to_string(value: &Value) -> String {
    serialize_with_python_json(value, None, JsonSeparators::python_default(false), false)
        .unwrap_or_else(|_| "null".to_string())
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
