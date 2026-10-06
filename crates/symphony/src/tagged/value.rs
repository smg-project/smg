//! What the text of a tagged parameter means as JSON.
//!
//! A tagged format writes each argument as text between an opening and a closing tag, and the text
//! does not say what type it is: `150` is the number 150 for an integer parameter and the string
//! `"150"` for a string one. The template that wrote the text knew the type. The parser gets it
//! back from the parameters the tool declares: [`Declared`] holds those types for one request, and
//! [`json`] turns one value's text into the JSON a client receives for it.
//!
//! The rules, in the order [`json`] applies them:
//!
//! - A parameter declared `string` is its text, every byte of it. Nothing is trimmed and nothing is
//!   decoded: `"exact phrase"` keeps its quotes, indentation and a trailing newline stay, and
//!   `&amp;` is five characters.
//! - A parameter declared `integer`, `number`, `boolean`, `array` or `object` is read as that type
//!   when its text is one, with the whitespace around it ignored. An integer may be spelled `+5` or
//!   `007` and a boolean `True` or `False`; they are written in JSON's spelling.
//! - Everything else is inferred: text that is JSON is that JSON, Python's `True`, `False` and
//!   `None` are `true`, `false` and `null`, and the rest is a string holding the text exactly.
//!   That covers a parameter the tool does not declare, a declared type this module does not know,
//!   and a value that is not of its declared type.
//!
//! A value read as JSON keeps the model's own bytes: `[1, 2.5]` stays spaced as written, and
//! `9007199254740993` keeps every digit. Whether text is JSON is decided without building the
//! value, so neither its size nor its depth costs more than one pass over it.
//!
//! Ported from `safe_val` and `coerce_value` in `crates/tool_parser/src/parsers/qwen_xml.rs` and
//! from `coerce_by_schema_type` and `param_types_for_function` in that crate's `helpers.rs`. Two
//! things differ, both because the templates write values this way and bellwether's references
//! record it:
//!
//! - A string keeps its text exactly. The old functions trimmed it, and unquoted a value declared
//!   `string` that happened to be a JSON string literal, so a search for `"exact phrase"` lost its
//!   quotes and a file's contents lost their final newline.
//! - JSON keeps the model's bytes. The old functions parsed the value and wrote it again compactly,
//!   so `{"a": 1}` reached the client as `{"a":1}`.

use std::collections::HashMap;

use openai_protocol::common::Tool;
use serde::de::IgnoredAny;
use serde_json::Value;

/// The type a tool declares for one parameter, of the types that change how its text is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `"type": "string"`.
    String,
    /// `"type": "integer"`.
    Integer,
    /// `"type": "number"`.
    Number,
    /// `"type": "boolean"`.
    Boolean,
    /// `"type": "array"`.
    Array,
    /// `"type": "object"`.
    Object,
}

impl Kind {
    /// The kind a parameter's schema declares, when its `type` is one name this module knows. A
    /// list of types, a missing `type` and any other name declare nothing here.
    fn of(schema: &Value) -> Option<Self> {
        match schema.get("type")?.as_str()? {
            "string" => Some(Self::String),
            "integer" => Some(Self::Integer),
            "number" => Some(Self::Number),
            "boolean" => Some(Self::Boolean),
            "array" => Some(Self::Array),
            "object" => Some(Self::Object),
            _ => None,
        }
    }
}

/// The parameter types a request's tools declare, by function name and then parameter name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Declared {
    functions: HashMap<String, HashMap<String, Kind>>,
}

impl Declared {
    /// Reads the types from each tool's `parameters.properties`. When two tools share a name, the
    /// first one's parameters count.
    pub fn of(tools: &[Tool]) -> Self {
        let mut functions = HashMap::new();
        for tool in tools {
            functions
                .entry(tool.function.name.clone())
                .or_insert_with(|| parameters_of(&tool.function.parameters));
        }
        Self { functions }
    }

    /// The kind `function` declares for `parameter`, if it declares one.
    pub fn kind(&self, function: &str, parameter: &str) -> Option<Kind> {
        self.functions.get(function)?.get(parameter).copied()
    }
}

fn parameters_of(schema: &Value) -> HashMap<String, Kind> {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return HashMap::new();
    };
    properties
        .iter()
        .filter_map(|(name, schema)| Some((name.clone(), Kind::of(schema)?)))
        .collect()
}

/// The JSON for one parameter's `text`, read as `kind` when the tool declares one. The result is
/// always one complete JSON value.
pub fn json(text: &str, kind: Option<Kind>) -> String {
    if kind == Some(Kind::String) {
        return string(text);
    }
    let trimmed = text.trim();
    let typed = match kind {
        Some(Kind::Integer) => trimmed.parse::<i64>().ok().map(|n| n.to_string()),
        Some(Kind::Number) => is_json_number(trimmed).then(|| trimmed.to_string()),
        Some(Kind::Boolean) => boolean(trimmed).map(str::to_string),
        Some(Kind::Array | Kind::Object) => is_json(trimmed).then(|| trimmed.to_string()),
        Some(Kind::String) | None => None,
    };
    typed.unwrap_or_else(|| inferred(text, trimmed))
}

/// What the text is when no declared type decided it.
fn inferred(text: &str, trimmed: &str) -> String {
    if is_json(trimmed) {
        return trimmed.to_string();
    }
    match trimmed {
        "True" => "true".to_string(),
        "False" => "false".to_string(),
        "None" => "null".to_string(),
        _ => string(text),
    }
}

fn boolean(trimmed: &str) -> Option<&'static str> {
    match trimmed {
        "true" | "True" => Some("true"),
        "false" | "False" => Some("false"),
        _ => None,
    }
}

fn is_json(text: &str) -> bool {
    serde_json::from_str::<IgnoredAny>(text).is_ok()
}

fn is_json_number(text: &str) -> bool {
    text.starts_with(|c: char| c == '-' || c.is_ascii_digit()) && is_json(text)
}

/// `text` as a JSON string.
fn string(text: &str) -> String {
    Value::String(text.to_string()).to_string()
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;
    use serde_json::json as value;

    use super::*;

    fn tool(name: &str, properties: Value) -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": properties}),
                strict: None,
            },
        }
    }

    #[test]
    fn a_tools_declared_types_are_found_by_function_and_parameter() {
        let declared = Declared::of(&[
            tool(
                "search",
                value!({"query": {"type": "string"}, "limit": {"type": "integer"}}),
            ),
            tool(
                "draw",
                value!({"points": {"type": "array"}, "style": {"type": "object"}}),
            ),
            tool(
                "set",
                value!({"ratio": {"type": "number"}, "on": {"type": "boolean"}}),
            ),
        ]);
        assert_eq!(declared.kind("search", "query"), Some(Kind::String));
        assert_eq!(declared.kind("search", "limit"), Some(Kind::Integer));
        assert_eq!(declared.kind("draw", "points"), Some(Kind::Array));
        assert_eq!(declared.kind("draw", "style"), Some(Kind::Object));
        assert_eq!(declared.kind("set", "ratio"), Some(Kind::Number));
        assert_eq!(declared.kind("set", "on"), Some(Kind::Boolean));
        assert_eq!(declared.kind("search", "points"), None);
        assert_eq!(declared.kind("missing", "query"), None);
    }

    #[test]
    fn a_schema_that_names_no_single_known_type_declares_nothing() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "nullable": {"type": ["string", "null"]},
                "untyped": {"description": "anything"},
                "other": {"type": "null"},
                "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            }),
        )]);
        for parameter in ["nullable", "untyped", "other", "choice"] {
            assert_eq!(declared.kind("f", parameter), None, "{parameter}");
        }
    }

    #[test]
    fn a_tool_without_properties_declares_nothing_and_the_first_of_two_names_counts() {
        let mut bare = tool("bare", value!({}));
        bare.function.parameters = value!({"type": "object"});
        let declared = Declared::of(&[
            bare,
            tool("twice", value!({"x": {"type": "integer"}})),
            tool("twice", value!({"x": {"type": "string"}})),
        ]);
        assert_eq!(declared.kind("bare", "x"), None);
        assert_eq!(declared.kind("twice", "x"), Some(Kind::Integer));
    }

    #[test]
    fn a_declared_string_is_its_text_exactly() {
        for (text, expected) in [
            ("Paris", r#""Paris""#),
            ("4", r#""4""#),
            ("true", r#""true""#),
            ("[60,30]", r#""[60,30]""#),
            (r#"{"a": 1}"#, r#""{\"a\": 1}""#),
            ("", r#""""#),
            // Quotes that are part of the value stay; the old parser unquoted this one.
            (r#""exact phrase""#, r#""\"exact phrase\"""#),
            // Whitespace is part of the value too; the old parser trimmed it.
            ("    indented\n", r#""    indented\n""#),
            (
                "Line one\nLine two with \"quotes\" and a backslash \\",
                r#""Line one\nLine two with \"quotes\" and a backslash \\""#,
            ),
            // No entity is decoded: Qwen's own API returns them as written.
            (
                "<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s",
                r#""<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s""#,
            ),
            // Text outside ASCII is written as it is, not as escapes.
            ("चाय कैसे बनाएं?", r#""चाय कैसे बनाएं?""#),
            ("tab\there", r#""tab\there""#),
            ("bell\u{7}", r#""bell\u0007""#),
        ] {
            assert_eq!(json(text, Some(Kind::String)), expected, "{text:?}");
        }
    }

    #[test]
    fn a_declared_integer_is_a_number_when_its_text_is_one() {
        for (text, expected) in [
            ("5", "5"),
            ("-3", "-3"),
            ("\n150\n", "150"),
            ("+5", "5"),
            ("007", "7"),
            // Not an integer, so it is inferred: JSON when the text is JSON, else a string.
            ("5.5", "5.5"),
            ("12345678901234567890", "12345678901234567890"),
            ("five", r#""five""#),
            ("", r#""""#),
        ] {
            assert_eq!(json(text, Some(Kind::Integer)), expected, "{text:?}");
        }
    }

    #[test]
    fn a_declared_number_keeps_the_digits_the_model_wrote() {
        for (text, expected) in [
            ("0.5", "0.5"),
            ("150", "150"),
            ("-1e-5", "-1e-5"),
            ("1.50", "1.50"),
            // A float could not hold this integer; the text does.
            ("9007199254740993", "9007199254740993"),
            (" 2.5 ", "2.5"),
            // JSON, but not a number: inferred, so it stays the JSON it is.
            ("true", "true"),
            ("[1]", "[1]"),
            ("fast", r#""fast""#),
        ] {
            assert_eq!(json(text, Some(Kind::Number)), expected, "{text:?}");
        }
    }

    #[test]
    fn a_declared_boolean_takes_json_and_python_spellings() {
        for (text, expected) in [
            ("true", "true"),
            ("True", "true"),
            ("false", "false"),
            ("False", "false"),
            (" true\n", "true"),
            ("yes", r#""yes""#),
            ("1", "1"),
        ] {
            assert_eq!(json(text, Some(Kind::Boolean)), expected, "{text:?}");
        }
    }

    #[test]
    fn a_declared_array_or_object_keeps_the_models_bytes() {
        for kind in [Kind::Array, Kind::Object] {
            for (text, expected) in [
                (r#"["a", "b"]"#, r#"["a", "b"]"#),
                (
                    r#"{"nested": {"deep": [1, 2.5, true, null]}}"#,
                    r#"{"nested": {"deep": [1, 2.5, true, null]}}"#,
                ),
                ("\n[60,30]\n", "[60,30]"),
                (r#"{"a":1}"#, r#"{"a":1}"#),
                // Not JSON: a string, since nothing better can be said of it.
                ("[1, 2", r#""[1, 2""#),
                ("{'a': 1}", r#""{'a': 1}""#),
            ] {
                assert_eq!(json(text, Some(kind)), expected, "{kind:?} {text:?}");
            }
        }
    }

    #[test]
    fn an_undeclared_value_is_inferred() {
        for (text, expected) in [
            ("42", "42"),
            ("1.5", "1.5"),
            ("true", "true"),
            ("false", "false"),
            ("null", "null"),
            (r#"{"key": "value"}"#, r#"{"key": "value"}"#),
            ("[1, 2, 3]", "[1, 2, 3]"),
            (r#""quoted""#, r#""quoted""#),
            ("True", "true"),
            ("False", "false"),
            ("None", "null"),
            ("hello world", r#""hello world""#),
            ("&lt;div&gt;", r#""&lt;div&gt;""#),
            ("it&#39;s", r#""it&#39;s""#),
            ("&#x3C;", r#""&#x3C;""#),
            (
                "<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s",
                r#""<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s""#,
            ),
            // A string keeps the whitespace the old parser trimmed; JSON does not need it.
            ("  spaces  ", r#""  spaces  ""#),
            (" 42 ", "42"),
            ("", r#""""#),
        ] {
            assert_eq!(json(text, None), expected, "{text:?}");
        }
    }

    #[test]
    fn deeply_nested_json_is_still_json() {
        let deep = format!("{}{}", "[".repeat(50_000), "]".repeat(50_000));
        assert_eq!(json(&deep, Some(Kind::Array)), deep);
        assert_eq!(json(&deep, None), deep);
        let unclosed = "[".repeat(50_000);
        assert_eq!(
            json(&unclosed, Some(Kind::Array)),
            format!("\"{unclosed}\"")
        );
    }

    /// A small deterministic generator, so the property below needs no dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as usize
        }
    }

    #[test]
    fn whatever_the_text_and_the_kind_the_result_is_one_json_value() {
        const PIECES: [&str; 24] = [
            "", " ", "\n", "0", "-", "1.5", "e9", "true", "True", "None", "null", "\"", "\\", "{",
            "}", "[", "]", ":", ",", "a", "é", "\u{1}", "<", "&amp;",
        ];
        const KINDS: [Option<Kind>; 7] = [
            None,
            Some(Kind::String),
            Some(Kind::Integer),
            Some(Kind::Number),
            Some(Kind::Boolean),
            Some(Kind::Array),
            Some(Kind::Object),
        ];
        let mut random = Lcg(7);
        for _ in 0..20_000 {
            let text: String = (0..random.next() % 7)
                .map(|_| PIECES[random.next() % PIECES.len()])
                .collect();
            for kind in KINDS {
                let written = json(&text, kind);
                let parsed: Result<Value, _> = serde_json::from_str(&written);
                assert!(parsed.is_ok(), "{kind:?} {text:?} gave {written:?}");
                if kind == Some(Kind::String) {
                    assert_eq!(parsed.ok(), Some(Value::String(text.clone())), "{text:?}");
                }
            }
        }
    }
}
