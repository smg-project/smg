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
//! - A parameter declared `integer` is that integer when its text is a sign and digits, with the
//!   whitespace around it ignored. It may be spelled `+5` or `007`; it is written in JSON's
//!   spelling, with the digits the model wrote.
//! - Everything else is inferred: text that is JSON is that JSON, Python's `True`, `False` and
//!   `None` are `true`, `false` and `null`, and the rest is a string holding the text exactly.
//!   That covers a parameter the tool does not declare, a declared integer whose text is not one,
//!   and every other declared type: a `number`, `boolean`, `array` or `object` is written by the
//!   templates as JSON, which is what inference reads, so declaring one changes nothing.
//!
//! A value read as JSON keeps the model's own bytes: `[1, 2.5]` stays spaced as written, and
//! `9007199254740993` keeps every digit. JSON here means what `serde_json` reads into a value from
//! inside the arguments object, because that is how the adapters read a call's arguments back.
//! Text it refuses there, such as a lone surrogate escape or nesting that reaches its depth limit
//! once the object is around it, is a string like any other text, so one odd value never costs a
//! call its other arguments.
//!
//! Ported from `safe_val` and `coerce_value` in `crates/tool_parser/src/parsers/qwen_xml.rs` and
//! from `coerce_by_schema_type` and `param_types_for_function` in that crate's `helpers.rs`. Three
//! things differ. The first two are how the templates write values and what bellwether's
//! references record:
//!
//! - A string keeps its text exactly. The old functions trimmed it, and unquoted a value declared
//!   `string` that happened to be a JSON string literal, so a search for `"exact phrase"` lost its
//!   quotes and a file's contents lost their final newline.
//! - JSON keeps the model's bytes. The old functions parsed the value and wrote it again compactly,
//!   so `{"a": 1}` reached the client as `{"a":1}`.
//! - The old functions had a rule each for `number`, `boolean`, `array` and `object`. Each gave
//!   exactly what inference gives, so they are not carried over.

use std::collections::HashMap;

use openai_protocol::common::Tool;
use serde_json::Value;

/// The type a tool declares for one parameter, of the two types that change how its text is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `"type": "string"`.
    String,
    /// `"type": "integer"`.
    Integer,
}

impl Kind {
    /// The kind a parameter's schema declares, when its `type` is `string` or `integer`. Any other
    /// name, a list of types and a missing `type` declare nothing here.
    fn of(schema: &Value) -> Option<Self> {
        match schema.get("type")?.as_str()? {
            "string" => Some(Self::String),
            "integer" => Some(Self::Integer),
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
/// always one JSON value that `serde_json` reads back.
pub fn json(text: &str, kind: Option<Kind>) -> String {
    match kind {
        Some(Kind::String) => string(text),
        Some(Kind::Integer) => integer(text.trim())
            .filter(|integer| reads_back_as_an_argument(integer))
            .unwrap_or_else(|| inferred(text)),
        None => inferred(text),
    }
}

/// `text` as a JSON integer when it is an optional sign and then digits: no sign for `+`, no
/// leading zeros. The digits are copied, not read as a number, so none of them is lost; whether
/// `serde_json` can read that many is for the caller to check.
fn integer(text: &str) -> Option<String> {
    let (minus, digits) = match text.strip_prefix('-') {
        Some(digits) => ("-", digits),
        None => ("", text.strip_prefix('+').unwrap_or(text)),
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let significant = digits.trim_start_matches('0');
    Some(if significant.is_empty() {
        "0".to_string()
    } else {
        format!("{minus}{significant}")
    })
}

/// What the text is when no declared type decided it.
fn inferred(text: &str) -> String {
    let trimmed = text.trim();
    if reads_back_as_an_argument(trimmed) {
        return trimmed.to_string();
    }
    match trimmed {
        "True" => "true".to_string(),
        "False" => "false".to_string(),
        "None" => "null".to_string(),
        _ => string(text),
    }
}

/// Whether `text` is one JSON value that `serde_json` still reads once it is a member of the
/// arguments object. The first parse says it is one value; the second reads it one level down,
/// where the object puts it, which is where nesting at the depth limit stops being readable.
fn reads_back_as_an_argument(text: &str) -> bool {
    serde_json::from_str::<Value>(text).is_ok()
        && serde_json::from_str::<Value>(&format!("[{text}]")).is_ok()
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
            tool("count", value!({"query": {"type": "integer"}})),
        ]);
        assert_eq!(declared.kind("search", "query"), Some(Kind::String));
        assert_eq!(declared.kind("search", "limit"), Some(Kind::Integer));
        assert_eq!(declared.kind("count", "query"), Some(Kind::Integer));
        assert_eq!(declared.kind("count", "limit"), None);
        assert_eq!(declared.kind("missing", "query"), None);
    }

    #[test]
    fn only_a_single_string_or_integer_type_declares_a_kind() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "ratio": {"type": "number"},
                "on": {"type": "boolean"},
                "points": {"type": "array"},
                "style": {"type": "object"},
                "nothing": {"type": "null"},
                "nullable": {"type": ["string", "null"]},
                "untyped": {"description": "anything"},
                "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            }),
        )]);
        for parameter in [
            "ratio", "on", "points", "style", "nothing", "nullable", "untyped", "choice",
        ] {
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
            ("-007", "-7"),
            ("0", "0"),
            ("-0", "0"),
            ("000", "0"),
            // Wider than any machine integer, with a sign and zeros to remove.
            ("+9223372036854775808", "9223372036854775808"),
            (
                "-00123456789012345678901234567890",
                "-123456789012345678901234567890",
            ),
            // Not an integer, so it is inferred: JSON when the text is JSON, else a string.
            ("5.5", "5.5"),
            ("12345678901234567890", "12345678901234567890"),
            ("five", r#""five""#),
            ("", r#""""#),
            ("+", r#""+""#),
            ("-", r#""-""#),
            ("1_000", r#""1_000""#),
            ("٣", r#""٣""#),
        ] {
            assert_eq!(json(text, Some(Kind::Integer)), expected, "{text:?}");
        }
    }

    #[test]
    fn any_other_value_is_inferred() {
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
            ("yes", r#""yes""#),
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
            ("\n[60,30]\n", "[60,30]"),
            ("", r#""""#),
            // Text that looks like JSON and is not one value: a string.
            ("[1, 2", r#""[1, 2""#),
            ("{'a': 1}", r#""{'a': 1}""#),
            ("1 2", r#""1 2""#),
        ] {
            assert_eq!(json(text, None), expected, "{text:?}");
        }
    }

    #[test]
    fn json_keeps_the_bytes_the_model_wrote() {
        for text in [
            "0.5",
            "-1e-5",
            "1.50",
            // A float could not hold this integer; the text does.
            "9007199254740993",
            r#"["a", "b"]"#,
            r#"{"a":1}"#,
            r#"{"nested": {"deep": [1, 2.5, true, null]}}"#,
        ] {
            assert_eq!(json(text, None), text, "{text:?}");
        }
    }

    #[test]
    fn a_declared_integer_with_more_digits_than_can_be_read_back_is_a_string() {
        let mut longest_kept = 0;
        for digits in [1, 19, 20, 21, 39, 100, 300, 308, 309, 310, 400, 5000] {
            let text = "9".repeat(digits);
            let written = json(&text, Some(Kind::Integer));
            assert!(
                read_back(&written).is_some(),
                "{digits} digits cost the call its arguments"
            );
            if written == text {
                assert!(longest_kept < digits);
                longest_kept = digits;
            } else {
                assert_eq!(written, format!("\"{text}\""), "{digits} digits");
            }
        }
        // Past what a machine integer holds the digits still come through, up to what a float can.
        assert!((100..400).contains(&longest_kept), "{longest_kept}");
    }

    /// The arguments object with `value` as one of its members, read the way the adapters read it.
    fn read_back(value: &str) -> Option<Value> {
        serde_json::from_str(&format!(r#"{{"p": {value}, "q": 1}}"#)).ok()
    }

    #[test]
    fn text_the_adapters_could_not_read_back_is_a_string() {
        // A lone surrogate escape is JSON by its grammar, and not a value serde_json reads.
        let surrogate = r#"["\ud83d"]"#;
        assert_eq!(json(surrogate, None), r#""[\"\\ud83d\"]""#);
        assert!(read_back(&json(surrogate, None)).is_some());
        // Two values are not one, whatever brackets around them would read as.
        assert_eq!(json("1, 2", None), r#""1, 2""#);
    }

    #[test]
    fn nesting_is_json_only_as_deep_as_the_arguments_object_can_hold_it() {
        let nested = |levels: usize| format!("{}{}", "[".repeat(levels), "]".repeat(levels));
        let mut deepest_kept = 0;
        for levels in 1..=140 {
            let text = nested(levels);
            let written = json(&text, None);
            assert!(
                read_back(&written).is_some(),
                "{levels} levels cost the call its arguments"
            );
            if written == text {
                assert_eq!(
                    deepest_kept,
                    levels - 1,
                    "{levels} kept after a shallower refusal"
                );
                deepest_kept = levels;
            } else {
                assert_eq!(written, format!("\"{text}\""), "{levels} levels");
            }
        }
        // The limit is serde_json's, less the level the object adds: one level deeper is still
        // JSON by itself, and no longer readable inside the object.
        assert!((100..140).contains(&deepest_kept), "{deepest_kept}");
        let one_deeper = nested(deepest_kept + 1);
        assert!(serde_json::from_str::<Value>(&one_deeper).is_ok());
        assert!(read_back(&one_deeper).is_none());
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
    fn whatever_the_text_and_the_kind_the_result_reads_back_as_one_json_value() {
        const SURROGATE: &str = r#""\ud800""#;
        const PIECES: [&str; 26] = [
            "", " ", "\n", "0", "-", "+", "1.5", "e9", "true", "True", "None", "null", "\"", "\\",
            "{", "}", "[", "]", ":", ",", "a", "é", "\u{1}", "<", "&amp;", SURROGATE,
        ];
        const KINDS: [Option<Kind>; 3] = [None, Some(Kind::String), Some(Kind::Integer)];
        // More digits than serde_json reads as a number, alone or after other digits.
        let digits = "9".repeat(400);
        let pieces = [PIECES.as_slice(), &[digits.as_str()]].concat();
        let mut random = Lcg(7);
        let mut json_but_for_a_surrogate = 0;
        for _ in 0..40_000 {
            let text: String = (0..random.next() % 8)
                .map(|_| pieces[random.next() % pieces.len()])
                .collect();
            // The string alone or in an array: JSON by its grammar, which serde_json refuses.
            let trimmed = text.trim();
            json_but_for_a_surrogate +=
                usize::from(trimmed == SURROGATE || trimmed == format!("[{SURROGATE}]"));
            for kind in KINDS {
                let written = json(&text, kind);
                let parsed: Result<Value, _> = serde_json::from_str(&written);
                assert!(parsed.is_ok(), "{kind:?} {text:?} gave {written:?}");
                if kind == Some(Kind::String) {
                    assert_eq!(parsed.ok(), Some(Value::String(text.clone())), "{text:?}");
                }
            }
        }
        assert!(
            json_but_for_a_surrogate > 0,
            "the generator never wrote text that only a surrogate escape keeps from being JSON"
        );
    }
}
