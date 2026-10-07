//! What the text of a tagged parameter means as JSON.
//!
//! A tagged format writes each argument as text between an opening and a closing tag, and the text
//! does not say what type it is: `150` is the number 150 for an integer parameter and the string
//! `"150"` for a string one. The template that wrote the text knew the type. The parser gets it
//! back from the parameters the tool declares: [`Declared`] holds those types for one request, and
//! [`json`] turns one value's text into the JSON a client receives for it.
//!
//! The text [`json`] takes is the value's own: the template writes one newline after the opening
//! tag and one before the closing tag, and the caller has removed exactly those two, as vLLM's
//! parsers do. Every other byte is the value's.
//!
//! The rules, in the order [`json`] applies them:
//!
//! - A parameter declared `string` is its text, every byte of it. Nothing is trimmed and nothing is
//!   decoded: `"exact phrase"` keeps its quotes, indentation and a trailing newline stay, and
//!   `&amp;` is five characters. A parameter is declared `string` when its schema admits that
//!   type at all: `"type": "string"`, a list of types or an `anyOf` that includes it, or, when the
//!   schema has no `type`, an `enum` of strings, with or without `null` among them. A `type`
//!   decides alone when it is there, as it does for vLLM: BFCL declares
//!   `{"type": "array", "enum": [...]}` for a list whose members come from the enum, and that is
//!   an array. When the schema admits `null` as well as `string`, the text `null`, and the `None`
//!   a template writes for a null argument, are null; any other text is the string. A string
//!   declared alone keeps `null` as its text, as vLLM's readers do; bellwether #56 refuses a case
//!   whose reference holds a null there, since no output carries it.
//! - A parameter declared `integer` is that integer when its text is a sign and digits, with the
//!   whitespace around it ignored. It may be spelled `+5` or `007`; it is written in JSON's
//!   spelling, with the digits the model wrote.
//! - Everything else is inferred: text that is JSON is that JSON; text that is a Python literal
//!   (`True`, `None`, `{'size': 'large'}`, `['n1', 'n2']`, a tuple) is the JSON it stands for,
//!   written with `json.dumps`'s separators; and the rest is a string holding the text exactly.
//!   That covers a parameter the tool does not declare, a declared integer whose text is not one,
//!   and every other declared type. Most templates write an array or object with `tojson` and a
//!   boolean or null as Python's `True`, `False` and `None`; Seed-OSS's writes every value with
//!   `{{ value }}`, so an object comes as Python's repr. Inference reads both, so declaring a
//!   `number`, `boolean`, `array` or `object` changes nothing.
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

/// The texts that mean null for a nullable string: JSON's word and the `None` a template writes
/// for a null argument.
pub(crate) const NULL_WORDS: [&str; 2] = ["null", "None"];

/// The type a tool declares for one parameter, of the types that change how its text is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A schema that admits `string`: the text is the string.
    String,
    /// A schema that admits `string` and `null`: `null` or `None` is null, any other text the
    /// string.
    NullableString,
    /// A schema that admits `integer` and nothing else.
    Integer,
}

impl Kind {
    /// The kind a parameter's schema declares. A schema that admits a string declares a string,
    /// nullable when it admits `null` too; one that admits an integer alone declares an integer;
    /// any other schema declares nothing here.
    fn of(schema: &Value) -> Option<Self> {
        let mut admitted = Vec::new();
        admitted_types(schema, &mut admitted);
        if admitted.contains(&"string") {
            Some(if admitted.contains(&"null") {
                Self::NullableString
            } else {
                Self::String
            })
        } else if admitted == ["integer"] {
            Some(Self::Integer)
        } else {
            None
        }
    }
}

/// The type names `schema` admits: its `type`, as one name or a list; the types of its `anyOf` or
/// `oneOf` members; and, when it has no `type`, `string` for an `enum` of strings, and `null` too
/// when `null` is among them (what Pydantic writes for `Literal["a", "b", None]`).
fn admitted_types<'a>(schema: &'a Value, into: &mut Vec<&'a str>) {
    let typed = match schema.get("type") {
        Some(Value::String(name)) => {
            into.push(name);
            true
        }
        Some(Value::Array(names)) => {
            into.extend(names.iter().filter_map(Value::as_str));
            true
        }
        _ => false,
    };
    for key in ["anyOf", "oneOf"] {
        let members = schema.get(key).and_then(Value::as_array);
        for member in members.into_iter().flatten() {
            admitted_types(member, into);
        }
    }
    if typed {
        return;
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let strings_or_null = values
            .iter()
            .all(|value| value.is_string() || value.is_null());
        if strings_or_null && values.iter().any(Value::is_string) {
            into.push("string");
            if values.iter().any(Value::is_null) {
                into.push("null");
            }
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
///
/// `text` is the value's own text. The caller has removed the one newline the template writes
/// after the opening tag and the one it writes before the closing tag, and nothing else: a
/// declared string keeps every byte it is given.
pub fn json(text: &str, kind: Option<Kind>) -> String {
    match kind {
        Some(Kind::String) => string(text),
        Some(Kind::NullableString) if NULL_WORDS.contains(&text) => "null".to_string(),
        Some(Kind::NullableString) => string(text),
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
    match python_literal(trimmed) {
        Some(json) if reads_back_as_an_argument(&json) => json,
        _ => string(text),
    }
}

/// The JSON a Python literal stands for, written with `json.dumps`'s separators, or `None` when
/// `text` is not one. The literals a template's `{{ value }}` writes for a JSON value: `None`,
/// `True`, `False`, numbers, strings in single or double quotes with Python's escapes, lists,
/// tuples and dicts, with trailing commas allowed. Anything else, a bare word or an apostrophe in
/// prose among them, is not a literal.
fn python_literal(text: &str) -> Option<String> {
    let mut reader = Literal {
        bytes: text.as_bytes(),
        text,
        at: 0,
        out: String::with_capacity(text.len()),
        depth: 0,
    };
    reader.value()?;
    reader.skip_space();
    (reader.at == text.len()).then_some(reader.out)
}

/// A reader over one Python literal, writing its JSON as it goes.
struct Literal<'a> {
    bytes: &'a [u8],
    text: &'a str,
    at: usize,
    out: String,
    /// Containers open around the value being read; past [`DEEPEST`] the literal is refused.
    depth: usize,
}

/// The deepest nesting the reader follows: `serde_json`'s own limit, which the result has to pass
/// anyway. The text is the model's, so a run of ten thousand `[` must not run the stack out.
const DEEPEST: usize = 128;

impl Literal<'_> {
    fn skip_space(&mut self) {
        while self.bytes.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.bytes.get(self.at) == Some(&byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn word(&mut self, word: &str, json: &str) -> bool {
        if self.text[self.at..].starts_with(word)
            && !self
                .bytes
                .get(self.at + word.len())
                .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            self.at += word.len();
            self.out.push_str(json);
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<()> {
        self.skip_space();
        if self.word("None", "null") || self.word("True", "true") || self.word("False", "false") {
            return Some(());
        }
        match *self.bytes.get(self.at)? {
            b'\'' | b'"' => self.string(),
            b'[' | b'(' | b'{' => {
                if self.depth >= DEEPEST {
                    return None;
                }
                self.depth += 1;
                let read = match self.bytes[self.at] {
                    b'[' => self.sequence(b'[', b']'),
                    b'(' => self.sequence(b'(', b')'),
                    _ => self.dict(),
                };
                self.depth -= 1;
                read
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn string(&mut self) -> Option<()> {
        let quote = *self.bytes.get(self.at)?;
        self.at += 1;
        let mut value = String::new();
        loop {
            let rest = &self.text[self.at..];
            let mut chars = rest.char_indices();
            let (_, c) = chars.next()?;
            self.at += c.len_utf8();
            if c == char::from(quote) {
                break;
            }
            if c != '\\' {
                value.push(c);
                continue;
            }
            let (_, escaped) = chars.next()?;
            self.at += escaped.len_utf8();
            match escaped {
                'n' => value.push('\n'),
                't' => value.push('\t'),
                'r' => value.push('\r'),
                'a' => value.push('\u{7}'),
                'b' => value.push('\u{8}'),
                'f' => value.push('\u{c}'),
                'v' => value.push('\u{b}'),
                // An octal escape: up to three octal digits, the first already read.
                '0'..='7' => {
                    let mut code = escaped.to_digit(8)?;
                    for _ in 0..2 {
                        let Some(digit) = self
                            .bytes
                            .get(self.at)
                            .and_then(|b| char::from(*b).to_digit(8))
                        else {
                            break;
                        };
                        code = code * 8 + digit;
                        self.at += 1;
                    }
                    value.push(char::from_u32(code)?);
                }
                'x' | 'u' | 'U' => {
                    let digits = match escaped {
                        'x' => 2,
                        'u' => 4,
                        _ => 8,
                    };
                    let hex = self.text.get(self.at..self.at + digits)?;
                    let code = u32::from_str_radix(hex, 16).ok()?;
                    value.push(char::from_u32(code)?);
                    self.at += digits;
                }
                '\\' | '\'' | '"' => value.push(escaped),
                // A backslash before a line ending is the continuation Python drops; Python reads
                // `\r\n` and a lone `\r` as line endings too.
                '\n' => {}
                '\r' => {
                    if self.bytes.get(self.at) == Some(&b'\n') {
                        self.at += 1;
                    }
                }
                // Python keeps the backslash on an escape it does not have: `'\d+'` is three
                // characters.
                other => {
                    value.push('\\');
                    value.push(other);
                }
            }
        }
        self.out.push_str(&string(&value));
        Some(())
    }

    fn number(&mut self) -> Option<()> {
        let start = self.at;
        if self.eat(b'-') {
            self.skip_space();
        }
        let digits = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'.' || *b == b'_')
        {
            self.at += 1;
        }
        // An exponent's sign: `1e-05` is how repr writes 0.00001.
        if self.at > digits
            && matches!(self.bytes.get(self.at - 1), Some(b'e' | b'E'))
            && matches!(self.bytes.get(self.at), Some(b'+' | b'-'))
        {
            self.at += 1;
            while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                self.at += 1;
            }
        }
        if self.at == digits {
            return None;
        }
        // Python allows a sign before a number and `_` between digits; JSON allows neither, and a
        // leading `+` or zeros are not written by `repr`, so the digits are checked as JSON.
        let text: String = self.text[digits..self.at].replace('_', "");
        let json = if self.text[start..].starts_with('-') {
            format!("-{text}")
        } else {
            text
        };
        serde_json::from_str::<serde_json::Number>(&json).ok()?;
        self.out.push_str(&json);
        Some(())
    }

    /// A list or a tuple. A parenthesis around one item with no comma is grouping, as Python
    /// reads it: `(1)` is `1`, and only `(1,)` is a tuple.
    fn sequence(&mut self, open: u8, close: u8) -> Option<()> {
        self.eat(open).then_some(())?;
        let bracket = self.out.len();
        self.out.push('[');
        let mut items = 0;
        let mut comma = false;
        loop {
            self.skip_space();
            if self.eat(close) {
                break;
            }
            if items > 0 {
                self.out.push_str(", ");
            }
            items += 1;
            self.value()?;
            self.skip_space();
            if !self.eat(b',') {
                self.skip_space();
                self.eat(close).then_some(())?;
                break;
            }
            comma = true;
        }
        if open == b'(' && items == 1 && !comma {
            self.out.remove(bracket);
        } else {
            self.out.push(']');
        }
        Some(())
    }

    fn dict(&mut self) -> Option<()> {
        self.eat(b'{').then_some(())?;
        self.out.push('{');
        let mut first = true;
        loop {
            self.skip_space();
            if self.eat(b'}') {
                break;
            }
            if !first {
                self.out.push_str(", ");
            }
            first = false;
            // A key is written as a JSON string, as `json.dumps` writes a non-string key.
            let key_start = self.out.len();
            self.value()?;
            if !self.out[key_start..].starts_with('"') {
                let key = self.out.split_off(key_start);
                self.out.push_str(&string(&key));
            }
            self.skip_space();
            self.eat(b':').then_some(())?;
            self.out.push_str(": ");
            self.value()?;
            self.skip_space();
            if !self.eat(b',') {
                self.skip_space();
                self.eat(b'}').then_some(())?;
                break;
            }
        }
        self.out.push('}');
        Some(())
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
    fn a_schema_that_admits_a_string_declares_one_nullable_when_it_admits_null_too() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "plain": {"type": "string"},
                "listed": {"type": ["string", "null"]},
                // What Pydantic writes for `Optional[str]`.
                "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "either": {"oneOf": [{"type": "null"}, {"type": "string"}]},
                "choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "named": {"enum": ["celsius", "fahrenheit"]},
                "named_or_null": {"enum": ["celsius", "fahrenheit"], "type": ["string", "null"]},
                "named_or_none": {"enum": ["celsius", "fahrenheit", null]},
                "nested": {"anyOf": [{"anyOf": [{"type": "string"}]}, {"type": "null"}]},
            }),
        )]);
        for (parameter, kind) in [
            ("plain", Kind::String),
            ("listed", Kind::NullableString),
            ("optional", Kind::NullableString),
            ("either", Kind::NullableString),
            ("choice", Kind::String),
            ("named", Kind::String),
            ("named_or_null", Kind::NullableString),
            ("named_or_none", Kind::NullableString),
            ("nested", Kind::NullableString),
        ] {
            assert_eq!(declared.kind("f", parameter), Some(kind), "{parameter}");
        }
    }

    #[test]
    fn a_type_decides_alone_and_an_enum_beside_it_adds_nothing() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                // BFCL's shape for a list whose members come from the enum.
                "members": {"type": "array", "items": {"type": "string"}, "enum": ["a", "b"]},
                "counted": {"type": "integer", "enum": ["one", "two"]},
                "named": {"type": "string", "enum": [1, 2]},
                "listed": {"type": ["string", "null"], "enum": ["a", "b"]},
            }),
        )]);
        assert_eq!(declared.kind("f", "members"), None);
        assert_eq!(declared.kind("f", "counted"), Some(Kind::Integer));
        assert_eq!(declared.kind("f", "named"), Some(Kind::String));
        assert_eq!(declared.kind("f", "listed"), Some(Kind::NullableString));
    }

    #[test]
    fn an_integer_is_declared_alone_and_every_other_schema_declares_nothing() {
        let declared = Declared::of(&[tool(
            "f",
            value!({
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "on": {"type": "boolean"},
                "points": {"type": "array"},
                "style": {"type": "object"},
                "nothing": {"type": "null"},
                "optional_count": {"type": ["integer", "null"]},
                "untyped": {"description": "anything"},
                "mixed": {"enum": [1, "two"]},
                "empty": {"enum": []},
                "only_null": {"enum": [null]},
                // Aliases and other spellings of a type name are not read today; vLLM reads
                // these as string and integer. Whether Symphony follows is the maintainer's
                // decision.
                "alias": {"type": "str"},
                "cased": {"type": "String"},
                "padded": {"type": " integer "},
                "width": {"type": "int32"},
            }),
        )]);
        assert_eq!(declared.kind("f", "count"), Some(Kind::Integer));
        for parameter in [
            "ratio",
            "on",
            "points",
            "style",
            "nothing",
            "optional_count",
            "untyped",
            "mixed",
            "empty",
            "only_null",
            "alias",
            "cased",
            "padded",
            "width",
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
    fn a_nullable_string_is_null_for_null_or_none_and_otherwise_its_text() {
        for (text, expected) in [
            ("null", "null"),
            ("None", "null"),
            ("Paris", r#""Paris""#),
            ("12345", r#""12345""#),
            ("true", r#""true""#),
            ("", r#""""#),
            // Only the bare words: the value's own text is kept as it is.
            (" null", r#"" null""#),
            ("NULL", r#""NULL""#),
            ("none", r#""none""#),
            (r#""null""#, r#""\"null\"""#),
        ] {
            assert_eq!(json(text, Some(Kind::NullableString)), expected, "{text:?}");
        }
    }

    #[test]
    fn a_declared_integer_is_a_number_when_its_text_is_one() {
        for (text, expected) in [
            ("5", "5"),
            ("-3", "-3"),
            (" 150 ", "150"),
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
            // Python reads `1_000` as a number, and so does inference now.
            ("1_000", "1000"),
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
            (" [60,30] ", "[60,30]"),
            ("", r#""""#),
            // Text that looks like JSON and is not one value: a string.
            ("[1, 2", r#""[1, 2""#),
            // Python's repr of an object, as Seed-OSS's template writes it.
            ("{'a': 1}", r#"{"a": 1}"#),
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
        let mut refused = false;
        for digits in [1, 19, 20, 21, 39, 100, 300, 308, 309, 310, 400, 5000] {
            let text = "9".repeat(digits);
            let written = json(&text, Some(Kind::Integer));
            assert!(
                read_back(&written).is_some(),
                "{digits} digits cost the call its arguments"
            );
            if written == text {
                assert!(
                    !refused,
                    "{digits} digits kept after a shorter count was refused"
                );
                longest_kept = digits;
            } else {
                refused = true;
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
        // A number past what a float holds is JSON by its grammar and not a value serde_json
        // reads; it stays a string, and would stop being one if `arbitrary_precision` were on.
        assert_eq!(json("1e400", None), r#""1e400""#);
        assert!(read_back(&json("1e400", None)).is_some());
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
    fn a_python_literal_is_the_json_it_stands_for_with_json_dumps_separators() {
        // Seed-OSS's template writes every value with `{{ value }}`: bellwether's
        // seed-oss-36b-instruct/parse/bfcl-live-multiple-0-0-0 and -146-58-0.
        for (text, expected) in [
            (
                "{'size': 'large', 'milk_type': 'coconut'}",
                r#"{"size": "large", "milk_type": "coconut"}"#,
            ),
            ("['n1', 'n2']", r#"["n1", "n2"]"#),
            ("[]", "[]"),
            ("{}", "{}"),
            ("(1, 2)", "[1, 2]"),
            ("()", "[]"),
            // A parenthesis around one item is grouping, not a tuple.
            ("(1)", "1"),
            ("('a')", "\"a\""),
            ("((1))", "1"),
            ("{'x': (1), 'y': (1,)}", "{\"x\": 1, \"y\": [1]}"),
            ("('solo',)", r#"["solo"]"#),
            (
                "{'a': None, 'b': True, 'c': False}",
                r#"{"a": null, "b": true, "c": false}"#,
            ),
            (
                "{'n': -3, 'x': 2.5, 'big': 1_000}",
                r#"{"n": -3, "x": 2.5, "big": 1000}"#,
            ),
            (
                "{'q': \"it's\", 'e': 'a\\'b\\n'}",
                r#"{"q": "it's", "e": "a'b\n"}"#,
            ),
            ("{'u': '\\u00e9\\x41'}", r#"{"u": "éA"}"#),
            (
                "[{'k': [1, {'d': 'e'}]}, 'tail',]",
                r#"[{"k": [1, {"d": "e"}]}, "tail"]"#,
            ),
            ("{1: 'one'}", r#"{"1": "one"}"#),
            ("None", "null"),
            // repr's spellings of small and large floats, with the exponent's sign.
            ("{'x': 1e-05}", r#"{"x": 1e-05}"#),
            ("{'x': 1.5e-07, 'n': 'a'}", r#"{"x": 1.5e-07, "n": "a"}"#),
            ("[-2.5e-05, 1e+16, 2E3]", "[-2.5e-05, 1e+16, 2E3]"),
        ] {
            assert_eq!(json(text, None), expected, "{text:?}");
        }
    }

    #[test]
    fn nesting_past_serde_jsons_depth_is_a_string_and_never_runs_the_stack_out() {
        // A model can write any number of brackets; the reader stops where serde_json would.
        for text in [
            "[".repeat(100_000),
            "{'a': ".repeat(50_000),
            "(".repeat(200) + &")".repeat(200),
        ] {
            assert_eq!(json(&text, None), string(&text), "{} bytes", text.len());
        }
        // Well inside the limit, a literal still reads. `None` is not JSON, so this goes through
        // the Python reader; serde_json counts the arguments object around the value too, so the
        // read-back refuses a little before the reader's own limit would.
        let nested = "[".repeat(100) + "None" + &"]".repeat(100);
        assert_eq!(
            json(&nested, None),
            "[".repeat(100) + "null" + &"]".repeat(100)
        );
    }

    #[test]
    fn an_escape_python_does_not_have_keeps_its_backslash() {
        let written = json(
            concat!(
                r"{'pattern': '\d+', 'path': 'C:\path', 'cut': 'a\",
                "\n",
                r"b', 'crlf': 'a\",
                "\r\n",
                r"b', 'cr': 'a\",
                "\r",
                r"b'}"
            ),
            None,
        );
        let read: Value = serde_json::from_str(&written).expect("an object");
        assert_eq!(read["pattern"], "\\d+");
        assert_eq!(read["path"], "C:\\path");
        assert_eq!(read["cut"], "ab");
        assert_eq!(read["crlf"], "ab");
        assert_eq!(read["cr"], "ab");
    }

    #[test]
    fn pythons_other_escapes_are_read_though_repr_never_writes_them() {
        // The text holds backslashes, written here as escapes so that no tool turns them into
        // the control characters themselves.
        let text = concat!("{'e': '", "\\a\\b\\f\\v\\101\\7\\0z", "'}");
        let written = json(text, None);
        let read: Value = serde_json::from_str(&written).expect("an object");
        assert_eq!(read["e"], "\u{7}\u{8}\u{c}\u{b}A\u{7}\0z");
        assert_eq!(
            json(r"1e-", None),
            string("1e-"),
            "an exponent with no digits"
        );
        assert_eq!(json(r"1e", None), string("1e"));
    }

    #[test]
    fn text_that_is_not_a_literal_stays_a_string() {
        for text in [
            "it's",
            "don't do that",
            "{'a': 1",
            "['a' 'b']",
            "{'a' 1}",
            "Nones",
            "1 2",
            "[1,, 2]",
            "{'a': }",
            "'unterminated",
            "x = {'a': 1}",
            "{'a': 1} and more",
            "- 5 apples",
        ] {
            assert_eq!(json(text, None), string(text), "{text:?}");
        }
        // JSON stays as the model wrote it, spacing and digits included; the literal reader does
        // not get to rewrite it.
        assert_eq!(json("[1,2]", None), "[1,2]");
        assert_eq!(json("9007199254740993", None), "9007199254740993");
    }

    #[test]
    fn whatever_the_text_and_the_kind_the_result_reads_back_as_one_json_value() {
        const SURROGATE: &str = r#""\ud800""#;
        const PIECES: [&str; 26] = [
            "", " ", "\n", "0", "-", "+", "1.5", "e9", "true", "True", "None", "null", "\"", "\\",
            "{", "}", "[", "]", ":", ",", "a", "é", "\u{1}", "<", "&amp;", SURROGATE,
        ];
        const KINDS: [Option<Kind>; 4] = [
            None,
            Some(Kind::String),
            Some(Kind::NullableString),
            Some(Kind::Integer),
        ];
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
                if kind == Some(Kind::String)
                    || (kind == Some(Kind::NullableString) && text != "null" && text != "None")
                {
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
