//! A parser for JSON that may stop anywhere.
//!
//! [`PartialJson::parse`] reads as much of a JSON value as the input contains and returns the value
//! it determines so far together with the number of bytes it consumed. In prefix mode (the default)
//! an input cut at any point parses: an open object or array closes where the input ends, a value
//! that has not started yet is `null`, and an unfinished string is returned as written or, when the
//! caller asks for complete strings only, left out together with its key. The byte count is always
//! on a character boundary, so the caller can slice the input by it.
//!
//! Ported from `crates/tool_parser/src/partial_json.rs` (smg 897ef9a9) under ground rule 27. What
//! changed in form: an error type of its own, `is_complete` as a function of this module, the
//! cursor's literal parsing shared by `true`, `false` and `null`, names and documentation. What
//! changed in behaviour, twice. A word of letters that is no literal never advances the cursor: the
//! original looked ahead over at most the longest literal of the kind (five bytes for either
//! boolean, four for `null`), so `truex` was rejected unseen but `false` or `null` followed by
//! letters passed the look-ahead, was consumed, and was rejected only then (`[1, falsex` consumed
//! ten bytes there and `[1, nullx` nine, both consume four here). And a surrogate pair written as
//! two `\u` escapes is one character: the original could not read one and stopped at its first
//! half, so an object holding such a string came back without it and with the parse ending there;
//! here the pair decodes, a cut inside it is an unfinished escape, and a surrogate without its
//! other half is invalid, as before. Everything else behaves as before, since the ported streaming
//! parser must reproduce the old one until bellwether's fixtures judge otherwise; the choices worth
//! a second look are named here so that the parity review finds them: an unknown escape keeps the
//! escaped character, an unfinished `\u` escape becomes U+FFFD, a number that cannot be read
//! becomes `0`, a prefix of `true`, `false` or `null` counts as the literal, and a trailing comma
//! before a closing bracket is accepted.

use serde::{de::IgnoredAny, Deserialize};
use serde_json::{Deserializer, Map, Number, Value};

/// Why a parse stopped with an error rather than a value.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PartialJsonError {
    /// The value nests deeper than the parser allows.
    #[error("JSON nests deeper than {0} levels")]
    TooDeep(usize),
    /// The input is not JSON, and the parser was not asked to accept a prefix.
    #[error("not JSON: {0}")]
    Invalid(&'static str),
}

type Parsed<T> = Result<T, PartialJsonError>;

/// Parses JSON prefixes. Cheap to construct and stateless between calls.
#[derive(Clone, Copy, Debug)]
pub struct PartialJson {
    max_depth: usize,
    allow_incomplete: bool,
}

impl PartialJson {
    /// The nesting depth [`PartialJson::default`] allows.
    pub const DEFAULT_MAX_DEPTH: usize = 32;

    /// A parser that rejects values nesting deeper than `max_depth` and, with `allow_incomplete`,
    /// accepts an input cut at any point; without it an incomplete input is an error.
    pub fn new(max_depth: usize, allow_incomplete: bool) -> Self {
        Self {
            max_depth,
            allow_incomplete,
        }
    }

    /// Parse `input` and return the value determined so far and the bytes consumed.
    ///
    /// With `allow_partial_strings` an unfinished string is returned as written; without it the
    /// parse stops before the string, and an object leaves out the key whose value it was, which
    /// is how an argument stream avoids emitting a string fragment that may still grow.
    pub fn parse(&self, input: &str, allow_partial_strings: bool) -> Parsed<(Value, usize)> {
        let mut cursor = Cursor {
            chars: input.chars().peekable(),
            position: 0,
            max_depth: self.max_depth,
            allow_incomplete: self.allow_incomplete,
            allow_partial_strings,
        };
        let value = cursor.value(0)?;
        Ok((value, cursor.position))
    }

    /// The nesting depth this parser allows.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
}

impl Default for PartialJson {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_DEPTH, true)
    }
}

/// Whether `input` is one complete JSON value with nothing but whitespace after it.
pub fn is_complete(input: &str) -> bool {
    let mut de = Deserializer::from_str(input);
    IgnoredAny::deserialize(&mut de).is_ok() && de.end().is_ok()
}

/// One pass over the input.
struct Cursor<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    position: usize,
    max_depth: usize,
    allow_incomplete: bool,
    allow_partial_strings: bool,
}

impl Cursor<'_> {
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn advance(&mut self) {
        if let Some(c) = self.chars.next() {
            self.position += c.len_utf8();
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.advance();
        }
    }

    fn value(&mut self, depth: usize) -> Parsed<Value> {
        if depth > self.max_depth {
            return Err(PartialJsonError::TooDeep(self.max_depth));
        }
        self.skip_whitespace();
        match self.peek() {
            Some('{') => self.object(depth + 1),
            Some('[') => self.array(depth + 1),
            Some('"') => self.string(),
            Some('t') => self.literal("true", Value::Bool(true)),
            Some('f') => self.literal("false", Value::Bool(false)),
            Some('n') => self.literal("null", Value::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            _ if self.allow_incomplete => Ok(Value::Null),
            _ => Err(PartialJsonError::Invalid("unexpected character")),
        }
    }

    fn object(&mut self, depth: usize) -> Parsed<Value> {
        if depth > self.max_depth {
            return Err(PartialJsonError::TooDeep(self.max_depth));
        }
        let mut object = Map::new();
        self.advance();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(Value::Object(object));
        }
        loop {
            let key = match self.string() {
                Ok(Value::String(key)) => key,
                Err(_) if self.allow_incomplete => return Ok(Value::Object(object)),
                Err(e) => return Err(e),
                Ok(_) => return Err(PartialJsonError::Invalid("expected a string key")),
            };
            self.skip_whitespace();
            if self.peek() != Some(':') {
                if self.allow_incomplete {
                    object.insert(key, Value::Null);
                    return Ok(Value::Object(object));
                }
                return Err(PartialJsonError::Invalid("expected ':'"));
            }
            self.advance();
            self.skip_whitespace();
            let value = match self.value(depth) {
                Ok(value) => value,
                Err(_) if self.allow_incomplete => {
                    // The value is unfinished. With partial strings allowed it stands as `null`;
                    // without them the key is left out, so the caller sees nothing that may grow.
                    if self.allow_partial_strings {
                        object.insert(key, Value::Null);
                    }
                    return Ok(Value::Object(object));
                }
                Err(e) => return Err(e),
            };
            object.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    if self.peek() == Some('}') {
                        self.advance();
                        return Ok(Value::Object(object));
                    }
                }
                Some('}') => {
                    self.advance();
                    return Ok(Value::Object(object));
                }
                _ if self.allow_incomplete => return Ok(Value::Object(object)),
                _ => return Err(PartialJsonError::Invalid("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Parsed<Value> {
        if depth > self.max_depth {
            return Err(PartialJsonError::TooDeep(self.max_depth));
        }
        let mut array = Vec::new();
        self.advance();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.advance();
            return Ok(Value::Array(array));
        }
        loop {
            let value = match self.value(depth) {
                Ok(value) => value,
                Err(_) if self.allow_incomplete => return Ok(Value::Array(array)),
                Err(e) => return Err(e),
            };
            array.push(value);
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    if self.peek() == Some(']') {
                        self.advance();
                        return Ok(Value::Array(array));
                    }
                }
                Some(']') => {
                    self.advance();
                    return Ok(Value::Array(array));
                }
                _ if self.allow_incomplete => return Ok(Value::Array(array)),
                _ => return Err(PartialJsonError::Invalid("expected ',' or ']'")),
            }
        }
    }

    fn string(&mut self) -> Parsed<Value> {
        if self.peek() != Some('"') {
            return Err(PartialJsonError::Invalid("expected '\"'"));
        }
        self.advance();
        let mut string = String::new();
        let mut escaped = false;
        while let Some(ch) = self.peek() {
            if escaped {
                let unescaped = match ch {
                    'b' => '\u{0008}',
                    'f' => '\u{000C}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'u' => {
                        self.advance();
                        string.push(self.unicode_escape()?);
                        escaped = false;
                        continue;
                    }
                    // `"`, `\` and `/` stand for themselves; so does an unknown escape, leniently.
                    other => other,
                };
                string.push(unescaped);
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                self.advance();
                return Ok(Value::String(string));
            } else {
                string.push(ch);
            }
            self.advance();
        }
        if self.allow_incomplete && self.allow_partial_strings {
            Ok(Value::String(string))
        } else {
            Err(PartialJsonError::Invalid("unterminated string"))
        }
    }

    /// The character after `\u`: four hex digits, or a surrogate pair written as two `\u` escapes.
    ///
    /// Fewer than four digits, or a pair cut after its first half, is an unfinished escape: U+FFFD
    /// in prefix mode, where the string is still arriving, and an error otherwise. A surrogate
    /// without its other half is invalid in both modes.
    fn unicode_escape(&mut self) -> Parsed<char> {
        let Some(high) = self.hex_unit() else {
            return self.unfinished_escape();
        };
        if !(0xD800..0xDC00).contains(&high) {
            return char::from_u32(high).ok_or(PartialJsonError::Invalid("invalid unicode escape"));
        }
        let mut ahead = self.chars.clone();
        match (ahead.next(), ahead.next()) {
            (Some('\\'), Some('u')) => {
                self.advance();
                self.advance();
                let Some(low) = self.hex_unit() else {
                    return self.unfinished_escape();
                };
                if (0xDC00..0xE000).contains(&low) {
                    let scalar = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                    return char::from_u32(scalar)
                        .ok_or(PartialJsonError::Invalid("invalid unicode escape"));
                }
                Err(PartialJsonError::Invalid("invalid unicode escape"))
            }
            (None, _) | (Some('\\'), None) => self.unfinished_escape(),
            _ => Err(PartialJsonError::Invalid("invalid unicode escape")),
        }
    }

    /// Four hex digits, consumed, as a number; `None` when fewer than four were there.
    fn hex_unit(&mut self) -> Option<u32> {
        let mut hex = String::new();
        while hex.len() < 4 && self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
            hex.push(self.peek().unwrap_or_default());
            self.advance();
        }
        (hex.len() == 4)
            .then(|| u32::from_str_radix(&hex, 16).ok())
            .flatten()
    }

    fn unfinished_escape(&self) -> Parsed<char> {
        if self.allow_incomplete {
            Ok('\u{FFFD}')
        } else {
            Err(PartialJsonError::Invalid("incomplete unicode escape"))
        }
    }

    fn number(&mut self) -> Parsed<Value> {
        let mut text = String::new();
        if self.peek() == Some('-') {
            text.push('-');
            self.advance();
        }
        if self.peek() == Some('0') {
            text.push('0');
            self.advance();
        } else {
            self.digits(&mut text);
        }
        if self.peek() == Some('.') {
            text.push('.');
            self.advance();
            self.digits(&mut text);
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            text.push('e');
            self.advance();
            if let Some(sign @ ('+' | '-')) = self.peek() {
                text.push(sign);
                self.advance();
            }
            self.digits(&mut text);
        }
        if let Ok(n) = text.parse::<i64>() {
            Ok(Value::Number(Number::from(n)))
        } else if let Ok(n) = text.parse::<f64>() {
            Ok(Value::Number(
                Number::from_f64(n).unwrap_or_else(|| Number::from(0)),
            ))
        } else if self.allow_incomplete {
            Ok(Value::Number(Number::from(0)))
        } else {
            Err(PartialJsonError::Invalid("invalid number"))
        }
    }

    fn digits(&mut self, into: &mut String) {
        while let Some(digit) = self.peek().filter(char::is_ascii_digit) {
            into.push(digit);
            self.advance();
        }
    }

    /// `true`, `false` or `null` (`expected`), or in prefix mode a prefix of it.
    ///
    /// The whole run of letters is looked at before anything is consumed, so a word that is no
    /// literal leaves the position where it was and the enclosing value closes before it:
    /// `[1, truex`, `[1, falsex` and `[1, nullx` all give `[1]` after four bytes. The original
    /// looked ahead over at most the longest literal of the kind (five bytes for either boolean,
    /// four for `null`), which let `false` or `null` followed by letters advance the cursor before
    /// the word was rejected; that is one of the two behaviours this port changes, and the module
    /// doc names both.
    fn literal(&mut self, expected: &'static str, value: Value) -> Parsed<Value> {
        let word: String = self
            .chars
            .clone()
            .take_while(|c| c.is_alphabetic())
            .collect();
        if !(word == expected || (self.allow_incomplete && expected.starts_with(&word))) {
            return Err(PartialJsonError::Invalid("invalid literal"));
        }
        for _ in word.chars() {
            self.advance();
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn prefix(input: &str) -> (Value, usize) {
        PartialJson::default()
            .parse(input, true)
            .expect("prefix mode accepts any cut")
    }

    #[test]
    fn a_complete_value_parses_whole_and_consumes_every_byte() {
        let input =
            r#"{"name": "test", "value": 42, "flags": [true, false, null], "ratio": -1.5e2}"#;
        let (value, consumed) = prefix(input);
        assert_eq!(
            value,
            json!({"name": "test", "value": 42, "flags": [true, false, null], "ratio": -150.0})
        );
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn a_cut_input_gives_the_value_determined_so_far() {
        assert_eq!(
            prefix(r#"{"name": "test", "value": "#).0,
            json!({"name": "test", "value": null})
        );
        assert_eq!(prefix(r#"{"name": "tes"#).0, json!({"name": "tes"}));
        assert_eq!(
            prefix("[1, 2, ").0,
            json!([1, 2, null]),
            "a value that has not started after a comma is null"
        );
        assert_eq!(
            prefix(r#"{"a": {"b": [1, {"c""#).0,
            json!({"a": {"b": [1, {"c": null}]}})
        );
    }

    #[test]
    fn every_cut_of_a_value_parses_in_prefix_mode_without_consuming_past_the_cut() {
        let input =
            r#"{"city": "Paris", "days": [1, 2.5, -3e1], "ok": true, "none": null, "é": "é\n"}"#;
        for cut in 0..=input.len() {
            if !input.is_char_boundary(cut) {
                continue;
            }
            let (_, consumed) = PartialJson::default()
                .parse(&input[..cut], true)
                .unwrap_or_else(|e| panic!("cut at {cut}: {e}"));
            assert!(consumed <= cut, "cut at {cut} consumed {consumed}");
            assert!(input.is_char_boundary(consumed));
        }
    }

    #[test]
    fn consumed_is_a_byte_offset_on_a_character_boundary() {
        for input in [r#"{"k":"é"}"#, "{\"k\":\"é", r#"{"k":"🌍"}"#] {
            let (value, consumed) = prefix(input);
            assert_eq!(
                value["k"].as_str().map(|s| s.chars().next()),
                Some(input.chars().nth(6))
            );
            assert!(input.is_char_boundary(consumed), "{input}: {consumed}");
            if input.ends_with('}') {
                assert_eq!(consumed, input.len());
            }
        }
    }

    #[test]
    fn complete_strings_only_leaves_out_the_key_of_an_unfinished_string() {
        let parser = PartialJson::default();
        let (value, consumed) = parser.parse(r#"{"name": ""#, false).expect("a prefix");
        assert_eq!(
            value,
            json!({}),
            "the pending key is left out, as Python's Allow.ALL & ~Allow.STR does"
        );
        assert_eq!(consumed, 10);
        let (nested, _) = parser
            .parse(r#"{"tool": {"name": ""#, false)
            .expect("a prefix");
        assert_eq!(nested, json!({"tool": {}}));
        let (finished, _) = parser.parse(r#"{"name": "te"#, true).expect("a prefix");
        assert_eq!(finished, json!({"name": "te"}));
    }

    #[test]
    fn the_string_flag_makes_no_difference_to_a_complete_value() {
        let input = r#"{"name": "test"}"#;
        let parser = PartialJson::default();
        assert_eq!(
            parser.parse(input, false).expect("complete"),
            parser.parse(input, true).expect("complete")
        );
        assert_eq!(parser.parse(input, false).expect("complete").1, input.len());
    }

    #[test]
    fn nesting_past_the_limit_is_an_error() {
        let parser = PartialJson::new(3, false);
        assert!(parser.parse(r#"{"a": 1}"#, true).is_ok());
        assert!(parser.parse(r#"{"a": {"b": {"c": 1}}}"#, true).is_ok());
        assert_eq!(
            parser.parse(r#"{"a": {"b": {"c": {"d": 1}}}}"#, true),
            Err(PartialJsonError::TooDeep(3))
        );
        assert_eq!(
            PartialJson::default().max_depth(),
            PartialJson::DEFAULT_MAX_DEPTH
        );
    }

    #[test]
    fn without_prefix_mode_an_unfinished_input_is_an_error() {
        let strict = PartialJson::new(8, false);
        assert_eq!(
            strict.parse(r#"{"a": "#, true),
            Err(PartialJsonError::Invalid("unexpected character"))
        );
        assert_eq!(
            strict.parse(r#"{"a": "x"#, true),
            Err(PartialJsonError::Invalid("unterminated string"))
        );
        assert_eq!(
            strict.parse("[1, 2", true),
            Err(PartialJsonError::Invalid("expected ',' or ']'"))
        );
        assert_eq!(
            strict.parse("tru", true), // codespell:ignore tru
            Err(PartialJsonError::Invalid("invalid literal"))
        );
        assert_eq!(
            strict.parse(r#"{"a": 1}"#, true).expect("complete").0,
            json!({"a": 1})
        );
    }

    #[test]
    fn literals_and_their_prefixes_read_as_the_literal_in_prefix_mode() {
        assert_eq!(prefix("tru").0, json!(true)); // codespell:ignore tru
        assert_eq!(prefix("f").0, json!(false));
        assert_eq!(prefix("nul").0, json!(null));
        assert_eq!(prefix("[true, fals").0, json!([true, false])); // codespell:ignore fals
    }

    #[test]
    fn a_word_that_is_no_literal_is_left_unconsumed_and_closes_the_enclosing_value() {
        // The whole word is checked before anything is consumed, so the array closes before it.
        // The original agreed for `truex` and `trué` but consumed `falsex` (ten bytes) because its
        // look-ahead stopped at the literal's length; this port treats every word alike.
        assert_eq!(prefix("[1, truex"), (json!([1]), 4));
        assert_eq!(prefix("[1, trué"), (json!([1]), 4));
        assert_eq!(prefix("[1, truex]"), (json!([1]), 4));
        assert_eq!(prefix("[1, falsex]"), (json!([1]), 4));
        assert_eq!(prefix("[1, nullx]"), (json!([1]), 4));
        assert_eq!(
            PartialJson::default().parse("truex", true),
            Err(PartialJsonError::Invalid("invalid literal")),
            "at the top level there is no enclosing value to close"
        );
    }

    #[test]
    fn escapes_are_decoded_and_an_unfinished_unicode_escape_is_the_replacement_character() {
        assert_eq!(prefix(r#""a\"b\\c\/d\n\té""#).0, json!("a\"b\\c/d\n\té"));
        assert_eq!(
            prefix(r#""x\q""#).0,
            json!("xq"),
            "an unknown escape keeps its character"
        );
        assert_eq!(prefix(r#""\u00"#).0, json!("\u{FFFD}"));
    }

    #[test]
    fn a_surrogate_pair_written_as_two_escapes_is_one_character() {
        let text = r#"{"e": "\ud83c\udf0d", "f": 1}"#;
        let (value, consumed) = prefix(text);
        assert_eq!(value, json!({"e": "🌍", "f": 1}));
        assert_eq!(consumed, text.len());
        for cut in 0..=text.len() {
            let (_, consumed) = prefix(&text[..cut]);
            assert_eq!(
                consumed, cut,
                "cut at {cut}: a pair cut in two is a string still arriving"
            );
        }
        assert_eq!(
            PartialJson::new(8, false)
                .parse(text, true)
                .expect("complete")
                .0,
            json!({"e": "🌍", "f": 1})
        );
    }

    #[test]
    fn a_surrogate_without_its_other_half_is_invalid_in_both_modes() {
        for text in [
            r#""\ud83c""#,
            r#""\udf0d""#,
            r#""\ud83c x""#,
            r#""\ud83c\u0041""#,
        ] {
            assert_eq!(
                PartialJson::default().parse(text, true),
                Err(PartialJsonError::Invalid("invalid unicode escape")),
                "{text}"
            );
            assert_eq!(
                PartialJson::new(8, false).parse(text, true),
                Err(PartialJsonError::Invalid("invalid unicode escape")),
                "{text}"
            );
        }
    }

    #[test]
    fn numbers_are_integers_when_they_can_be_and_floats_otherwise() {
        assert_eq!(prefix("12").0, json!(12));
        assert_eq!(prefix("-0").0, json!(0));
        assert_eq!(prefix("-12.5e3").0, json!(-12500.0));
        assert_eq!(prefix("1E2").0, json!(100.0));
        assert_eq!(
            prefix("-").0,
            json!(0),
            "a sign alone reads as 0 in prefix mode, as the original did"
        );
    }

    #[test]
    fn trailing_commas_are_accepted_as_the_original_did() {
        assert_eq!(prefix(r#"{"a": 1,}"#).0, json!({"a": 1}));
        assert_eq!(prefix("[1,]").0, json!([1]));
    }

    #[test]
    fn is_complete_accepts_one_whole_value_and_nothing_else() {
        assert!(is_complete(r#"{"a": [1, "b"]}"#));
        assert!(is_complete("  42 \n"));
        assert!(!is_complete(r#"{"a": [1, "b"]"#));
        assert!(!is_complete(r#"{"a": 1} x"#));
        assert!(!is_complete(""));
    }
}
