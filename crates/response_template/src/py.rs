//! The Python built-ins transformers applies to model output, reproduced
//! exactly (Python 3.12): `str.strip()`, `int()`, `float()`, truthiness and
//! `json.loads`. serde_json's parser is not used for `json.loads`: it rounds
//! some floats differently (the workspace does not enable `float_roundtrip`),
//! turns integers beyond u64 into floats, stops 128 levels deep, and rejects
//! `NaN`, `Infinity` and lone surrogates, which Python accepts.

use std::cell::Cell;

use serde_json::{Map, Value};

use crate::error::Taint;

thread_local! {
    static OTHER_PLACEHOLDERS: Cell<bool> = const { Cell::new(false) };
    /// A JSON key holding a stand-in was given twice where the text also
    /// holds one of its characters.
    static KEY_CLASH: Cell<bool> = const { Cell::new(false) };
}

/// `compute`'s value, and why it holds a value JSON cannot hold, if it still
/// does. A placeholder can be dropped on the way (by a transform that does not
/// use it, or a key given twice), so a marked value is computed again with
/// other placeholders; one no placeholder reached comes out the same, keys in
/// the same order (`Map`'s `==` ignores the order). The mark stays where a
/// JSON key with a lone surrogate's stand-in may have met the same text as a
/// real key. The flags this sets are thread-local: `compute` must run to its
/// end on this thread, without calling this again.
pub(crate) fn unless_unreached<E>(
    compute: impl Fn() -> Result<(Value, Taint), E>,
) -> Result<(Value, Taint), E> {
    KEY_CLASH.set(false);
    let (value, taint) = compute()?;
    if taint.is_none() {
        return Ok((value, None));
    }
    OTHER_PLACEHOLDERS.set(true);
    let other = compute();
    OTHER_PLACEHOLDERS.set(false);
    let unreached = other
        .is_ok_and(|(other, _)| serde_json::to_vec(&other).ok() == serde_json::to_vec(&value).ok());
    Ok((value, taint.filter(|_| !unreached || KEY_CLASH.get())))
}

fn other_placeholders() -> bool {
    OTHER_PLACEHOLDERS.get()
}

/// The first of the private-use characters that stand in for the 2,048 lone
/// surrogates, one each.
fn stand_in_base() -> u32 {
    if other_placeholders() {
        0x10_0000
    } else {
        0xf_0000
    }
}

fn is_stand_in(c: char) -> bool {
    let base = stand_in_base();
    (base..base + 0x800).contains(&u32::from(c))
}

/// `str.isspace()`: White_Space and U+001C..U+001F.
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
pub(crate) fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `bool(value)`.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `value == 1` (true for 1, 1.0 and True).
pub(crate) fn equals_one(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() == Some(1.0),
        _ => false,
    }
}

/// `type(value).__name__`, for messages.
pub(crate) fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// The zero of every run of decimal digits (`str.isdecimal()` in Python 3.12,
/// Unicode 15.0); `int()` and `float()` accept all of them.
const DIGIT_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
    0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950,
    0x1fbf0,
];

/// The ASCII digit for a Unicode decimal digit.
fn ascii_digit(c: char) -> Option<char> {
    let c = u32::from(c);
    let i = DIGIT_ZEROS.partition_point(|&zero| zero <= c);
    let digit = c - DIGIT_ZEROS[i.checked_sub(1)?];
    (digit < 10).then(|| char::from_digit(digit, 10))?
}

/// What `int()` and `float()` parse: the text without surrounding whitespace
/// (White_Space, unlike `strip()`), with decimal digits made ASCII.
fn numeric_ascii(s: &str) -> Option<String> {
    s.trim_matches(char::is_whitespace)
        .chars()
        .map(|c| {
            if c.is_ascii() {
                Some(c)
            } else {
                ascii_digit(c)
            }
        })
        .collect()
}

/// ASCII digits, with single underscores between them (PEP 515).
fn is_digits(s: &str) -> bool {
    s.split('_')
        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Python 3.12 refuses to convert integer strings longer than this.
const MAX_STR_DIGITS: usize = 4300;

/// An integer as JSON, or a placeholder (with the reason) when it does not fit.
pub(crate) fn int_value(negative: bool, digits: &str) -> (Value, Taint) {
    let magnitude = digits.trim_start_matches('0');
    let value = match magnitude.parse::<u64>() {
        Ok(n) if !negative => Some(Value::from(n)),
        Ok(n) if n <= 1 << 63 => Some(Value::from(0i64.wrapping_sub_unsigned(n))),
        Err(_) if magnitude.is_empty() => Some(Value::from(0)),
        _ => None,
    };
    match value {
        Some(value) => (value, None),
        None => (
            Value::from(u8::from(other_placeholders())),
            Some("an integer outside i64 and u64"),
        ),
    }
}

/// `int(s)`, or `None` where Python raises `ValueError`.
pub(crate) fn int(s: &str) -> Option<(Value, Taint)> {
    let s = numeric_ascii(s)?;
    let (negative, body) = match s.strip_prefix('-') {
        Some(body) => (true, body),
        None => (false, s.strip_prefix('+').unwrap_or(&s)),
    };
    if !is_digits(body) {
        return None;
    }
    let digits = body.replace('_', "");
    (digits.len() <= MAX_STR_DIGITS).then(|| int_value(negative, &digits))
}

/// `float(s)`, or `None` where Python raises `ValueError`.
pub(crate) fn float(s: &str) -> Option<f64> {
    let s = numeric_ascii(s)?;
    let body = s
        .strip_prefix(['+', '-'])
        .unwrap_or(&s)
        .to_ascii_lowercase();
    if !matches!(body.as_str(), "inf" | "infinity" | "nan") {
        let (mantissa, exponent) = match body.split_once('e') {
            Some((m, e)) => (m, Some(e.strip_prefix(['+', '-']).unwrap_or(e))),
            None => (body.as_str(), None),
        };
        let (int_part, frac) = match mantissa.split_once('.') {
            Some((i, f)) => (i, f),
            None => (mantissa, ""),
        };
        let ok = (int_part.is_empty() || is_digits(int_part))
            && (frac.is_empty() || is_digits(frac))
            && !(int_part.is_empty() && frac.is_empty())
            && exponent.is_none_or(is_digits);
        if !ok {
            return None;
        }
    }
    s.replace('_', "").parse().ok()
}

/// A float as JSON, or a placeholder (with the reason) for NaN and infinities.
pub(crate) fn float_value(f: f64) -> (Value, Taint) {
    match serde_json::Number::from_f64(f) {
        Some(n) => (Value::Number(n), None),
        None => (
            Value::from(f64::from(u8::from(other_placeholders()))),
            Some("NaN or an infinity"),
        ),
    }
}

/// Why `json.loads` failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JsonError {
    /// `json.JSONDecodeError`.
    Decode,
    /// A plain `ValueError`: an integer longer than 4,300 digits.
    Digits,
    /// Nested [`JSON_DEPTH_LIMIT`] deep; Python raises `RecursionError` only
    /// near its C recursion limit.
    Recursion,
}

/// The nesting depth at which `json_loads` stops.
pub(crate) const JSON_DEPTH_LIMIT: usize = 512;

/// `json.loads(s)`. A value JSON cannot hold is replaced by a placeholder of
/// the same Python type and reported through `taint`.
pub(crate) fn json_loads(s: &str, taint: &mut Taint) -> Result<Value, JsonError> {
    let mut json = Json {
        s,
        i: 0,
        taint: None,
        stand_in_text: false,
        stand_in_key_twice: false,
    };
    json.ws();
    let value = json.value(0)?;
    json.ws();
    if json.i != s.len() {
        return Err(JsonError::Decode);
    }
    if json.stand_in_text && json.stand_in_key_twice {
        KEY_CLASH.set(true);
    }
    if json.taint.is_some() {
        *taint = json.taint;
    }
    Ok(value)
}

struct Json<'a> {
    s: &'a str,
    i: usize,
    taint: Taint,
    /// A stand-in character that is not a lone surrogate's was read.
    stand_in_text: bool,
    /// An object key holding a stand-in character was given twice.
    stand_in_key_twice: bool,
}

impl Json<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, word: &str) -> bool {
        let found = self.s[self.i..].starts_with(word);
        if found {
            self.i += word.len();
        }
        found
    }

    fn tainted(&mut self, value: (Value, Taint)) -> Value {
        if value.1.is_some() {
            self.taint = value.1;
        }
        value.0
    }

    fn value(&mut self, depth: usize) -> Result<Value, JsonError> {
        match self.peek() {
            Some(b'{' | b'[') if depth + 1 >= JSON_DEPTH_LIMIT => Err(JsonError::Recursion),
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => self.string().map(Value::String),
            _ if self.eat("null") => Ok(Value::Null),
            _ if self.eat("true") => Ok(Value::Bool(true)),
            _ if self.eat("false") => Ok(Value::Bool(false)),
            _ if self.eat("NaN") => Ok(self.tainted(float_value(f64::NAN))),
            _ if self.eat("Infinity") || self.eat("-Infinity") => {
                Ok(self.tainted(float_value(f64::INFINITY)))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(JsonError::Decode),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    /// The number grammar of Python's C scanner: a fraction or exponent that
    /// is not followed by a digit is left unread.
    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.i;
        let negative = self.eat("-");
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(JsonError::Decode),
        }
        let int_end = self.i;
        let mut is_float = false;
        if self.peek() == Some(b'.')
            && self
                .s
                .as_bytes()
                .get(self.i + 1)
                .is_some_and(u8::is_ascii_digit)
        {
            self.i += 1;
            self.digits();
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let mark = self.i;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                self.i = mark;
            } else {
                is_float = true;
            }
        }
        let text = &self.s[start..self.i];
        Ok(if is_float {
            let f = text.parse().map_err(|_| JsonError::Decode)?;
            self.tainted(float_value(f))
        } else {
            let digits = &self.s[start + usize::from(negative)..int_end];
            if digits.len() > MAX_STR_DIGITS {
                return Err(JsonError::Digits);
            }
            self.tainted(int_value(negative, digits))
        })
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let hex = self.s.get(self.i..self.i + 4).ok_or(JsonError::Decode)?;
        if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(JsonError::Decode);
        }
        self.i += 4;
        u32::from_str_radix(hex, 16).map_err(|_| JsonError::Decode)
    }

    /// A string; control characters are errors (`strict=True`).
    fn string(&mut self) -> Result<String, JsonError> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let rest = &self.s[self.i..];
            let end = rest
                .find(|c: char| c == '"' || c == '\\' || c < ' ')
                .ok_or(JsonError::Decode)?;
            out.push_str(&rest[..end]);
            self.stand_in_text |= rest[..end].chars().any(is_stand_in);
            self.i += end;
            match self.peek() {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => self.i += 1,
                _ => return Err(JsonError::Decode),
            }
            let escaped = self.peek().ok_or(JsonError::Decode)?;
            self.i += 1;
            out.push(match escaped {
                b'"' => '"',
                b'\\' => '\\',
                b'/' => '/',
                b'b' => '\u{8}',
                b'f' => '\u{c}',
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'u' => self.unicode_escape()?,
                _ => return Err(JsonError::Decode),
            });
        }
    }

    /// `\uXXXX`, joined with a following low surrogate escape; a lone
    /// surrogate becomes its stand-in and taints the value.
    fn unicode_escape(&mut self) -> Result<char, JsonError> {
        let unit = self.hex4()?;
        if (0xd800..0xdc00).contains(&unit) && self.s[self.i..].starts_with("\\u") {
            let mark = self.i;
            self.i += 2;
            let low = self.hex4()?;
            if (0xdc00..0xe000).contains(&low) {
                let c = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
                let c = char::from_u32(c).ok_or(JsonError::Decode)?;
                self.stand_in_text |= is_stand_in(c);
                return Ok(c);
            }
            self.i = mark;
        }
        Ok(char::from_u32(unit).unwrap_or_else(|| {
            self.taint = Some("a lone surrogate");
            char::from_u32(stand_in_base() + unit - 0xd800).unwrap_or_default()
        }))
    }

    fn object(&mut self, depth: usize) -> Result<Value, JsonError> {
        self.i += 1;
        let mut map = Map::new();
        self.ws();
        if self.eat("}") {
            return Ok(Value::Object(map));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(JsonError::Decode);
            }
            let key = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(JsonError::Decode);
            }
            self.ws();
            let value = self.value(depth)?;
            self.stand_in_key_twice |= key.contains(is_stand_in) && map.contains_key(&key);
            map.insert(key, value);
            self.ws();
            if self.eat("}") {
                return Ok(Value::Object(map));
            }
            if !self.eat(",") {
                return Err(JsonError::Decode);
            }
            self.ws();
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, JsonError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth)?);
            self.ws();
            if self.eat("]") {
                return Ok(Value::Array(items));
            }
            if !self.eat(",") {
                return Err(JsonError::Decode);
            }
            self.ws();
        }
    }
}
