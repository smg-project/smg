//! Reading the transformers recordings in `tests/fixtures/hf/`, written by
//! `scripts/generate_hf_fixtures.py`. Shared by the replay test and the
//! crate's unit tests.

#![allow(
    clippy::allow_attributes,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    dead_code,
    reason = "test helpers shared by the replay test and the unit tests; a bad fixture should panic"
)]

use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

/// The fixture directory: `RESPONSE_TEMPLATE_HF_FIXTURES` for local extra
/// cases, else the committed recordings.
pub fn fixtures_dir() -> PathBuf {
    match std::env::var_os("RESPONSE_TEMPLATE_HF_FIXTURES") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hf"),
    }
}

/// The rows of `<name>.jsonl`, or of `<name>-0.jsonl`, `<name>-1.jsonl`, ... in order.
pub fn read_jsonl(dir: &Path, name: &str) -> Vec<Value> {
    let mut files = Vec::new();
    let single = dir.join(format!("{name}.jsonl"));
    if single.exists() {
        files.push(single);
    }
    for k in 0.. {
        let shard = dir.join(format!("{name}-{k}.jsonl"));
        if !shard.exists() {
            break;
        }
        files.push(shard);
    }
    files
        .iter()
        .flat_map(|path| {
            let text = fs::read_to_string(path).unwrap();
            text.lines()
                .filter(|l| !l.is_empty())
                .map(|l| serde_json::from_str::<Value>(l).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn read_json(dir: &Path, name: &str) -> Value {
    let path = dir.join(name);
    if !path.exists() {
        return Value::Object(Map::new());
    }
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// Python's `float.hex()`.
pub fn float_hex(f: f64) -> String {
    let bits = f.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = bits & ((1u64 << 52) - 1);
    if exponent == 0 && mantissa == 0 {
        return format!("{sign}0x0.0p+0");
    }
    let (lead, exp) = if exponent == 0 {
        (0, -1022)
    } else {
        (1, exponent - 1023)
    };
    let exp_sign = if exp < 0 { "-" } else { "+" };
    format!("{sign}0x{lead}.{mantissa:013x}p{exp_sign}{}", exp.abs())
}

/// The inverse of `float.hex()` for the forms it writes.
pub fn float_from_hex(s: &str) -> f64 {
    let (negative, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let rest = rest.strip_prefix("0x").unwrap();
    let (mantissa, exp) = rest.split_once('p').unwrap();
    let (lead, frac) = mantissa.split_once('.').unwrap();
    let exp: i64 = exp.parse().unwrap();
    let lead: u64 = lead.parse().unwrap();
    let frac_bits = u64::from_str_radix(&format!("{frac:0<13}"), 16).unwrap();
    let bits = if lead == 0 {
        assert!(frac_bits == 0 || exp == -1022);
        frac_bits
    } else {
        (((exp + 1023) as u64) << 52) | frac_bits
    };
    let f = f64::from_bits(bits);
    if negative {
        -f
    } else {
        f
    }
}

/// Decode a fixture value: `{"$float": hex}` is a float, `{"$dict": [[k, v], ...]}` a dict.
pub fn untag(v: &Value) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(untag).collect()),
        Value::Object(map) => {
            if map.len() == 1 {
                if let Some(Value::String(hex)) = map.get("$float") {
                    return Value::Number(Number::from_f64(float_from_hex(hex)).unwrap());
                }
                if let Some(Value::Array(items)) = map.get("$dict") {
                    let mut out = Map::new();
                    for item in items {
                        out.insert(item[0].as_str().unwrap().to_owned(), untag(&item[1]));
                    }
                    return Value::Object(out);
                }
            }
            Value::Object(map.iter().map(|(k, v)| (k.clone(), untag(v))).collect())
        }
        other => other.clone(),
    }
}

/// Encode a value as the fixtures do (the inverse of [`untag`]).
pub fn tag(v: &Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => {
            let mut map = Map::new();
            map.insert(
                "$float".into(),
                Value::String(float_hex(n.as_f64().unwrap())),
            );
            Value::Object(map)
        }
        Value::Array(items) => Value::Array(items.iter().map(tag).collect()),
        Value::Object(map) => {
            if map.keys().any(|k| k.starts_with('$')) {
                let items = map
                    .iter()
                    .map(|(k, v)| Value::Array(vec![Value::String(k.clone()), tag(v)]))
                    .collect();
                let mut out = Map::new();
                out.insert("$dict".into(), Value::Array(items));
                return Value::Object(out);
            }
            Value::Object(map.iter().map(|(k, v)| (k.clone(), tag(v))).collect())
        }
        other => other.clone(),
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `json.dumps(v, ensure_ascii=False, separators=(",", ":"))` of a
/// tagged value (so no floats).
pub fn canonical(v: &Value) -> String {
    let mut out = String::new();
    write_canonical(&mut out, v);
    out
}

fn write_canonical(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            assert!(!n.is_f64(), "untagged float in canonical JSON");
            let _ = write!(out, "{n}");
        }
        Value::String(s) => write_json_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(out, k);
                out.push(':');
                write_canonical(out, v);
            }
            out.push('}');
        }
    }
}

const DIGEST_LETTERS: &[u8; 16] = b"ghjkmnpqrstvwxyz";

/// The first 8 bytes of SHA-256, one letter per nibble.
pub fn digest_text(text: &str) -> String {
    let hash = Sha256::digest(text.as_bytes());
    hash[..8]
        .iter()
        .flat_map(|b| {
            [
                DIGEST_LETTERS[(b >> 4) as usize],
                DIGEST_LETTERS[(b & 15) as usize],
            ]
        })
        .map(char::from)
        .collect()
}

/// The digest of a list of traces (each canonical JSON on its own line).
pub fn digest_traces(traces: &[Value]) -> String {
    let mut text = String::new();
    for t in traces {
        text.push_str(&canonical(t));
        text.push('\n');
    }
    digest_text(&text)
}

/// Chunk lengths in characters for a feeds spec: "whole", "chars", "step:K",
/// or lengths separated by spaces.
pub fn chunk_lengths(text: &str, spec: &str) -> Vec<usize> {
    let n = text.chars().count();
    if spec == "whole" {
        return vec![n];
    }
    if spec == "chars" {
        return vec![1; n];
    }
    if let Some(step) = spec.strip_prefix("step:") {
        let step: usize = step.parse().unwrap();
        return (0..n).step_by(step).map(|i| step.min(n - i)).collect();
    }
    spec.split_whitespace()
        .map(|k| k.parse().unwrap())
        .collect()
}

/// Split `text` into chunks of the given lengths in characters.
pub fn split_chars<'a>(text: &'a str, lengths: &[usize]) -> Vec<&'a str> {
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut out = Vec::new();
    let mut at = 0;
    for &k in lengths {
        out.push(&text[bounds[at]..bounds[at + k]]);
        at += k;
    }
    assert_eq!(at, bounds.len() - 1, "chunk lengths do not cover the text");
    out
}

/// Byte offset of character index `i`.
pub fn byte_offset(text: &str, i: usize) -> usize {
    text.char_indices()
        .map(|(b, _)| b)
        .nth(i)
        .unwrap_or(text.len())
}

/// Character index of byte offset `b`.
pub fn char_index(text: &str, b: usize) -> usize {
    text[..b].chars().count()
}
