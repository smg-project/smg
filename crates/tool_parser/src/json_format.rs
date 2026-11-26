//! Serialize a [`serde_json::Value`] like Python's `json.dumps(value, ensure_ascii=False)`:
//! spaced `", "` / `": "` separators, single line, raw UTF-8.
//!
//! serde_json only ships compact (`{"a":1}`) and pretty (multi-line) formatters,
//! neither of which matches `json.dumps`. The DeepSeek-V4.1 reference parser
//! (and the serving engines' Python parsers) hand tool-call `arguments` back as `json.dumps`
//! output, so a compact string would differ byte-for-byte from theirs. This is
//! the same formatter `llm-tokenizer` uses for prompt rendering (`json_dumps.rs`),
//! duplicated here so the parser crate does not pull in the tokenizer crate.

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
#[path = "../tests/unit/json_format/tests.rs"]
mod tests;
