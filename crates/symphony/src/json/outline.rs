//! The outline of a tool-call object as it arrives: where its name is and where its arguments are.
//!
//! A model writes a call as a JSON object, `{"name": "get_weather", "arguments": {"city": "Paris"}}`
//! in some order and with its own spacing, and the object arrives in pieces. [`outline`] reads the
//! object so far and reports the call's name once its string is complete, the byte span of the
//! arguments value from the moment its first byte has arrived, and whether the object has closed.
//! Because the span is a range into the model's own text, an argument stream can emit exactly the
//! bytes the model wrote, as they arrive, and nothing it emitted ever has to change: a byte of the
//! value, once there, stays. That replaces the old crate's way of streaming arguments, which parsed
//! the partial object, re-serialized the arguments and emitted the difference to the previous
//! serialization, so that clients saw `{"city":"Paris"}` where the model had written
//! `{"city": "Paris"}`.
//!
//! The member names a call uses are the ones the old crate accepted: `name` or `tool_name` for the
//! name, `arguments` or `parameters` for the arguments; the first of each that appears counts. The
//! outline says nothing about whether the name is a declared tool or whether the arguments are an
//! object; those are the format's decisions, made with the span in hand.

use std::ops::Range;

/// What the object so far reveals about the call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outline {
    /// The call's name, once its string has arrived whole.
    pub name: Option<String>,
    /// The arguments value's bytes, from its first byte to where it ends or to where the text ends.
    pub arguments: Option<Span>,
    /// Whether the object's closing brace has arrived.
    pub complete: bool,
}

/// A value's place in the text: byte offsets, the end exclusive and known only once the value is
/// complete.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// The byte at which the value starts.
    pub start: usize,
    /// One past the value's last byte, once the value is complete.
    pub end: Option<usize>,
}

impl Span {
    /// The value's bytes so far: up to its end when it is complete, else to the end of `text`.
    pub fn text<'a>(&self, text: &'a str) -> &'a str {
        &text[self.range(text)]
    }

    /// The bytes of the value as a range, for callers that index the text themselves.
    pub fn range(&self, text: &str) -> Range<usize> {
        self.start..self.end.unwrap_or(text.len())
    }
}

const NAME_KEYS: [&str; 2] = ["name", "tool_name"];
const ARGUMENTS_KEYS: [&str; 2] = ["arguments", "parameters"];

/// Read the object in `text`, which may be cut anywhere after its opening brace.
///
/// Text before the brace is skipped if it is whitespace; anything else, or no brace, gives an empty
/// outline. Members are read in order; a member whose key or value has not finished arriving ends
/// the reading, with the arguments span open if that member is the arguments.
pub fn outline(text: &str) -> Outline {
    let mut found = Outline::default();
    let Some(open) = after_whitespace(text, 0) else {
        return found;
    };
    if !text[open..].starts_with('{') {
        return found;
    }
    let mut at = open + 1;
    loop {
        let Some(key_start) = after_whitespace(text, at) else {
            return found;
        };
        if text[key_start..].starts_with('}') {
            found.complete = true;
            return found;
        }
        if text[key_start..].starts_with(',') {
            at = key_start + 1;
            continue;
        }
        let Some(key_end) = string_end(text, key_start) else {
            return found;
        };
        let Some(colon) = after_whitespace(text, key_end) else {
            return found;
        };
        if !text[colon..].starts_with(':') {
            return found;
        }
        let Some(value_start) = after_whitespace(text, colon + 1) else {
            return found;
        };
        let value_end = value_end(text, value_start);
        let key = decode_string(&text[key_start..key_end]);
        if found.name.is_none() && key.as_deref().is_some_and(|k| NAME_KEYS.contains(&k)) {
            if let Some(end) = value_end {
                found.name = decode_string(&text[value_start..end]);
            }
        }
        if found.arguments.is_none() && key.as_deref().is_some_and(|k| ARGUMENTS_KEYS.contains(&k))
        {
            found.arguments = Some(Span {
                start: value_start,
                end: value_end,
            });
        }
        match value_end {
            Some(end) => at = end,
            None => return found,
        }
    }
}

/// The first byte at or after `at` that is not whitespace, if any.
fn after_whitespace(text: &str, at: usize) -> Option<usize> {
    text[at..]
        .char_indices()
        .find(|(_, c)| !c.is_whitespace())
        .map(|(i, _)| at + i)
}

/// One past the closing quote of the string starting at `at`, if the string is complete.
fn string_end(text: &str, at: usize) -> Option<usize> {
    if !text[at..].starts_with('"') {
        return None;
    }
    let mut escaped = false;
    for (i, c) in text[at + 1..].char_indices() {
        match (escaped, c) {
            (true, _) => escaped = false,
            (false, '\\') => escaped = true,
            (false, '"') => return Some(at + 1 + i + 1),
            _ => {}
        }
    }
    None
}

/// One past the last byte of the value starting at `at`, if the value is complete.
///
/// Objects and arrays end at their matching bracket, strings at their closing quote, and numbers
/// and literals at the first byte that cannot continue them, which exists only once something
/// follows them: a number cut at the end of the text may still grow.
fn value_end(text: &str, at: usize) -> Option<usize> {
    let rest = &text[at..];
    match rest.chars().next()? {
        '"' => string_end(text, at),
        '{' | '[' => {
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escaped = false;
            for (i, c) in rest.char_indices() {
                match (in_string, escaped, c) {
                    (true, true, _) => escaped = false,
                    (true, false, '\\') => escaped = true,
                    (true, false, '"') => in_string = false,
                    (true, false, _) => {}
                    (false, _, '"') => in_string = true,
                    (false, _, '{' | '[') => depth += 1,
                    (false, _, '}' | ']') => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(at + i + c.len_utf8());
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        _ => rest
            .char_indices()
            .find(|(_, c)| matches!(c, ',' | '}' | ']') || c.is_whitespace())
            .map(|(i, _)| at + i),
    }
}

/// The string the quoted JSON text at `quoted` denotes, if it is a complete, valid string.
fn decode_string(quoted: &str) -> Option<String> {
    serde_json::from_str(quoted).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALL: &str = r#"{"name": "get_weather", "arguments": {"city": "Paris", "days": [1, 2]}}"#;

    #[test]
    fn a_complete_call_gives_its_name_its_arguments_bytes_and_the_close() {
        let found = outline(CALL);
        assert_eq!(found.name.as_deref(), Some("get_weather"));
        let arguments = found.arguments.expect("arguments");
        assert_eq!(
            arguments.text(CALL),
            r#"{"city": "Paris", "days": [1, 2]}"#,
            "the model's own bytes, spacing included"
        );
        assert_eq!(arguments.end, Some(CALL.len() - 1));
        assert!(found.complete);
    }

    #[test]
    fn every_cut_of_a_call_is_a_prefix_of_the_outline_of_the_whole() {
        let whole = outline(CALL);
        let whole_arguments = whole.arguments.clone().expect("arguments");
        for cut in 0..=CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let found = outline(&CALL[..cut]);
            assert!(
                found.name.is_none() || found.name == whole.name,
                "cut at {cut}: a name, once seen, is the final name"
            );
            if let Some(arguments) = &found.arguments {
                assert_eq!(arguments.start, whole_arguments.start, "cut at {cut}");
                assert!(
                    arguments.end.is_none() || arguments.end == whole_arguments.end,
                    "cut at {cut}: an end, once seen, is the final end"
                );
                assert!(
                    whole_arguments
                        .text(CALL)
                        .starts_with(arguments.text(&CALL[..cut])),
                    "cut at {cut}: the bytes so far are a prefix of the final bytes"
                );
            }
            assert_eq!(found.complete, cut == CALL.len(), "cut at {cut}");
        }
    }

    #[test]
    fn the_arguments_span_opens_with_the_first_byte_of_the_value() {
        let text = r#"{"name": "f", "arguments": "#;
        assert_eq!(outline(text).arguments, None, "the value has not started");
        let text = r#"{"name": "f", "arguments": {"#;
        assert_eq!(
            outline(text).arguments,
            Some(Span {
                start: 27,
                end: None
            })
        );
        let text = r#"{"name": "f", "arguments": {"a": "#;
        assert_eq!(
            outline(text).arguments.map(|s| s.text(text).to_string()),
            Some(r#"{"a": "#.into())
        );
    }

    #[test]
    fn the_name_appears_only_when_its_string_is_whole() {
        assert_eq!(outline(r#"{"name": "get_wea"#).name, None);
        assert_eq!(outline(r#"{"name": "get_weather"#).name, None);
        assert_eq!(
            outline(r#"{"name": "get_weather""#).name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            outline(r#"{"name": "café \"x\""}"#).name.as_deref(),
            Some("café \"x\""),
            "the name is the decoded string"
        );
    }

    #[test]
    fn arguments_may_come_before_the_name() {
        let text = r#"{"arguments": {"city": "Paris"}, "name": "get_weather"}"#;
        let found = outline(text);
        assert_eq!(found.name.as_deref(), Some("get_weather"));
        assert_eq!(
            found.arguments.expect("arguments").text(text),
            r#"{"city": "Paris"}"#
        );
        assert!(found.complete);
        let cut = &text[..r#"{"arguments": {"city": "Paris"}, "na"#.len()];
        let found = outline(cut);
        assert_eq!(found.name, None);
        assert_eq!(found.arguments.expect("arguments").end, Some(31));
    }

    #[test]
    fn the_old_crates_other_member_names_count_too() {
        let text = r#"{"tool_name": "f", "parameters": {"a": 1}}"#;
        let found = outline(text);
        assert_eq!(found.name.as_deref(), Some("f"));
        assert_eq!(
            found.arguments.expect("arguments").text(text),
            r#"{"a": 1}"#
        );
    }

    #[test]
    fn string_number_and_literal_arguments_end_where_json_says() {
        let text = r#"{"name": "f", "arguments": "{\"a\": 1}"}"#;
        assert_eq!(
            outline(text).arguments.expect("arguments").text(text),
            r#""{\"a\": 1}""#,
            "a string value is reported quoted, as written; decoding it is the format's call"
        );
        let text = r#"{"name": "f", "arguments": 42}"#;
        assert_eq!(outline(text).arguments.expect("arguments").text(text), "42");
        let text = r#"{"name": "f", "arguments": 42"#;
        assert_eq!(
            outline(text).arguments.expect("arguments").end,
            None,
            "a number at the end of the text may still grow"
        );
        let text = r#"{"name": "f", "arguments": null, "x": 1}"#;
        assert_eq!(
            outline(text).arguments.expect("arguments").text(text),
            "null"
        );
    }

    #[test]
    fn brackets_and_quotes_inside_strings_do_not_count() {
        let text = r#"{"name": "f", "arguments": {"q": "a } ] \" {", "n": [1, "]"]}}"#;
        let found = outline(text);
        assert_eq!(
            found.arguments.expect("arguments").text(text),
            r#"{"q": "a } ] \" {", "n": [1, "]"]}"#
        );
        assert!(found.complete);
    }

    #[test]
    fn text_that_is_not_an_object_gives_an_empty_outline() {
        for text in ["", "   ", "[1, 2]", "hello {", r#""name""#] {
            assert_eq!(outline(text), Outline::default(), "{text:?}");
        }
        assert_eq!(
            outline("  \n{"),
            Outline::default(),
            "an open brace alone reveals nothing yet"
        );
    }

    #[test]
    fn multibyte_text_keeps_offsets_on_character_boundaries() {
        let text = r#"{"name": "météo", "arguments": {"ville": "Zürich 🌍"}}"#;
        let found = outline(text);
        assert_eq!(found.name.as_deref(), Some("météo"));
        let arguments = found.arguments.expect("arguments");
        assert!(text.is_char_boundary(arguments.start));
        assert_eq!(arguments.text(text), r#"{"ville": "Zürich 🌍"}"#);
        for cut in 0..=text.len() {
            if text.is_char_boundary(cut) {
                let _ = outline(&text[..cut]);
            }
        }
    }
}
