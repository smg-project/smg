//! The outline of a tool-call object as it arrives: where its name is and where its arguments are.
//!
//! A model writes a call as a JSON object,
//! `{"name": "get_weather", "arguments": {"city": "Paris"}}` in some order and with its own
//! spacing, and the object arrives in pieces. [`outline`] reads the object so far and reports the
//! call's name once its string is complete, the byte span of the arguments value from the moment
//! its first byte has arrived, and where the object closed, so the caller knows how many of its
//! bytes the object took. Because the span is a range into the model's own text, an argument stream
//! can emit exactly the bytes the model wrote, as they arrive, and nothing it emitted ever has to
//! change: a byte of the value, once there, stays. That replaces the old crate's way of streaming
//! arguments, which parsed the partial object, re-serialized the arguments and emitted the
//! difference to the previous serialization, so that clients saw `{"city":"Paris"}` where the model
//! had written `{"city": "Paris"}`.
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
    /// One past the object's closing brace, once it has arrived.
    pub close: Option<usize>,
}

impl Outline {
    /// Whether the object's closing brace has arrived.
    pub fn complete(&self) -> bool {
        self.close.is_some()
    }
}

/// A value's place in the text the outline was taken from: byte offsets, the end exclusive and
/// known only once the value is complete. `text` and `range` take that same text.
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
    let mut scanner = Scanner::default();
    scanner.advance(text);
    scanner.found().clone()
}

/// The outline taken as the object arrives: it keeps its place and its state between pieces, so
/// the text grows by appending and each byte is read once. `advance` takes the whole text so
/// far, reads only what came after the last call, and the outline in `found` is the one
/// [`outline`] gives for that text.
#[derive(Clone, Debug, Default)]
pub struct Scanner {
    found: Outline,
    /// Bytes read so far.
    at: usize,
    stage: Stage,
}

#[derive(Clone, Debug, Default)]
enum Stage {
    /// Before the opening brace.
    #[default]
    Start,
    /// After the brace or a comma, before a key.
    Member,
    /// Inside the key's string, which opened at `start`.
    Key { start: usize, escaped: bool },
    /// After the key, before the colon.
    Colon { key: Member },
    /// After the colon, before the value.
    ValueStart { key: Member },
    /// Inside the value, which started at `start`.
    Value {
        key: Member,
        start: usize,
        shape: Shape,
    },
    /// Nothing more is read: the object closed, or the outline stopped where it could say nothing
    /// it would have to take back.
    Done,
}

/// Which of the call's members a key names, decided once the key is whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Member {
    Name,
    Arguments,
    Other,
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    String {
        escaped: bool,
    },
    /// A container, with the brackets open inside it and the string state.
    Container {
        depth: usize,
        in_string: bool,
        escaped: bool,
    },
    /// A number or a literal: it ends at a delimiter or whitespace, and may still grow at the
    /// end of the text.
    Scalar,
}

impl Scanner {
    /// What the text so far reveals.
    pub fn found(&self) -> &Outline {
        &self.found
    }

    /// Read the bytes of `text` that came after the last call; `text` holds every byte fed so
    /// far, in order.
    pub fn advance(&mut self, text: &str) {
        while self.at < text.len() {
            if matches!(self.stage, Stage::Done) {
                return;
            }
            let c = text[self.at..].chars().next().unwrap_or('\0');
            let consumed = self.step(text, c);
            if consumed {
                self.at += c.len_utf8();
            }
        }
    }

    /// One character; whether it was consumed, or is to be read again in the next stage.
    fn step(&mut self, text: &str, c: char) -> bool {
        let at = self.at;
        match &mut self.stage {
            Stage::Start => {
                if c.is_whitespace() {
                    return true;
                }
                self.stage = if c == '{' { Stage::Member } else { Stage::Done };
                c == '{'
            }
            Stage::Member => match c {
                _ if c.is_whitespace() || c == ',' => true,
                '}' => {
                    self.found.close = Some(at + 1);
                    self.stage = Stage::Done;
                    true
                }
                '"' => {
                    self.stage = Stage::Key {
                        start: at,
                        escaped: false,
                    };
                    true
                }
                _ => {
                    self.stage = Stage::Done;
                    false
                }
            },
            Stage::Key { start, escaped } => {
                match (*escaped, c) {
                    (true, _) => *escaped = false,
                    (false, '\\') => *escaped = true,
                    (false, '"') => {
                        let key = match decode_string(&text[*start..=at]).as_deref() {
                            Some(key) if NAME_KEYS.contains(&key) && self.found.name.is_none() => {
                                Member::Name
                            }
                            Some(key)
                                if ARGUMENTS_KEYS.contains(&key)
                                    && self.found.arguments.is_none() =>
                            {
                                Member::Arguments
                            }
                            _ => Member::Other,
                        };
                        self.stage = Stage::Colon { key };
                    }
                    _ => {}
                }
                true
            }
            Stage::Colon { key } => {
                if c.is_whitespace() {
                    return true;
                }
                self.stage = if c == ':' {
                    Stage::ValueStart { key: *key }
                } else {
                    Stage::Done
                };
                c == ':'
            }
            Stage::ValueStart { key } => {
                if c.is_whitespace() {
                    return true;
                }
                if matches!(c, ',' | '}' | ']' | ':') {
                    // A delimiter where a value should start: the object is malformed from
                    // here, and the outline says nothing it would have to take back.
                    self.stage = Stage::Done;
                    return false;
                }
                let key = *key;
                if key == Member::Arguments {
                    self.found.arguments = Some(Span {
                        start: at,
                        end: None,
                    });
                }
                let shape = match c {
                    '"' => Shape::String { escaped: false },
                    '{' | '[' => Shape::Container {
                        depth: 1,
                        in_string: false,
                        escaped: false,
                    },
                    _ => Shape::Scalar,
                };
                self.stage = Stage::Value {
                    key,
                    start: at,
                    shape,
                };
                true
            }
            Stage::Value { key, start, shape } => {
                let (key, start) = (*key, *start);
                match shape {
                    Shape::String { escaped } => {
                        match (*escaped, c) {
                            (true, _) => *escaped = false,
                            (false, '\\') => *escaped = true,
                            (false, '"') => self.value_ends(text, key, start, at + 1),
                            _ => {}
                        }
                        true
                    }
                    Shape::Container {
                        depth,
                        in_string,
                        escaped,
                    } => {
                        match (*in_string, *escaped, c) {
                            (true, true, _) => *escaped = false,
                            (true, false, '\\') => *escaped = true,
                            (true, false, '"') => *in_string = false,
                            (true, false, _) => {}
                            (false, _, '"') => *in_string = true,
                            (false, _, '{' | '[') => *depth += 1,
                            (false, _, '}' | ']') => {
                                *depth -= 1;
                                if *depth == 0 {
                                    self.value_ends(text, key, start, at + c.len_utf8());
                                }
                            }
                            _ => {}
                        }
                        true
                    }
                    Shape::Scalar => {
                        if matches!(c, ',' | '}' | ']') || c.is_whitespace() {
                            self.value_ends(text, key, start, at);
                            return false;
                        }
                        true
                    }
                }
            }
            Stage::Done => false,
        }
    }

    /// The value from `start` is complete at `end`: the name is decoded, the arguments span
    /// closes, and the next member follows.
    fn value_ends(&mut self, text: &str, key: Member, start: usize, end: usize) {
        match key {
            Member::Name => self.found.name = decode_string(&text[start..end]),
            Member::Arguments => {
                if let Some(span) = &mut self.found.arguments {
                    span.end = Some(end);
                }
            }
            Member::Other => {}
        }
        self.stage = Stage::Member;
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
        assert_eq!(found.close, Some(CALL.len()));
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
            assert_eq!(found.complete(), cut == CALL.len(), "cut at {cut}");
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
        assert_eq!(found.close, Some(text.len()));
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
        assert_eq!(found.close, Some(text.len()));
    }

    #[test]
    fn a_delimiter_where_a_value_should_start_ends_the_outline_without_a_span() {
        for text in [
            r#"{"arguments": }"#,
            r#"{"arguments": ,}"#,
            r#"{"arguments": ]"#,
            r#"{"arguments": :"#,
        ] {
            let found = outline(text);
            assert_eq!(found.arguments, None, "{text:?}: no value, so no span");
            assert!(
                !found.complete(),
                "{text:?}: a malformed object does not close"
            );
        }
        assert_eq!(outline(r#"{"name": }"#).name, None);
    }

    #[test]
    fn the_close_offset_says_how_many_bytes_the_object_took() {
        let text = r#"{"name": "f", "arguments": {}}  </tool_call> more"#;
        let found = outline(text);
        assert_eq!(found.close, Some(r#"{"name": "f", "arguments": {}}"#.len()));
        assert_eq!(&text[found.close.expect("closed")..], "  </tool_call> more");
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
