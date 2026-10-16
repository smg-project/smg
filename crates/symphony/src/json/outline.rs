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
//! The member names a call uses are the ones the old crate accepted, `name` or `tool_name` for the
//! name and `arguments` or `parameters` for the arguments, and `args`, which Inkling's template
//! writes; the first of each that appears counts. The outline says nothing about whether the name
//! is a declared tool or whether the arguments are an object; those are the format's decisions,
//! made with the span in hand.

use std::ops::Range;

use super::prefix::Prefix;

/// What the object so far reveals about the call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outline {
    /// The call's name, once its string has arrived whole.
    pub name: Option<String>,
    /// The arguments value's bytes, from its first byte to where it ends or to where the text ends.
    pub arguments: Option<Span>,
    /// An arguments member's key has arrived, whether or not its value began: an object whose
    /// outline stops at a delimiter where the value should start has the key and no span.
    pub arguments_named: bool,
    /// One past the object's closing brace, once it has arrived.
    pub close: Option<usize>,
}

impl Outline {
    /// Whether the object's closing brace has arrived.
    pub fn complete(&self) -> bool {
        self.close.is_some()
    }

    /// Which of the call's members `key` names: the name while no name has come, the arguments
    /// while no arguments have, so the first member of each kind counts; anything else another
    /// member.
    fn member_named(&self, key: &str) -> Member {
        if NAME_KEYS.contains(&key) && self.name.is_none() {
            Member::Name
        } else if ARGUMENTS_KEYS.contains(&key) && self.arguments.is_none() {
            Member::Arguments
        } else {
            Member::Other
        }
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
const ARGUMENTS_KEYS: [&str; 3] = ["arguments", "parameters", "args"];

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
/// [`outline`] gives for that text, unless the scanner judges the members' values too
/// ([`Self::judging_values`]).
#[derive(Clone, Debug, Default)]
pub struct Scanner {
    found: Outline,
    /// Bytes read so far.
    at: usize,
    stage: Stage,
    /// Whether a member's value other than the arguments is judged as a JSON prefix
    /// ([`Self::judging_values`]).
    judging: bool,
    /// The judge of the value being read, while one is judged.
    judge: Option<Prefix>,
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

/// What a byte is to the value being read.
enum Read {
    /// The value's, and the value goes on.
    Inside,
    /// The value's last: the value ends after it.
    Last,
    /// Not the value's: the value ended before it, and the byte is the next stage's to read.
    Past,
}

impl Shape {
    /// The shape of a value whose first byte is `c`.
    fn opened_by(c: char) -> Self {
        match c {
            '"' => Self::String { escaped: false },
            '{' | '[' => Self::Container {
                depth: 1,
                in_string: false,
                escaped: false,
            },
            _ => Self::Scalar,
        }
    }

    /// Read the value's next byte: a string ends at its closing quote, a container where its
    /// last bracket closes, brackets and quotes inside a string not counting, and a scalar before
    /// a delimiter or whitespace.
    fn read(&mut self, c: char) -> Read {
        match self {
            Self::String { escaped } => match (*escaped, c) {
                (true, _) => {
                    *escaped = false;
                    Read::Inside
                }
                (false, '\\') => {
                    *escaped = true;
                    Read::Inside
                }
                (false, '"') => Read::Last,
                _ => Read::Inside,
            },
            Self::Container {
                depth,
                in_string,
                escaped,
            } => match (*in_string, *escaped, c) {
                (true, true, _) => {
                    *escaped = false;
                    Read::Inside
                }
                (true, false, '\\') => {
                    *escaped = true;
                    Read::Inside
                }
                (true, false, '"') => {
                    *in_string = false;
                    Read::Inside
                }
                (true, false, _) => Read::Inside,
                (false, _, '"') => {
                    *in_string = true;
                    Read::Inside
                }
                (false, _, '{' | '[') => {
                    *depth += 1;
                    Read::Inside
                }
                (false, _, '}' | ']') => {
                    *depth -= 1;
                    if *depth == 0 {
                        Read::Last
                    } else {
                        Read::Inside
                    }
                }
                _ => Read::Inside,
            },
            Self::Scalar => {
                if matches!(c, ',' | '}' | ']') || c.is_whitespace() {
                    Read::Past
                } else {
                    Read::Inside
                }
            }
        }
    }
}

impl Scanner {
    /// What the text so far reveals.
    pub fn found(&self) -> &Outline {
        &self.found
    }

    /// Where the outline stopped reading short of the object's close, if it did: the byte it could
    /// not place as a member, a colon, a value or the closing brace, and so the first byte that is
    /// not the object's. `None` while the object is being read, and once it has closed.
    pub fn stopped_at(&self) -> Option<usize> {
        (matches!(self.stage, Stage::Done) && self.found.close.is_none()).then_some(self.at)
    }

    /// Stop too where a member's value other than the arguments stops being a JSON prefix: a
    /// literal that runs into the text after an object whose brace was left off (`true</f>`), or
    /// a container that text cuts (`{"x": 1</f>`), at the byte that broke it or at the first
    /// letter of a word that is no literal. A string's bytes are a prefix until it closes, so a
    /// string never stops the outline. The arguments are the caller's own to judge, as the
    /// assembler does with every byte of them. Without this, a value's bytes are read for their
    /// brackets and quotes only, as [`outline`] reads them.
    #[must_use]
    pub fn judging_values(mut self) -> Self {
        self.judging = true;
        self
    }

    /// Feed the value's byte `c` at `at` to the value's judge, while the value is judged, and
    /// return where the value stopped being a JSON prefix, as an offset from its start, if it has.
    fn judged(&mut self, text: &str, at: usize, c: char) -> Option<usize> {
        self.judge
            .as_mut()
            .and_then(|judge| judge.feed(&text[at..at + c.len_utf8()]))
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
                        let key = decode_string(&text[*start..=at])
                            .map_or(Member::Other, |key| self.found.member_named(&key));
                        if key == Member::Arguments {
                            self.found.arguments_named = true;
                        }
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
                // Another member's value is judged where the caller asked for it; the arguments
                // are the caller's own to judge, as the assembler does with every byte of them.
                self.judge = (self.judging && key != Member::Arguments).then(Prefix::default);
                if self.judged(text, at, c).is_some() {
                    // No value starts with this byte: the outline stops at it.
                    self.stage = Stage::Done;
                    return false;
                }
                self.stage = Stage::Value {
                    key,
                    start: at,
                    shape: Shape::opened_by(c),
                };
                true
            }
            Stage::Value { key, start, shape } => {
                let (key, start) = (*key, *start);
                let end = match shape.read(c) {
                    Read::Inside => None,
                    Read::Last => Some(at + c.len_utf8()),
                    Read::Past => {
                        self.value_ends(text, key, start, at);
                        return false;
                    }
                };
                // The byte is the value's. A judged value stops the outline where it stops being
                // a JSON prefix: at this byte, or at the first letter of a word that is no literal.
                if let Some(from) = self.judged(text, at, c) {
                    self.at = start + from;
                    self.stage = Stage::Done;
                    return false;
                }
                if let Some(end) = end {
                    self.value_ends(text, key, start, end);
                }
                true
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
        self.judge = None;
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
    fn a_scanner_fed_piece_by_piece_finds_what_the_whole_texts_outline_finds() {
        let texts = [
            CALL,
            r#"{"name": "f", "arguments": "{\"a\": 1}", "n": 42, "x": [1, {"y": "]"}]}"#,
            r#"{"name": "f", "arguments": {"a": 1}, "strict": true</f>"#,
            r#"{"name": "f", "arguments": {"a": 1}, "meta": {"x": 1</f>"#,
            r#"{"name": "f", "arguments": {"a": 1}, "flag": nux}"#,
            r#"{"arguments": {"a": [1, 2]}, "name": "f", junk}"#,
            r#"{"name": "météo", "arguments": {"ville": "Zürich 🌍"}}  after"#,
        ];
        for text in texts {
            for judging in [false, true] {
                let scanner = || {
                    let scanner = Scanner::default();
                    if judging {
                        scanner.judging_values()
                    } else {
                        scanner
                    }
                };
                let mut whole = scanner();
                whole.advance(text);
                for cut in (0..=text.len()).filter(|&cut| text.is_char_boundary(cut)) {
                    let mut pieces = scanner();
                    pieces.advance(&text[..cut]);
                    pieces.advance(text);
                    let place = format!("{text:?} cut at {cut}, judging {judging}");
                    assert_eq!(pieces.found(), whole.found(), "{place}");
                    assert_eq!(pieces.stopped_at(), whole.stopped_at(), "{place}");
                }
            }
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
    fn inklings_args_names_the_arguments_for_every_syntax_that_reads_the_outline() {
        // `args` is Inkling's spelling, and the outline is shared, so a JSON call of any family
        // written with it has its arguments; the members are read in their order, so the first
        // member named for the arguments is the arguments.
        let text = r#"{"name": "f", "args": {"a": 1}}"#;
        let found = outline(text);
        assert_eq!(found.name.as_deref(), Some("f"));
        assert_eq!(
            found.arguments.expect("arguments").text(text),
            r#"{"a": 1}"#
        );
        let text = r#"{"name": "f", "arguments": {"a": 1}, "args": {"b": 2}}"#;
        assert_eq!(
            outline(text).arguments.expect("arguments").text(text),
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
    fn the_outline_says_where_it_stopped_short_of_the_close() {
        let stopped = |text: &str| {
            let mut scanner = Scanner::default();
            scanner.advance(text);
            scanner.stopped_at()
        };
        let text = r#"{"name": "f", "arguments": {"a": 1}</f>"#;
        assert_eq!(stopped(text), text.find("</f>"), "a brace left off");
        assert_eq!(
            stopped(r#"{"name": "f", "arguments": {"a": 1}, junk}"#),
            Some(r#"{"name": "f", "arguments": {"a": 1}, "#.len()),
            "text where a member should be"
        );
        assert_eq!(
            stopped(r#"{"name": "f", "arguments": }"#),
            Some(r#"{"name": "f", "arguments": "#.len()),
            "a delimiter where a value should start"
        );
        assert_eq!(stopped("  x"), Some(2), "no object at all");
        assert_eq!(stopped(r#"{"name": "f"}"#), None, "closed, not stopped");
        assert_eq!(stopped(r#"{"name": "f", "arg"#), None, "still reading");
        assert_eq!(stopped(""), None);
    }

    #[test]
    fn a_judged_value_stops_the_outline_where_it_stops_being_json() {
        let stopped = |text: &str| {
            let mut scanner = Scanner::default().judging_values();
            scanner.advance(text);
            scanner.stopped_at()
        };
        let text = r#"{"name": "f", "arguments": {"a": 1}, "strict": true</f>"#;
        assert_eq!(
            stopped(text),
            text.find("</f>"),
            "a literal runs into the trailer"
        );
        let text = r#"{"name": "f", "arguments": {"a": 1}, "meta": {"x": 1</f>"#;
        assert_eq!(
            stopped(text),
            text.find("</f>"),
            "a container the text cuts"
        );
        let text = r#"{"name": "f", "arguments": {"a": 1}, "flag": nux}"#;
        assert_eq!(
            stopped(text),
            text.find("nux"),
            "a word that is no literal, from its first letter"
        );
        assert_eq!(
            stopped(r#"{"name": "f", "arguments": {"a": 1}, "strict": true}"#),
            None,
            "a whole object closes"
        );
        assert_eq!(
            stopped(r#"{"name": "f", "arguments": 12abc}"#),
            None,
            "the arguments are the caller's to judge"
        );
        let mut scanner = Scanner::default();
        scanner.advance(r#"{"name": "f", "arguments": {"a": 1}, "strict": true</f>"#);
        assert_eq!(
            scanner.stopped_at(),
            None,
            "unjudged, the literal runs on, as the JSON syntax reads it"
        );
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
