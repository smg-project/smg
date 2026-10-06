//! Events from one tagged call as it arrives.
//!
//! [`Assembler`] takes a call written as tags, `<function=NAME>` and then `<parameter=KEY>` around
//! each value's text, in pieces as the format cuts it out of the model's output, and turns it into
//! the events of one call: `ToolCallStart` once the function tag is whole, `ToolCallArguments` as
//! the arguments object is built, `ToolCallEnd` at `</function>`. The model writes no JSON object,
//! so the assembler writes one, member by member, in the spelling the references use: `{"city":
//! "Paris", "limit": 5}`. What each value's text means as JSON is [`json`]'s decision, from the
//! type the tool declares for the parameter ([`Declared`]).
//!
//! A value declared a string is streamed: its opening fragment `{"key": "` (or `, "key": "`) is
//! pushed at the parameter tag, each piece of its text as it arrives, and `"` at `</parameter>`.
//! Any other value is pushed whole at its close, because its text has to be read before it is
//! written. A string that may also be null streams from the first byte that rules null out.
//!
//! The template writes one newline after `<parameter=KEY>` and one before `</parameter>`; neither
//! is the value's, and the assembler takes exactly those two away, as vLLM's parsers do. The
//! second cannot be told from a newline the value ends with until the closing tag follows it, so a
//! newline at the end of a streamed piece is held back until the next piece says which it is.
//!
//! Every byte of the call lands in exactly one event: the leading whitespace and the function tag
//! are the `source` of `ToolCallStart`; a parameter tag, the template's newlines and a value's
//! text are the `source` of the fragments they produce; bytes no event has used yet go into the
//! source of the next event, so sources stay in the output's order. Inside a value only
//! `</parameter>` is a tag; `<function=`, `<parameter=` and `</function>` there are the value's
//! text (vLLM ends a value at the next `<parameter=` or `</function>` as well, a tolerance to judge
//! with adversarial fixtures rather than port).
//!
//! Two endings. [`Assembler::close`] is for a block the model ended before `</function>` (its own
//! closing marker came, or the next call began): the call is over as far as the model is
//! concerned, so an open string is closed, the object is closed, and what was held comes back as
//! `Malformed`. [`Assembler::finish`] is for a stream that was cut: nothing is closed, as in
//! [`crate::json::Assembler`], because arguments cut short must not look complete to a client.
//! A value that is written whole and never closed comes back as `Malformed` in both, since its key
//! was never written.
//!
//! `feed` takes one piece and returns how many of its bytes the call took: all of them before
//! `</function>`, and up to that tag in the piece that carries it, so the format routes whatever
//! follows (the block's closing marker, more text) itself.

use serde_json::Value;

use crate::{
    event::{Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::value::{json, Declared, Kind},
};

const FUNCTION_OPEN: usize = 0;
const PARAMETER_OPEN: usize = 1;
const PARAMETER_CLOSE: usize = 2;
const FUNCTION_CLOSE: usize = 3;
const TAGS: [&str; 4] = ["<function=", "<parameter=", "</parameter>", "</function>"];
const WITHOUT_A_FUNCTION: &str = "a tool call without a function tag";
const SECOND_FUNCTION: &str = "a second function tag inside a call";
const CLOSED_EARLY: &str = "a tool-call block that closed before its function tag closed";

/// The events of one tool call, from the bytes of its tags.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    scanner: Scanner,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// The function's name once its tag is whole, which is when `ToolCallStart` was pushed.
    function: Option<String>,
    /// Members written to the arguments object so far.
    written: u32,
    done: bool,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Before `<function=`.
    Opening,
    /// Inside `<function=…>`, reading the name up to `>`.
    FunctionName(String),
    /// After the function tag or a `</parameter>`.
    Between,
    /// Inside `<parameter=…>`, reading the key up to `>`.
    ParameterName(String),
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Whether the newline the template writes after the tag may still come: no value text yet.
    leading: bool,
    /// Where the value's text begins in `carried`, for a value written whole at its close.
    text_start: usize,
    mode: Mode,
}

#[derive(Clone, Debug)]
enum Mode {
    /// Pushed piece by piece; `held_newline` says `carried` ends with a newline that may be the
    /// template's.
    Streaming { held_newline: bool },
    /// Written at the close with [`json`].
    Whole,
    /// A string that may be null: written whole if its text is `null` or `None`, streamed from the
    /// first byte that rules both out.
    Undecided,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            scanner: Scanner::new(TAGS),
            carried: String::new(),
            stage: Stage::Opening,
            function: None,
            written: 0,
            done: false,
        }
    }

    /// Whether the call has closed, at `</function>` or as a block that never named a function;
    /// `feed` takes nothing after that, so the format stops here.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// Append the next bytes of the call, push the events they complete, and return how many of
    /// the bytes the call took: all of them before `</function>`, up to that tag in the piece that
    /// carries it, none after it.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) -> usize {
        if self.done {
            return 0;
        }
        let held_before = self.scanner.held().len();
        let mut seen = 0;
        for piece in self.scanner.feed(bytes) {
            seen += match &piece {
                Piece::Text(text) => text.len(),
                Piece::Marker(tag) => TAGS[*tag].len(),
            };
            self.take(piece, declared, out);
            if self.done {
                return seen - held_before;
            }
        }
        bytes.len()
    }

    /// The block the model ended before `</function>`: an open string is closed, the object is
    /// closed, and the bytes of a value or a tag cut short come back as `Malformed`. A block that
    /// never named a function comes back as `Malformed` whole.
    pub fn close(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.gather_held();
        if !self.started() {
            self.leftover(MalformedReason::Other(CLOSED_EARLY.to_string()), out);
            return;
        }
        let between = matches!(self.stage, Stage::Between);
        let open_string = self.open_string();
        if !between {
            self.leftover(MalformedReason::Other(CLOSED_EARLY.to_string()), out);
        }
        if open_string {
            self.push_fragment("\"", Text::default(), out);
        }
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.close_object(source, out);
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::default(),
        });
    }

    /// No more bytes will come, and the call never closed. Nothing is closed for the client: an
    /// open string stays open, the bytes of a value or a tag cut short come back as `Malformed`,
    /// and a started call ends. A block that never named a function comes back as `Malformed`
    /// whole.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.gather_held();
        self.leftover(MalformedReason::UnterminatedRegion, out);
        if self.started() {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
        }
    }

    /// The bytes the scanner held as a possible tag join the carried bytes: at an ending nothing
    /// completes them, and a whole tag is never held.
    fn gather_held(&mut self) {
        let held = self.scanner.held().to_string();
        self.carried.push_str(&held);
    }

    fn open_string(&self) -> bool {
        matches!(
            &self.stage,
            Stage::Value(ValueState {
                mode: Mode::Streaming { .. },
                ..
            })
        )
    }

    /// The carried bytes as `Malformed`, when there are any.
    fn leftover(&mut self, why: MalformedReason, out: &mut Events) {
        let text = std::mem::take(&mut self.carried);
        if !text.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(text),
                why,
            });
        }
    }

    fn take(&mut self, piece: Piece, declared: &Declared, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, declared, out),
            Piece::Marker(tag) => self.tag(tag, declared, out),
        }
    }

    fn tag(&mut self, tag: usize, declared: &Declared, out: &mut Events) {
        let bytes = TAGS[tag];
        let opening = matches!(self.stage, Stage::Opening);
        let between = matches!(self.stage, Stage::Between);
        match tag {
            FUNCTION_OPEN if opening => {
                self.carried.push_str(bytes);
                self.stage = Stage::FunctionName(String::new());
            }
            FUNCTION_CLOSE if opening => {
                self.carried.push_str(bytes);
                self.leftover(MalformedReason::Other(WITHOUT_A_FUNCTION.to_string()), out);
                self.done = true;
            }
            PARAMETER_OPEN if between => {
                self.carried.push_str(bytes);
                self.stage = Stage::ParameterName(String::new());
            }
            FUNCTION_CLOSE if between => {
                let source = Text::uncounted(std::mem::take(&mut self.carried));
                self.close_object(source, out);
                out.push(Event::ToolCallEnd {
                    index: self.index,
                    source: Text::uncounted(bytes),
                });
                self.done = true;
            }
            FUNCTION_OPEN if between => {
                self.carried.push_str(bytes);
                self.leftover(MalformedReason::Other(SECOND_FUNCTION.to_string()), out);
            }
            PARAMETER_CLOSE if matches!(self.stage, Stage::Value(_)) => {
                self.close_value(bytes, out);
            }
            // Inside a name or a value, and a closing tag with nothing open: the model's bytes,
            // where the model put them.
            _ => self.text(bytes, declared, out),
        }
    }

    fn text(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        match &mut self.stage {
            Stage::Opening | Stage::Between => self.carried.push_str(text),
            Stage::FunctionName(name) => match text.split_once('>') {
                Some((head, rest)) => {
                    name.push_str(head);
                    let name = std::mem::take(name);
                    self.carried.push_str(head);
                    self.carried.push('>');
                    out.push(Event::ToolCallStart {
                        index: self.index,
                        id: self.id.clone(),
                        name: name.clone(),
                        source: Text::uncounted(std::mem::take(&mut self.carried)),
                    });
                    self.function = Some(name);
                    self.stage = Stage::Between;
                    self.text(rest, declared, out);
                }
                None => {
                    name.push_str(text);
                    self.carried.push_str(text);
                }
            },
            Stage::ParameterName(key) => match text.split_once('>') {
                Some((head, rest)) => {
                    key.push_str(head);
                    let key = std::mem::take(key);
                    self.carried.push_str(head);
                    self.carried.push('>');
                    self.open_value(key, declared, out);
                    self.text(rest, declared, out);
                }
                None => {
                    key.push_str(text);
                    self.carried.push_str(text);
                }
            },
            Stage::Value(_) => self.value_text(text, out),
        }
    }

    /// The parameter's tag is whole: the value begins. A declared string opens its fragment now.
    fn open_value(&mut self, key: String, declared: &Declared, out: &mut Events) {
        let function = self.function.as_deref().unwrap_or_default();
        let kind = declared.kind(function, &key);
        let mode = match kind {
            Some(Kind::String) => Mode::Streaming {
                held_newline: false,
            },
            Some(Kind::NullableString) => Mode::Undecided,
            Some(Kind::Integer) | None => Mode::Whole,
        };
        if matches!(mode, Mode::Streaming { .. }) {
            let opening = format!("{}{}: \"", self.separator(), quoted(&key));
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.push_fragment(&opening, source, out);
            self.written += 1;
        }
        let text_start = self.carried.len();
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            leading: true,
            text_start,
            mode,
        });
    }

    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        if text.is_empty() {
            return;
        }
        let mut text = text;
        if value.leading {
            value.leading = false;
            if let Some(rest) = text.strip_prefix('\n') {
                self.carried.push('\n');
                value.text_start = self.carried.len();
                text = rest;
            }
        }
        if text.is_empty() {
            return;
        }
        match value.mode.clone() {
            Mode::Streaming { .. } => self.stream(text, out),
            Mode::Whole => self.carried.push_str(text),
            Mode::Undecided => {
                self.carried.push_str(text);
                let so_far = self.whole_text();
                if ["null", "None"].iter().any(|word| word.starts_with(so_far)) {
                    return;
                }
                self.open_string_late(out);
            }
        }
    }

    /// A string that may have been null is not: open it now, then stream what has arrived.
    fn open_string_late(&mut self, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        let key = std::mem::take(&mut value.key);
        let arrived = self.carried.split_off(value.text_start);
        value.mode = Mode::Streaming {
            held_newline: false,
        };
        value.text_start = 0;
        let opening = format!("{}{}: \"", self.separator(), quoted(&key));
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.push_fragment(&opening, source, out);
        self.written += 1;
        if let Stage::Value(value) = &mut self.stage {
            value.key = key;
        }
        self.stream(&arrived, out);
    }

    /// A piece of a streamed string: the held newline, if any, turns out to be the value's; a
    /// newline the piece ends with is held in its turn.
    fn stream(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(ValueState {
            mode: Mode::Streaming { held_newline },
            ..
        }) = &mut self.stage
        else {
            return;
        };
        let mut value = if *held_newline {
            format!("\n{text}")
        } else {
            text.to_string()
        };
        *held_newline = value.ends_with('\n');
        if *held_newline {
            value.pop();
        }
        let mut source = std::mem::take(&mut self.carried) + text;
        if *held_newline {
            source.pop();
            self.carried.push('\n');
        }
        if !value.is_empty() {
            self.push_fragment(&escaped(&value), Text::uncounted(source), out);
        } else if !source.is_empty() {
            // Only the template's newline arrived; it waits for the fragment that follows.
            self.carried.insert_str(0, &source);
        }
    }

    /// `</parameter>`: a streamed string gets its closing quote, any other value is written whole.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = match value.mode {
            Mode::Streaming { .. } => "\"".to_string(),
            Mode::Whole | Mode::Undecided => {
                let member = format!(
                    "{}{}: {}",
                    self.separator(),
                    quoted(&value.key),
                    json(self.whole_text(), value.kind)
                );
                self.written += 1;
                member
            }
        };
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(tag);
        self.push_fragment(&fragment, Text::uncounted(source), out);
        self.stage = Stage::Between;
    }

    /// The value's text so far, for a value written whole: what came after the tag and its
    /// newline, less the newline the template writes before `</parameter>`.
    fn whole_text(&self) -> &str {
        let Stage::Value(value) = &self.stage else {
            return "";
        };
        let text = &self.carried[value.text_start..];
        text.strip_suffix('\n').unwrap_or(text)
    }

    /// `, ` before every member but the first, which opens the object.
    fn separator(&self) -> &'static str {
        if self.written == 0 {
            "{"
        } else {
            ", "
        }
    }

    /// `}` after the members, or `{}` when there were none.
    fn close_object(&mut self, source: Text, out: &mut Events) {
        let closing = if self.written == 0 { "{}" } else { "}" };
        self.push_fragment(closing, source, out);
    }

    fn push_fragment(&self, json: &str, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: json.to_string(),
            source,
        });
    }
}

/// `text` as a JSON string, quotes included.
fn quoted(text: &str) -> String {
    Value::String(text.to_string()).to_string()
}

/// `text` as the inside of a JSON string: escaped, without the quotes.
fn escaped(text: &str) -> String {
    let quoted = quoted(text);
    quoted[1..quoted.len() - 1].to_string()
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;

    /// A tool `f` with a string `city`, an integer `limit`, a nullable string `note` and nothing
    /// declared for `extra`.
    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "limit": {"type": "integer"},
                    "note": {"type": ["string", "null"]},
                }}),
                strict: None,
            },
        }])
    }

    const CALL: &str = "\n<function=f>\n<parameter=city>\nParis\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>";

    /// Feeds the pieces, then the ending asked for, and returns the events with how many bytes
    /// each feed took.
    fn run(pieces: &[&str], ending: fn(Assembler, &mut Events)) -> (Vec<Event>, Vec<usize>) {
        let declared = declared();
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        let mut taken = Vec::new();
        for piece in pieces {
            taken.push(assembler.feed(piece, &declared, &mut out));
        }
        ending(assembler, &mut out);
        (out.drain(), taken)
    }

    fn finished(pieces: &[&str]) -> Vec<Event> {
        run(pieces, Assembler::finish).0
    }

    fn closed(pieces: &[&str]) -> Vec<Event> {
        run(pieces, Assembler::close).0
    }

    /// The arguments as the client sees them: every fragment, concatenated.
    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Every byte the events account for, in order.
    fn bytes(events: &[Event]) -> String {
        let mut events_list = Events::new();
        for event in events {
            events_list.push(event.clone());
        }
        events_list.text()
    }

    fn name_of(events: &[Event]) -> Option<&str> {
        events.iter().find_map(|event| match event {
            Event::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
    }

    fn malformed(events: &[Event]) -> Vec<(String, MalformedReason)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((text.text.clone(), why.clone())),
                _ => None,
            })
            .collect()
    }

    fn fragment(json: &str, source: &str) -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(source),
        }
    }

    #[test]
    fn a_call_streams_its_string_and_writes_its_integer_at_the_close() {
        let (events, taken) = run(&[CALL], Assembler::finish);
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".to_string(),
                    name: "f".to_string(),
                    source: Text::uncounted("\n<function=f>"),
                },
                fragment("{\"city\": \"", "\n<parameter=city>"),
                fragment("Paris", "\nParis"),
                fragment("\"", "\n</parameter>"),
                fragment(", \"limit\": 5", "\n<parameter=limit>\n5\n</parameter>"),
                fragment("}", "\n"),
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("</function>"),
                },
            ]
        );
        assert_eq!(arguments(&events), r#"{"city": "Paris", "limit": 5}"#);
        assert_eq!(bytes(&events), CALL);
        assert_eq!(taken, vec![CALL.len()]);
    }

    #[test]
    fn every_chunking_gives_the_same_arguments_and_accounts_for_every_byte() {
        let whole = finished(&[CALL]);
        for cut in 1..CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let (events, taken) = run(&[&CALL[..cut], &CALL[cut..]], Assembler::finish);
            assert_eq!(arguments(&events), arguments(&whole), "cut at {cut}");
            assert_eq!(bytes(&events), CALL, "cut at {cut}");
            assert_eq!(name_of(&events), Some("f"), "cut at {cut}");
            assert_eq!(taken.iter().sum::<usize>(), CALL.len(), "cut at {cut}");
            assert!(malformed(&events).is_empty(), "cut at {cut}");
        }
        // Byte by byte as well.
        let pieces: Vec<&str> = CALL
            .char_indices()
            .map(|(at, c)| &CALL[at..at + c.len_utf8()])
            .collect();
        let (events, _) = run(&pieces, Assembler::finish);
        assert_eq!(arguments(&events), arguments(&whole));
        assert_eq!(bytes(&events), CALL);
    }

    #[test]
    fn a_string_keeps_its_own_newlines_and_loses_the_templates_two() {
        let call = "<function=f>\n<parameter=city>\n\nTwo\nlines\n\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"city": "\nTwo\nlines\n"}"#);
        assert_eq!(bytes(&events), call);
        // The same with the value's text split after each newline, where the hold-back decides.
        let pieces = [
            "<function=f>\n<parameter=city>",
            "\n",
            "\n",
            "Two\n",
            "lines\n",
            "\n",
            "</parameter>\n</function>",
        ];
        let events = finished(&pieces);
        assert_eq!(arguments(&events), r#"{"city": "\nTwo\nlines\n"}"#);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn a_value_without_the_templates_newlines_is_read_as_written() {
        let call = "<function=f><parameter=city>Paris</parameter><parameter=limit>5</parameter></function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"city": "Paris", "limit": 5}"#);
        assert_eq!(bytes(&events), call);
        let empty = "<function=f>\n<parameter=city>\n</parameter>\n</function>";
        assert_eq!(arguments(&finished(&[empty])), r#"{"city": ""}"#);
    }

    #[test]
    fn a_string_that_may_be_null_streams_once_its_text_rules_null_out() {
        for (text, expected) in [
            ("null", "null"),
            ("None", "null"),
            ("Paris", r#""Paris""#),
            ("Nonesuch", r#""Nonesuch""#),
            ("nul", r#""nul""#),
            ("", r#""""#),
        ] {
            let call = format!("<function=f>\n<parameter=note>\n{text}\n</parameter>\n</function>");
            let events = finished(&[&call]);
            assert_eq!(
                arguments(&events),
                format!(r#"{{"note": {expected}}}"#),
                "{text:?}"
            );
            assert_eq!(bytes(&events), call, "{text:?}");
        }
        // `No` is held as a possible `None`; `tes` settles it, and the string streams from there.
        let events = finished(&[
            "<function=f>\n<parameter=note>\nNo",
            "tes",
            " here\n</parameter>\n</function>",
        ]);
        assert_eq!(
            events[1..5],
            [
                fragment("{\"note\": \"", "\n<parameter=note>\n"),
                fragment("Notes", "Notes"),
                fragment(" here", " here"),
                fragment("\"", "\n</parameter>"),
            ]
        );
    }

    #[test]
    fn an_undeclared_value_is_inferred_and_a_second_member_opens_with_a_comma() {
        let call = "<function=f>\n<parameter=extra>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=limit>\n007\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"extra": {"a": [1, 2]}, "limit": 7}"#
        );
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn a_call_without_parameters_is_an_empty_object() {
        let call = "\n<function=f>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            events[1..],
            [
                fragment("{}", "\n"),
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("</function>"),
                },
            ]
        );
    }

    #[test]
    fn a_function_the_request_did_not_declare_has_every_value_inferred() {
        let call = "<function=g>\n<parameter=city>\nParis\n</parameter>\n<parameter=n>\n5\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(name_of(&events), Some("g"));
        assert_eq!(arguments(&events), r#"{"city": "Paris", "n": 5}"#);
    }

    #[test]
    fn keys_and_text_are_escaped_as_json() {
        let call = "<function=f>\n<parameter=a\"b>\nx\n</parameter>\n<parameter=city>\nsay \"hi\" \\ tab\tend\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"a\"b": "x", "city": "say \"hi\" \\ tab\tend"}"#
        );
        assert!(serde_json::from_str::<Value>(&arguments(&events)).is_ok());
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn tags_inside_a_value_are_its_text() {
        let call = "<function=f>\n<parameter=city>\na <function=x> b <parameter=y> c </function>\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"city": "a <function=x> b <parameter=y> c </function>"}"#
        );
        assert_eq!(bytes(&events), call);
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn bytes_after_the_function_close_are_not_taken() {
        let declared = declared();
        let mut assembler = Assembler::new(3, "call_3");
        let mut out = Events::new();
        let piece = "<function=f>\n</function>\n</tool_call>more";
        let taken = assembler.feed(piece, &declared, &mut out);
        assert_eq!(&piece[..taken], "<function=f>\n</function>");
        assert!(assembler.done());
        assert_eq!(assembler.feed("x", &declared, &mut out), 0);
        let events = out.drain();
        assert_eq!(bytes(&events), "<function=f>\n</function>");
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 3,
                source: Text::uncounted("</function>"),
            })
        );
    }

    #[test]
    fn a_block_that_closed_before_its_function_tag_closed_gets_its_arguments_closed() {
        // Inside a streamed string: the string and the object are closed, the held bytes returned.
        let events = closed(&["<function=f>\n<parameter=city>\nPar", "is\n</param"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n</param".to_string(),
                MalformedReason::Other(CLOSED_EARLY.to_string())
            )]
        );
        assert_eq!(
            bytes(&events),
            "<function=f>\n<parameter=city>\nParis\n</param"
        );
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            })
        );
        // Inside a value written whole: its key was never written, so its bytes come back.
        let events =
            closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n<parameter=limit>\n5"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<parameter=limit>\n5".to_string(),
                MalformedReason::Other(CLOSED_EARLY.to_string())
            )]
        );
        // Without any parameter: an empty object.
        let events = closed(&["<function=f>\n"]);
        assert_eq!(arguments(&events), "{}");
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_cut_stream_leaves_an_open_string_open_and_returns_what_was_held() {
        let events = finished(&["<function=f>\n<parameter=city>\nPar", "is\n</param"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"#);
        assert_eq!(
            malformed(&events),
            vec![("\n</param".to_string(), MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(
            bytes(&events),
            "<function=f>\n<parameter=city>\nParis\n</param"
        );
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            })
        );
        // A value written whole, cut short.
        let events = finished(&["<function=f>\n<parameter=limit>\n5"]);
        assert_eq!(arguments(&events), "");
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<parameter=limit>\n5".to_string(),
                MalformedReason::UnterminatedRegion
            )]
        );
    }

    #[test]
    fn a_block_without_a_function_tag_is_malformed_whole() {
        for (pieces, why) in [
            (
                vec!["\n{\"name\": \"f\"}\n"],
                MalformedReason::UnterminatedRegion,
            ),
            (vec!["\n<function=f"], MalformedReason::UnterminatedRegion),
        ] {
            let events = finished(&pieces);
            assert_eq!(malformed(&events), vec![(pieces.concat(), why)]);
            assert_eq!(events.len(), 1);
        }
        let events = closed(&["\n<parameter=city>\nParis\n"]);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<parameter=city>\nParis\n".to_string(),
                MalformedReason::Other(CLOSED_EARLY.to_string())
            )]
        );
        assert_eq!(events.len(), 1);
        // `</function>` with no function open ends the block as one that named none.
        let (events, taken) = run(&["\nx</function>rest"], Assembler::finish);
        assert_eq!(
            malformed(&events),
            vec![(
                "\nx</function>".to_string(),
                MalformedReason::Other(WITHOUT_A_FUNCTION.to_string())
            )]
        );
        assert_eq!(taken, vec!["\nx</function>".len()]);
    }

    #[test]
    fn a_second_function_tag_inside_a_call_is_malformed_and_the_call_goes_on() {
        let call = "<function=f>\n<function=g>\n<parameter=limit>\n5\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(name_of(&events), Some("f"));
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<function=".to_string(),
                MalformedReason::Other(SECOND_FUNCTION.to_string())
            )]
        );
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn whitespace_between_tags_is_carried_into_the_next_events_source() {
        let call = "<function=f>  \n\n<parameter=limit>\n5\n</parameter>  \n</function>";
        let events = finished(&[call]);
        assert_eq!(
            events[1..],
            [
                fragment("{\"limit\": 5", "  \n\n<parameter=limit>\n5\n</parameter>"),
                fragment("}", "  \n"),
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("</function>"),
                },
            ]
        );
    }
}
