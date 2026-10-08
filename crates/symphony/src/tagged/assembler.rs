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
//! Every byte of the call lands in exactly one event. Whitespace between tags is the template's
//! and goes into the source of the next event: the function tag's into `ToolCallStart`, a
//! parameter tag's and the value's text into the fragments they produce, the rest into the
//! closing `}`. Anything else between tags, a tag out of its place, a tag that cuts a name short
//! and a name that is empty come back as `Malformed` with a reason, so the model's bytes are never
//! read as something they are not and never vanish into a source. Inside a value only
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

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::value::{json, Declared, Kind, NULL_WORDS},
};

const FUNCTION_OPEN: usize = 0;
const PARAMETER_OPEN: usize = 1;
const PARAMETER_CLOSE: usize = 2;
const FUNCTION_CLOSE: usize = 3;
/// The four tags of the syntax: a function's opening up to its name, a parameter's opening up to
/// its key, a parameter's close, a function's close.
pub const TAGS: [&str; 4] = ["<function=", "<parameter=", "</parameter>", "</function>"];

/// How a family's template writes a value that is not a string. Qwen 3.5, Qwen3-Coder and their
/// kin write an object or a list as JSON (`tojson`) and a boolean or null as Python's word
/// (`True`, `None`); Seed-OSS writes every value with Python's `str`, an object or a list as its
/// repr (`{'size': 'large'}`). The assembler reads all of these as the JSON they stand for
/// ([`json`]); the grammar derived from a table pins an object's or a list's
/// shape only where the template writes it as JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// An object or a list as JSON, a boolean or null as Python's word: Qwen 3.5 and later,
    /// Qwen3-Coder.
    Json,
    /// Every value as Python's text, an object or a list as its repr: Seed-OSS.
    Python,
}
const WITHOUT_A_FUNCTION: &str = "a tool call without a function tag";
const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const EMPTY_NAME: &str = "a tag without a name";
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
    /// Inside `<function=…>`: the name is `carried[start..]` until `>`.
    FunctionName { start: usize },
    /// After the function tag or a `</parameter>`.
    Between,
    /// Inside `<parameter=…>`: the key is `carried[start..]` until `>`.
    ParameterName { start: usize },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Whether the newline the template writes after the tag may still come: no value text yet.
    leading: bool,
    mode: Mode,
}

#[derive(Clone, Debug)]
enum Mode {
    /// Pushed piece by piece; `held_newline` says `carried` ends with a newline that may be the
    /// template's.
    Streaming { held_newline: bool },
    /// Written at the close with [`json`]; the text is `carried[start..]`.
    Whole { start: usize },
    /// A string that may be null: written whole if its text is a null word, streamed from the
    /// first byte that rules the words out; the text is `carried[start..]` until then.
    Undecided { start: usize },
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
    /// closed, and the bytes of a value, a name or a tag cut short come back as `Malformed`. A
    /// block that never named a function has had its text reported as it came; what is left of it
    /// comes back as `Malformed` here.
    pub fn close(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.held_tag();
        if !self.started() {
            self.carried.push_str(&held);
            if matches!(self.stage, Stage::Opening) {
                self.leftover(MalformedReason::Other(WITHOUT_A_FUNCTION.to_string()), out);
            } else {
                self.report_tag(CLOSED_EARLY, out);
            }
            return;
        }
        if matches!(self.stage, Stage::Between) {
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.close_object(source, out);
            self.carried = held;
            self.leftover(MalformedReason::Other(CLOSED_EARLY.to_string()), out);
        } else {
            let open_string = self.open_string();
            self.carried.push_str(&held);
            self.report_tag(CLOSED_EARLY, out);
            if open_string {
                self.push_fragment("\"".to_string(), Text::default(), out);
            }
            self.close_object(Text::default(), out);
        }
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::default(),
        });
    }

    /// No more bytes will come, and the call never closed. Nothing is closed for the client: an
    /// open string stays open, the bytes of a value, a name or a tag cut short come back as
    /// `Malformed`, and a started call ends. A block that never named a function has had its text
    /// reported as it came; what is left of it comes back as `Malformed` here.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.held_tag();
        self.carried.push_str(&held);
        self.leftover(MalformedReason::UnterminatedRegion, out);
        if self.started() {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
        }
    }

    /// The bytes the scanner held as the start of a tag. At an ending nothing completes them, and
    /// a whole tag is never held, so they are text no event has used.
    fn held_tag(&mut self) -> String {
        self.scanner.held().to_string()
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

    /// A tag that is reported, with the carried bytes: the whitespace before it is the template's
    /// and is dropped, the tag's own bytes come back as `Malformed`.
    fn report_tag(&mut self, why: &str, out: &mut Events) {
        let text = std::mem::take(&mut self.carried);
        let tag_at = text
            .char_indices()
            .find(|(_, c)| !c.is_whitespace())
            .map_or(text.len(), |(at, _)| at);
        if tag_at > 0 {
            out.push(Event::Dropped {
                text: Text::uncounted(&text[..tag_at]),
                why: DropReason::Wrapper,
            });
        }
        if tag_at < text.len() {
            out.push(Event::Malformed {
                text: Text::uncounted(&text[tag_at..]),
                why: MalformedReason::Other(why.to_string()),
            });
        }
    }

    /// The carried bytes as the template's wrapper, when there are any: they come before text
    /// that is reported on its own, so they cannot wait for a later event's source.
    fn drop_carried(&mut self, out: &mut Events) {
        let text = std::mem::take(&mut self.carried);
        if !text.is_empty() {
            out.push(Event::Dropped {
                text: Text::uncounted(text),
                why: DropReason::Wrapper,
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
        match self.stage {
            Stage::Opening | Stage::Between => self.tag_between(tag, out),
            // A tag cuts the name short: the tag's bytes so far are reported, and the tag is read
            // where the name began.
            Stage::FunctionName { .. } => {
                self.report_tag(TAG_CUT_SHORT, out);
                self.stage = if self.started() {
                    Stage::Between
                } else {
                    Stage::Opening
                };
                self.tag_between(tag, out);
            }
            Stage::ParameterName { .. } => {
                self.report_tag(TAG_CUT_SHORT, out);
                self.stage = Stage::Between;
                self.tag_between(tag, out);
            }
            Stage::Value(_) if tag == PARAMETER_CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tags are the value's text.
            Stage::Value(_) => self.text(bytes, declared, out),
        }
    }

    /// A tag where the syntax expects one: before the function tag, or between parameters.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = TAGS[tag];
        let between = matches!(self.stage, Stage::Between);
        match tag {
            FUNCTION_OPEN => {
                self.carried.push_str(bytes);
                self.stage = Stage::FunctionName {
                    start: self.carried.len(),
                };
            }
            PARAMETER_OPEN if between => {
                self.carried.push_str(bytes);
                self.stage = Stage::ParameterName {
                    start: self.carried.len(),
                };
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
            FUNCTION_CLOSE => {
                self.carried.push_str(bytes);
                self.leftover(MalformedReason::Other(WITHOUT_A_FUNCTION.to_string()), out);
                self.done = true;
            }
            _ => {
                self.drop_carried(out);
                out.push(Event::Malformed {
                    text: Text::uncounted(bytes),
                    why: MalformedReason::Other(TAG_OUT_OF_PLACE.to_string()),
                });
            }
        }
    }

    fn text(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        match self.stage {
            Stage::Opening | Stage::Between => self.text_between(text, out),
            Stage::FunctionName { start } => match text.split_once('>') {
                Some((head, rest)) => {
                    self.carried.push_str(head);
                    let name = self.carried[start..].to_string();
                    self.carried.push('>');
                    self.function_named(name, out);
                    self.text(rest, declared, out);
                }
                None => self.carried.push_str(text),
            },
            Stage::ParameterName { start } => match text.split_once('>') {
                Some((head, rest)) => {
                    self.carried.push_str(head);
                    let key = self.carried[start..].to_string();
                    self.carried.push('>');
                    self.parameter_named(key, declared, out);
                    self.text(rest, declared, out);
                }
                None => self.carried.push_str(text),
            },
            Stage::Value(_) => self.value_text(text, out),
        }
    }

    /// Text where the syntax has tags: whitespace is the template's, anything else is reported.
    fn text_between(&mut self, text: &str, out: &mut Events) {
        let mut rest = text;
        while let Some(first) = rest.chars().next() {
            let space = first.is_whitespace();
            let length = rest
                .char_indices()
                .find(|(_, c)| c.is_whitespace() != space)
                .map_or(rest.len(), |(at, _)| at);
            if space {
                self.carried.push_str(&rest[..length]);
            } else {
                self.drop_carried(out);
                out.push(Event::Malformed {
                    text: Text::uncounted(&rest[..length]),
                    why: MalformedReason::Other(TEXT_BETWEEN_TAGS.to_string()),
                });
            }
            rest = &rest[length..];
        }
    }

    /// The function tag is whole: the call starts, unless the name is empty or a call has started
    /// already, in which case the tag is reported.
    fn function_named(&mut self, name: String, out: &mut Events) {
        if name.is_empty() {
            self.report_tag(EMPTY_NAME, out);
            self.stage = if self.started() {
                Stage::Between
            } else {
                Stage::Opening
            };
        } else if self.started() {
            self.report_tag(SECOND_FUNCTION, out);
            self.stage = Stage::Between;
        } else {
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name: name.clone(),
                source: Text::uncounted(std::mem::take(&mut self.carried)),
            });
            self.function = Some(name);
            self.stage = Stage::Between;
        }
    }

    /// The parameter's tag is whole: the value begins. A declared string opens its fragment now.
    fn parameter_named(&mut self, key: String, declared: &Declared, out: &mut Events) {
        if key.is_empty() {
            // vLLM keeps an empty parameter name and writes `"": value`; here it is reported.
            self.report_tag(EMPTY_NAME, out);
            self.stage = Stage::Between;
            return;
        }
        let function = self.function.as_deref().unwrap_or_default();
        let kind = declared.kind(function, &key);
        let mode = match kind {
            Some(Kind::String) => {
                self.open_string_fragment(&key, out);
                Mode::Streaming {
                    held_newline: false,
                }
            }
            Some(Kind::NullableString) => Mode::Undecided {
                start: self.carried.len(),
            },
            Some(Kind::Integer | Kind::Array | Kind::Object) | None => Mode::Whole {
                start: self.carried.len(),
            },
        };
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            leading: true,
            mode,
        });
    }

    /// `{"key": "` or `, "key": "`, with the carried bytes as its source.
    fn open_string_fragment(&mut self, key: &str, out: &mut Events) {
        let mut opening = String::with_capacity(key.len() + 8);
        opening.push_str(self.separator());
        push_quoted(&mut opening, key);
        opening.push_str(": \"");
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.push_fragment(opening, source, out);
        self.written += 1;
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
                if let Mode::Whole { start } | Mode::Undecided { start } = &mut value.mode {
                    *start = self.carried.len();
                }
                text = rest;
            }
        }
        if text.is_empty() {
            // The piece was the template's newline alone.
            return;
        }
        match value.mode {
            Mode::Streaming { .. } => self.stream(text, out),
            Mode::Whole { .. } => self.carried.push_str(text),
            Mode::Undecided { start } => {
                self.carried.push_str(text);
                let so_far = self.whole_text(start);
                if !NULL_WORDS.iter().any(|word| word.starts_with(so_far)) {
                    self.open_string_late(start, out);
                }
            }
        }
    }

    /// A string that may have been null is not: open it now, then stream what has arrived.
    fn open_string_late(&mut self, start: usize, out: &mut Events) {
        let arrived = self.carried.split_off(start);
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        let key = value.key.clone();
        value.mode = Mode::Streaming {
            held_newline: false,
        };
        self.open_string_fragment(&key, out);
        self.stream(&arrived, out);
    }

    /// A piece of a streamed string. A newline the piece ends with is held, since it may be the
    /// template's; a newline held before it turns out to be the value's.
    fn stream(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(ValueState {
            mode: Mode::Streaming { held_newline },
            ..
        }) = &mut self.stage
        else {
            return;
        };
        let had_newline = *held_newline;
        let (body, ends_with_newline) = match text.strip_suffix('\n') {
            Some(body) => (body, true),
            None => (text, false),
        };
        *held_newline = ends_with_newline;
        if body.is_empty() && !had_newline {
            // The piece is one newline; it waits with the carried bytes for the piece that follows.
            self.carried.push('\n');
            return;
        }
        let mut json = String::with_capacity(body.len() + 2);
        if had_newline {
            json.push_str("\\n");
        }
        push_escaped(&mut json, body);
        // The source: the carried bytes (the template's newline after the tag, or the newline that
        // was held and is now the value's) and the piece, less the newline held in its turn.
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(body);
        if ends_with_newline {
            self.carried.push('\n');
        }
        self.push_fragment(json, Text::uncounted(source), out);
    }

    /// `</parameter>`: a streamed string gets its closing quote, any other value is written whole.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = match value.mode {
            Mode::Streaming { .. } => "\"".to_string(),
            Mode::Whole { start } | Mode::Undecided { start } => {
                let written = json(self.whole_text(start), value.kind);
                let mut member = String::with_capacity(value.key.len() + written.len() + 8);
                member.push_str(self.separator());
                push_quoted(&mut member, &value.key);
                member.push_str(": ");
                member.push_str(&written);
                self.written += 1;
                member
            }
        };
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(tag);
        self.push_fragment(fragment, Text::uncounted(source), out);
        self.stage = Stage::Between;
    }

    /// The text of a value written whole: what came after the tag and its newline, less the
    /// newline the template writes before `</parameter>`.
    fn whole_text(&self, start: usize) -> &str {
        let text = &self.carried[start..];
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
        self.push_fragment(closing.to_string(), source, out);
    }

    fn push_fragment(&self, json: String, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}

/// Appends `text` as a JSON string, quotes included.
pub(crate) fn push_quoted(out: &mut String, text: &str) {
    out.push('"');
    push_escaped(out, text);
    out.push('"');
}

/// Appends `text` as the inside of a JSON string, escaped as `serde_json` escapes it: the quote,
/// the backslash and the control characters, nothing else.
pub(crate) fn push_escaped(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => {
                out.push_str("\\u00");
                out.push(char::from_digit(u32::from(c) >> 4, 16).unwrap_or('0'));
                out.push(char::from_digit(u32::from(c) & 0xf, 16).unwrap_or('0'));
            }
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::{json as value, Value};

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

    const CALL: &str = concat!(
        "\n<function=f>\n<parameter=city>\nParis\n</parameter>\n<parameter=limit>\n5\n</parameter>",
        "\n<parameter=note>\nNo rain\n</parameter>\n</function>"
    );
    const CALL_ARGUMENTS: &str = r#"{"city": "Paris", "limit": 5, "note": "No rain"}"#;

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
        let mut list = Events::new();
        for event in events {
            list.push(event.clone());
        }
        list.text()
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

    fn dropped(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(text.text.clone()),
                _ => None,
            })
            .collect()
    }

    fn ends(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, Event::ToolCallEnd { .. }))
            .count()
    }

    fn other(reason: &str) -> MalformedReason {
        MalformedReason::Other(reason.to_string())
    }

    fn fragment(json: &str, source: &str) -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(source),
        }
    }

    fn end(source: &str) -> Event {
        Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted(source),
        }
    }

    /// No fragment is empty and no `Malformed` or `Dropped` text is empty.
    fn nothing_empty(events: &[Event]) {
        for event in events {
            match event {
                Event::ToolCallArguments { json, .. } => assert!(!json.is_empty(), "{event:?}"),
                Event::Malformed { text, .. } | Event::Dropped { text, .. } => {
                    assert!(!text.text.is_empty(), "{event:?}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn a_call_streams_its_strings_and_writes_its_integer_at_the_close() {
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
                fragment(", \"note\": \"", "\n<parameter=note>\n"),
                fragment("No rain", "No rain"),
                fragment("\"", "\n</parameter>"),
                fragment("}", "\n"),
                end("</function>"),
            ]
        );
        assert_eq!(arguments(&events), CALL_ARGUMENTS);
        assert_eq!(bytes(&events), CALL);
        assert_eq!(taken, vec![CALL.len()]);
    }

    #[test]
    fn every_chunking_gives_the_same_arguments_and_accounts_for_every_byte() {
        for cut in 1..CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let (events, taken) = run(&[&CALL[..cut], &CALL[cut..]], Assembler::finish);
            assert_eq!(arguments(&events), CALL_ARGUMENTS, "cut at {cut}");
            assert_eq!(bytes(&events), CALL, "cut at {cut}");
            assert_eq!(name_of(&events), Some("f"), "cut at {cut}");
            assert_eq!(taken.iter().sum::<usize>(), CALL.len(), "cut at {cut}");
            assert!(malformed(&events).is_empty(), "cut at {cut}");
            nothing_empty(&events);
        }
        // Byte by byte as well.
        let pieces: Vec<&str> = CALL
            .char_indices()
            .map(|(at, c)| &CALL[at..at + c.len_utf8()])
            .collect();
        let (events, _) = run(&pieces, Assembler::finish);
        assert_eq!(arguments(&events), CALL_ARGUMENTS);
        assert_eq!(bytes(&events), CALL);
        nothing_empty(&events);
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
        nothing_empty(&events);
    }

    #[test]
    fn a_value_without_the_templates_newlines_is_read_as_written() {
        let call = concat!(
            "<function=f><parameter=city>Paris</parameter>",
            "<parameter=limit>5</parameter></function>"
        );
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"city": "Paris", "limit": 5}"#);
        assert_eq!(bytes(&events), call);
        let empty = "<function=f>\n<parameter=city>\n</parameter>\n</function>";
        assert_eq!(arguments(&finished(&[empty])), r#"{"city": ""}"#);
    }

    #[test]
    fn a_type_beside_an_enum_decides_and_an_array_stays_one() {
        // bellwether's qwen3-coder-next/parse/bfcl-live-multiple-146-58-0: BFCL declares the list
        // `{"type": "array", "items": {"type": "string"}, "enum": [...]}`, and vLLM reads it as an
        // array; before this the enum made it a string holding `[]`.
        let declared = Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["temperature", "humidity"],
                    },
                }}),
                strict: None,
            },
        }]);
        let call = "<function=f>\n<parameter=metrics>\n[]\n</parameter>\n</function>";
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assembler.feed(call, &declared, &mut out);
        assembler.finish(&mut out);
        let events = out.drain();
        assert_eq!(arguments(&events), r#"{"metrics": []}"#);
        assert_eq!(bytes(&events), call);
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
        let call = concat!(
            "<function=f>\n<parameter=extra>\n{\"a\": [1, 2]}\n</parameter>",
            "\n<parameter=limit>\n007\n</parameter>\n</function>"
        );
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
        assert_eq!(events[1..], [fragment("{}", "\n"), end("</function>")]);
    }

    #[test]
    fn a_function_the_request_did_not_declare_has_every_value_inferred() {
        let call = concat!(
            "<function=g>\n<parameter=city>\nParis\n</parameter>",
            "\n<parameter=n>\n5\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(name_of(&events), Some("g"));
        assert_eq!(arguments(&events), r#"{"city": "Paris", "n": 5}"#);
    }

    #[test]
    fn keys_and_text_are_escaped_as_json() {
        let call = concat!(
            "<function=f>\n<parameter=a\"b>\nx\n</parameter>",
            "\n<parameter=city>\nsay \"hi\" \\ tab\tend \u{1}\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"a\"b": "x", "city": "say \"hi\" \\ tab\tend \u0001"}"#
        );
        assert!(serde_json::from_str::<Value>(&arguments(&events)).is_ok());
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn the_escaping_is_serde_jsons() {
        let texts = [
            "plain",
            "quote \" backslash \\ slash / tab \t newline \n return \r",
            "\u{8}\u{c}\u{1}\u{1f}\u{7f}",
            "Café 🌍 \u{200d} \u{10ffff}",
            "",
        ];
        for text in texts {
            let mut escaped = String::new();
            push_escaped(&mut escaped, text);
            let expected = serde_json::to_string(text).unwrap_or_default();
            assert_eq!(escaped, expected[1..expected.len() - 1], "{text:?}");
        }
    }

    #[test]
    fn tags_inside_a_value_are_its_text() {
        let call = concat!(
            "<function=f>\n<parameter=city>\na <function=x> b <parameter=y> c </function>",
            "\n</parameter>\n</function>"
        );
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
        // Neither ending adds anything to a call that closed.
        let mut after = Events::new();
        assembler.clone().close(&mut after);
        assembler.finish(&mut after);
        assert!(after.is_empty());
    }

    #[test]
    fn text_between_tags_is_malformed_and_the_whitespace_before_it_is_dropped() {
        // A parameter tag the model misspelt is reported, not read as a parameter and not hidden.
        let call = concat!(
            "<function=f>\n<parameter city>\nParis\n</parameter>",
            "\n<parameter=limit>\n5\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![
                ("<parameter".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("city>".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("Paris".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("</parameter>".to_string(), other(TAG_OUT_OF_PLACE)),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", " ", "\n", "\n"]);
        assert_eq!(bytes(&events), call);
        nothing_empty(&events);
        // Prose before the function tag is reported too; the whitespace before the tag stays its
        // source.
        let events = finished(&["I will call it now <function=f>\n</function>"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("I".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("will".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("call".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("it".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("now".to_string(), other(TEXT_BETWEEN_TAGS)),
            ]
        );
        assert_eq!(
            events
                .iter()
                .find(|e| matches!(e, Event::ToolCallStart { .. })),
            Some(&Event::ToolCallStart {
                index: 0,
                id: "call_0".to_string(),
                name: "f".to_string(),
                source: Text::uncounted(" <function=f>"),
            })
        );
        assert_eq!(
            bytes(&events),
            "I will call it now <function=f>\n</function>"
        );
    }

    #[test]
    fn a_tag_inside_a_name_cuts_the_name_short() {
        // `</function>` inside the function's name ends the block, which named no function.
        let (events, taken) = run(&["<function=f</function>rest"], Assembler::finish);
        assert_eq!(name_of(&events), None);
        assert_eq!(
            malformed(&events),
            vec![
                ("<function=f".to_string(), other(TAG_CUT_SHORT)),
                ("</function>".to_string(), other(WITHOUT_A_FUNCTION)),
            ]
        );
        assert_eq!(taken, vec!["<function=f</function>".len()]);
        assert_eq!(ends(&events), 0);
        // Inside a parameter's key, the call goes on and `</function>` ends it.
        let call = "<function=f>\n<parameter=a</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            malformed(&events),
            vec![("<parameter=a".to_string(), other(TAG_CUT_SHORT))]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(events.last(), Some(&end("</function>")));
        assert_eq!(bytes(&events), call);
        // A parameter tag inside the key starts the next parameter.
        let call = "<function=f>\n<parameter=a<parameter=limit>\n5\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn an_empty_name_is_malformed() {
        let events = finished(&["<function=>\n<parameter=limit>\n5\n</parameter>\n</function>"]);
        assert_eq!(name_of(&events), None);
        assert_eq!(
            malformed(&events)[0],
            ("<function=>".to_string(), other(EMPTY_NAME))
        );
        assert_eq!(arguments(&events), "");
        let call = concat!(
            "<function=f>\n<parameter=>\nx\n</parameter>",
            "\n<parameter=limit>\n5\n</parameter>\n</function>"
        );
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![
                ("<parameter=>".to_string(), other(EMPTY_NAME)),
                ("x".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("</parameter>".to_string(), other(TAG_OUT_OF_PLACE)),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", "\n", "\n"]);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn a_block_that_closed_before_its_function_tag_closed_gets_its_arguments_closed() {
        // Inside a streamed string: the string and the object are closed, the held bytes returned.
        let events = closed(&["<function=f>\n<parameter=city>\nPar", "is\n</param"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![("</param".to_string(), other(CLOSED_EARLY))]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(
            bytes(&events),
            "<function=f>\n<parameter=city>\nParis\n</param"
        );
        assert_eq!(events.last(), Some(&end("")));
        assert_eq!(ends(&events), 1);
        // Inside a value written whole: its key was never written, so its bytes come back.
        let events =
            closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n<parameter=limit>\n5"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![("<parameter=limit>\n5".to_string(), other(CLOSED_EARLY))]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        // A string that may still be null is a value written whole.
        let events = closed(&["<function=f>\n<parameter=note>\nnul"]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            malformed(&events),
            vec![("<parameter=note>\nnul".to_string(), other(CLOSED_EARLY))]
        );
        // Inside a key, and a tag the scanner held between parameters: the same.
        let events = closed(&["<function=f>\n<parameter=ci"]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            malformed(&events),
            vec![("<parameter=ci".to_string(), other(CLOSED_EARLY))]
        );
        let events = closed(&["<function=f>\n<param"]);
        assert_eq!(
            events[1..],
            [
                fragment("{}", "\n"),
                Event::Malformed {
                    text: Text::uncounted("<param"),
                    why: other(CLOSED_EARLY),
                },
                end(""),
            ]
        );
        // Without any parameter: an empty object, with the whitespace as its source.
        let events = closed(&["<function=f>\n"]);
        assert_eq!(events[1..], [fragment("{}", "\n"), end("")]);
        assert_eq!(bytes(&events), "<function=f>\n");
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
        assert_eq!(events.last(), Some(&end("")));
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
        // A started call with no parameter still ends, and only what arrived is reported.
        let events = finished(&["<function=f>\n"]);
        assert_eq!(
            events[1..],
            [
                Event::Malformed {
                    text: Text::uncounted("\n"),
                    why: MalformedReason::UnterminatedRegion
                },
                end(""),
            ]
        );
        let events = finished(&["<function=f>"]);
        assert_eq!(events[1..], [end("")]);
        nothing_empty(&events);
    }

    #[test]
    fn a_block_without_a_function_tag_reports_every_byte() {
        // A JSON object where tags were expected: its words are text between tags, the whitespace
        // around them the wrapper, and what is left at the end never closed.
        let events = finished(&["\n{\"name\": \"f\"}\n"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("{\"name\":".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("\"f\"}".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("\n".to_string(), MalformedReason::UnterminatedRegion),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n", " "]);
        assert_eq!(ends(&events), 0);
        assert_eq!(bytes(&events), "\n{\"name\": \"f\"}\n");
        // A function tag cut short by the end of the stream.
        let events = finished(&["\n<function=f"]);
        assert_eq!(
            malformed(&events),
            vec![(
                "\n<function=f".to_string(),
                MalformedReason::UnterminatedRegion
            )]
        );
        assert_eq!(events.len(), 1);
        let events = closed(&["\n<parameter=city>\nParis\n"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("<parameter=".to_string(), other(TAG_OUT_OF_PLACE)),
                ("city>".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("Paris".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("\n".to_string(), other(WITHOUT_A_FUNCTION)),
            ]
        );
        assert_eq!(bytes(&events), "\n<parameter=city>\nParis\n");
        let events = closed(&["\n"]);
        assert_eq!(
            malformed(&events),
            vec![("\n".to_string(), other(WITHOUT_A_FUNCTION))]
        );
        assert!(closed(&[]).is_empty());
        // Held bytes at the ending are reported too, with the whitespace before them.
        let events = closed(&["\n<para"]);
        assert_eq!(
            malformed(&events),
            vec![("\n<para".to_string(), other(WITHOUT_A_FUNCTION))]
        );
        let events = closed(&["<function=f</fun"]);
        assert_eq!(
            malformed(&events),
            vec![("<function=f</fun".to_string(), other(CLOSED_EARLY))]
        );
        assert_eq!(ends(&events), 0);
        // `</function>` with no function open ends the block as one that named none.
        let (events, taken) = run(&["\nx</function>rest"], Assembler::finish);
        assert_eq!(
            malformed(&events),
            vec![
                ("x".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("</function>".to_string(), other(WITHOUT_A_FUNCTION)),
            ]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(taken, vec!["\nx</function>".len()]);
    }

    #[test]
    fn a_second_function_tag_inside_a_call_is_reported_and_the_call_goes_on() {
        let call = "<function=f>\n<function=g>\n<parameter=limit>\n5\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(name_of(&events), Some("f"));
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![("<function=g>".to_string(), other(SECOND_FUNCTION))]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
        assert_eq!(bytes(&events), call);
        // The block ending inside the second function tag still ends the call, closed.
        let events = closed(&["<function=f>\n<parameter=city>\nParis\n</parameter>\n<function=g"]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            malformed(&events),
            vec![("<function=g".to_string(), other(CLOSED_EARLY))]
        );
        assert_eq!(events.last(), Some(&end("")));
        assert_eq!(ends(&events), 1);
        let events = closed(&["<function=f>\n<function="]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(ends(&events), 1);
    }

    #[test]
    fn a_report_with_no_whitespace_before_it_drops_nothing() {
        let events = finished(&["<function=f>x<function=g>y</function>"]);
        assert_eq!(
            malformed(&events),
            vec![
                ("x".to_string(), other(TEXT_BETWEEN_TAGS)),
                ("<function=g>".to_string(), other(SECOND_FUNCTION)),
                ("y".to_string(), other(TEXT_BETWEEN_TAGS)),
            ]
        );
        assert!(dropped(&events).is_empty());
        nothing_empty(&events);
        // Whitespace outside ASCII counts as the template's too.
        let events = finished(&["<function=f>\u{3000}\u{a0}x</function>"]);
        assert_eq!(dropped(&events), vec!["\u{3000}\u{a0}"]);
        assert_eq!(
            malformed(&events),
            vec![("x".to_string(), other(TEXT_BETWEEN_TAGS))]
        );
    }

    #[test]
    fn a_string_that_may_be_null_is_read_without_the_templates_newlines_too() {
        for (text, expected) in [
            ("null", "null"),
            ("Paris", r#""Paris""#),
            ("nul", r#""nul""#),
        ] {
            let call = format!("<function=f><parameter=note>{text}</parameter></function>");
            let events = finished(&[&call]);
            assert_eq!(
                arguments(&events),
                format!(r#"{{"note": {expected}}}"#),
                "{text:?}"
            );
            assert_eq!(bytes(&events), call, "{text:?}");
        }
    }

    #[test]
    fn a_closing_parameter_tag_between_parameters_is_out_of_place() {
        let call = "<function=f>\n<parameter=limit>\n5\n</parameter>\n</parameter>\n</function>";
        let events = finished(&[call]);
        assert_eq!(arguments(&events), r#"{"limit": 5}"#);
        assert_eq!(
            malformed(&events),
            vec![("</parameter>".to_string(), other(TAG_OUT_OF_PLACE))]
        );
        assert_eq!(dropped(&events), vec!["\n"]);
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
                end("</function>"),
            ]
        );
    }
}
