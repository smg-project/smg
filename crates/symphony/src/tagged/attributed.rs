//! Parameters named by an attribute: the events of one `<invoke name="NAME">` block whose
//! arguments are `<parameter name="KEY">VALUE</parameter>` tags, typed by the request's tools.
//!
//! MiniMax M2 writes a call as
//!
//! ```text
//! <invoke name="get_weather">
//! <parameter name="city">Paris</parameter>
//! <parameter name="days">3</parameter>
//! </invoke>
//! ```
//!
//! inside a `<minimax:tool_call>` block that holds one invoke or several, each tag on a line of
//! its own. MiniCPM5 writes the same shape with its own tags and no block,
//!
//! ```text
//! <function name="weather"><param name="city">Paris</param><param name="days">3</param></function>
//! ```
//!
//! one call after another on lines of their own; dots3 writes M2's tags with a block per call and
//! every value on a line of its own. The engine's table owns the block's and the invoke's markers
//! ([`formats::minimax_m2`], [`formats::minicpm5`], [`formats::dots3`]): it enters this
//! assembler at `<invoke name="` (`<function name="`) and leaves it at the invoke's closing
//! tag, so the assembler starts inside the function's name and never sees that tag; the engine
//! hands it the tag's bytes as the call's end. Between those two the assembler reads the
//! parameter tags itself, in the family's spelling ([`Tags`]).
//!
//! What the syntax says, and what the assembler does with it:
//!
//! - **The request's tools type the value**, as in the keyed assembler: a declared string is its
//!   text, every byte of it, streamed as it arrives; every other value is written whole at
//!   `</parameter>` through [`json`], which reads the JSON the template writes for a number, a
//!   boolean, a list, an object or null (`tojson`), and infers the rest. A parameter tag with
//!   anything between the name's closing quote and `>` is no parameter tag the syntax has: the
//!   whole tag comes back as `Malformed`, and no parameter opens.
//! - **Inside a value only `</parameter>` is a tag**; `<parameter name="` there is the value's
//!   text.
//! - **A value may be a CDATA section** where the family's template writes one ([`Tags::cdata`]):
//!   MiniCPM5 wraps a value that holds `<`, `&` or a newline as `<![CDATA[…]]>`. The section's
//!   opener and closer are the template's wrapping and go into the fragments' sources, not the
//!   value; the text between them is the value as it is, streamed as it arrives for a declared
//!   string, read whole at the closing tag for any other type. Text between `]]>` and the closing
//!   tag, which the template never writes, is reported.
//! - **A value may stand on its own line** ([`Tags::value_on_its_own_line`]): dots3's template
//!   writes a newline after the parameter tag and one before its closing tag. Those two are the
//!   template's framing and go into the fragments' sources, not the value: the first byte of a
//!   value is dropped when it is a newline, and a trailing newline is held until the closing tag
//!   says it was the framing, or more text says it was the value's.
//! - **The newlines between the tags are the template's** and go into the next event's source;
//!   text where the syntax has tags is reported.
//! - **Two endings**, as DSML's: [`Assembler::close`] is the invoke's closing tag, or the block's
//!   end before it; [`Assembler::finish`] is a stream that was cut, and closes nothing.
//! - **Every byte of the call lands in exactly one event.**
//!
//! [`formats::minimax_m2`]: crate::formats::minimax_m2()
//! [`formats::minicpm5`]: crate::formats::minicpm5()
//! [`formats::dots3`]: crate::formats::dots3()

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::{json, Declared, Kind, NULL_WORDS},
    },
};

/// A CDATA section's opener and closer, where the family's template writes one around a value.
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";

const OPEN: usize = 0;
const CLOSE: usize = 1;
const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const EMPTY_NAME: &str = "a tag without a name";
const TAG_TAIL: &str = "text after a tag's name";
const CLOSED_EARLY: &str = "an invoke that closed before its function name closed";

/// How a family spells the syntax: the parameter tag's opening up to the name's quote and its
/// closing tag, the invoke's closing tag, and whether a value may be a CDATA section. The invoke's
/// opening is the table's terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tags {
    pub parameter_open: &'static str,
    pub parameter_close: &'static str,
    /// The invoke's closing tag: the one terminal the call's end carries. The next invoke's
    /// opening and a block's close end a call too, but they belong to the region, and the engine
    /// drops them.
    pub invoke_close: &'static str,
    /// Whether the template wraps a value as `<![CDATA[…]]>` when it holds `<`, `&` or a newline.
    pub cdata: bool,
    /// Whether the template writes a newline after the parameter tag and one before its closing
    /// tag, so a value stands on a line of its own.
    pub value_on_its_own_line: bool,
}

impl Tags {
    /// MiniMax M2's: `<parameter name="…">`, `</parameter>` and `</invoke>`, no CDATA.
    pub const MINIMAX_M2: Self = Self {
        parameter_open: "<parameter name=\"",
        parameter_close: "</parameter>",
        invoke_close: "</invoke>",
        cdata: false,
        value_on_its_own_line: false,
    };
    /// dots3's: MiniMax M2's tags, every value on a line of its own.
    pub const DOTS3: Self = Self {
        parameter_open: "<parameter name=\"",
        parameter_close: "</parameter>",
        invoke_close: "</invoke>",
        cdata: false,
        value_on_its_own_line: true,
    };
    /// MiniCPM5's: `<param name="…">`, `</param>` and `</function>`, a value a CDATA section when
    /// it holds `<`, `&` or a newline.
    pub const MINICPM5: Self = Self {
        parameter_open: "<param name=\"",
        parameter_close: "</param>",
        invoke_close: "</function>",
        cdata: true,
        value_on_its_own_line: false,
    };

    fn text(&self, tag: usize) -> &'static str {
        [self.parameter_open, self.parameter_close][tag]
    }
}

/// The events of one invoke, from the bytes after `<invoke name="`.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    tags: Tags,
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
    /// Inside the invoke tag: the name is `carried[..]` until `"`, then the tail until `>`.
    FunctionName,
    /// After the name's quote, before the tag's `>`: the tail is `carried[start..]`.
    FunctionTail { name: String, start: usize },
    /// After the invoke tag or a `</parameter>`.
    Between,
    /// Inside a parameter tag: the key is `carried[start..]` until `"`.
    ParameterName { start: usize },
    /// After the key's quote, before the tag's `>`: the tail, `carried[start..]`, which the
    /// syntax has empty.
    ParameterTail { key: String, start: usize },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Where the value's text starts in `carried`, while it is held.
    start: usize,
    /// Where it ends, once a CDATA section closed; `carried` holds the closer after it.
    end: Option<usize>,
    mode: Mode,
    cdata: Cdata,
    /// Inside a CDATA section, the trailing bytes that may be the start of its closer; for a value
    /// on its own line, a trailing newline that may be the template's framing: held until the next
    /// bytes or the closing tag say.
    held: String,
    /// For a value on its own line: whether its first byte, the template's newline, is still to
    /// come.
    leading_newline_due: bool,
}

/// Whether a value is a CDATA section, decided from its first bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cdata {
    /// The family writes no CDATA sections, or this value's first bytes are not an opener.
    Plain,
    /// Fewer bytes than the opener has, all of them the opener's so far.
    Undecided,
    /// Inside the section: the text up to `]]>` is the value's.
    Inside,
    /// The section closed: only the closing tag should follow.
    Done,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// A declared string, pushed piece by piece once its opening fragment is out.
    Streaming { opened: bool },
    /// Written at the close with [`json`].
    Whole,
    /// A string that may be null: held until its text rules the null words out.
    Undecided,
}

impl Assembler {
    /// An assembler for the invoke at `index` with the id the format minted for it, positioned
    /// just after the invoke tag's `name="`.
    pub fn new(index: u32, id: impl Into<String>, tags: Tags) -> Self {
        Self {
            index,
            id: id.into(),
            tags,
            scanner: Scanner::new([tags.parameter_open, tags.parameter_close]),
            carried: String::new(),
            stage: Stage::FunctionName,
            function: None,
            written: 0,
            done: false,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// The invoke's closing tag in the family's spelling: the one terminal a started call's end
    /// carries.
    pub fn invoke_close(&self) -> &'static str {
        self.tags.invoke_close
    }

    /// Append the next bytes of the invoke and push the events they complete. Every byte is the
    /// invoke's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        if self.done {
            return;
        }
        for piece in self.scanner.feed(bytes) {
            self.take(piece, declared, out);
        }
    }

    /// The invoke's end: `terminal` is the closing tag the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. After the last parameter the
    /// object is closed and the call ends; inside a value an open string is closed, the object is
    /// closed, and what was held comes back as `Malformed`; an invoke that never named a function
    /// has its bytes reported.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.scanner.held().to_string();
        self.carried.push_str(&held);
        if !self.started() {
            let why = if self.carried.is_empty() && terminal.is_empty() {
                None
            } else if matches!(self.stage, Stage::FunctionName) && self.carried.is_empty() {
                Some(EMPTY_NAME)
            } else {
                Some(CLOSED_EARLY)
            };
            self.carried.push_str(terminal);
            if let Some(why) = why {
                self.report(why, out);
            }
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between => {}
            Stage::Value(value) => self.cut_value(&value, out),
            // A name or a tag the end cut short: its bytes are reported, and the object closes.
            _ => self.report(TAG_CUT_SHORT, out),
        }
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.close_object(source, out);
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(terminal),
        });
        self.done = true;
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A streamed string's open fragment stays open; everything held comes back as
    /// `Malformed { UnterminatedRegion }`; a call that started still ends, with no bytes of its
    /// own, as every assembler ends one.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.scanner.held().to_string();
        self.carried.push_str(&held);
        let held = self.take_held();
        self.carried.push_str(&held);
        if !self.carried.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(std::mem::take(&mut self.carried)),
                why: MalformedReason::UnterminatedRegion,
            });
        }
        if self.started() {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
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
        let bytes = self.tags.text(tag);
        match &self.stage {
            Stage::Between => self.tag_between(tag, out),
            Stage::Value(_) if tag == CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tag is the value's text.
            Stage::Value(_) => self.text(bytes, declared, out),
            // A tag cuts a name or a tag's tail short: the bytes so far are reported, and the tag
            // is read where they began.
            _ => {
                self.report(TAG_CUT_SHORT, out);
                self.stage = Stage::Between;
                if self.started() {
                    self.tag_between(tag, out);
                } else {
                    out.push(Event::Malformed {
                        text: Text::uncounted(bytes),
                        why: MalformedReason::Other(TAG_OUT_OF_PLACE.to_string()),
                    });
                }
            }
        }
    }

    /// A tag between parameters: the parameter tag opens a key, once the function is named; the
    /// closing tag has no place, and neither does a parameter before the function's name closed.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = self.tags.text(tag);
        if tag == OPEN && self.started() {
            self.carried.push_str(bytes);
            self.stage = Stage::ParameterName {
                start: self.carried.len(),
            };
        } else {
            self.drop_carried(out);
            out.push(Event::Malformed {
                text: Text::uncounted(bytes),
                why: MalformedReason::Other(TAG_OUT_OF_PLACE.to_string()),
            });
        }
    }

    fn text(&mut self, text: &str, declared: &Declared, out: &mut Events) {
        match &self.stage {
            Stage::FunctionName => match text.split_once('"') {
                Some((head, rest)) => {
                    self.carried.push_str(head);
                    // The name's bytes stay carried: they are the start's source.
                    let name = self.carried.clone();
                    self.carried.push('"');
                    if name.is_empty() {
                        self.report(EMPTY_NAME, out);
                        self.stage = Stage::Between;
                    } else {
                        self.stage = Stage::FunctionTail {
                            name,
                            start: self.carried.len(),
                        };
                    }
                    self.text(rest, declared, out);
                }
                None => self.carried.push_str(text),
            },
            Stage::FunctionTail { name, start } => {
                let (name, start) = (name.clone(), *start);
                match text.split_once('>') {
                    Some((head, rest)) => {
                        self.carried.push_str(head);
                        let tail = self.carried[start..].to_string();
                        self.carried.push('>');
                        self.start_call(name, &tail, out);
                        self.text(rest, declared, out);
                    }
                    None => self.carried.push_str(text),
                }
            }
            Stage::Between => self.text_between(text, out),
            Stage::ParameterName { start } => {
                let start = *start;
                match text.split_once('"') {
                    Some((head, rest)) => {
                        self.carried.push_str(head);
                        let key = self.carried[start..].to_string();
                        self.carried.push('"');
                        if key.is_empty() {
                            self.report(EMPTY_NAME, out);
                            self.stage = Stage::Between;
                        } else {
                            self.stage = Stage::ParameterTail {
                                key,
                                start: self.carried.len(),
                            };
                        }
                        self.text(rest, declared, out);
                    }
                    None => self.carried.push_str(text),
                }
            }
            Stage::ParameterTail { key, start } => {
                let (key, start) = (key.clone(), *start);
                match text.split_once('>') {
                    Some((head, rest)) => {
                        self.carried.push_str(head);
                        let tail = self.carried[start..].to_string();
                        self.carried.push('>');
                        self.open_value(key, &tail, declared, out);
                        self.text(rest, declared, out);
                    }
                    None => self.carried.push_str(text),
                }
            }
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

    /// The invoke tag is whole: the call starts. A tail after the name's quote is reported first.
    fn start_call(&mut self, name: String, tail: &str, out: &mut Events) {
        self.function = Some(name.clone());
        if tail.is_empty() {
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(std::mem::take(&mut self.carried)),
            });
        } else {
            // The tag's bytes up to the tail stay as the start's source; the tail is reported.
            let tag_end = self.carried.len() - tail.len() - 1;
            let mut after = self.carried.split_off(tag_end);
            let close = after.pop();
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(std::mem::take(&mut self.carried)),
            });
            out.push(Event::Malformed {
                text: Text::uncounted(after),
                why: MalformedReason::Other(TAG_TAIL.to_string()),
            });
            self.carried.extend(close);
        }
        self.stage = Stage::Between;
    }

    /// The parameter tag is whole: the value begins, typed by what the tool declares for the key.
    /// A tag with a tail after the name's quote (`<parameter name="x" type="int">`) is not one the
    /// syntax has: the whole tag comes back as `Malformed`, and no parameter opens.
    fn open_value(&mut self, key: String, tail: &str, declared: &Declared, out: &mut Events) {
        if !tail.is_empty() {
            self.report(TAG_TAIL, out);
            self.stage = Stage::Between;
            return;
        }
        let function = self.function.as_deref().unwrap_or_default();
        let kind = declared.kind(function, &key);
        let mode = match kind {
            Some(Kind::String) => Mode::Streaming { opened: false },
            Some(Kind::NullableString) => Mode::Undecided,
            Some(Kind::Integer | Kind::Array | Kind::Object) | None => Mode::Whole,
        };
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            start: self.carried.len(),
            end: None,
            mode,
            cdata: if self.tags.cdata {
                Cdata::Undecided
            } else {
                Cdata::Plain
            },
            held: String::new(),
            leading_newline_due: self.tags.value_on_its_own_line,
        });
    }

    /// The next bytes of a value. Where the family writes CDATA sections, the first bytes decide
    /// whether this value is one: they are held until the opener's length has arrived or a byte
    /// rules the opener out, then read as they would have been.
    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        if text.is_empty() {
            return;
        }
        if value.leading_newline_due {
            // The newline after the parameter tag is the template's: into the source, not the
            // value. A value that starts otherwise has no framing to drop.
            value.leading_newline_due = false;
            if let Some(rest) = text.strip_prefix('\n') {
                self.carried.push('\n');
                value.start += 1;
                self.value_text(rest, out);
                return;
            }
        }
        match value.cdata {
            Cdata::Undecided => {
                self.carried.push_str(text);
                let so_far = &self.carried[value.start..];
                if so_far.len() < CDATA_OPEN.len() {
                    if CDATA_OPEN.starts_with(so_far) {
                        return;
                    }
                    value.cdata = Cdata::Plain;
                } else if so_far.starts_with(CDATA_OPEN) {
                    value.cdata = Cdata::Inside;
                    value.start += CDATA_OPEN.len();
                } else {
                    value.cdata = Cdata::Plain;
                }
                let arrived = self.carried.split_off(value.start);
                self.section_text(&arrived, out);
            }
            _ => self.section_text(text, out),
        }
    }

    /// The next bytes of a value whose CDATA question is settled: inside a section the text up to
    /// `]]>` is the value's and a trailing start of the closer is held; after the closer, text is
    /// out of place; a plain value's bytes are the value's.
    fn section_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        match value.cdata {
            Cdata::Inside => {
                let mut pending = std::mem::take(&mut value.held);
                pending.push_str(text);
                if let Some(at) = pending.find(CDATA_CLOSE) {
                    let inside = pending[..at].to_string();
                    let after = pending[at + CDATA_CLOSE.len()..].to_string();
                    self.value_bytes(&inside, out);
                    if let Stage::Value(value) = &mut self.stage {
                        value.cdata = Cdata::Done;
                        if matches!(value.mode, Mode::Whole | Mode::Undecided) {
                            value.end = Some(self.carried.len());
                        }
                    }
                    self.carried.push_str(CDATA_CLOSE);
                    self.section_text(&after, out);
                } else {
                    // A trailing `]]` or `]` may be the closer's start: held. Both are ASCII, so
                    // the split falls on a character boundary.
                    let keep = if pending.ends_with("]]") {
                        2
                    } else if pending.ends_with(']') {
                        1
                    } else {
                        0
                    };
                    let (give, hold) = pending.split_at(pending.len() - keep);
                    let give = give.to_string();
                    value.held = hold.to_string();
                    self.value_bytes(&give, out);
                }
            }
            Cdata::Done => self.text_between(text, out),
            Cdata::Plain | Cdata::Undecided if self.tags.value_on_its_own_line => {
                // A trailing newline may be the template's framing before the closing tag: held.
                let mut pending = std::mem::take(&mut value.held);
                pending.push_str(text);
                let (give, hold) = match pending.strip_suffix('\n') {
                    Some(give) => (give.to_string(), "\n".to_string()),
                    None => (pending, String::new()),
                };
                value.held = hold;
                self.value_bytes(&give, out);
            }
            Cdata::Plain | Cdata::Undecided => self.value_bytes(text, out),
        }
    }

    /// The value's held start of a CDATA closer, if any, taken: value text after all.
    fn take_held(&mut self) -> String {
        match &mut self.stage {
            Stage::Value(value) => std::mem::take(&mut value.held),
            _ => String::new(),
        }
    }

    /// The value's own bytes, by its mode.
    fn value_bytes(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        if text.is_empty() {
            return;
        }
        match value.mode {
            Mode::Whole => self.carried.push_str(text),
            Mode::Undecided => {
                self.carried.push_str(text);
                let so_far = &self.carried[value.start..];
                if !NULL_WORDS.iter().any(|word| word.starts_with(so_far)) {
                    // Not null after all: open the string and stream what arrived.
                    let arrived = self.carried.split_off(value.start);
                    value.mode = Mode::Streaming { opened: false };
                    self.stream(&arrived, out);
                }
            }
            Mode::Streaming { .. } => self.stream(text, out),
        }
    }

    /// A piece of a streamed string, after its opening fragment.
    fn stream(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(ValueState {
            key,
            mode: Mode::Streaming { opened },
            ..
        }) = &mut self.stage
        else {
            return;
        };
        let key = key.clone();
        if !*opened {
            *opened = true;
            let mut opening = String::with_capacity(key.len() + 8);
            opening.push_str(self.separator());
            push_quoted(&mut opening, &key);
            opening.push_str(": \"");
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.push_fragment(opening, source, out);
            self.written += 1;
        }
        if text.is_empty() {
            return;
        }
        let mut json = String::with_capacity(text.len());
        push_escaped(&mut json, text);
        self.push_fragment(json, Text::uncounted(text), out);
    }

    /// `</parameter>`: a streamed string gets its closing quote, any other value is written whole.
    /// Bytes held as a CDATA closer's start that never completed are the value's text.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let held = self.take_held();
        if self.tags.value_on_its_own_line && held == "\n" {
            // The newline before the closing tag: the template's framing.
            if let Stage::Value(value) = &mut self.stage {
                if matches!(value.mode, Mode::Whole | Mode::Undecided) && value.end.is_none() {
                    value.end = Some(self.carried.len());
                }
            }
            self.carried.push('\n');
        } else {
            self.value_bytes(&held, out);
        }
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = match value.mode {
            Mode::Streaming { opened: true } => "\"".to_string(),
            Mode::Streaming { opened: false } => {
                // An empty string: its opening and closing quotes come together.
                let mut member = String::with_capacity(value.key.len() + 8);
                member.push_str(self.separator());
                push_quoted(&mut member, &value.key);
                member.push_str(": \"\"");
                self.written += 1;
                member
            }
            Mode::Whole | Mode::Undecided => {
                let end = value.end.unwrap_or(self.carried.len());
                let written = json(&self.carried[value.start..end], value.kind);
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

    /// The call ended inside a value: an open string is closed, and a value never written comes
    /// back as `Malformed`, since its member was never written.
    fn cut_value(&mut self, value: &ValueState, out: &mut Events) {
        self.carried.push_str(&value.held);
        self.report(TAG_CUT_SHORT, out);
        if matches!(value.mode, Mode::Streaming { opened: true }) {
            self.push_fragment("\"".to_string(), Text::default(), out);
        }
    }

    /// Reports the carried bytes as `Malformed` with `why`, whitespace included.
    fn report(&mut self, why: &str, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Malformed {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: MalformedReason::Other(why.to_string()),
        });
    }

    /// Whitespace carried before text that is reported: the template's, dropped so the report
    /// holds only the text.
    fn drop_carried(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Dropped {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: DropReason::Wrapper,
        });
    }

    fn separator(&self) -> &'static str {
        if self.written == 0 {
            "{"
        } else {
            ", "
        }
    }

    /// `}` or `{}`, with `source` as its bytes.
    fn close_object(&mut self, source: Text, out: &mut Events) {
        let json = if self.written == 0 { "{}" } else { "}" }.to_string();
        self.push_fragment(json, source, out);
    }

    fn push_fragment(&self, json: String, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;

    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "s": {"type": "string"},
                    "n": {"type": "integer"},
                    "ns": {"type": ["string", "null"]},
                    "o": {"type": "object"},
                    "b": {"type": "boolean"},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The events of one invoke fed in `pieces`, closed by its tag or cut.
    fn run(pieces: &[&str], cut: bool) -> Vec<Event> {
        run_with(Tags::MINIMAX_M2, pieces, cut)
    }

    fn run_with(tags: Tags, pieces: &[&str], cut: bool) -> Vec<Event> {
        let mut assembler = Assembler::new(0, "call_0", tags);
        let mut out = Events::new();
        let declared = declared();
        for piece in pieces {
            assembler.feed(piece, &declared, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(tags.invoke_close, &mut out);
        }
        out.drain()
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => t.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<(String, String)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed {
                    text,
                    why: MalformedReason::Other(why),
                } => Some((text.text.clone(), why.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_rendered_invoke_gives_its_call_typed_by_the_tool_and_every_byte() {
        let text = concat!(
            "f\">\n<parameter name=\"s\">Paris</parameter>\n<parameter name=\"n\">3</parameter>\n",
            "<parameter name=\"b\">true</parameter>\n",
            "<parameter name=\"o\">{\"a\": [1, 2]}</parameter>\n"
        );
        let events = run(&[text], false);
        assert_eq!(
            bytes(&events),
            format!("{text}{}", Tags::MINIMAX_M2.invoke_close)
        );
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { index: 0, name, source, .. }
                if name == "f" && source.text == "f\">"
        ));
        assert_eq!(
            arguments(&events),
            r#"{"s": "Paris", "n": 3, "b": true, "o": {"a": [1, 2]}}"#
        );
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text == Tags::MINIMAX_M2.invoke_close
        ));
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_declared_string_streams_as_it_arrives_and_a_cut_leaves_it_open() {
        let pieces = ["f\">\n<parameter name=\"s\">Par", "is, Fr", "ance"];
        let events = run(&pieces, true);
        assert_eq!(bytes(&events), pieces.concat());
        // The opening fragment, then every piece of the string as it came; no closing quote.
        assert_eq!(arguments(&events), r#"{"s": "Paris, France"#);
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text.is_empty()
        ));
    }

    #[test]
    fn a_nullable_string_is_null_or_the_string() {
        let as_null = run(&["f\">\n<parameter name=\"ns\">null</parameter>\n"], false);
        assert_eq!(arguments(&as_null), r#"{"ns": null}"#);
        let as_text = run(
            &[
                "f\">\n<parameter name=\"ns\">nu",
                "ll and void</parameter>\n",
            ],
            false,
        );
        assert_eq!(arguments(&as_text), r#"{"ns": "null and void"}"#);
    }

    #[test]
    fn an_empty_string_and_an_invoke_without_parameters_write_their_objects() {
        let empty = run(&["f\">\n<parameter name=\"s\"></parameter>\n"], false);
        assert_eq!(arguments(&empty), r#"{"s": ""}"#);
        let none = run(&["f\">\n"], false);
        assert_eq!(arguments(&none), "{}");
        assert!(matches!(&none[0], Event::ToolCallStart { name, .. } if name == "f"));
    }

    #[test]
    fn a_parameter_tag_with_a_tail_is_no_parameter_and_the_text_is_reported() {
        let text = "f\">\n<parameter name=\"n\" type=\"int\">1</parameter>\n";
        let events = run(&[text], false);
        assert_eq!(
            bytes(&events),
            format!("{text}{}", Tags::MINIMAX_M2.invoke_close)
        );
        let reported = malformed(&events);
        assert_eq!(reported[0].1, TAG_TAIL);
        // The template's newline before the tag rides in the report, as every report carries
        // what was held before it.
        assert_eq!(reported[0].0, "\n<parameter name=\"n\" type=\"int\">");
        // `1` is text where the syntax has tags, and the stray close has no place.
        assert_eq!(
            reported[1],
            ("1".to_string(), TEXT_BETWEEN_TAGS.to_string())
        );
        assert_eq!(reported[2].1, TAG_OUT_OF_PLACE);
        assert_eq!(arguments(&events), "{}");
    }

    #[test]
    fn an_invoke_that_closes_inside_its_name_is_reported_and_starts_no_call() {
        let events = run(&["get_wea"], false);
        assert_eq!(
            bytes(&events),
            format!("get_wea{}", Tags::MINIMAX_M2.invoke_close)
        );
        assert_eq!(malformed(&events)[0].1, CLOSED_EARLY);
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { .. })));
    }

    #[test]
    fn a_tag_inside_a_value_is_the_values_text_and_a_close_inside_a_value_closes_it() {
        let events = run(
            &["f\">\n<parameter name=\"s\">a <parameter name=\"x\"> b</parameter>\n"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"s": "a <parameter name=\"x\"> b"}"#);
        let cut = run(&["f\">\n<parameter name=\"n\">12"], false);
        // The close cuts a whole value: the tag and the digits, held for the value's close that
        // never came, are reported together, and the object closes empty.
        assert_eq!(arguments(&cut), "{}");
        assert_eq!(
            malformed(&cut)[0],
            (
                "\n<parameter name=\"n\">12".to_string(),
                TAG_CUT_SHORT.to_string()
            )
        );
    }

    /// MiniCPM5's spelling: `<param name="…">`, `</param>`, `</function>`, CDATA sections.
    const MINICPM5_CALL: &str = concat!(
        "f\">",
        "<param name=\"s\"><![CDATA[café <b> & \"q\"\nligne]]></param>",
        "<param name=\"o\"><![CDATA[{\"k\": \"<v>\"}]]></param>",
        "<param name=\"n\">3</param>"
    );

    #[test]
    fn a_cdata_section_is_the_value_inside_it_for_a_string_and_for_a_whole_value() {
        let events = run_with(Tags::MINICPM5, &[MINICPM5_CALL], false);
        assert_eq!(bytes(&events), format!("{MINICPM5_CALL}</function>"));
        assert_eq!(
            arguments(&events),
            r#"{"s": "café <b> & \"q\"\nligne", "o": {"k": "<v>"}, "n": 3}"#
        );
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn a_cdata_section_reads_the_same_however_the_bytes_are_cut() {
        let whole = run_with(Tags::MINICPM5, &[MINICPM5_CALL], false);
        let expected = arguments(&whole);
        // Every split at a character boundary, one byte of UTF-8 at a time where the text allows,
        // the closer's `]]` and `>` apart, the opener in halves.
        let boundaries: Vec<usize> = MINICPM5_CALL.char_indices().map(|(at, _)| at).collect();
        for &at in &boundaries[1..] {
            let events = run_with(
                Tags::MINICPM5,
                &[&MINICPM5_CALL[..at], &MINICPM5_CALL[at..]],
                false,
            );
            assert_eq!(
                bytes(&events),
                format!("{MINICPM5_CALL}</function>"),
                "cut at {at}"
            );
            assert_eq!(arguments(&events), expected, "cut at {at}");
            assert!(malformed(&events).is_empty(), "cut at {at}");
        }
        let one_by_one: Vec<&str> = MINICPM5_CALL
            .char_indices()
            .map(|(at, c)| &MINICPM5_CALL[at..at + c.len_utf8()])
            .collect();
        let events = run_with(Tags::MINICPM5, &one_by_one, false);
        assert_eq!(arguments(&events), expected);
    }

    #[test]
    fn a_bracket_inside_a_cdata_value_and_a_short_plain_value_are_read_as_they_are() {
        // `]` and `]]` that do not start the closer are the value's text.
        let call = "f\">\n<param name=\"s\"><![CDATA[a[1]] and b]]></param>";
        let events = run_with(Tags::MINICPM5, &[call], false);
        assert_eq!(arguments(&events), r#"{"s": "a[1]] and b"}"#);
        // A value shorter than the opener, and one that starts like it but is not one.
        let short = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"s\">Rome</param>"],
            false,
        );
        assert_eq!(arguments(&short), r#"{"s": "Rome"}"#);
        let alike = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"s\"><![CDATB[x]]></param>"],
            false,
        );
        assert_eq!(arguments(&alike), r#"{"s": "<![CDATB[x]]>"}"#);
    }

    #[test]
    fn text_after_a_cdata_closer_is_reported_and_a_cut_section_keeps_its_held_bytes() {
        let events = run_with(
            Tags::MINICPM5,
            &["f\"><param name=\"s\"><![CDATA[x]]>tail</param>"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"s": "x"}"#);
        assert_eq!(
            malformed(&events),
            vec![("tail".to_string(), TEXT_BETWEEN_TAGS.to_string())]
        );
        // Cut with a `]` held: the byte is value text, not lost.
        let cut = run_with(Tags::MINICPM5, &["f\"><param name=\"s\"><![CDATA[x]"], true);
        assert_eq!(bytes(&cut), "f\"><param name=\"s\"><![CDATA[x]");
        assert_eq!(arguments(&cut), r#"{"s": "x"#);
    }

    /// dots3's spelling: MiniMax M2's tags, every value on a line of its own.
    const DOTS3_CALL: &str = concat!(
        "f\">\n<parameter name=\"s\">\ncafé\nà deux\n</parameter>\n",
        "<parameter name=\"n\">\n3\n</parameter>\n",
        "<parameter name=\"o\">\n{\"k\": [1]}\n</parameter>\n"
    );

    #[test]
    fn a_value_on_its_own_line_sheds_its_framing_newlines_however_the_bytes_are_cut() {
        let whole = run_with(Tags::DOTS3, &[DOTS3_CALL], false);
        assert_eq!(bytes(&whole), format!("{DOTS3_CALL}</invoke>"));
        assert_eq!(
            arguments(&whole),
            r#"{"s": "café\nà deux", "n": 3, "o": {"k": [1]}}"#
        );
        assert!(malformed(&whole).is_empty());
        let boundaries: Vec<usize> = DOTS3_CALL.char_indices().map(|(at, _)| at).collect();
        for &at in &boundaries[1..] {
            let events = run_with(Tags::DOTS3, &[&DOTS3_CALL[..at], &DOTS3_CALL[at..]], false);
            assert_eq!(
                bytes(&events),
                format!("{DOTS3_CALL}</invoke>"),
                "cut at {at}"
            );
            assert_eq!(arguments(&events), arguments(&whole), "cut at {at}");
            assert!(malformed(&events).is_empty(), "cut at {at}");
        }
        // Without the framing the value is read as it is; an empty value is an empty string.
        let bare = run_with(
            Tags::DOTS3,
            &["f\">\n<parameter name=\"s\">Rome</parameter>\n"],
            false,
        );
        assert_eq!(arguments(&bare), r#"{"s": "Rome"}"#);
        let empty = run_with(
            Tags::DOTS3,
            &["f\">\n<parameter name=\"s\">\n\n</parameter>\n"],
            false,
        );
        assert_eq!(arguments(&empty), r#"{"s": ""}"#);
    }
}
