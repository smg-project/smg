//! DeepSeek's DSML: the events of one `<｜DSML｜ invoke name="NAME">` block, from its bytes.
//!
//! DeepSeek V4 and V4.1 write a call as
//!
//! ```text
//! <｜DSML｜ invoke name="get_weather">
//! <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
//! <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! ```
//!
//! inside a `<｜DSML｜ calls>` block that holds one invoke or several. The engine's table owns the
//! `calls` and `invoke` markers ([`formats::deepseek_v4_1`]): it enters this assembler at
//! `<｜DSML｜ invoke name="` and leaves it at `</｜DSML｜ invoke>`, so the assembler starts inside
//! the function's name and never sees the invoke's closing tag; the engine hands it that tag's
//! bytes as the call's end. Between those two the assembler reads the parameter tags itself.
//!
//! What the syntax says, and what the assembler does with it:
//!
//! - **The `string` attribute types the value.** `string="true"` is the text, every byte of it,
//!   streamed as it arrives like a declared string in [`assembler`](super::assembler);
//!   `string="false"` is JSON the model wrote, written whole at the value's close through
//!   [`json`]'s inference (`null`, `true`, `[1, 2]`, `{"a": 1}` as written; text that is not JSON
//!   is a string holding it, so one odd value never costs a call its other arguments). The
//!   request's tools are not consulted: the model says the type. A parameter tag with another
//!   attribute, or none, is no parameter tag: it comes back whole as `Malformed`, and what
//!   follows it is text between tags until the next tag the syntax has.
//! - **No template newline around a value.** The template writes the value directly between the
//!   tags, so nothing is taken away from it; the newline between tags is the template's and goes
//!   into the next event's source, as in the Qwen assembler.
//! - **A name ends at its quote.** The function name runs to `"`, the key to `"`; the rest of the
//!   tag runs to `>`.
//! - **Inside a value only `</｜DSML｜ parameter>` is a tag for the assembler**; a parameter tag
//!   there is the value's text, as for Qwen. The invoke tags and the block's close are the
//!   engine's terminals, so they end the call wherever they stand, a value included, as
//!   `</tool_call>` does for Qwen (the table's `invoke + invoke_open` and `invoke + calls_close`
//!   rows).
//! - **Two endings.** [`Assembler::close`] is the invoke's closing tag ([`INVOKE_CLOSE`], the one
//!   terminal the call's end carries), or the next invoke's opening or the block's close, which
//!   end the call with no bytes of their own: after the last parameter it closes the object and
//!   pushes `ToolCallEnd`; inside a value it reports what was held, closes an open string and the
//!   object, as the Qwen assembler does. [`Assembler::finish`] is a stream that was cut: nothing
//!   is closed.
//! - **Every byte of the invoke lands in exactly one event**, with the same accounting as the Qwen
//!   assembler: the name's bytes in `ToolCallStart`, each tag's and value's bytes in the
//!   fragments they produce, text where the syntax has tags as `Malformed` run by run with the
//!   whitespace before it as `Dropped { Wrapper }`, as the Qwen assembler reports it. A client
//!   that shows malformed text sees such words without the spaces between them (an invoke whose
//!   name never closed shows as `name="a"string="true"`); the bytes are all accounted for, in the
//!   dropped and malformed events.
//!
//! This is the second tagged assembler beside the Qwen one; the two share the value module and the
//! escaping, and will share one core when a third dialect (GLM, MiniMax, Hy4) arrives and shows
//! where the seams are.
//!
//! [`formats::deepseek_v4_1`]: crate::formats::deepseek_v4_1()

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::json,
    },
};

/// The invoke's closing tag: the one terminal the call's end carries. The next invoke's opening
/// and the block's close end a call too, but they belong to the region, and the engine drops them.
pub const INVOKE_CLOSE: &str = "</｜DSML｜ invoke>";

const OPEN: usize = 0;
const CLOSE: usize = 1;
/// The parameter tag's opening, up to the name's quote, and its closing tag: the two markers the
/// assembler reads inside an invoke, and the ones a grammar for the syntax spells.
pub const PARAMETER_OPEN: &str = "<｜DSML｜ parameter name=\"";
pub const PARAMETER_CLOSE: &str = "</｜DSML｜ parameter>";
const TAGS: [&str; 2] = [PARAMETER_OPEN, PARAMETER_CLOSE];
const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const EMPTY_NAME: &str = "a tag without a name";
const TAG_TAIL: &str = "text after a tag's name";
const CLOSED_EARLY: &str = "an invoke that closed before its function name closed";
const STRING_TRUE: &str = " string=\"true\"";
const STRING_FALSE: &str = " string=\"false\"";

/// The events of one DSML invoke, from the bytes after `<｜DSML｜ invoke name="`.
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
    /// Inside the invoke tag: the name is `carried[..]` until `"`, then the tail until `>`.
    FunctionName,
    /// After the name's quote, before the tag's `>`: the tail is `carried[start..]`.
    FunctionTail { name: String, start: usize },
    /// After the invoke tag or a `</｜DSML｜ parameter>`.
    Between,
    /// Inside a parameter tag: the key is `carried[start..]` until `"`.
    ParameterName { start: usize },
    /// After the key's quote, before the tag's `>`: the attribute, `carried[start..]`.
    ParameterTail { key: String, start: usize },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    /// `string="true"`: streamed; `string="false"`: written whole at the close.
    string: bool,
    /// For a whole value, where its text starts in `carried`.
    start: usize,
    /// For a streamed value, whether its opening fragment has been pushed.
    opened: bool,
}

impl Assembler {
    /// An assembler for the invoke at `index` with the id the format minted for it, positioned
    /// just after `<｜DSML｜ invoke name="`.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            scanner: Scanner::new(TAGS),
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

    /// Append the next bytes of the invoke and push the events they complete. Every byte is the
    /// invoke's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        if self.done {
            return;
        }
        for piece in self.scanner.feed(bytes) {
            self.take(piece, out);
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
                self.leftover(why, out);
            }
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between => {}
            Stage::Value(value) => self.cut_value(&value, out),
            // A name or a tag the end cut short: its bytes are reported, and the object closes.
            _ => self.leftover(TAG_CUT_SHORT, out),
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
    /// client. A streamed string's open fragment stays open; a whole value never written, a name
    /// and the held bytes come back as `Malformed { UnterminatedRegion }`; a call that started
    /// still ends, with no bytes of its own, as every assembler ends one.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.scanner.held().to_string();
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

    fn take(&mut self, piece: Piece, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, out),
            Piece::Marker(tag) => self.tag(tag, out),
        }
    }

    fn tag(&mut self, tag: usize, out: &mut Events) {
        let bytes = TAGS[tag];
        match &self.stage {
            Stage::Between => self.tag_between(tag, out),
            Stage::Value(_) if tag == CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tag is the value's text.
            Stage::Value(_) => self.text(bytes, out),
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
    /// closing tag has no place, and neither does a parameter before the function's name closed
    /// (a missing quote on the name), so no fragment comes before the call's start.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = TAGS[tag];
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

    fn text(&mut self, text: &str, out: &mut Events) {
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
                    self.text(rest, out);
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
                        self.text(rest, out);
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
                        self.text(rest, out);
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
                        self.open_value(key, &tail, out);
                        self.text(rest, out);
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

    /// The parameter tag is whole: the value begins, typed by its `string` attribute. A tag with
    /// any other tail (`<｜DSML｜ parameter name="x" kind="y">`) is not a parameter tag the
    /// syntax has: the whole tag comes back as `Malformed`, and no parameter opens, so a tag the
    /// syntax does not spell is never read as one.
    fn open_value(&mut self, key: String, tail: &str, out: &mut Events) {
        let string = match tail {
            STRING_TRUE => true,
            STRING_FALSE => false,
            _ => {
                self.report(TAG_TAIL, out);
                self.stage = Stage::Between;
                return;
            }
        };
        self.stage = Stage::Value(ValueState {
            key,
            string,
            start: self.carried.len(),
            opened: false,
        });
    }

    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        if text.is_empty() {
            return;
        }
        if !value.string {
            self.carried.push_str(text);
            return;
        }
        let key = value.key.clone();
        if !value.opened {
            value.opened = true;
            let mut opening = String::with_capacity(key.len() + 8);
            opening.push_str(self.separator());
            push_quoted(&mut opening, &key);
            opening.push_str(": \"");
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.push_fragment(opening, source, out);
            self.written += 1;
        }
        let mut json = String::with_capacity(text.len());
        push_escaped(&mut json, text);
        self.push_fragment(json, Text::uncounted(text), out);
    }

    /// `</｜DSML｜ parameter>`: a streamed string gets its closing quote, a whole value is written.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = if value.string {
            if value.opened {
                "\"".to_string()
            } else {
                // An empty string: its opening and closing quotes come together.
                let mut member = String::with_capacity(value.key.len() + 8);
                member.push_str(self.separator());
                push_quoted(&mut member, &value.key);
                member.push_str(": \"\"");
                self.written += 1;
                member
            }
        } else {
            let written = json(&self.carried[value.start..], None);
            let mut member = String::with_capacity(value.key.len() + written.len() + 8);
            member.push_str(self.separator());
            push_quoted(&mut member, &value.key);
            member.push_str(": ");
            member.push_str(&written);
            self.written += 1;
            member
        };
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(tag);
        self.push_fragment(fragment, Text::uncounted(source), out);
        self.stage = Stage::Between;
    }

    /// The invoke ended inside a value: an open string is closed, and a whole value's text comes
    /// back as `Malformed`, since its member was never written.
    fn cut_value(&mut self, value: &ValueState, out: &mut Events) {
        if value.string && value.opened {
            // Bytes the end cut short (the beginning of a tag) are reported, not hidden in the
            // quote's source.
            self.report(TAG_CUT_SHORT, out);
            self.push_fragment("\"".to_string(), Text::default(), out);
        } else {
            self.leftover(TAG_CUT_SHORT, out);
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

    fn leftover(&mut self, why: &str, out: &mut Events) {
        self.report(why, out);
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
