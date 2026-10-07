//! Keyed arguments: the events of one call written as its name and then `<arg_key>` and
//! `<arg_value>` pairs, from its bytes.
//!
//! GLM, Hy4, Ling and IQuest write a call as
//!
//! ```text
//! <tool_call>get_weather
//! <arg_key>city</arg_key>
//! <arg_value>Paris</arg_value><arg_key>days</arg_key>
//! <arg_value>3</arg_value>
//! </tool_call>
//! ```
//!
//! each with its own spellings of the four tags ([`Tags`]) and its own newlines between them (Ling
//! writes one after the name and one after `</arg_key>`, Hy4 and IQuest none). The engine's table
//! owns the call markers: it enters this assembler after the opening marker and leaves it at the
//! closing one, which it hands over as the call's end. Between the two the assembler reads the
//! name and the tags itself.
//!
//! What the syntax says, and what the assembler does with it:
//!
//! - **The name is the text before the first `<arg_key>`**, or before the call's end when the call
//!   has no arguments, less the whitespace around it, which is the template's and goes into the
//!   next event's source. The name is known only when that tag arrives, so `ToolCallStart` waits
//!   for it; names are short.
//! - **A value's type comes from the request's tools**, as in the Qwen assembler: a declared string
//!   is its text, streamed as it arrives; every other value is written whole at `</arg_value>`
//!   through [`json`], which reads the JSON the templates write for a number, a boolean, a list,
//!   an object or null, and infers the rest. Nothing is taken from a value: the templates write it
//!   directly between its tags.
//! - **Inside a value only `</arg_value>` is a tag**; the other three there are the value's text.
//! - **Two endings.** [`Assembler::close`] is the call's closing marker, or the block's end before
//!   it: after the last value it names the call if it has not, closes the object and pushes
//!   `ToolCallEnd` with the marker's bytes; inside a value it closes an open string and the object
//!   and reports what was held. [`Assembler::finish`] is a stream that was cut: nothing is closed.
//! - **Every byte of the call lands in exactly one event**, with the Qwen assembler's accounting:
//!   the name's bytes in `ToolCallStart`, each tag's and value's bytes in the fragments they
//!   produce, text where the syntax has tags as `Malformed` run by run with the whitespace before
//!   it as `Dropped { Wrapper }`.
//!
//! This is the third tagged assembler, beside the Qwen one and DSML's; the three share the value
//! module and the escaping, and the seams for one core are now visible: the tag spellings, how a
//! name ends, where the template's newlines are, and what types a value.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::{json, Declared, Kind, NULL_WORDS},
    },
};

const KEY_OPEN: usize = 0;
const KEY_CLOSE: usize = 1;
const VALUE_OPEN: usize = 2;
const VALUE_CLOSE: usize = 3;
const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const EMPTY_KEY: &str = "a key with no text";
const WITHOUT_A_NAME: &str = "a call block without a name";

/// The four tags a family spells its keyed arguments with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tags {
    pub key_open: &'static str,
    pub key_close: &'static str,
    pub value_open: &'static str,
    pub value_close: &'static str,
}

impl Tags {
    /// GLM's, Ling's and most families' spelling.
    pub const PLAIN: Self = Self {
        key_open: "<arg_key>",
        key_close: "</arg_key>",
        value_open: "<arg_value>",
        value_close: "</arg_value>",
    };

    /// Hy4's spelling, every tag suffixed `:opensource`.
    pub const HY4: Self = Self {
        key_open: "<arg_key:opensource>",
        key_close: "</arg_key:opensource>",
        value_open: "<arg_value:opensource>",
        value_close: "</arg_value:opensource>",
    };

    fn text(&self, tag: usize) -> &'static str {
        [
            self.key_open,
            self.key_close,
            self.value_open,
            self.value_close,
        ][tag]
    }
}

/// The events of one keyed call, from the bytes after the call's opening marker.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    tags: Tags,
    scanner: Scanner,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// The function's name once `ToolCallStart` was pushed.
    function: Option<String>,
    /// Members written to the arguments object so far.
    written: u32,
    done: bool,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Before the first tag: the name is the carried text, less the whitespace around it.
    Name,
    /// After a value, before the next key.
    Between,
    /// Inside `<arg_key>…</arg_key>`: the key is `carried[start..]`.
    Key { start: usize },
    /// After `</arg_key>`, before `<arg_value>`.
    Keyed { key: String },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Where the value's text starts in `carried`, while it is held.
    start: usize,
    mode: Mode,
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
    /// An assembler for the call at `index` with the id the format minted for it, positioned
    /// just after the call's opening marker.
    pub fn new(index: u32, id: impl Into<String>, tags: Tags) -> Self {
        Self {
            index,
            id: id.into(),
            scanner: Scanner::new([
                tags.key_open,
                tags.key_close,
                tags.value_open,
                tags.value_close,
            ]),
            tags,
            carried: String::new(),
            stage: Stage::Name,
            function: None,
            written: 0,
            done: false,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// Append the next bytes of the call and push the events they complete. Every byte is the
    /// call's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        if self.done {
            return;
        }
        for piece in self.scanner.feed(bytes) {
            self.take(piece, declared, out);
        }
    }

    /// The call's end: `terminal` is the closing marker the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. A call that has not been
    /// named yet is named from the text so far; after the last value the object is closed; inside
    /// a value an open string is closed, the object is closed, and what was held comes back as
    /// `Malformed`; a block with no name at all has its bytes reported.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.scanner.held().to_string();
        self.carried.push_str(&held);
        if matches!(self.stage, Stage::Name) {
            self.name_call(out);
        }
        if !self.started() {
            self.carried.push_str(terminal);
            self.leftover(WITHOUT_A_NAME, out);
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between | Stage::Name => {}
            Stage::Value(value) => self.cut_value(&value, out),
            // A key or a tag the end cut short: its bytes are reported, and the object closes.
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
    /// client. A streamed string's open fragment stays open; everything held comes back as
    /// `Malformed { UnterminatedRegion }`.
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
            Stage::Name => {
                // The first tag ends the name; the call starts here, and the tag is read as
                // between keys.
                self.name_call(out);
                if self.started() {
                    self.tag_between(tag, out);
                } else {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(bytes),
                        why: MalformedReason::Other(TAG_OUT_OF_PLACE.to_string()),
                    });
                }
            }
            Stage::Between => self.tag_between(tag, out),
            Stage::Key { start } if tag == KEY_CLOSE => {
                let key = self.carried[*start..].to_string();
                self.carried.push_str(bytes);
                if key.is_empty() {
                    self.report(EMPTY_KEY, out);
                    self.stage = Stage::Between;
                } else {
                    self.stage = Stage::Keyed { key };
                }
            }
            Stage::Keyed { key } if tag == VALUE_OPEN => {
                let key = key.clone();
                self.carried.push_str(bytes);
                self.open_value(key, declared, out);
            }
            Stage::Value(_) if tag == VALUE_CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tags are the value's text.
            Stage::Value(_) => self.text(bytes, declared, out),
            // A tag cuts a key short, or stands where the syntax has another: the bytes so far
            // are reported, and the tag is read as between keys.
            _ => {
                self.report(TAG_CUT_SHORT, out);
                self.stage = Stage::Between;
                self.tag_between(tag, out);
            }
        }
    }

    /// A tag between keys: `<arg_key>` opens one; the other three have no place here.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = self.tags.text(tag);
        if tag == KEY_OPEN {
            self.carried.push_str(bytes);
            self.stage = Stage::Key {
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
        let _ = declared;
        match &self.stage {
            Stage::Name | Stage::Key { .. } => self.carried.push_str(text),
            Stage::Between | Stage::Keyed { .. } => self.text_between(text, out),
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

    /// The text so far is the call's name, less the whitespace around it: the call starts, and
    /// the whitespace after the name is carried into the next source. A name with no text is
    /// reported, and no call starts.
    fn name_call(&mut self, out: &mut Events) {
        let name = self.carried.trim().to_string();
        if name.is_empty() {
            self.stage = Stage::Between;
            return;
        }
        let end = self.carried.trim_end().len();
        let after = self.carried.split_off(end);
        out.push(Event::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: name.clone(),
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
        self.carried = after;
        self.function = Some(name);
        self.stage = Stage::Between;
    }

    /// `<arg_value>` after a key: the value begins, typed by what the tool declares for it.
    fn open_value(&mut self, key: String, declared: &Declared, out: &mut Events) {
        let _ = out;
        let function = self.function.as_deref().unwrap_or_default();
        let kind = declared.kind(function, &key);
        let mode = match kind {
            Some(Kind::String) => Mode::Streaming { opened: false },
            Some(Kind::NullableString) => Mode::Undecided,
            Some(Kind::Integer) | None => Mode::Whole,
        };
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            start: self.carried.len(),
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

    /// `</arg_value>`: a streamed string gets its closing quote, any other value is written whole.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
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
                let written = json(&self.carried[value.start..], value.kind);
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
        if matches!(value.mode, Mode::Streaming { opened: true }) {
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.push_fragment("\"".to_string(), source, out);
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
