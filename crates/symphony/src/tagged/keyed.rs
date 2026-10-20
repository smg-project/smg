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
//! each with its own spellings of the four tags ([`Tags`]) and its own newlines around them (Ling
//! writes one after the name, after each `</arg_key>` and before the close, GLM 4.5 and 4.6 one
//! after the name, after each `</arg_key>` and after each `</arg_value>`, K2-Horizon the same under
//! `ifm|`-prefixed tags with the block's own around its calls, Hy4 and IQuest none;
//! [`Newlines`]). The engine's table
//! owns the call markers: it enters this assembler after the opening marker and leaves it at the
//! closing one, which it hands over as the call's end. Between the two the assembler reads the
//! name and the tags itself.
//!
//! What the assembler cannot read as the call is never content: the block's markers and tags are
//! the format's own tokens, and the text between them where the syntax has no place for text is
//! the model's failed call, which the families' own parsers drop whole. Such bytes are refused
//! ([`refuse`](super::refuse)): a `Malformed` event of no bytes with the reason counts the failure
//! where it stands, the bytes leave as `Dropped { MalformedCall }`, and a call that started keeps
//! the pairs it read.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use super::object::{Mode, Object};
use crate::{
    event::{runs, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        drop_refused, end_cut_call, refuse,
        value::{json, Kind},
    },
    Declared,
};

const KEY_OPEN: usize = 0;
const KEY_CLOSE: usize = 1;
const VALUE_OPEN: usize = 2;
const VALUE_CLOSE: usize = 3;

/// Whether `text` can be a function's name: no whitespace, quote, brace, bracket, parenthesis or
/// angle bracket, the delimiters of prose, a string, JSON, a list, a pythonic call and a tag.
fn is_name(text: &str) -> bool {
    !text.chars().any(|c| {
        c.is_whitespace()
            || matches!(
                c,
                '"' | '\'' | '{' | '}' | '[' | ']' | '(' | ')' | '<' | '>'
            )
    })
}

/// The four tags a family spells its keyed arguments with, and where its template writes a newline
/// around them ([`Newlines`]). The assembler reads a call whatever the newlines, as the template's
/// whitespace between tags; the grammar derived from a table spells the ones its template writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tags {
    pub key_open: &'static str,
    pub key_close: &'static str,
    pub value_open: &'static str,
    pub value_close: &'static str,
    pub newlines: Newlines,
}

/// Where a template writes a newline around a call's keyed arguments: each field is the text it
/// writes at that place, a newline or nothing. What the recorded calls show, place by place, is
/// in the constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Newlines {
    /// After the call's name.
    pub after_name: &'static str,
    /// Between `</arg_key>` and `<arg_value>`.
    pub after_key: &'static str,
    /// After `</arg_value>`, whether or not a pair follows.
    pub after_value: &'static str,
    /// Before the call's close, whether or not a pair came.
    pub before_close: &'static str,
    /// Inside a block of calls, before each call's opener. The block's two places are the
    /// grammar's alone: the table owns the block's markers, and the assembler never sees the
    /// bytes between them.
    pub block_before_call: &'static str,
    /// Inside a block of calls, before the block's close.
    pub block_before_close: &'static str,
}

impl Newlines {
    /// Nothing anywhere: GLM 4.7 and later, IQuest, Hy4.
    pub const NONE: Self = Self {
        after_name: "",
        after_key: "",
        after_value: "",
        before_close: "",
        block_before_call: "",
        block_before_close: "",
    };

    /// Ling's: after the name, after each `</arg_key>` and before the close, nothing between
    /// pairs, so a call without arguments reads `<tool_call>NAME\n\n</tool_call>` (every one of
    /// the 259 in ling-3.0-flash's recorded parse sets) and two pairs read
    /// `</arg_value><arg_key>`.
    pub const LING: Self = Self {
        after_name: "\n",
        after_key: "\n",
        after_value: "",
        before_close: "\n",
        block_before_call: "",
        block_before_close: "",
    };

    /// GLM 4.5 and 4.6's: after the name, after each `</arg_key>` and after each `</arg_value>`,
    /// nothing of its own before the close, so a call without arguments reads
    /// `<tool_call>NAME\n</tool_call>` (280 of 280 in glm-4.6's recorded parse sets), two pairs
    /// read `</arg_value>\n<arg_key>` (23,167 of 23,167) and the close follows the last pair's
    /// newline (19,254 of 19,254).
    pub const GLM4: Self = Self {
        after_name: "\n",
        after_key: "\n",
        after_value: "\n",
        before_close: "",
        block_before_call: "",
        block_before_close: "",
    };

    /// K2-Horizon's: GLM 4.5 and 4.6's places, and the block's own newline before each call and
    /// before the block's close, so a turn's calls read
    /// `<ifm|tool_calls>\n<ifm|tool_call>NAME\n…</ifm|tool_call>\n</ifm|tool_calls>` (in
    /// k2-horizon-36b's recorded shapes set: 4 of 4 zero-argument calls `NAME\n</ifm|tool_call>`,
    /// 2,667 pairs followed by `\n<ifm|arg_key>`, 1,588 last pairs by `\n</ifm|tool_call>`).
    pub const K2_HORIZON: Self = Self {
        block_before_call: "\n",
        block_before_close: "\n",
        ..Self::GLM4
    };
}

impl Tags {
    /// The plain tags with nothing between them: GLM 4.7 and later, IQuest.
    pub const PLAIN: Self = Self {
        key_open: "<arg_key>",
        key_close: "</arg_key>",
        value_open: "<arg_value>",
        value_close: "</arg_value>",
        newlines: Newlines::NONE,
    };

    /// The plain tags with Ling's newlines.
    pub const LING: Self = Self {
        newlines: Newlines::LING,
        ..Self::PLAIN
    };

    /// The plain tags with GLM 4.5 and 4.6's newlines.
    pub const GLM4: Self = Self {
        newlines: Newlines::GLM4,
        ..Self::PLAIN
    };

    /// K2-Horizon's spelling, every tag prefixed `ifm|`, with its newlines.
    pub const K2_HORIZON: Self = Self {
        key_open: "<ifm|arg_key>",
        key_close: "</ifm|arg_key>",
        value_open: "<ifm|arg_value>",
        value_close: "</ifm|arg_value>",
        newlines: Newlines::K2_HORIZON,
    };

    /// Hy4's spelling, every tag suffixed `:opensource`, nothing between them.
    pub const HY4: Self = Self {
        key_open: "<arg_key:opensource>",
        key_close: "</arg_key:opensource>",
        value_open: "<arg_value:opensource>",
        value_close: "</arg_value:opensource>",
        newlines: Newlines::NONE,
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
    id: String,
    tags: Tags,
    scanner: Scanner,
    /// The arguments object as it is written, with the bytes carried for the next event's source.
    object: Object,
    stage: Stage,
    /// The function's name once `ToolCallStart` was pushed.
    function: Option<String>,
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

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned
    /// just after the call's opening marker.
    pub fn new(index: u32, id: impl Into<String>, tags: Tags) -> Self {
        Self {
            id: id.into(),
            scanner: Scanner::new([
                tags.key_open,
                tags.key_close,
                tags.value_open,
                tags.value_close,
            ]),
            tags,
            object: Object::new(index),
            stage: Stage::Name,
            function: None,
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
        let scanned = self.scanner.feed(bytes);
        for piece in scanned.pieces() {
            self.take(piece, declared, out);
        }
        self.scanner.reclaim(scanned);
    }

    /// The call's end: `terminal` is the closing marker the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. A call that has not been
    /// named yet is named from the text so far; after the last value the object is closed; inside
    /// a value held for its close what was held is refused and the object closes without that
    /// member; inside a streamed string nothing is closed, since nothing is made up for a value
    /// the model never finished: the fragments stand, what was held is refused, and one
    /// `Malformed` event of no bytes says that the arguments are no whole value before the call's
    /// end ([`end_cut_call`]); a block with no name at all has its bytes, the marker's included,
    /// refused.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.scanner.held().to_string();
        self.object.carried.push_str(&held);
        if matches!(self.stage, Stage::Name) {
            self.name_call(out);
        }
        if !self.started() {
            self.object.carried.push_str(terminal);
            self.object.refuse(MalformedReason::CallWithoutName, out);
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between | Stage::Name => {}
            Stage::Value(ValueState {
                mode: Mode::Streaming { opened: true },
                ..
            }) => {
                // The start of a tag held at the end is refused; the one word before the end
                // says the arguments are no whole value.
                drop_refused(std::mem::take(&mut self.object.carried), out);
                end_cut_call(self.object.index, String::new(), terminal, out);
                self.done = true;
                return;
            }
            // A key, a tag or a value held for its close that the end cut short: its bytes are
            // refused, its member is never written, and the object closes.
            _ => self.object.refuse(MalformedReason::TagCutShort, out),
        }
        let source = Text::uncounted(std::mem::take(&mut self.object.carried));
        self.object.close(source, out);
        out.push(Event::ToolCallEnd {
            index: self.object.index,
            source: Text::uncounted(terminal.to_owned()),
        });
        self.done = true;
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A streamed string's open fragment stays open; a call that started still ends, with
    /// no bytes of its own, as every assembler ends one, everything it held refused, after the one
    /// `Malformed { UnterminatedRegion }` event of no bytes that says that its arguments are no
    /// whole value ([`end_cut_call`]); what a block that named no call held is refused as
    /// `UnterminatedRegion` too.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let held = self.scanner.held().to_string();
        self.object.carried.push_str(&held);
        if self.started() {
            drop_refused(std::mem::take(&mut self.object.carried), out);
            end_cut_call(self.object.index, String::new(), "", out);
        } else {
            self.object.refuse(MalformedReason::UnterminatedRegion, out);
        }
    }

    fn take(&mut self, piece: Piece<'_>, declared: &Declared, out: &mut Events) {
        match piece {
            Piece::Text(text) => self.text(&text, out),
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
                    self.object.drop_carried(out);
                    refuse(bytes.to_string(), MalformedReason::TagOutOfPlace, out);
                }
            }
            Stage::Between => self.tag_between(tag, out),
            Stage::Key { start } if tag == KEY_CLOSE => {
                let key = self.object.carried[*start..].to_string();
                self.object.carried.push_str(bytes);
                if key.is_empty() {
                    self.object.refuse(MalformedReason::EmptyKey, out);
                    self.stage = Stage::Between;
                } else {
                    self.stage = Stage::Keyed { key };
                }
            }
            Stage::Keyed { key } if tag == VALUE_OPEN => {
                let key = key.clone();
                self.object.carried.push_str(bytes);
                self.open_value(key, declared);
            }
            Stage::Value(_) if tag == VALUE_CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tags are the value's text.
            Stage::Value(_) => self.text(bytes, out),
            // A tag cuts a key short, or stands where the syntax has another: the bytes so far
            // are refused, and the tag is read as between keys.
            _ => {
                self.object.refuse(MalformedReason::TagCutShort, out);
                self.stage = Stage::Between;
                self.tag_between(tag, out);
            }
        }
    }

    /// A tag between keys: `<arg_key>` opens one, once the call has started; the other three, and
    /// any tag in a block that named no call, have no place here and are refused.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = self.tags.text(tag);
        if tag == KEY_OPEN && self.started() {
            self.object.carried.push_str(bytes);
            self.stage = Stage::Key {
                start: self.object.carried.len(),
            };
        } else {
            self.object.drop_carried(out);
            refuse(bytes.to_string(), MalformedReason::TagOutOfPlace, out);
        }
    }

    fn text(&mut self, text: &str, out: &mut Events) {
        match &self.stage {
            Stage::Name | Stage::Key { .. } => self.object.carried.push_str(text),
            Stage::Between | Stage::Keyed { .. } => self.text_between(text, out),
            Stage::Value(_) => self.value_text(text, out),
        }
    }

    /// Text where the syntax has tags: whitespace is the template's, anything else is refused.
    fn text_between(&mut self, text: &str, out: &mut Events) {
        for (run, space) in runs(text) {
            if space {
                self.object.carried.push_str(run);
            } else {
                self.object.drop_carried(out);
                refuse(run.to_string(), MalformedReason::TextBetweenTags, out);
            }
        }
    }

    /// The text so far is the call's name, less the whitespace around it: the call starts, and
    /// the whitespace after the name is carried into the next source. Text that is no name is
    /// refused, and no call starts.
    fn name_call(&mut self, out: &mut Events) {
        let name = self.object.carried.trim().to_string();
        if name.is_empty() {
            self.stage = Stage::Between;
            return;
        }
        if !is_name(&name) {
            // Text where the name should be, a JSON call inside a code fence or prose: refused,
            // and no call starts.
            self.object
                .refuse(MalformedReason::TextWhereTheNameShouldBe, out);
            self.stage = Stage::Between;
            return;
        }
        let end = self.object.carried.trim_end().len();
        let after = self.object.carried.split_off(end);
        out.push(Event::ToolCallStart {
            index: self.object.index,
            id: self.id.clone(),
            name: name.clone(),
            source: Text::uncounted(std::mem::take(&mut self.object.carried)),
        });
        self.object.carried = after;
        self.function = Some(name);
        self.stage = Stage::Between;
    }

    /// `<arg_value>` after a key: the value begins, typed by what the tool declares for it.
    fn open_value(&mut self, key: String, declared: &Declared) {
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
            start: self.object.carried.len(),
            mode,
        });
    }

    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        self.object
            .value_bytes(&value.key, value.start, &mut value.mode, text, out);
    }

    /// `</arg_value>`: a streamed string gets its closing quote, any other value is written whole.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = match value.mode {
            Mode::Streaming { opened: true } => "\"".to_string(),
            // An empty string: its opening and closing quotes come together.
            Mode::Streaming { opened: false } => self.object.member(&value.key, "\"\""),
            Mode::Whole | Mode::Undecided => {
                let written = json(&self.object.carried[value.start..], value.kind);
                self.object.member(&value.key, &written)
            }
        };
        self.object.close_member(fragment, tag, out);
        self.stage = Stage::Between;
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.feed(text, declared, out);
        text.len()
    }

    /// The call's closing marker is the call's own: its bytes are the call's end.
    fn close(self: Box<Self>, terminal: &str, out: &mut Events) -> bool {
        (*self).close(terminal, out);
        true
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;
    use crate::{event::DropReason, testing::bytes};

    const CLOSE: &str = "</tool_call>";

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
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    /// The events of one block fed in `pieces`, closed by the call's marker or cut.
    fn run(pieces: &[&str], cut: bool) -> Vec<Event<'static>> {
        let mut assembler = Assembler::new(0, "call_0", Tags::PLAIN);
        let mut out = Events::new();
        let declared = declared();
        for piece in pieces {
            assembler.feed(piece, &declared, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(CLOSE, &mut out);
        }
        out.drain()
    }

    fn fragments<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn arguments(events: &[Event]) -> String {
        fragments(events).concat()
    }

    fn malformed<'a>(events: &'a [Event<'_>]) -> Vec<(&'a str, MalformedReason)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why } => Some((&*text.text, *why)),
                _ => None,
            })
            .collect()
    }

    /// The text of a refused call block, in order: the `Dropped { MalformedCall }` events.
    fn refused<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::MalformedCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The malformed events' reasons, each run of one reason once, so that a cut inside a run of
    /// text (two events for one) reads as the whole does.
    fn reasons(events: &[Event]) -> Vec<MalformedReason> {
        let mut reasons: Vec<MalformedReason> = Vec::new();
        for (_, why) in malformed(events) {
            if reasons.last() != Some(&why) {
                reasons.push(why);
            }
        }
        reasons
    }

    fn starts(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, Event::ToolCallStart { .. }))
            .count()
    }

    fn ends<'a>(events: &'a [Event<'_>]) -> Vec<&'a str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(&*source.text),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_declared_string_streams_and_every_other_value_is_written_at_its_close() {
        let events = run(
            &[
                "f\n<arg_key>s</arg_key>\n<arg_value>Pa",
                "ris</arg_value><arg_key>n</arg_key><arg_value>3</arg_value>",
            ],
            false,
        );
        assert_eq!(starts(&events), 1);
        assert_eq!(
            fragments(&events),
            [r#"{"s": ""#, "Pa", "ris", "\"", r#", "n": 3"#, "}"]
        );
        assert_eq!(ends(&events), [CLOSE]);
        assert_eq!(
            bytes(&events),
            concat!(
                "f\n<arg_key>s</arg_key>\n<arg_value>Paris</arg_value>",
                "<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>"
            )
        );
    }

    #[test]
    fn a_block_that_names_no_call_starts_none_and_has_every_byte_refused() {
        // Two pairs: the second once reached `Stage::Key` and wrote arguments for a call that
        // never started (smg #2842, claude[bot]).
        let output = concat!(
            "<arg_key>a</arg_key><arg_value>1</arg_value>",
            "<arg_key>b</arg_key><arg_value>2</arg_value>"
        );
        let events = run(&[output], false);
        assert_eq!(starts(&events), 0);
        assert!(fragments(&events).is_empty());
        assert!(ends(&events).is_empty());
        assert_eq!(bytes(&events), format!("{output}{CLOSE}"));
        assert!(malformed(&events)
            .iter()
            .any(|(text, why)| text.is_empty() && *why == MalformedReason::TagOutOfPlace));
        // Every byte of the block, the closing marker included, is refused: none of it is an
        // event a client reads as content.
        assert_eq!(refused(&events).concat(), format!("{output}{CLOSE}"));
        assert!(malformed(&events).iter().all(|(text, _)| text.is_empty()));
    }

    #[test]
    fn text_that_is_not_a_name_starts_no_call() {
        // A JSON call quoted in a code fence (bellwether's probe), prose, a pythonic call written
        // where the name goes (whole with a string, a number or nothing, or cut by the first
        // tag), and a subscript.
        for text in [
            "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n",
            "get weather<arg_key>n</arg_key><arg_value>1</arg_value>",
            "f(s=\"x\")",
            "Read(offset=10)",
            "f()",
            "f(s</arg_key><arg_value>x</arg_value>",
            "f[0]<arg_key>n</arg_key><arg_value>1</arg_value>",
        ] {
            let events = run(&[text], false);
            assert_eq!(starts(&events), 0, "{text:?}");
            assert!(fragments(&events).is_empty(), "{text:?}");
            assert!(ends(&events).is_empty(), "{text:?}");
            assert!(
                malformed(&events)
                    .iter()
                    .any(|(_, why)| *why == MalformedReason::TextWhereTheNameShouldBe),
                "{text:?}"
            );
            assert_eq!(bytes(&events), format!("{text}{CLOSE}"));
            // The block is refused whole, the marker that closed it included.
            assert_eq!(
                refused(&events).concat(),
                format!("{text}{CLOSE}"),
                "{text:?}"
            );
            assert!(
                malformed(&events).iter().all(|(text, _)| text.is_empty()),
                "{text:?}"
            );
        }
    }

    #[test]
    fn an_empty_key_is_reported_and_the_next_pair_is_read() {
        let events = run(
            &[concat!(
                "f<arg_key></arg_key><arg_value>x</arg_value>",
                "<arg_key>n</arg_key><arg_value>1</arg_value>"
            )],
            false,
        );
        assert_eq!(arguments(&events), r#"{"n": 1}"#);
        assert!(malformed(&events)
            .iter()
            .any(|(_, why)| *why == MalformedReason::EmptyKey));
        // The empty pair's tags, and the value that followed with no key before it, are refused.
        assert_eq!(
            refused(&events),
            ["<arg_key></arg_key>", "<arg_value>", "x", "</arg_value>"]
        );
    }

    #[test]
    fn a_tag_inside_a_key_cuts_it_short_and_is_read_between_keys() {
        let events = run(
            &["f<arg_key>na<arg_key>n</arg_key><arg_value>1</arg_value>"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"n": 1}"#);
        assert!(malformed(&events)
            .iter()
            .any(|(text, why)| text.is_empty() && *why == MalformedReason::TagCutShort));
        assert_eq!(refused(&events), ["<arg_key>na"]);
    }

    #[test]
    fn a_nullable_string_is_null_when_its_text_says_so_and_streams_otherwise() {
        let events = run(
            &["f<arg_key>ns</arg_key><arg_value>nu", "ll</arg_value>"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"ns": null}"#);
        let events = run(
            &[
                "f<arg_key>ns</arg_key><arg_value>nu",
                "mber one</arg_value>",
            ],
            false,
        );
        // The held `nu` and the piece that ruled null out go out together.
        assert_eq!(fragments(&events), [r#"{"ns": ""#, "number one", "\"", "}"]);
        let events = run(&["f<arg_key>ns</arg_key><arg_value></arg_value>"], false);
        assert_eq!(arguments(&events), r#"{"ns": ""}"#);
    }

    #[test]
    fn an_empty_declared_string_is_an_empty_string() {
        let events = run(&["f<arg_key>s</arg_key><arg_value></arg_value>"], false);
        assert_eq!(arguments(&events), r#"{"s": ""}"#);
    }

    #[test]
    fn a_cut_stream_ends_a_started_call_and_closes_nothing() {
        let events = run(&["f<arg_key>s</arg_key><arg_value>Pa"], true);
        assert_eq!(starts(&events), 1);
        assert_eq!(fragments(&events), [r#"{"s": ""#, "Pa"]);
        assert_eq!(ends(&events), [""]);
        // The arguments are no whole value: one `Malformed` event says so before the end, of no
        // bytes since nothing was held at the cut.
        assert_eq!(
            malformed(&events),
            [("", MalformedReason::UnterminatedRegion)]
        );
        assert!(matches!(
            &events[events.len() - 2..],
            [Event::Malformed { .. }, Event::ToolCallEnd { .. }]
        ));
        assert_eq!(bytes(&events), "f<arg_key>s</arg_key><arg_value>Pa");
        // A value held for its close is refused, and the one event has no bytes.
        let events = run(&["f<arg_key>n</arg_key><arg_value>12"], true);
        assert_eq!(fragments(&events), Vec::<&str>::new());
        assert_eq!(
            malformed(&events),
            [("", MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(refused(&events), ["<arg_key>n</arg_key><arg_value>12"]);
        assert_eq!(ends(&events), [""]);
        // A block with no name at all, cut: the tag before any name and the text after it are
        // refused as they arrive, nothing is held at the cut, and no call started.
        let events = run(&["<arg_key>a"], true);
        assert_eq!(starts(&events), 0);
        assert!(ends(&events).is_empty());
        assert_eq!(
            malformed(&events),
            [
                ("", MalformedReason::TagOutOfPlace),
                ("", MalformedReason::TextBetweenTags)
            ]
        );
        assert_eq!(refused(&events), ["<arg_key>", "a"]);
    }

    /// The call's closing marker inside a streamed string: no closing quote and no closing brace
    /// are made up for a value the model never closed; the fragments stand, what was held (`<`
    /// could begin `<arg_key>`) is refused, and one `Malformed` event of no bytes before the end
    /// says that the arguments are no whole value. The marker's bytes are still the call's end.
    #[test]
    fn the_calls_end_inside_a_streamed_string_leaves_it_open_and_says_so() {
        let events = run(&["f<arg_key>s</arg_key><arg_value>a <"], false);
        assert_eq!(fragments(&events), [r#"{"s": ""#, "a "]);
        assert_eq!(
            malformed(&events),
            [("", MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(refused(&events), ["<"]);
        assert!(matches!(
            &events[events.len() - 2..],
            [Event::Malformed { .. }, Event::ToolCallEnd { .. }]
        ));
        assert_eq!(ends(&events), [CLOSE]);
        assert_eq!(
            bytes(&events),
            format!("f<arg_key>s</arg_key><arg_value>a <{CLOSE}")
        );
        // Nothing held: the one event has no bytes.
        let events = run(&["f<arg_key>s</arg_key><arg_value>Par"], false);
        assert_eq!(fragments(&events), [r#"{"s": ""#, "Par"]);
        assert_eq!(
            malformed(&events),
            [("", MalformedReason::UnterminatedRegion)]
        );
        assert_eq!(ends(&events), [CLOSE]);
        assert_eq!(
            bytes(&events),
            format!("f<arg_key>s</arg_key><arg_value>Par{CLOSE}")
        );
        // A string its tag closed is whole: no word, however the block ended.
        let events = run(
            &["f<arg_key>s</arg_key><arg_value>Paris</arg_value>"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"s": "Paris"}"#);
        assert!(malformed(&events).is_empty());
    }

    #[test]
    fn text_between_tags_is_refused_with_its_whitespace_dropped() {
        let events = run(
            &["f\n<arg_key>n</arg_key>\n junk \n<arg_value>1</arg_value>\n"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"n": 1}"#);
        assert!(malformed(&events)
            .iter()
            .any(|(text, why)| text.is_empty() && *why == MalformedReason::TextBetweenTags));
        assert_eq!(refused(&events), ["junk"]);
        // The whitespace before the text is the template's; the key's tags carried before it go
        // out with it, since nothing before the text can take them.
        assert!(events.iter().any(|e| matches!(
            e,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text.ends_with("\n ")
        )));
    }

    #[test]
    fn keys_and_values_are_escaped() {
        let events = run(
            &["f<arg_key>q\"k</arg_key><arg_value>say \"hi\" \\ then\nnew</arg_value>"],
            false,
        );
        assert_eq!(arguments(&events), r#"{"q\"k": "say \"hi\" \\ then\nnew"}"#);
        let read: serde_json::Value = serde_json::from_str(&arguments(&events)).expect("json");
        assert_eq!(read["q\"k"], "say \"hi\" \\ then\nnew");
    }

    /// Every malformed shape of the module, whole and at every two-way cut: no event the
    /// assembler pushes carries bytes a client reads as content, the bytes are refused behind an
    /// event of no bytes, and a cut changes nothing but where a run of text splits.
    #[test]
    fn what_the_syntax_cannot_read_is_refused_never_carried_by_the_event() {
        for (output, cut) in [
            // The shapes of a live cell's botched blocks: a name with a space, a key tag that
            // never opened, a key a value close cut short, a value close written twice, a call
            // written as Python with tags inside, an empty key; and the cut ones.
            (
                "probability_dice roll<arg_key>n</arg_key><arg_value>2</arg_value>",
                false,
            ),
            ("f</arg_key><arg_value>1975</arg_value>", false),
            ("f<arg_key>format=2D</arg_value>", false),
            (
                "f<arg_key>s</arg_key><arg_value>x</arg_value></arg_value>",
                false,
            ),
            ("f(base</arg_key>3,exponent</arg_key>5)", false),
            ("f<arg_key></arg_key><arg_value>x</arg_value>", false),
            ("f<arg_key>n</arg_key><arg_value>12", true),
            ("<arg_key>a", true),
            ("f<arg_key>s</arg_key><arg_value>a <", false),
        ] {
            let end = if cut { "" } else { CLOSE };
            let whole = run(&[output], cut);
            assert_eq!(bytes(&whole), format!("{output}{end}"), "{output:?}");
            assert!(!malformed(&whole).is_empty(), "{output:?}");
            assert!(
                malformed(&whole).iter().all(|(text, _)| text.is_empty()),
                "{output:?}: {whole:?}"
            );
            assert!(!refused(&whole).is_empty(), "{output:?}");
            let said = (
                arguments(&whole),
                reasons(&whole),
                refused(&whole).concat(),
                ends(&whole).len(),
            );
            for at in 1..output.len() {
                let events = run(&[&output[..at], &output[at..]], cut);
                assert_eq!(
                    bytes(&events),
                    format!("{output}{end}"),
                    "{output:?} cut at {at}"
                );
                assert_eq!(
                    (
                        arguments(&events),
                        reasons(&events),
                        refused(&events).concat(),
                        ends(&events).len()
                    ),
                    said,
                    "{output:?} cut at {at}"
                );
            }
        }
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let output = concat!(
            "f\n<arg_key>s</arg_key>\n<arg_value>Paris</arg_value>",
            "<arg_key>n</arg_key><arg_value>3</arg_value>\n"
        );
        let whole = run(&[output], false);
        let said = (
            arguments(&whole),
            malformed(&whole).len(),
            ends(&whole).len(),
        );
        for cut in 1..output.len() {
            let events = run(&[&output[..cut], &output[cut..]], false);
            assert_eq!(bytes(&events), format!("{output}{CLOSE}"), "cut at {cut}");
            assert_eq!(
                (
                    arguments(&events),
                    malformed(&events).len(),
                    ends(&events).len()
                ),
                said,
                "cut at {cut}"
            );
        }
    }
}
