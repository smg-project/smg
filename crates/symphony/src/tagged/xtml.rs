//! Kimi K3's XTML: the events of one `<|open|>call tool="NAME" index="N"<|sep|>` block, from
//! its bytes.
//!
//! Kimi K3 writes a turn as tagged regions, each tag an `<|open|>` or `<|close|>` token, a word,
//! and the `<|sep|>` token, and its calls as
//!
//! ```text
//! <|open|>tools<|sep|>
//! <|open|>call tool="get_weather" index="1"<|sep|>
//! <|open|>argument key="city" type="string"<|sep|>Paris<|close|>argument<|sep|>
//! <|open|>argument key="days" type="number"<|sep|>3<|close|>argument<|sep|>
//! <|close|>call<|sep|>
//! <|close|>tools<|sep|>
//! ```
//!
//! without the newlines, which stand here for the reader. The engine's table owns the `tools` and
//! `call` tags ([`formats::kimi_k3`]): it enters this assembler at `<|open|>call tool="` and
//! leaves it at `<|close|>call<|sep|>`, so the assembler starts inside the function's name and
//! never sees the call's closing tag; the engine hands it that tag's bytes as the call's end.
//! Between those two the assembler reads the argument tags itself.
//!
//! What the syntax says, and what the assembler does with it:
//!
//! - **The `type` attribute types the value**, and carries every type: `string` is the text,
//!   every byte of it, streamed as it arrives like a declared string in
//!   [`assembler`](super::assembler); `number`, `boolean`, `null`, `array` and `object` are JSON
//!   the model wrote, written whole at the value's close through [`json`]'s inference (`3`,
//!   `true`, `null`, `[1, 2]`, `{"a": 1}` as written; text that is not JSON is a string holding
//!   it, so one odd value never costs a call its other arguments). The request's tools are not
//!   consulted: the model says the type. A tag whose `type` is none of the six is reported
//!   whole as `Malformed`, and no argument opens.
//! - **The call tag carries an index**, `index="1"`, the model's count of the turn's calls. The
//!   assembler keeps its own count, which the format mints, as every assembler does: a tag without
//!   the index is a call all the same, and a tail that is neither is reported as `Malformed`.
//! - **A name ends at its quote.** The function name runs to `"`, the key to `"`; the rest of
//!   each tag runs to its `<|sep|>`, which is a token of its own and may arrive in pieces, so it
//!   is one of the markers the scanner holds.
//! - **Inside a value only `<|close|>argument<|sep|>` is a tag for the assembler**; an argument
//!   tag or a `<|sep|>` there is the value's text, as for DSML. The call tags and the block's close
//!   are the engine's terminals, so they end the call wherever they stand, a value included.
//! - **Two endings.** [`Assembler::close`] is the call's closing tag ([`CALL_CLOSE`], the one
//!   terminal the call's end carries), or the next call's opening or the block's close, which end
//!   the call with no bytes of their own: after the last argument it closes the object and pushes
//!   `ToolCallEnd`; inside a value it reports what was held, closes an open string and the
//!   object. [`Assembler::finish`] is a stream that was cut: nothing is closed.
//! - **Every byte of the call lands in exactly one event**, with the DSML assembler's accounting:
//!   the name's bytes in `ToolCallStart`, each tag's and value's bytes in the fragments they
//!   produce, text where the syntax has tags as `Malformed` run by run with the whitespace before
//!   it as `Dropped { Wrapper }`.
//!
//! This is the DSML assembler ([`dsml`](super::dsml)) with Kimi's tags and six types in place of
//! DeepSeek's two; the two keep their shape side by side until a shared core shows where the
//! seams are.
//!
//! [`formats::kimi_k3`]: crate::formats::kimi_k3()

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::json,
    },
};

/// The call's closing tag: the one terminal the call's end carries. The next call's opening and
/// the block's close end a call too, but they belong to the region, and the engine drops them.
pub const CALL_CLOSE: &str = "<|close|>call<|sep|>";

const ARGUMENT_OPEN: usize = 0;
const ARGUMENT_CLOSE: usize = 1;
const SEP: usize = 2;
const TAGS: [&str; 3] = [
    "<|open|>argument key=\"",
    "<|close|>argument<|sep|>",
    "<|sep|>",
];
const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const EMPTY_NAME: &str = "a tag without a name";
const TAG_TAIL: &str = "text after a tag's name";
const CLOSED_EARLY: &str = "a call that closed before its function name closed";
/// The types a value may carry; `string` streams, the rest are written whole.
const TYPES: [&str; 6] = ["string", "number", "boolean", "null", "array", "object"];

/// The events of one XTML call, from the bytes after `<|open|>call tool="`.
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
    /// Inside the call tag: the name is `carried[..]` until `"`, then the tail until `<|sep|>`.
    FunctionName,
    /// After the name's quote, before the tag's `<|sep|>`: the tail is `carried[start..]`.
    FunctionTail { name: String, start: usize },
    /// After the call tag or a `<|close|>argument<|sep|>`.
    Between,
    /// Inside an argument tag: the key is `carried[start..]` until `"`.
    ArgumentKey { start: usize },
    /// After the key's quote, before the tag's `<|sep|>`: the type attribute, `carried[start..]`.
    ArgumentTail { key: String, start: usize },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    /// `type="string"`: streamed; any other type: written whole at the close.
    string: bool,
    /// For a whole value, where its text starts in `carried`.
    start: usize,
    /// For a streamed value, whether its opening fragment has been pushed.
    opened: bool,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned
    /// just after `<|open|>call tool="`.
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

    /// Append the next bytes of the call and push the events they complete. Every byte is the
    /// call's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        if self.done {
            return;
        }
        for piece in self.scanner.feed(bytes) {
            self.take(piece, out);
        }
    }

    /// The call's end: `terminal` is the closing tag the engine read (its bytes go into
    /// `ToolCallEnd`), or nothing when the block ended another way. After the last argument the
    /// object is closed and the call ends; inside a value an open string is closed, the object is
    /// closed, and what was held comes back as `Malformed`; a call that never named a function has
    /// its bytes reported.
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
            // The tail of a call or argument tag ends at its separator.
            Stage::FunctionTail { .. } | Stage::ArgumentTail { .. } if tag == SEP => {
                self.end_tail(out);
            }
            Stage::Value(_) if tag == ARGUMENT_CLOSE => self.close_value(bytes, out),
            // Inside a value, the other tags are the value's text.
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

    /// A tag between arguments: the argument tag opens a key, once the function is named; the
    /// closing tag and a stray separator have no place, and neither does an argument before the
    /// function's name closed (a missing quote on the name), so no fragment comes before the
    /// call's start.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = TAGS[tag];
        if tag == ARGUMENT_OPEN && self.started() {
            self.carried.push_str(bytes);
            self.stage = Stage::ArgumentKey {
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

    /// The separator that ends a call tag's or an argument tag's tail.
    fn end_tail(&mut self, out: &mut Events) {
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::FunctionTail { name, start } => {
                let tail = self.carried[start..].to_string();
                self.carried.push_str(TAGS[SEP]);
                self.start_call(name, &tail, out);
            }
            Stage::ArgumentTail { key, start } => {
                let tail = self.carried[start..].to_string();
                self.carried.push_str(TAGS[SEP]);
                self.open_value(key, &tail, out);
            }
            other => self.stage = other,
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
            // A tail runs to its separator, which arrives as a marker; its text is carried.
            Stage::FunctionTail { .. } | Stage::ArgumentTail { .. } => self.carried.push_str(text),
            Stage::Between => self.text_between(text, out),
            Stage::ArgumentKey { start } => {
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
                            self.stage = Stage::ArgumentTail {
                                key,
                                start: self.carried.len(),
                            };
                        }
                        self.text(rest, out);
                    }
                    None => self.carried.push_str(text),
                }
            }
            Stage::Value(_) => self.value_text(text, out),
        }
    }

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

    /// The call tag is whole: the call starts, with the tag's bytes as its source. The tail is
    /// the model's index of the call, ` index="1"`, or nothing; any other tail is reported.
    fn start_call(&mut self, name: String, tail: &str, out: &mut Events) {
        self.function = Some(name.clone());
        if tail.is_empty() || is_index(tail) {
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(std::mem::take(&mut self.carried)),
            });
        } else {
            let tag_end = self.carried.len() - tail.len() - TAGS[SEP].len();
            let mut after = self.carried.split_off(tag_end);
            let close = after.split_off(after.len() - TAGS[SEP].len());
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
            self.carried.push_str(&close);
        }
        self.stage = Stage::Between;
    }

    /// The argument tag is whole: the value begins, typed by its `type` attribute. A tag whose
    /// type is none of the six is not an argument tag the syntax has: the whole tag comes back
    /// as `Malformed`, and no argument opens.
    fn open_value(&mut self, key: String, tail: &str, out: &mut Events) {
        let Some(kind) = tail
            .strip_prefix(" type=\"")
            .and_then(|rest| rest.strip_suffix('"'))
        else {
            self.report(TAG_TAIL, out);
            self.stage = Stage::Between;
            return;
        };
        if !TYPES.contains(&kind) {
            self.report(TAG_TAIL, out);
            self.stage = Stage::Between;
            return;
        }
        self.stage = Stage::Value(ValueState {
            key,
            string: kind == "string",
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

    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        let fragment = if value.string {
            if value.opened {
                "\"".to_string()
            } else {
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

    /// The call ended inside a value: an open string is closed, and a whole value's text comes
    /// back as `Malformed`, since its member was never written.
    fn cut_value(&mut self, value: &ValueState, out: &mut Events) {
        if value.string && value.opened {
            self.report(TAG_CUT_SHORT, out);
            self.push_fragment("\"".to_string(), Text::default(), out);
        } else {
            self.report(TAG_CUT_SHORT, out);
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

/// Whether a call tag's tail is the model's index of the call, ` index="N"`.
fn is_index(tail: &str) -> bool {
    tail.strip_prefix(" index=\"")
        .and_then(|rest| rest.strip_suffix('"'))
        .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recorded call, from after `<|open|>call tool="`: a string, a number and an object.
    const CALL: &str = concat!(
        "get_user_info\" index=\"1\"<|sep|>",
        "<|open|>argument key=\"user_id\" type=\"number\"<|sep|>7890<|close|>argument<|sep|>",
        "<|open|>argument key=\"special\" type=\"string\"<|sep|>black<|close|>argument<|sep|>",
        "<|open|>argument key=\"prefs\" type=\"object\"<|sep|>{\"size\": \"large\"}",
        "<|close|>argument<|sep|>"
    );

    fn run(pieces: &[&str]) -> Vec<Event> {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.close(CALL_CLOSE, &mut out);
        let events = out.drain();
        nothing_empty(&events);
        events
    }

    /// No fragment is empty and no `Malformed` or `Dropped` text is empty: an empty report would
    /// claim a byte where none was.
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

    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|event| match event {
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::Malformed { text, .. } | Event::Dropped { text, .. } => text.text.as_str(),
                _ => "",
            })
            .collect()
    }

    #[test]
    fn a_whole_call_gives_its_name_its_typed_arguments_and_every_byte() {
        let events = run(&[CALL]);
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { name, id, .. } if name == "get_user_info" && id == "call_0"
        ));
        assert_eq!(
            arguments(&events),
            r#"{"user_id": 7890, "special": "black", "prefs": {"size": "large"}}"#
        );
        assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
        assert_eq!(bytes(&events), format!("{CALL}{CALL_CLOSE}"));
    }

    #[test]
    fn every_cut_of_the_call_says_the_same() {
        let whole = arguments(&run(&[CALL]));
        for cut in 1..CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let events = run(&[&CALL[..cut], &CALL[cut..]]);
            assert_eq!(arguments(&events), whole, "cut at {cut}");
            assert_eq!(
                bytes(&events),
                format!("{CALL}{CALL_CLOSE}"),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn the_six_types_are_written_as_the_model_wrote_them_and_a_string_streams() {
        let call = concat!(
            "f\" index=\"2\"<|sep|>",
            "<|open|>argument key=\"s\" type=\"string\"<|sep|>12<|close|>argument<|sep|>",
            "<|open|>argument key=\"n\" type=\"number\"<|sep|>1.5<|close|>argument<|sep|>",
            "<|open|>argument key=\"b\" type=\"boolean\"<|sep|>true<|close|>argument<|sep|>",
            "<|open|>argument key=\"z\" type=\"null\"<|sep|>null<|close|>argument<|sep|>",
            "<|open|>argument key=\"a\" type=\"array\"<|sep|>[1, \"x\"]<|close|>argument<|sep|>",
            "<|open|>argument key=\"o\" type=\"object\"<|sep|>{\"k\": null}<|close|>argument<|sep|>"
        );
        let events = run(&[call]);
        assert_eq!(
            arguments(&events),
            r#"{"s": "12", "n": 1.5, "b": true, "z": null, "a": [1, "x"], "o": {"k": null}}"#
        );
        // The string streamed in pieces, each its own fragment.
        let streamed = run(&[
            "f\" index=\"2\"<|sep|><|open|>argument key=\"s\" type=\"string\"<|sep|>he",
            "llo<|close|>argument<|sep|>",
        ]);
        let fragments: Vec<&str> = streamed
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(fragments, [r#"{"s": ""#, "he", "llo", "\"", "}"]);
    }

    #[test]
    fn a_type_the_syntax_lacks_and_a_tail_that_is_no_index_are_reported() {
        let call = concat!(
            "f\" index=\"x\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"date\"<|sep|>2026<|close|>argument<|sep|>",
            "<|open|>argument key=\"b\" type=\"number\"<|sep|>2<|close|>argument<|sep|>"
        );
        let events = run(&[call]);
        assert_eq!(arguments(&events), r#"{"b": 2}"#);
        let malformed: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert!(malformed.contains(&" index=\"x\""), "{malformed:?}");
        // The call tag's separator, carried after its tail was reported, leads the bad tag's bytes.
        assert!(
            malformed.contains(&"<|sep|><|open|>argument key=\"a\" type=\"date\"<|sep|>"),
            "{malformed:?}"
        );
        assert_eq!(bytes(&events), format!("{call}{CALL_CLOSE}"));
    }

    #[test]
    fn a_call_tag_without_an_index_is_a_call_all_the_same() {
        let call = concat!(
            "f\"<|sep|>",
            "<|open|>argument key=\"a\" type=\"number\"<|sep|>1<|close|>argument<|sep|>"
        );
        let events = run(&[call]);
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { name, source, .. } if name == "f" && source.text == "f\"<|sep|>"
        ));
        assert_eq!(arguments(&events), r#"{"a": 1}"#);
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::Malformed { .. })));
        assert_eq!(bytes(&events), format!("{call}{CALL_CLOSE}"));
    }

    #[test]
    fn a_stream_cut_inside_a_value_closes_nothing_and_a_call_cut_short_closes_its_object() {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assembler.feed(
            "f\" index=\"1\"<|sep|><|open|>argument key=\"s\" type=\"string\"<|sep|>par",
            &mut out,
        );
        assembler.finish(&mut out);
        let events = out.drain();
        nothing_empty(&events);
        assert_eq!(arguments(&events), r#"{"s": "par"#);
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text.is_empty()
        ));

        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assembler.feed(
            "f\" index=\"1\"<|sep|><|open|>argument key=\"n\" type=\"number\"<|sep|>12",
            &mut out,
        );
        assembler.close("", &mut out);
        let events = out.drain();
        nothing_empty(&events);
        assert_eq!(arguments(&events), "{}");
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, .. } if text.text.ends_with("12")
        )));
    }

    #[test]
    fn a_separator_split_across_pieces_still_ends_the_tag() {
        let whole = arguments(&run(&[CALL]));
        let at = CALL.find("<|sep|>").expect("a separator") + 3;
        let events = run(&[&CALL[..at], &CALL[at..]]);
        assert_eq!(arguments(&events), whole);
    }
}
