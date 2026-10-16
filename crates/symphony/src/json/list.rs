//! A JSON list of calls with no markers around it: the whole output is either
//! `[{"name": …, "arguments": {…}}, …]` or content. xLAM writes this; so does any model whose
//! template puts the calls alone in the turn.
//!
//! The format's one state is an arguments state entered at the output's start, so this assembler
//! sees every byte of the output and decides at the first one that is not whitespace:
//!
//! - `[` opens the list. Each object in it goes to a [`json::Assembler`](super::Assembler) of its
//!   own, with the next free index, which is spent only when the object starts a call, so an
//!   object with no name leaves no gap; every call streams as a Qwen3 call does. The `[`, the
//!   commas between objects, the whitespace around them, the `]` and whitespace after it are the
//!   template's and come back as `Dropped { Wrapper }`; any other text between objects or after
//!   the `]` is `Malformed`, one event per run.
//! - Anything else makes the whole output content: that byte and every later one come back as
//!   `Content`, since the model answered in prose. The whitespace held before it comes back as
//!   content too. A `[` that is followed by anything but an object or the `]` is prose as well
//!   (`[Note: draft]`, a markdown link), and the `[` comes back as content with the rest.
//!
//! With [`Assembler::object_or_list`] (the `json` table), a `{` at the first byte is one call
//! alone: the object goes to a [`json::Assembler`](super::Assembler) as a list's object does, and
//! what follows it is the template's wrapping, whitespace dropped and anything else `Malformed`.
//! A list reads as before. [`Assembler::new`] (the `xlam` table) keeps a `{` as prose.
//!
//! A region with no markers never closes by one, so [`Assembler::close`] is for a table that puts
//! the list between markers after all; [`Assembler::finish`] is the end of the stream: an object
//! still open is finished as the JSON assembler finishes one (a started call closed with what
//! arrived, an object that never named a call `Malformed`), and a list left open reports what
//! it held.
//!
//! Between markers (Llama Nemotron's `<TOOLCALL>[…]</TOOLCALL>`) a row of the table opens the
//! region and its closing marker ends it, so prose has no place inside: [`Assembler::framed`]
//! reports text where the list should be, a `[` that opens no list of calls included, as
//! `Malformed` where the bare list takes it for content. The engine picks it for a table with a
//! row into the arguments state.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    json,
};

const TEXT_BETWEEN_OBJECTS: &str = "text between a list's calls";
const TEXT_AFTER_THE_LIST: &str = "text after the list's end";
const TEXT_AFTER_THE_OBJECT: &str = "text after the call's object";
const TEXT_IN_PLACE_OF_THE_LIST: &str = "text where the list of calls should be";

/// The events of a JSON list of calls, each call with its own index from `first`.
#[derive(Debug)]
pub struct Assembler {
    next_index: u32,
    stage: Stage,
    /// Bytes no event has accounted for yet; the next event takes them as its source.
    carried: String,
    /// Objects that began, so that prose after the `[` is told from text between objects.
    objects: u32,
    /// Whether a `{` at the first byte is one call alone ([`Self::object_or_list`]).
    lone_object: bool,
    /// Whether a `[` opened a list, so that a finished object returns to the list or ends the
    /// output's calls.
    listed: bool,
    /// Whether markers frame the list: text in place of it is malformed, not prose.
    framed: bool,
}

#[derive(Debug)]
enum Stage {
    /// Before the first byte that is not whitespace.
    Deciding,
    /// The output is prose.
    Content,
    /// Text where the list should be, between markers: every byte malformed.
    Malformed,
    /// Inside the list, between objects.
    Between,
    /// Inside an object.
    /// Boxed: the assembler holds the object's scanner and its prefix, far more than the other
    /// stages carry.
    Object(Box<json::Assembler>),
    /// After the `]`.
    After,
}

impl Assembler {
    /// An assembler positioned at the output's start, the first call taking index `first`: the
    /// bare list, where anything but a list is prose.
    pub fn new(first: u32) -> Self {
        Self {
            next_index: first,
            stage: Stage::Deciding,
            carried: String::new(),
            objects: 0,
            lone_object: false,
            listed: false,
            framed: false,
        }
    }

    /// [`Self::new`], also reading a `{` at the first byte as one call alone.
    pub fn object_or_list(first: u32) -> Self {
        Self {
            lone_object: true,
            ..Self::new(first)
        }
    }

    /// An assembler positioned after the marker that opens the list's region, the first call
    /// taking index `first`: anything but a list is malformed.
    pub fn framed(first: u32) -> Self {
        Self {
            framed: true,
            ..Self::new(first)
        }
    }

    /// Append the next bytes and push the events they complete; every byte is taken.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        let mut rest = bytes;
        while !rest.is_empty() {
            rest = self.take(rest, out);
        }
    }

    /// The region closed by a marker: a table that puts the list between markers.
    pub fn close(self, out: &mut Events) {
        self.finish(out);
    }

    /// The end of the stream: an open object is finished as the JSON assembler finishes one, and
    /// what the list held comes back as `Malformed { UnterminatedRegion }`.
    pub fn finish(mut self, out: &mut Events) {
        match std::mem::replace(&mut self.stage, Stage::After) {
            Stage::Object(object) => {
                self.drop_carried(out);
                object.finish(out);
            }
            Stage::Deciding | Stage::Content | Stage::After => {
                if !self.carried.is_empty() {
                    out.push(Event::Content(Text::uncounted(std::mem::take(
                        &mut self.carried,
                    ))));
                }
            }
            // Every byte was reported as it arrived.
            Stage::Malformed => {}
            Stage::Between => {
                if !self.carried.is_empty() {
                    out.push(Event::Malformed {
                        text: Text::uncounted(std::mem::take(&mut self.carried)),
                        why: MalformedReason::UnterminatedRegion,
                    });
                }
            }
        }
    }

    /// Reads as much of `text` as the stage takes and returns the rest.
    fn take<'a>(&mut self, text: &'a str, out: &mut Events) -> &'a str {
        match &mut self.stage {
            Stage::Deciding => {
                let Some((at, first)) = text.char_indices().find(|(_, c)| !c.is_whitespace())
                else {
                    self.carried.push_str(text);
                    return "";
                };
                self.carried.push_str(&text[..at]);
                if first == '[' {
                    // Held until the first object or the `]` says this is a list: prose after a
                    // `[` takes it back as content.
                    self.carried.push('[');
                    self.listed = true;
                    self.stage = Stage::Between;
                    &text[at + 1..]
                } else if first == '{' && self.lone_object {
                    self.drop_carried(out);
                    let index = self.next_index;
                    self.objects += 1;
                    self.stage = Stage::Object(Box::new(json::Assembler::new(
                        index,
                        format!("call_{index}"),
                    )));
                    &text[at..]
                } else {
                    self.not_a_list(&text[at..], out);
                    ""
                }
            }
            Stage::Content => {
                out.push(Event::Content(Text::uncounted(text)));
                ""
            }
            Stage::Malformed => {
                out.push(Event::Malformed {
                    text: Text::uncounted(text),
                    why: MalformedReason::Other(TEXT_IN_PLACE_OF_THE_LIST.to_string()),
                });
                ""
            }
            Stage::Between => {
                let Some((at, first)) = text
                    .char_indices()
                    .find(|(_, c)| !c.is_whitespace() && *c != ',')
                else {
                    self.carried.push_str(text);
                    return "";
                };
                self.carried.push_str(&text[..at]);
                match first {
                    '{' => {
                        self.drop_carried(out);
                        let index = self.next_index;
                        self.objects += 1;
                        self.stage = Stage::Object(Box::new(json::Assembler::new(
                            index,
                            format!("call_{index}"),
                        )));
                        &text[at..]
                    }
                    ']' => {
                        self.carried.push(']');
                        self.drop_carried(out);
                        self.stage = Stage::After;
                        &text[at + 1..]
                    }
                    _ if self.objects == 0 => {
                        // Not a list of calls after all: the `[` and everything held are prose,
                        // or malformed between markers.
                        self.not_a_list(&text[at..], out);
                        ""
                    }
                    _ => {
                        // A run of text between objects, reported whole.
                        let end = text[at..]
                            .char_indices()
                            .find(|(_, c)| c.is_whitespace() || matches!(c, ',' | '{' | ']'))
                            .map_or(text.len(), |(i, _)| at + i);
                        self.drop_carried(out);
                        out.push(Event::Malformed {
                            text: Text::uncounted(&text[at..end]),
                            why: MalformedReason::Other(TEXT_BETWEEN_OBJECTS.to_string()),
                        });
                        &text[end..]
                    }
                }
            }
            Stage::Object(object) => {
                let taken = object.feed(text, out);
                if object.done() {
                    // The index is spent only by an object that started a call.
                    if object.started() {
                        self.next_index += 1;
                    }
                    self.stage = if self.listed {
                        Stage::Between
                    } else {
                        Stage::After
                    };
                }
                &text[taken..]
            }
            Stage::After => {
                // As the engine's wrapping: whitespace is the template's, anything else malformed.
                let mut rest = text;
                while let Some(first) = rest.chars().next() {
                    let space = first.is_whitespace();
                    let length = rest
                        .char_indices()
                        .find(|(_, c)| c.is_whitespace() != space)
                        .map_or(rest.len(), |(i, _)| i);
                    out.push(if space {
                        Event::Dropped {
                            text: Text::uncounted(&rest[..length]),
                            why: DropReason::Wrapper,
                        }
                    } else {
                        Event::Malformed {
                            text: Text::uncounted(&rest[..length]),
                            why: MalformedReason::Other(
                                if self.listed {
                                    TEXT_AFTER_THE_LIST
                                } else {
                                    TEXT_AFTER_THE_OBJECT
                                }
                                .to_string(),
                            ),
                        }
                    });
                    rest = &rest[length..];
                }
                ""
            }
        }
    }

    /// The bytes held and `text` are not a list of calls: content under the bare list, malformed
    /// between markers; every later byte comes back the same way.
    fn not_a_list(&mut self, text: &str, out: &mut Events) {
        self.carried.push_str(text);
        let text = Text::uncounted(std::mem::take(&mut self.carried));
        if self.framed {
            self.stage = Stage::Malformed;
            out.push(Event::Malformed {
                text,
                why: MalformedReason::Other(TEXT_IN_PLACE_OF_THE_LIST.to_string()),
            });
        } else {
            self.stage = Stage::Content;
            out.push(Event::Content(text));
        }
    }

    fn drop_carried(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Dropped {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: DropReason::Wrapper,
        });
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, _declared: &crate::tagged::Declared, out: &mut Events) -> usize {
        self.feed(text, out);
        text.len()
    }

    fn close(self: Box<Self>, _terminal: &str, out: &mut Events) -> bool {
        (*self).close(out);
        false
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIST: &str = concat!(
        "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, ",
        "{\"name\": \"get_stock_price\", \"arguments\": {}}]"
    );

    fn run(pieces: &[&str]) -> Vec<Event> {
        let mut assembler = Assembler::new(0);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.finish(&mut out);
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

    fn calls(events: &[Event]) -> Vec<(u32, String, String)> {
        let mut calls: Vec<(u32, String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { index, name, .. } => {
                    calls.push((*index, name.clone(), String::new()));
                }
                Event::ToolCallArguments { json, .. } => {
                    if let Some(last) = calls.last_mut() {
                        last.2.push_str(json);
                    }
                }
                _ => {}
            }
        }
        calls
    }

    #[test]
    fn an_object_without_arguments_is_a_call_with_an_empty_object_in_a_list_and_alone() {
        // The object's assembler says `{}` for a call whose object has no arguments member, so a
        // list's object and a lone object read as one written `"arguments": {}`, whole and byte
        // by byte.
        let list = "[{\"name\": \"f\"}, {\"name\": \"g\", \"arguments\": {}}]";
        let per_char: Vec<&str> = list
            .char_indices()
            .map(|(i, c)| &list[i..i + c.len_utf8()])
            .collect();
        for events in [run(&[list]), run(&per_char)] {
            assert_eq!(bytes(&events), list);
            assert_eq!(
                calls(&events),
                [
                    (0, "f".to_string(), "{}".to_string()),
                    (1, "g".to_string(), "{}".to_string()),
                ]
            );
        }
        let alone = "{\"name\": \"f\"}";
        let events = run_alone(&[alone]);
        assert_eq!(bytes(&events), alone);
        assert_eq!(calls(&events), [(0, "f".to_string(), "{}".to_string())]);
    }

    #[test]
    fn a_list_gives_each_object_as_a_call_with_the_brackets_and_commas_dropped() {
        let events = run(&[LIST]);
        assert_eq!(bytes(&events), LIST);
        assert_eq!(
            calls(&events),
            [
                (
                    0,
                    "get_weather".to_string(),
                    r#"{"city": "Paris"}"#.to_string()
                ),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, ["[", ", ", "]"]);
    }

    #[test]
    fn prose_is_content_whole_and_a_cut_list_closes_nothing() {
        let events = run(&["  Hello, ", "[not a list]"]);
        let content: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "  Hello, [not a list]");
        assert!(calls(&events).is_empty());

        let events = run(&["[{\"name\": \"get_weather\", \"argu"]);
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallEnd { .. })));
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::UnterminatedRegion,
                ..
            }
        )));
        assert_eq!(bytes(&events), "[{\"name\": \"get_weather\", \"argu");
    }

    #[test]
    fn an_object_that_names_no_call_spends_no_index_and_wrapping_after_the_list_is_dropped() {
        let output = "[{\"foo\": 1}, {\"name\": \"f\", \"arguments\": {}}]\n";
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(calls(&events), [(0, "f".to_string(), "{}".to_string())]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "\n"
        )));
        // Text between objects comes back as one run, text after the list too.
        let output =
            "[{\"name\": \"f\", \"arguments\": {}} junk {\"name\": \"g\", \"arguments\": {}}] tail";
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(calls(&events).len(), 2);
        let malformed: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(malformed, ["junk", "tail"]);
    }

    #[test]
    fn a_bracket_that_opens_no_list_of_calls_is_prose() {
        for output in [
            "[Note: draft] ok",
            "[1] First, then second.",
            "[docs](https://x) and more",
        ] {
            let events = run(&[output]);
            let content: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::Content(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(content, output, "{output:?}");
            assert!(calls(&events).is_empty(), "{output:?}");
        }
    }

    fn run_alone(pieces: &[&str]) -> Vec<Event> {
        let mut assembler = Assembler::object_or_list(0);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.finish(&mut out);
        out.drain()
    }

    #[test]
    fn a_lone_object_is_one_call_and_what_follows_it_is_wrapping() {
        let output = "  {\"name\": \"f\", \"parameters\": {\"a\": 1}}\n";
        let events = run_alone(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(0, "f".to_string(), r#"{"a": 1}"#.to_string())]
        );
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(dropped, ["  ", "\n"]);
        let output = "{\"name\": \"f\", \"arguments\": {}} tail";
        let events = run_alone(&[output]);
        assert_eq!(bytes(&events), output);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::Other(why) }
                if text.text == "tail" && why == TEXT_AFTER_THE_OBJECT
        )));
        // A list reads as before, prose is content, and a cut object closes nothing.
        assert_eq!(calls(&run_alone(&[LIST])).len(), 2);
        assert!(calls(&run_alone(&["Sure: {\"name\": \"f\"}"])).is_empty());
        let events = run_alone(&["{\"name\": \"f\", \"argu"]);
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallEnd { .. })));
        assert_eq!(bytes(&events), "{\"name\": \"f\", \"argu");
        // The list-only assembler keeps a `{` as prose.
        let events = run(&["{\"name\": \"f\", \"arguments\": {}}"]);
        assert!(calls(&events).is_empty());
        let whole = calls(&run_alone(&[output]));
        for cut in 1..output.len() {
            let events = run_alone(&[&output[..cut], &output[cut..]]);
            assert_eq!(bytes(&events), output, "cut at {cut}");
            assert_eq!(calls(&events), whole, "cut at {cut}");
        }
    }

    #[test]
    fn between_markers_text_in_place_of_the_list_is_malformed() {
        for output in [
            "not a list",
            "  [Note: draft] ok",
            "{\"name\": \"f\", \"arguments\": {}}",
        ] {
            let mut assembler = Assembler::framed(0);
            let mut out = Events::new();
            assembler.feed(output, &mut out);
            assembler.close(&mut out);
            let events = out.drain();
            assert_eq!(bytes(&events), output, "{output:?}");
            assert!(calls(&events).is_empty(), "{output:?}");
            assert!(
                events
                    .iter()
                    .all(|event| matches!(event, Event::Malformed { .. })),
                "{output:?}: {events:?}"
            );
        }
        // The list itself reads the same between markers.
        let mut assembler = Assembler::framed(0);
        let mut out = Events::new();
        assembler.feed(LIST, &mut out);
        assembler.close(&mut out);
        let events = out.drain();
        assert_eq!(bytes(&events), LIST);
        assert_eq!(calls(&events).len(), 2);
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        let whole = calls(&run(&[LIST]));
        for cut in 1..LIST.len() {
            let events = run(&[&LIST[..cut], &LIST[cut..]]);
            assert_eq!(bytes(&events), LIST, "cut at {cut}");
            assert_eq!(calls(&events), whole, "cut at {cut}");
        }
    }
}
