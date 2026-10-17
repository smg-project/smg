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
//!   commas between objects, the whitespace around them, the `]` and whitespace that ends the
//!   output after it are the template's and come back as `Dropped { Wrapper }`; any other text
//!   between objects is `Malformed`, one event per run, and so is text after the `]`, with the
//!   whitespace before and between its words: once text follows the list, every byte after the
//!   `]` is the model's prose, which the template's reference keeps as written.
//! - Anything else makes the whole output content: that byte and every later one come back as
//!   `Content`, since the model answered in prose. The whitespace held before it comes back as
//!   content too. A `[` that is followed by anything but an object or the `]` is prose as well
//!   (`[Note: draft]`, a markdown link), and the `[` comes back as content with the rest.
//!
//! With [`Framing::ObjectOrList`] (the `json` table), a `{` at the first byte is one call
//! alone: the object goes to a [`json::Assembler`](super::Assembler) as a list's object does, and
//! what follows it reads as what follows a list's `]`: whitespace that ends the output is the
//! template's wrapping, dropped; text, with the whitespace before and between its words, is
//! `Malformed`. A list reads as before. [`Framing::Bare`] (the `xlam` table) keeps a `{` as
//! prose.
//!
//! A region with no markers never closes by one, so [`Assembler::close`] is for a table that puts
//! the list between markers after all; [`Assembler::finish`] is the end of the stream: an object
//! still open is finished as the JSON assembler finishes one (a started call closed with what
//! arrived, an object that never named a call `Malformed`), and a list left open reports what
//! it held.
//!
//! Between markers (Llama Nemotron's `<TOOLCALL>[…]</TOOLCALL>`) a row of the table opens the
//! region and its closing marker ends it, so prose has no place inside: [`Framing::Framed`]
//! reports text where the list should be, a `[` that opens no list of calls included, as
//! `Malformed` where the bare list takes it for content. The engine picks it for a table with a
//! row into the arguments state.
//!
//! The three are the one [`Framing`] the assembler takes at construction, and the engine's factory
//! maps the table's call syntax to it; the object reader and the stages around it are the same
//! under every framing.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    ids::CallIds,
    json,
};

/// The events of a JSON list of calls, each call with its own index from `first` and its id from
/// the engine's [`CallIds`].
#[derive(Debug)]
pub struct Assembler {
    next_index: u32,
    /// Where the calls' ids count from.
    ids: CallIds,
    stage: Stage,
    /// Bytes no event has accounted for yet; the next event takes them as its source.
    carried: String,
    /// Objects that began, so that prose after the `[` is told from text between objects.
    objects: u32,
    /// What frames the calls: what a `{` at the first byte is, and what text in place of a list
    /// is.
    framing: Framing,
    /// Whether a `[` opened a list, so that a finished object returns to the list or ends the
    /// output's calls.
    listed: bool,
    /// Whether text followed the lone object or the list's `]`: from there on every byte,
    /// whitespace included, is the model's prose after its calls. Before it, whitespace is held,
    /// the template's wrapping if nothing follows.
    prose_after: bool,
}

/// What frames the calls: how the first byte that is not whitespace reads, and what text in place
/// of a list is. The engine's factory maps the table's call syntax to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// The list alone in the output (xLAM): a `[` opens it, and anything else is prose.
    Bare,
    /// The list between the table's markers (Llama Nemotron): a `[` opens it, and anything else
    /// is malformed, since prose has no place inside the markers.
    Framed,
    /// One object or a list alone in the output (the `json` table): a `{` is one call alone, a
    /// `[` opens the list, and anything else is prose.
    ObjectOrList,
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
    /// An assembler positioned at the output's start, or after the marker that opens the list's
    /// region, the first call taking index `first` and the ids counting from `ids`; `framing` says
    /// what frames the calls.
    pub fn new(first: u32, ids: CallIds, framing: Framing) -> Self {
        Self {
            next_index: first,
            ids,
            stage: Stage::Deciding,
            carried: String::new(),
            objects: 0,
            framing,
            listed: false,
            prose_after: false,
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
            Stage::Deciding | Stage::Content => {
                if !self.carried.is_empty() {
                    out.push(Event::Content(Text::uncounted(std::mem::take(
                        &mut self.carried,
                    ))));
                }
            }
            // Whitespace held after the object or the list, with nothing after it: the template's
            // wrapping.
            Stage::After => self.drop_carried(out),
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
                } else if first == '{' && self.framing == Framing::ObjectOrList {
                    self.drop_carried(out);
                    let index = self.next_index;
                    self.objects += 1;
                    self.stage =
                        Stage::Object(Box::new(json::Assembler::new(index, self.ids.id(index))));
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
                    why: MalformedReason::TextInPlaceOfTheList,
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
                            self.ids.id(index),
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
                            why: MalformedReason::TextBetweenCalls,
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
                // Whitespace after the object or the list is held: the template's wrapping if
                // nothing follows it, the model's prose if text does. Once text follows, every
                // byte is prose after the calls, reported malformed with its whitespace, since
                // the template's reference keeps it as written.
                if !self.prose_after {
                    if text.chars().all(char::is_whitespace) {
                        self.carried.push_str(text);
                        return "";
                    }
                    self.prose_after = true;
                }
                self.carried.push_str(text);
                out.push(Event::Malformed {
                    text: Text::uncounted(std::mem::take(&mut self.carried)),
                    why: if self.listed {
                        MalformedReason::TextAfterTheList
                    } else {
                        MalformedReason::TextAfterTheArguments
                    },
                });
                ""
            }
        }
    }

    /// The bytes held and `text` are not a list of calls: content under the bare list, malformed
    /// between markers; every later byte comes back the same way.
    fn not_a_list(&mut self, text: &str, out: &mut Events) {
        self.carried.push_str(text);
        let text = Text::uncounted(std::mem::take(&mut self.carried));
        if self.framing == Framing::Framed {
            self.stage = Stage::Malformed;
            out.push(Event::Malformed {
                text,
                why: MalformedReason::TextInPlaceOfTheList,
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

    fn ids() -> CallIds {
        CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex")
    }

    const LIST: &str = concat!(
        "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, ",
        "{\"name\": \"get_stock_price\", \"arguments\": {}}]"
    );

    fn run(pieces: &[&str]) -> Vec<Event> {
        let mut assembler = Assembler::new(0, ids(), Framing::Bare);
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
        // Text between objects comes back as one run; text after the list too, with the
        // whitespace before it, which is the model's once text follows.
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
        assert_eq!(malformed, ["junk", " tail"]);
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
        let mut assembler = Assembler::new(0, ids(), Framing::ObjectOrList);
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
            Event::Malformed { text, why: MalformedReason::TextAfterTheArguments }
                if text.text == " tail"
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

    /// The malformed text of `events` under `why`, joined.
    fn malformed(events: &[Event], why: MalformedReason) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, why: reason } if *reason == why => {
                    Some(text.text.as_str())
                }
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

    #[test]
    fn prose_after_a_lone_object_keeps_its_whitespace_whole_and_in_pieces() {
        // A JSON answer the model explains: the reference keeps the sentence as written.
        let output = "{\n \"a\": true\n} \nA baseball game can be described as exciting.";
        let prose = " \nA baseball game can be described as exciting.";
        let whole = run_alone(&[output]);
        assert_eq!(bytes(&whole), output);
        assert_eq!(
            malformed(&whole, MalformedReason::TextAfterTheArguments),
            prose
        );
        assert!(dropped(&whole).is_empty(), "{whole:?}");
        let pieces: Vec<String> = output.chars().map(String::from).collect();
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let cut = run_alone(&refs);
        assert_eq!(bytes(&cut), output);
        assert_eq!(
            malformed(&cut, MalformedReason::TextAfterTheArguments),
            prose
        );
        assert!(dropped(&cut).is_empty(), "{cut:?}");
        // After a call too, and after a list's `]`.
        let output = "{\"name\": \"f\", \"arguments\": {}}\n\nDone, see above.";
        for events in [
            run_alone(&[output]),
            run_alone(&[
                "{\"name\": \"f\", \"arguments\": {}}\n",
                "\nDone, see above.",
            ]),
        ] {
            assert_eq!(calls(&events).len(), 1);
            assert_eq!(
                malformed(&events, MalformedReason::TextAfterTheArguments),
                "\n\nDone, see above."
            );
            assert!(dropped(&events).is_empty(), "{events:?}");
        }
        let output = "[{\"name\": \"f\", \"arguments\": {}}] and so on";
        for events in [
            run(&[output]),
            run(&["[{\"name\": \"f\", \"arguments\": {}}]", " ", "and so on"]),
        ] {
            assert_eq!(
                malformed(&events, MalformedReason::TextAfterTheList),
                " and so on"
            );
            assert_eq!(dropped(&events), ["[", "]"]);
        }
        // Whitespace alone after the object or the list stays the template's wrapping.
        for events in [
            run_alone(&["{\"name\": \"f\", \"arguments\": {}} \n"]),
            run_alone(&["{\"name\": \"f\", \"arguments\": {}}", " ", "\n"]),
        ] {
            assert_eq!(dropped(&events), [" \n"]);
            assert!(malformed(&events, MalformedReason::TextAfterTheArguments).is_empty());
        }
    }

    #[test]
    fn between_markers_text_in_place_of_the_list_is_malformed() {
        for output in [
            "not a list",
            "  [Note: draft] ok",
            "{\"name\": \"f\", \"arguments\": {}}",
        ] {
            let mut assembler = Assembler::new(0, ids(), Framing::Framed);
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
        let mut assembler = Assembler::new(0, ids(), Framing::Framed);
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
