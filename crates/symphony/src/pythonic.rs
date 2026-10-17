//! Pythonic calls: the events of the calls a model writes as Python, `name(key=value, ...)`.
//!
//! Olmo 3 writes a turn's calls between `<function_calls>` and `</function_calls>`, one per line:
//!
//! ```text
//! <function_calls>get_weather(city="Paris", days=3)
//! get_stock_price()</function_calls>
//! ```
//!
//! LFM2.5 writes them as a Python list between `<|tool_call_start|>` and `<|tool_call_end|>`:
//!
//! ```text
//! <|tool_call_start|>[get_weather(city='Paris', days=3), get_stock_price()]<|tool_call_end|>
//! ```
//!
//! Llama 3.2's pythonic template writes the list with no markers at all. The engine's table owns
//! the markers; this assembler reads what lies between them: one call or several, bare or in a
//! list, and gives each its own index.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    ids::CallIds,
    tagged::{
        end_cut_call,
        value::{json, python_literal},
    },
};

/// How a template frames the calls of one turn, which the grammar derived from the table writes
/// and the assembler, which reads either, does not need: Olmo 3 writes them one per line between
/// its markers, LFM2.5 as a Python list, `[` and `]` around calls joined by `, `.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// One call per line: `get_weather(city="Paris")\nget_stock_price()`.
    OnePerLine,
    /// A Python list: `[get_weather(city='Paris'), get_stock_price()]`.
    List,
}

/// How a template writes a value, which the grammar derived from the table writes and the
/// assembler, which reads both, does not need: LFM2.5's template writes Python (`True`, `None`,
/// a string in either quote); Olmo 3's writes every value through `tojson`, so JSON (`true`,
/// `null`, a string in double quotes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Values {
    /// `True`, `False`, `None`, `'Paris'` or `"Paris"`.
    Python,
    /// `true`, `false`, `null`, `"Paris"`.
    Json,
}

/// How a template spells its pythonic calls: how it frames them and how it writes a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dialect {
    pub framing: Framing,
    pub values: Values,
}

impl Dialect {
    /// Olmo 3: one call per line, each value as `tojson` writes it.
    pub const OLMO3: Self = Self {
        framing: Framing::OnePerLine,
        values: Values::Json,
    };
    /// LFM2.5: a Python list of calls with Python values.
    pub const LFM2_5: Self = Self {
        framing: Framing::List,
        values: Values::Python,
    };
}

/// The events of the pythonic calls in one region, each call with its own index from `first` and
/// its id from the engine's [`CallIds`].
#[derive(Clone, Debug)]
pub struct Assembler {
    /// The index of the next call to start.
    next_index: u32,
    /// Where the calls' ids count from.
    ids: CallIds,
    /// The current call's index, once it started.
    index: u32,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// Members written to the current call's arguments object so far.
    written: u32,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Before a call's name: the list's brackets, separators and whitespace pass as wrapping.
    Between,
    /// Inside a name: `carried` holds it so far; `spaced` once whitespace followed it, after which
    /// only `(` can continue the call.
    Name { spaced: bool },
    /// After `(` or an argument's `,`: a key or `)` comes next.
    Arguments,
    /// Inside a key: the key is `carried[start..]`, less whitespace before the `=`; `spaced` once
    /// whitespace followed it.
    Key { start: usize, spaced: bool },
    /// Inside a value: its text is `carried[start..]`, scanned for its extent.
    Value {
        key: String,
        start: usize,
        scan: Extent,
    },
}

/// Where a value ends: at a `,` or `)` outside every string and bracket.
#[derive(Clone, Copy, Debug, Default)]
struct Extent {
    depth: u32,
    quote: Option<char>,
    escaped: bool,
}

impl Extent {
    /// Reads one character and says whether it ends the value (it is then not the value's).
    fn ends_at(&mut self, c: char) -> bool {
        if let Some(quote) = self.quote {
            if self.escaped {
                self.escaped = false;
            } else if c == '\\' {
                self.escaped = true;
            } else if c == quote {
                self.quote = None;
            }
            return false;
        }
        match c {
            '\'' | '"' => self.quote = Some(c),
            '(' | '[' | '{' => self.depth += 1,
            ')' | ']' | '}' if self.depth > 0 => self.depth -= 1,
            ',' | ')' if self.depth == 0 => return true,
            _ => {}
        }
        false
    }
}

impl Assembler {
    /// An assembler for the calls of one region, the first taking index `first`, their ids
    /// counting from `ids`.
    pub fn new(first: u32, ids: CallIds) -> Self {
        Self {
            next_index: first,
            ids,
            index: first,
            carried: String::new(),
            stage: Stage::Between,
            written: 0,
        }
    }

    /// Append the next bytes of the region and push the events they complete. Every byte is the
    /// region's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        for c in bytes.chars() {
            self.take(c, out);
        }
    }

    /// The region's end: its closing marker, or the next opener. A call still open is one the
    /// model never finished, since every template writes the call's `)` before the marker: it
    /// ends as it does at the stream's end, with the fragments it sent, nothing made up for the
    /// object the model never closed, and one `Malformed` event that carries what was held, a
    /// key, a value or a separator cut short, or no bytes ([`end_cut_call`]; the engine names the
    /// reason for a region a marker closed). A name that never got its `(` comes back as
    /// `Malformed`, and the wrapping before any call is dropped. The marker that closed the
    /// region is not passed: it belongs to the region, not to the last call, and the engine
    /// drops it.
    pub fn close(mut self, out: &mut Events) {
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between => self.drop_carried(out),
            Stage::Name { .. } => self.report(MalformedReason::NameWithoutACall, out),
            Stage::Arguments | Stage::Key { .. } | Stage::Value { .. } => {
                end_cut_call(self.index, std::mem::take(&mut self.carried), "", out);
            }
        }
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A call that started still ends, with no bytes of its own, as every assembler ends
    /// one and as [`Self::close`] ends a call the marker cut, after the one `Malformed {
    /// UnterminatedRegion }` event that says its arguments are no whole value
    /// ([`end_cut_call`]): it carries what was held, a key, a value or a separator cut short, or
    /// no bytes when the cut came right after the `(`, so that a call the end cut before its `)`
    /// is never handed over as finished without a word. Outside a call, what was held, a name
    /// that never got its `(` or the list's wrapping, comes back as `Malformed {
    /// UnterminatedRegion }` too.
    pub fn finish(mut self, out: &mut Events) {
        let held = std::mem::take(&mut self.carried);
        if matches!(
            self.stage,
            Stage::Arguments | Stage::Key { .. } | Stage::Value { .. }
        ) {
            end_cut_call(self.index, held, "", out);
        } else if !held.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(held),
                why: MalformedReason::UnterminatedRegion,
            });
        }
    }

    fn take(&mut self, c: char, out: &mut Events) {
        match &mut self.stage {
            Stage::Between => match c {
                c if c.is_whitespace() || c == '[' || c == ']' || c == ',' => self.carried.push(c),
                c if c.is_alphanumeric() || c == '_' => {
                    self.drop_carried(out);
                    self.carried.push(c);
                    self.stage = Stage::Name { spaced: false };
                }
                c => {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(c.to_string()),
                        why: MalformedReason::TextBetweenCalls,
                    });
                }
            },
            Stage::Name { spaced } => match c {
                c if !*spaced && (c.is_alphanumeric() || c == '_' || c == '.' || c == '-') => {
                    self.carried.push(c);
                }
                // Python allows whitespace between the name and its `(`.
                c if c.is_whitespace() => {
                    *spaced = true;
                    self.carried.push(c);
                }
                '(' => {
                    let source = std::mem::take(&mut self.carried);
                    let name = source.trim_end().to_string();
                    self.index = self.next_index;
                    self.next_index += 1;
                    self.written = 0;
                    out.push(Event::ToolCallStart {
                        index: self.index,
                        id: self.ids.id(self.index),
                        name,
                        source: Text::uncounted(format!("{source}(")),
                    });
                    self.stage = Stage::Arguments;
                }
                c => {
                    // Not a call after all: the name's bytes are reported, and the character is
                    // read where a call could begin.
                    self.report(MalformedReason::NameWithoutACall, out);
                    self.stage = Stage::Between;
                    self.take(c, out);
                }
            },
            Stage::Arguments => match c {
                c if c.is_whitespace() => self.carried.push(c),
                ')' => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.end_call(source, out);
                    self.stage = Stage::Between;
                }
                c if c.is_alphanumeric() || c == '_' => {
                    let start = self.carried.len();
                    self.carried.push(c);
                    self.stage = Stage::Key {
                        start,
                        spaced: false,
                    };
                }
                // A positional argument, a spread, or anything else where a key should begin.
                c => {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(c.to_string()),
                        why: MalformedReason::TextWhereAnArgumentShouldBegin,
                    });
                }
            },
            Stage::Key { start, spaced } => match c {
                c if !*spaced && (c.is_alphanumeric() || c == '_') => self.carried.push(c),
                // Python allows whitespace around the `=`; the value's extent trims its own.
                c if c.is_whitespace() => {
                    *spaced = true;
                    self.carried.push(c);
                }
                '=' => {
                    let key = self.carried[*start..].trim_end().to_string();
                    self.carried.push('=');
                    let start = self.carried.len();
                    self.stage = Stage::Value {
                        key,
                        start,
                        scan: Extent::default(),
                    };
                }
                c => {
                    self.report(MalformedReason::KeyWithoutAValue, out);
                    self.stage = Stage::Arguments;
                    self.take(c, out);
                }
            },
            Stage::Value { key, start, scan } => {
                if scan.ends_at(c) {
                    let (key, start) = (key.clone(), *start);
                    self.write_value(&key, start, out);
                    if c == ')' {
                        self.end_call(Text::uncounted(String::new()), out);
                        self.stage = Stage::Between;
                    } else {
                        self.carried.push(',');
                        self.stage = Stage::Arguments;
                    }
                } else {
                    self.carried.push(c);
                }
            }
        }
    }

    /// The value's extent is complete: `"key": <json>` with the key, the `=`, the value and the
    /// whitespace before them as its source.
    fn write_value(&mut self, key: &str, start: usize, out: &mut Events) {
        let text = self.carried[start..].trim().to_string();
        let value = python_literal(&text)
            .filter(|json| serde_json::from_str::<serde_json::Value>(json).is_ok())
            .unwrap_or_else(|| json(&text, None));
        let mut fragment = String::with_capacity(key.len() + value.len() + 8);
        fragment.push_str(if self.written == 0 { "{" } else { ", " });
        fragment.push_str(&serde_json::Value::String(key.to_string()).to_string());
        fragment.push_str(": ");
        fragment.push_str(&value);
        self.written += 1;
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: fragment,
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
    }

    /// The call's own `)`: `}` (or `{}`) with `source`, then `ToolCallEnd` with the parenthesis.
    /// It is the one end that closes a call's object; a region's marker or the stream's end
    /// never does.
    fn end_call(&mut self, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: if self.written == 0 { "{}" } else { "}" }.to_string(),
            source,
        });
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(")"),
        });
    }

    fn report(&mut self, why: MalformedReason, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Malformed {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why,
        });
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

    /// The region's marker belongs to the region, not to its last call.
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
    /// How a region ends: its closing marker (`close`) or the stream's end (`finish`).
    #[derive(Clone, Copy, Debug)]
    enum Ending {
        Close,
        Stream,
    }

    impl Ending {
        fn run(self, pieces: &[&str]) -> Vec<Event> {
            let mut assembler = Assembler::new(0, ids());
            let mut out = Events::new();
            for piece in pieces {
                assembler.feed(piece, &mut out);
            }
            match self {
                Ending::Close => assembler.close(&mut out),
                Ending::Stream => assembler.finish(&mut out),
            }
            out.drain()
        }
    }

    /// The region closed by its marker after `pieces`.
    fn run(pieces: &[&str]) -> Vec<Event> {
        Ending::Close.run(pieces)
    }

    fn dropped(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The text of every `Malformed` event, whatever its reason.
    fn malformed(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
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

    const OLMO: &str = concat!(
        "get_weather(city=\"Paris\", days=3, ",
        "prefs={\"size\": \"large\", \"deep\": [1, 2.5, true, null]})\n",
        "get_stock_price()"
    );
    const OLMO_ARGUMENTS: &str = concat!(
        r#"{"city": "Paris", "days": 3, "#,
        r#""prefs": {"size": "large", "deep": [1, 2.5, true, null]}}"#
    );
    const LFM: &str = "[get_weather(city='Paris', days=3, note='it\\'s fine'), get_stock_price()]";

    #[test]
    fn olmos_bare_calls_give_two_calls_with_their_values_and_every_byte() {
        let events = run(&[OLMO]);
        assert_eq!(bytes(&events), OLMO);
        assert_eq!(
            calls(&events),
            [
                (0, "get_weather".to_string(), OLMO_ARGUMENTS.to_string()),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        assert_eq!(
            events[0],
            Event::ToolCallStart {
                index: 0,
                id: ids().id(0),
                name: "get_weather".into(),
                source: Text::uncounted("get_weather("),
            }
        );
    }

    #[test]
    fn lfms_list_gives_the_same_calls_with_the_brackets_and_commas_dropped() {
        let events = run(&[LFM]);
        assert_eq!(bytes(&events), LFM);
        assert_eq!(
            calls(&events),
            [
                (
                    0,
                    "get_weather".to_string(),
                    r#"{"city": "Paris", "days": 3, "note": "it's fine"}"#.to_string()
                ),
                (1, "get_stock_price".to_string(), "{}".to_string()),
            ]
        );
        assert_eq!(dropped(&events), ["[", ", ", "]"]);
    }

    #[test]
    fn every_chunking_says_the_same_and_accounts_for_every_byte() {
        for text in [OLMO, LFM] {
            let whole = calls(&run(&[text]));
            for cut in 1..text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let events = run(&[&text[..cut], &text[cut..]]);
                assert_eq!(bytes(&events), text, "cut at {cut}");
                assert_eq!(calls(&events), whole, "cut at {cut}");
            }
        }
    }

    #[test]
    fn a_call_either_ending_cuts_inside_a_value_closes_nothing_and_says_so() {
        for ending in [Ending::Close, Ending::Stream] {
            let events = ending.run(&["get_weather(city=\"Par"]);
            assert_eq!(bytes(&events), "get_weather(city=\"Par", "{ending:?}");
            // The call ends with no bytes of its own and no fragment: its one word carries the
            // value cut short.
            assert!(
                matches!(
                    &events[..],
                    [
                        Event::ToolCallStart { index: 0, .. },
                        Event::Malformed { why: MalformedReason::UnterminatedRegion, .. },
                        Event::ToolCallEnd { index: 0, source },
                    ] if source.text.is_empty()
                ),
                "{ending:?}: {events:?}"
            );
            assert_eq!(said(&events), ["city=\"Par"], "{ending:?}");
        }
    }

    #[test]
    fn prose_where_a_call_should_begin_is_malformed_and_the_calls_around_it_still_come() {
        let text = "I will call:\nget_weather(city=\"Paris\")";
        let events = run(&[text]);
        assert_eq!(bytes(&events), text);
        assert_eq!(calls(&events).len(), 1);
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::Malformed { .. })));
    }

    #[test]
    fn pythons_whitespace_around_the_parenthesis_and_the_equals_sign_is_allowed() {
        let output = "get_weather (city = \"Paris\", days= 3 )";
        let events = run(&[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [(
                0,
                "get_weather".to_string(),
                r#"{"city": "Paris", "days": 3}"#.to_string()
            )]
        );
    }

    #[test]
    fn a_positional_argument_is_reported_as_text_where_an_argument_should_begin() {
        let events = run(&["get_weather(\"Paris\", days=3)"]);
        assert_eq!(
            calls(&events),
            [(0, "get_weather".to_string(), r#"{"days": 3}"#.to_string())]
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed {
                why: MalformedReason::TextWhereAnArgumentShouldBegin,
                ..
            }
        )));
    }

    fn by_the_character(text: &str) -> Vec<String> {
        text.chars().map(String::from).collect()
    }

    /// The words the end said: the text of every `Malformed { UnterminatedRegion }` event.
    fn said(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed {
                    text,
                    why: MalformedReason::UnterminatedRegion,
                } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_call_either_ending_cuts_right_after_its_parenthesis_says_so_before_its_end() {
        let start = |index: u32, name: &str| Event::ToolCallStart {
            index,
            id: ids().id(index),
            name: name.to_string(),
            source: Text::uncounted(format!("{name}(")),
        };
        let word = Event::Malformed {
            text: Text::uncounted(""),
            why: MalformedReason::UnterminatedRegion,
        };
        let end = |index: u32| Event::ToolCallEnd {
            index,
            source: Text::uncounted(""),
        };
        for ending in [Ending::Close, Ending::Stream] {
            assert_eq!(
                ending.run(&["[get_weather("]),
                [
                    Event::Dropped {
                        text: Text::uncounted("["),
                        why: DropReason::Wrapper,
                    },
                    start(0, "get_weather"),
                    word.clone(),
                    end(0),
                ],
                "{ending:?}"
            );
            // The same by the character, for a second call cut there: the first stands whole.
            let text = "[get_weather(city='Paris'), get_stock_price(";
            let pieces = by_the_character(text);
            let events = ending.run(&pieces.iter().map(String::as_str).collect::<Vec<_>>());
            assert_eq!(bytes(&events), text, "{ending:?}");
            assert_eq!(
                calls(&events),
                [
                    (
                        0,
                        "get_weather".to_string(),
                        r#"{"city": "Paris"}"#.to_string()
                    ),
                    (1, "get_stock_price".to_string(), String::new()),
                ],
                "{ending:?}"
            );
            assert_eq!(
                events[events.len() - 3..],
                [start(1, "get_stock_price"), word.clone(), end(1)],
                "{ending:?}"
            );
        }
    }

    /// What an ending finds at the cut: a call open (the one word carries what was held), whole
    /// calls alone, the list's wrapping after a whole call, or a name that never got its `(`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Found {
        Open,
        Whole,
        Wrapping,
        Name,
    }

    #[test]
    fn either_ending_says_so_wherever_it_cuts_a_call_and_carries_what_was_held() {
        // Where the ending cuts, what was held there, and what it finds.
        let cuts: &[(&str, &str, Found)] = &[
            ("[get_weather", "get_weather", Found::Name),
            ("[get_weather(", "", Found::Open),
            ("[get_weather( ", " ", Found::Open),
            ("[get_weather(ci", "ci", Found::Open),
            ("[get_weather(city ", "city ", Found::Open),
            ("[get_weather(city=", "city=", Found::Open),
            ("[get_weather(city='Par", "city='Par", Found::Open),
            ("[get_weather(city='Paris'", "city='Paris'", Found::Open),
            ("[get_weather(city='Paris',", ",", Found::Open),
            ("[get_weather(city='Paris', ", ", ", Found::Open),
            ("[get_weather(city='Paris', da", ", da", Found::Open),
            (
                "[get_weather(prefs={'a': [1, ",
                "prefs={'a': [1, ",
                Found::Open,
            ),
            ("[get_weather(city='Paris')", "", Found::Whole),
            ("[get_weather(city='Paris'), ", ", ", Found::Wrapping),
            (
                "[get_weather(city='Paris'), get_stock_price(",
                "",
                Found::Open,
            ),
        ];
        for ending in [Ending::Close, Ending::Stream] {
            for (text, held, found) in cuts {
                let pieces = by_the_character(text);
                let whole = ending.run(&[text]);
                let by_the_character =
                    ending.run(&pieces.iter().map(String::as_str).collect::<Vec<_>>());
                for events in [whole, by_the_character] {
                    let place = format!("{ending:?} after {text:?}");
                    assert_eq!(bytes(&events), *text, "{place}");
                    let starts = events
                        .iter()
                        .filter(|event| matches!(event, Event::ToolCallStart { .. }))
                        .count();
                    let ends = events
                        .iter()
                        .filter(|event| matches!(event, Event::ToolCallEnd { .. }))
                        .count();
                    assert_eq!(starts, ends, "{place}: every call that started ends");
                    match (found, ending) {
                        (Found::Open, _) => {
                            assert_eq!(said(&events), [*held], "{place}: one word, what was held");
                            assert!(
                                matches!(
                                    &events[events.len() - 2..],
                                    [Event::Malformed { .. }, Event::ToolCallEnd { source, .. }]
                                        if source.text.is_empty()
                                ),
                                "{place}: the word comes right before an end of no bytes"
                            );
                            // Nothing is made up: a cut call's fragments never close its object,
                            // so they are no whole JSON value (a member's own `}` may end them).
                            assert!(
                                calls(&events).last().is_some_and(|(_, _, arguments)| {
                                    serde_json::from_str::<serde_json::Value>(arguments).is_err()
                                }),
                                "{place}: the cut call's arguments stay open"
                            );
                        }
                        (Found::Whole, _) => {
                            assert!(malformed(&events).is_empty(), "{place}: no word");
                        }
                        // The wrapping after a whole call is the template's at the close and a
                        // cut region's bytes at the stream's end.
                        (Found::Wrapping, Ending::Close) => {
                            assert!(malformed(&events).is_empty(), "{place}: no word");
                            assert!(dropped(&events).contains(held), "{place}: wrapping dropped");
                        }
                        (Found::Wrapping, Ending::Stream) => {
                            assert_eq!(said(&events), [*held], "{place}: the held wrapping");
                        }
                        // A name that never got its `(` is no call either way, and the close
                        // names its reason.
                        (Found::Name, Ending::Close) => {
                            assert_eq!(malformed(&events), [*held], "{place}: the name");
                            assert!(said(&events).is_empty(), "{place}: the close's own reason");
                        }
                        (Found::Name, Ending::Stream) => {
                            assert_eq!(said(&events), [*held], "{place}: the name");
                        }
                    }
                    if *found != Found::Open {
                        assert!(
                            calls(&events)
                                .iter()
                                .all(|(_, _, arguments)| arguments.ends_with('}')),
                            "{place}: every call is whole"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_call_either_ending_cuts_anywhere_keeps_the_fragments_it_sent_and_nothing_more() {
        let whole = calls(&run(&[LFM]));
        for (ending, at) in [Ending::Close, Ending::Stream]
            .into_iter()
            .flat_map(|ending| (1..LFM.len()).map(move |at| (ending, at)))
        {
            if !LFM.is_char_boundary(at) {
                continue;
            }
            let text = &LFM[..at];
            let events = ending.run(&[text]);
            assert_eq!(bytes(&events), text, "{ending:?} at {at}");
            for (index, name, arguments) in calls(&events) {
                let (_, whole_name, whole_arguments) = &whole[index as usize];
                assert_eq!(&name, whole_name, "cut at {at}");
                // A call's fragments are a prefix of what the whole call gives, and the one word
                // stands between a cut call's start and its end, never inside a whole one.
                assert!(
                    whole_arguments.starts_with(&arguments),
                    "cut at {at}: {arguments:?} is no prefix of {whole_arguments:?}"
                );
                let start = events
                    .iter()
                    .position(|event| matches!(event, Event::ToolCallStart { index: i, .. } if *i == index))
                    .unwrap_or_else(|| panic!("cut at {at}: call {index} has no start"));
                let end = events
                    .iter()
                    .position(
                        |event| matches!(event, Event::ToolCallEnd { index: i, .. } if *i == index),
                    )
                    .unwrap_or_else(|| panic!("cut at {at}: call {index} has no end"));
                assert_eq!(
                    said(&events[start..end]).len(),
                    usize::from(&arguments != whole_arguments),
                    "cut at {at}: a cut call says so once, a whole one not at all"
                );
            }
        }
    }
}
