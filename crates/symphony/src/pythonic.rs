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
//! What the syntax says, and what the assembler does with it:
//!
//! - **A call is a name, `(`, arguments, `)`.** The name's bytes and the `(` are `ToolCallStart`'s
//!   source; the `)` is `ToolCallEnd`'s. A list's brackets, the commas between calls and the
//!   whitespace around them are the template's and come back as `Dropped { Wrapper }`; any other
//!   text where a call should begin is `Malformed`.
//! - **Each argument is `key=value`.** The value is a Python expression the template wrote for a
//!   JSON value: a string in either quote, a number, `True`, `False`, `None`, a list, a tuple, a
//!   dict, or the JSON a template writes with `tojson` for a nested value. Its extent is known
//!   when a `,` or the call's `)` arrives outside every string and bracket, so a value is held
//!   until then and written whole, through the value module's literal reader; a value that is
//!   not a literal is a string holding its text. Streaming a long string value as it arrives is
//!   a later step: the templates that write Python write short values, and the extent rule is
//!   what makes every chunking say the same.
//! - **Every byte lands in exactly one event**, as in the other assemblers: the key, the `=`, the
//!   value and the separator after it in the fragment they produce.
//! - **Two endings.** [`Assembler::close`] is the region's closing marker (or the next opener):
//!   a call still open is closed for the client, an open argument's bytes come back as
//!   `Malformed`, and the bytes before any call as well. [`Assembler::finish`] is a stream that
//!   was cut: nothing is closed, and what was held comes back as
//!   `Malformed { UnterminatedRegion }`.
//!
//! Ported in spirit from the old crate's pythonic parser, which compiled a regular expression per
//! call and read the arguments with a Python-literal evaluator at the end; here the arguments are
//! read as the stream arrives, at the value's extent.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    tagged::value::{json, python_literal},
};

const TEXT_BETWEEN_CALLS: &str = "text where a call should begin";
const NAME_WITHOUT_A_CALL: &str = "a name without a call";
const ARGUMENT_CUT_SHORT: &str = "an argument the call's end cut short";
const KEY_WITHOUT_A_VALUE: &str = "a key without a value";

/// The events of the pythonic calls in one region, each call with its own index from `first`.
#[derive(Clone, Debug)]
pub struct Assembler {
    /// The index of the next call to start.
    next_index: u32,
    /// The current call's index, once it started.
    index: u32,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// Members written to the current call's arguments object so far.
    written: u32,
    done: bool,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Before a call's name: the list's brackets, separators and whitespace pass as wrapping.
    Between,
    /// Inside a name: `carried` holds it so far.
    Name,
    /// After `(` or an argument's `,`: a key or `)` comes next.
    Arguments,
    /// Inside a key: the key is `carried[start..]`.
    Key { start: usize },
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
    /// An assembler for the calls of one region, the first taking index `first`.
    pub fn new(first: u32) -> Self {
        Self {
            next_index: first,
            index: first,
            carried: String::new(),
            stage: Stage::Between,
            written: 0,
            done: false,
        }
    }

    /// Append the next bytes of the region and push the events they complete. Every byte is the
    /// region's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        if self.done {
            return;
        }
        for c in bytes.chars() {
            self.take(c, out);
        }
    }

    /// The region's end: a call still open is closed for the client, and bytes that made no call
    /// come back as `Malformed`. `terminal` is the marker that closed the region; the engine
    /// drops it, since it belongs to the region and not to the last call.
    pub fn close(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        match std::mem::replace(&mut self.stage, Stage::Between) {
            Stage::Between => self.drop_carried(out),
            Stage::Name => self.report(NAME_WITHOUT_A_CALL, out),
            Stage::Arguments => {
                let source = Text::uncounted(std::mem::take(&mut self.carried));
                self.end_call(source, "", out);
            }
            Stage::Key { .. } => {
                self.report(KEY_WITHOUT_A_VALUE, out);
                self.end_call(Text::uncounted(String::new()), "", out);
            }
            Stage::Value { .. } => {
                self.report(ARGUMENT_CUT_SHORT, out);
                self.end_call(Text::uncounted(String::new()), "", out);
            }
        }
        self.done = true;
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client; what was held comes back as `Malformed { UnterminatedRegion }`.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        if !self.carried.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(std::mem::take(&mut self.carried)),
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
                    self.stage = Stage::Name;
                }
                c => {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(c.to_string()),
                        why: MalformedReason::Other(TEXT_BETWEEN_CALLS.to_string()),
                    });
                }
            },
            Stage::Name => match c {
                c if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' => {
                    self.carried.push(c);
                }
                '(' => {
                    let name = std::mem::take(&mut self.carried);
                    self.index = self.next_index;
                    self.next_index += 1;
                    self.written = 0;
                    out.push(Event::ToolCallStart {
                        index: self.index,
                        id: format!("call_{}", self.index),
                        name: name.clone(),
                        source: Text::uncounted(format!("{name}(")),
                    });
                    self.stage = Stage::Arguments;
                }
                c => {
                    // Not a call after all: the name's bytes are reported, and the character is
                    // read where a call could begin.
                    self.report(NAME_WITHOUT_A_CALL, out);
                    self.stage = Stage::Between;
                    self.take(c, out);
                }
            },
            Stage::Arguments => match c {
                c if c.is_whitespace() => self.carried.push(c),
                ')' => {
                    let source = Text::uncounted(std::mem::take(&mut self.carried));
                    self.end_call(source, ")", out);
                    self.stage = Stage::Between;
                }
                c if c.is_alphanumeric() || c == '_' => {
                    let start = self.carried.len();
                    self.carried.push(c);
                    self.stage = Stage::Key { start };
                }
                c => {
                    self.drop_carried(out);
                    out.push(Event::Malformed {
                        text: Text::uncounted(c.to_string()),
                        why: MalformedReason::Other(TEXT_BETWEEN_CALLS.to_string()),
                    });
                }
            },
            Stage::Key { start } => match c {
                c if c.is_alphanumeric() || c == '_' => self.carried.push(c),
                '=' => {
                    let key = self.carried[*start..].to_string();
                    self.carried.push('=');
                    let start = self.carried.len();
                    self.stage = Stage::Value {
                        key,
                        start,
                        scan: Extent::default(),
                    };
                }
                c => {
                    self.report(KEY_WITHOUT_A_VALUE, out);
                    self.stage = Stage::Arguments;
                    self.take(c, out);
                }
            },
            Stage::Value { key, start, scan } => {
                if scan.ends_at(c) {
                    let (key, start) = (key.clone(), *start);
                    self.write_value(&key, start, out);
                    if c == ')' {
                        self.end_call(Text::uncounted(String::new()), ")", out);
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

    /// `}` (or `{}`) with `source`, then `ToolCallEnd` with the closing parenthesis.
    fn end_call(&mut self, source: Text, close: &str, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json: if self.written == 0 { "{}" } else { "}" }.to_string(),
            source,
        });
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(close),
        });
    }

    fn report(&mut self, why: &str, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Malformed {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: MalformedReason::Other(why.to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[&str]) -> Vec<Event> {
        let mut assembler = Assembler::new(0);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.close(&mut out);
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
                id: "call_0".into(),
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
    fn a_call_cut_by_the_regions_end_is_closed_and_a_cut_stream_closes_nothing() {
        let events = run(&["get_weather(city=\"Par"]);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { why: MalformedReason::Other(why), .. } if why == ARGUMENT_CUT_SHORT
        )));
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::ToolCallEnd { .. })));
        assert_eq!(bytes(&events), "get_weather(city=\"Par");

        let mut assembler = Assembler::new(3);
        let mut out = Events::new();
        assembler.feed("get_weather(city=\"Par", &mut out);
        assembler.finish(&mut out);
        let events = out.drain();
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
        assert!(matches!(&events[0], Event::ToolCallStart { index: 3, .. }));
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
}
