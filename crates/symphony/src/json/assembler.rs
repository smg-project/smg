//! Events from one tool-call object as it arrives.
//!
//! [`Assembler`] takes a call's JSON object in pieces, as the format cuts it out of the model's
//! text, and turns it into the events of one call: `ToolCallStart` once the name is whole and the
//! arguments value has begun (or the object has closed without one), `ToolCallArguments` for every
//! run of new argument bytes, `ToolCallEnd` when the object closes. The fragments are the model's
//! own bytes: what a client receives, concatenated, is exactly what the model wrote as the arguments
//! value, spacing included, and no fragment ever has to be revised. The old crate had no equivalent;
//! it parsed the partial object on every chunk, re-serialized the arguments and emitted the
//! difference, which is how `{"city": "Paris"}` reached clients as `{"city":"Paris"}`.
//!
//! Every byte of the object lands in exactly one event: the head up to the arguments value is the
//! `source` of `ToolCallStart`, the argument bytes are the `source` of their fragments, the tail
//! after the value is the `source` of `ToolCallEnd`. When the arguments come before the name they
//! are held until the name is whole, since a call starts before its arguments.
//!
//! What the assembler does not decide: the call's index and id (the format mints them and passes
//! them to [`Assembler::new`]), whether the name is a declared tool, and whether a string-valued
//! arguments member should be decoded. The format decides those with the events in hand.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::outline::{outline, Span},
};

/// The events of one tool call, from the bytes of its object.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    text: String,
    started: bool,
    emitted: usize,
    done: bool,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            text: String::new(),
            started: false,
            emitted: 0,
            done: false,
        }
    }

    /// Whether the object has closed; bytes fed after that are ignored, so the format stops here.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Append the next bytes of the object and push the events they complete.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        if self.done {
            return;
        }
        self.text.push_str(bytes);
        let found = outline(&self.text);
        if !self.started {
            let Some(name) = found.name.clone() else {
                return;
            };
            let head_end = match (&found.arguments, found.complete) {
                (Some(span), _) => span.start,
                (None, true) => self.text.len(),
                (None, false) => return,
            };
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(&self.text[..head_end]),
            });
            self.started = true;
        }
        self.emit_new_argument_bytes(found.arguments.as_ref(), out);
        if found.complete {
            let tail_start = found
                .arguments
                .as_ref()
                .and_then(|span| span.end)
                .unwrap_or(self.text.len());
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(&self.text[tail_start..]),
            });
            self.done = true;
        }
    }

    /// No more bytes will come. A call that started is closed with what arrived; an object that
    /// never named its call, or never closed, comes back as `Malformed`, so its bytes are not lost.
    pub fn finish(self, out: &mut Events) {
        if self.done {
            return;
        }
        let found = outline(&self.text);
        if self.started {
            let mut this = self;
            this.emit_new_argument_bytes(found.arguments.as_ref(), out);
            let tail_start = found
                .arguments
                .as_ref()
                .and_then(|span| span.end)
                .unwrap_or(this.text.len());
            out.push(Event::ToolCallEnd {
                index: this.index,
                source: Text::uncounted(&this.text[tail_start..]),
            });
            return;
        }
        let why = if found.complete {
            MalformedReason::Other("a tool call without a name".to_string())
        } else {
            MalformedReason::UnterminatedRegion
        };
        if !self.text.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(self.text),
                why,
            });
        }
    }

    /// The argument bytes that arrived since the last fragment, as one fragment.
    fn emit_new_argument_bytes(&mut self, arguments: Option<&Span>, out: &mut Events) {
        let Some(span) = arguments else {
            return;
        };
        let bytes = span.text(&self.text);
        if bytes.len() > self.emitted {
            let fresh = &bytes[self.emitted..];
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: fresh.to_string(),
                source: Text::uncounted(fresh),
            });
            self.emitted = bytes.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CALL: &str = r#"{"name": "get_weather", "arguments": {"city": "Paris", "days": [1, 2]}}"#;
    const ARGUMENTS: &str = r#"{"city": "Paris", "days": [1, 2]}"#;

    fn run(pieces: &[&str]) -> Vec<Event> {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &mut out);
        }
        assembler.finish(&mut out);
        out.drain()
    }

    fn sources(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::Malformed { text, .. } => text.text.as_str(),
                _ => "",
            })
            .collect()
    }

    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn starts(events: &[Event]) -> Vec<(&str, &str)> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallStart { name, source, .. } => {
                    Some((name.as_str(), source.text.as_str()))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_whole_object_gives_start_arguments_and_end_with_every_byte_accounted_for() {
        let events = run(&[CALL]);
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "get_weather".into(),
                    source: Text::uncounted(r#"{"name": "get_weather", "arguments": "#),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: ARGUMENTS.into(),
                    source: Text::uncounted(ARGUMENTS),
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("}"),
                },
            ]
        );
        assert_eq!(sources(&events), CALL);
    }

    #[test]
    fn every_cut_gives_the_same_call_with_the_arguments_in_the_models_bytes() {
        let whole = run(&[CALL]);
        let mut cuts: Vec<Vec<&str>> = (1..CALL.len())
            .filter(|&cut| CALL.is_char_boundary(cut))
            .map(|cut| vec![&CALL[..cut], &CALL[cut..]])
            .collect();
        let bytes: Vec<&str> = CALL
            .char_indices()
            .map(|(i, c)| &CALL[i..i + c.len_utf8()])
            .collect();
        cuts.push(bytes);
        for pieces in cuts {
            let events = run(&pieces);
            assert_eq!(starts(&events), starts(&whole), "{pieces:?}");
            assert_eq!(arguments(&events), ARGUMENTS, "{pieces:?}");
            assert_eq!(
                sources(&events),
                CALL,
                "{pieces:?}: every byte in exactly one event"
            );
            assert_eq!(
                events.last(),
                whole.last(),
                "{pieces:?}: the same end, with the same tail"
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::ToolCallEnd { .. }))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn arguments_before_the_name_are_held_until_the_name_is_whole() {
        let text = r#"{"arguments": {"city": "Paris"}, "name": "get_weather"}"#;
        let cut = r#"{"arguments": {"city": "Paris"}, "na"#.len();
        let mut assembler = Assembler::new(1, "call_1");
        let mut out = Events::new();
        assembler.feed(&text[..cut], &mut out);
        assert!(out.is_empty(), "nothing is said before the name is known");
        assembler.feed(&text[cut..], &mut out);
        let events = out.drain();
        assert_eq!(starts(&events), vec![("get_weather", r#"{"arguments": "#)]);
        assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 1,
                source: Text::uncounted(r#", "name": "get_weather"}"#),
            })
        );
        assert_eq!(sources(&events), text);
    }

    #[test]
    fn an_object_without_arguments_starts_and_ends_at_its_close() {
        let events = run(&[r#"{"name": "f"}"#]);
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "f".into(),
                    source: Text::uncounted(r#"{"name": "f"}"#),
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::default(),
                },
            ]
        );
    }

    #[test]
    fn the_start_waits_for_the_arguments_to_begin_or_the_object_to_close() {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assembler.feed(r#"{"name": "f", "arguments":"#, &mut out);
        assert!(
            out.is_empty(),
            "the head is not complete until the value begins"
        );
        assembler.feed(" {", &mut out);
        assert_eq!(
            starts(out.as_slice()),
            vec![("f", r#"{"name": "f", "arguments": "#)]
        );
    }

    #[test]
    fn string_valued_arguments_stream_as_written() {
        let text = r#"{"name": "f", "arguments": "{\"a\": 1}"}"#;
        let events = run(&[text]);
        assert_eq!(arguments(&events), r#""{\"a\": 1}""#);
        assert_eq!(sources(&events), text);
    }

    #[test]
    fn an_object_that_never_names_its_call_comes_back_malformed() {
        assert_eq!(
            run(&[r#"{"foo": 1}"#]),
            vec![Event::Malformed {
                text: Text::uncounted(r#"{"foo": 1}"#),
                why: MalformedReason::Other("a tool call without a name".into()),
            }]
        );
        assert_eq!(
            run(&[r#"{"arguments": {"a"#]),
            vec![Event::Malformed {
                text: Text::uncounted(r#"{"arguments": {"a"#),
                why: MalformedReason::UnterminatedRegion,
            }]
        );
        assert_eq!(run(&[]), vec![], "nothing fed, nothing said");
    }

    #[test]
    fn a_started_call_cut_short_is_closed_with_what_arrived() {
        let events = run(&[r#"{"name": "f", "arguments": {"a": 1"#]);
        assert_eq!(arguments(&events), r#"{"a": 1"#);
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            })
        );
        let events = run(&[r#"{"name": "f", "arguments": {},"#]);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 0,
                source: Text::uncounted(","),
            })
        );
    }

    #[test]
    fn bytes_after_the_close_are_ignored_and_done_says_so() {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assembler.feed(CALL, &mut out);
        assert!(assembler.done());
        let before = out.len();
        assembler.feed("}}", &mut out);
        assert_eq!(out.len(), before);
        assembler.finish(&mut out);
        assert_eq!(out.len(), before);
    }

    #[test]
    fn the_index_and_id_ride_on_every_event() {
        let mut assembler = Assembler::new(7, "call_seven");
        let mut out = Events::new();
        assembler.feed(CALL, &mut out);
        for event in out.as_slice() {
            match event {
                Event::ToolCallStart { index, id, .. } => {
                    assert_eq!((*index, id.as_str()), (7, "call_seven"));
                }
                Event::ToolCallArguments { index, .. } | Event::ToolCallEnd { index, .. } => {
                    assert_eq!(*index, 7);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
    }
}
