//! Events from one tool-call object as it arrives.
//!
//! [`Assembler`] takes a call's JSON object in pieces, as the format cuts it out of the model's
//! text, and turns it into the events of one call: `ToolCallStart` once the name is whole and the
//! arguments value has begun (or the object has closed without one), `ToolCallArguments` for every
//! run of new argument bytes, `ToolCallEnd` when the object closes. The fragments are the model's
//! own bytes: what a client receives, concatenated, is exactly what the model wrote as the
//! arguments value, spacing included, and no fragment ever has to be revised. The old crate had no
//! equivalent; it parsed the partial object on every chunk, re-serialized the arguments and emitted
//! the difference, which is how `{"city": "Paris"}` reached clients as `{"city":"Paris"}`.
//!
//! Every byte of the object lands in exactly one event: the head up to the arguments value is the
//! `source` of `ToolCallStart`, the argument bytes are the `source` of their fragments, the tail
//! after the value is the `source` of `ToolCallEnd`. When the arguments come before the name they
//! are held until the name is whole, since a call starts before its arguments.
//!
//! The assembler keeps the promise `ToolCallArguments` makes, that the fragments so far always form
//! a valid JSON prefix, as far as the prefix parser can tell: each new run of argument bytes is
//! checked with [`PartialJson`] in prefix mode before it is emitted, and from the first byte the
//! parser cannot take, the argument bytes come back as `Malformed` with `InvalidArguments` instead,
//! nothing already emitted being revised. The prefix parser tolerates what the old crate tolerated
//! (a bracket closed by the wrong kind, a literal's prefix), so the promise is exactly as strong as
//! that parser.
//!
//! A started call whose object never closes is closed at `finish` with what arrived, and the bytes
//! after its arguments value (a comma cut short, a complete member, or bytes that are no member)
//! come back as `Malformed` with `UnterminatedRegion`.
//!
//! `feed` takes one piece and returns how many of its bytes the object took: all of them before the
//! close, and up to the closing brace in the piece that carries it, so the format routes whatever
//! follows the object (a closing marker, more text) itself. Bytes fed after the close are not
//! taken.
//!
//! What the assembler does not decide: the call's index and id (the format mints them and passes
//! them to [`Assembler::new`]), whether the name is a declared tool, and whether a string-valued
//! arguments member should be decoded. The format decides those with the events in hand.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::{
        outline::{outline, Span},
        partial::PartialJson,
    },
};

/// The events of one tool call, from the bytes of its object.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    text: String,
    started: bool,
    /// Argument bytes accounted for so far, as fragments or as malformed text.
    emitted: usize,
    /// The argument byte from which the arguments stopped being a valid JSON prefix, if they did.
    invalid_from: Option<usize>,
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
            invalid_from: None,
            done: false,
        }
    }

    /// Whether the object has closed, as a call or as an object that named none; `feed` takes
    /// nothing after that, so the format stops here.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Append the next bytes of the object, push the events they complete, and return how many of
    /// the bytes the object took: all of them before the close, up to the closing brace in the
    /// piece that carries it, none after it.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        if self.done {
            return 0;
        }
        let before = self.text.len();
        self.text.push_str(bytes);
        let mut found = outline(&self.text);
        let taken = match found.close {
            Some(close) => {
                self.text.truncate(close);
                found = outline(&self.text);
                close - before
            }
            None => bytes.len(),
        };
        if !self.started {
            let Some(name) = found.name.clone() else {
                if let Some(close) = found.close {
                    // Closed without a name: not a call. Said now, so `done` is true at the close.
                    out.push(Event::Malformed {
                        text: Text::uncounted(&self.text[..close]),
                        why: MalformedReason::Other("a tool call without a name".to_string()),
                    });
                    self.done = true;
                }
                return taken;
            };
            let head_end = match (&found.arguments, found.close) {
                (Some(span), _) => span.start,
                (None, Some(close)) => close,
                (None, None) => return taken,
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
        if let Some(close) = found.close {
            let tail_start = found
                .arguments
                .as_ref()
                .and_then(|span| span.end)
                .unwrap_or(close);
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(&self.text[tail_start..close]),
            });
            self.done = true;
        }
        taken
    }

    /// No more bytes will come, and the object never closed. A call that started is closed with
    /// what arrived, the bytes after its arguments value returned as `Malformed`; an object that
    /// never named its call comes back as `Malformed` whole. Either way no byte is lost.
    pub fn finish(self, out: &mut Events) {
        if self.done {
            return;
        }
        let found = outline(&self.text);
        if self.started {
            // `feed` has emitted every argument byte that arrived, as fragments or as malformed
            // text. The object never closed, so what follows the arguments value is the tail of an
            // unterminated region: a comma cut short, a complete member, or bytes that are no
            // member at all. A closed object's tail is the source of its `ToolCallEnd` instead.
            let tail_start = found
                .arguments
                .as_ref()
                .and_then(|span| span.end)
                .unwrap_or(self.text.len());
            if tail_start < self.text.len() {
                out.push(Event::Malformed {
                    text: Text::uncounted(&self.text[tail_start..]),
                    why: MalformedReason::UnterminatedRegion,
                });
            }
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
            return;
        }
        // `feed` answers an object that closed without a name at its brace, so what is left here
        // never closed.
        if !self.text.is_empty() {
            out.push(Event::Malformed {
                text: Text::uncounted(self.text),
                why: MalformedReason::UnterminatedRegion,
            });
        }
    }

    /// The argument bytes that arrived since the last call: a fragment for the part that keeps the
    /// arguments a valid JSON prefix, malformed text for the rest, once the prefix has broken.
    fn emit_new_argument_bytes(&mut self, arguments: Option<&Span>, out: &mut Events) {
        let Some(span) = arguments else {
            return;
        };
        let bytes = span.text(&self.text);
        if bytes.len() <= self.emitted {
            return;
        }
        let valid_end = match self.invalid_from {
            Some(from) => from,
            None => match PartialJson::default().parse(bytes, true) {
                Ok((_, consumed)) if consumed == bytes.len() => bytes.len(),
                Ok((_, consumed)) => *self.invalid_from.insert(consumed.max(self.emitted)),
                Err(_) => *self.invalid_from.insert(self.emitted),
            },
        };
        if self.emitted < valid_end {
            let fresh = &bytes[self.emitted..valid_end];
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: fresh.to_string(),
                source: Text::uncounted(fresh),
            });
            self.emitted = valid_end;
        }
        if self.emitted < bytes.len() {
            out.push(Event::Malformed {
                text: Text::uncounted(&bytes[self.emitted..]),
                why: MalformedReason::InvalidArguments,
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
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assert_eq!(assembler.feed(r#"{"foo": 1}  more"#, &mut out), 10);
        assert!(assembler.done(), "closed, as an object that named no call");
        assert_eq!(
            out.drain(),
            vec![Event::Malformed {
                text: Text::uncounted(r#"{"foo": 1}"#),
                why: MalformedReason::Other("a tool call without a name".into()),
            }]
        );
        assembler.finish(&mut out);
        assert!(out.is_empty(), "nothing more to say at the end");
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
    fn a_surrogate_pair_escape_in_the_arguments_streams_as_written() {
        let text = r#"{"name": "f", "arguments": {"e": "\ud83c\udf0d", "f": 1}}"#;
        let whole = run(&[text]);
        assert_eq!(arguments(&whole), r#"{"e": "\ud83c\udf0d", "f": 1}"#);
        assert!(
            !whole.iter().any(|e| matches!(e, Event::Malformed { .. })),
            "valid JSON, nothing malformed"
        );
        let pieces: Vec<&str> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        let bytewise = run(&pieces);
        assert_eq!(arguments(&bytewise), arguments(&whole));
        assert!(!bytewise
            .iter()
            .any(|e| matches!(e, Event::Malformed { .. })));
        assert_eq!(sources(&bytewise), text);
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
            &events[events.len() - 2..],
            [
                Event::Malformed {
                    text: Text::uncounted(","),
                    why: MalformedReason::UnterminatedRegion,
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::default(),
                },
            ]
        );
    }

    #[test]
    fn feed_says_how_many_bytes_the_object_took_and_takes_none_after_the_close() {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        let piece = format!("{CALL}\n</tool_call>\nmore text");
        assert_eq!(assembler.feed(&piece, &mut out), CALL.len());
        assert!(assembler.done());
        assert_eq!(
            out.as_slice().last(),
            Some(&Event::ToolCallEnd {
                index: 0,
                source: Text::uncounted("}"),
            }),
            "the surplus after the brace is the format's, not the end's"
        );
        assert_eq!(sources(out.as_slice()), CALL);
        let before = out.len();
        assert_eq!(assembler.feed("}}", &mut out), 0);
        assert_eq!(out.len(), before);
        assembler.finish(&mut out);
        assert_eq!(out.len(), before);
        let mut assembler = Assembler::new(0, "call_0");
        let cut = r#"{"name": "f", "arg"#.len();
        assert_eq!(
            assembler.feed(&CALL[..cut], &mut out),
            cut,
            "before the close every byte is taken"
        );
    }

    #[test]
    fn arguments_that_stop_being_a_json_prefix_come_back_malformed_from_that_byte() {
        let events = run(&[r#"{"name": "f", "arguments": 12abc}"#]);
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".into(),
                    name: "f".into(),
                    source: Text::uncounted(r#"{"name": "f", "arguments": "#),
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "12".into(),
                    source: Text::uncounted("12"),
                },
                Event::Malformed {
                    text: Text::uncounted("abc"),
                    why: MalformedReason::InvalidArguments,
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted("}"),
                },
            ]
        );
        // Byte by byte, with the junk after the value rather than inside it: the arguments stay
        // whole, the object never closes, and the junk comes back as the unterminated region's tail.
        let text = r#"{"name": "f", "arguments": {"a": 1} junk}"#;
        let pieces: Vec<&str> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        let events = run(&pieces);
        assert_eq!(arguments(&events), r#"{"a": 1}"#);
        let malformed: String = events
            .iter()
            .filter_map(|e| match e {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(malformed, " junk}");
        assert_eq!(sources(&events), text);
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
