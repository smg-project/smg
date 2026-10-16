//! Events from one tool-call object as it arrives.
//!
//! [`Assembler`] takes a call's JSON object in pieces, as the format cuts it out of the model's
//! text, and turns it into the events of one call: `ToolCallStart` once the name is whole and the
//! arguments value has begun (or the object has closed without one), `ToolCallArguments` for every
//! run of new argument bytes, `ToolCallEnd` when the object closes. The fragments are the model's
//! own bytes: what a client receives, concatenated, is exactly what the model wrote as the
//! arguments value, spacing included, and no fragment ever has to be revised; an object that ends
//! without an arguments member is a call with none, `{}`, one fragment of no bytes, as every
//! other syntax writes a call with nothing in it. The old crate had no
//! equivalent; it parsed the partial object on every chunk, re-serialized the arguments and emitted
//! the difference, which is how `{"city": "Paris"}` reached clients as `{"city":"Paris"}`.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::{
        outline::{Scanner, Span},
        prefix::Prefix,
    },
};

/// The events of one tool call, from the bytes of its object.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    text: String,
    /// The outline of `text`, read as it grows.
    scanner: Scanner,
    started: bool,
    /// Argument bytes accounted for so far, as fragments or as malformed text.
    emitted: usize,
    /// Argument bytes the prefix has judged so far.
    judged: usize,
    /// Whether the argument bytes are still a JSON prefix, judged as they arrive.
    prefix: Prefix,
    /// The argument byte from which the arguments stopped being a valid JSON prefix, if they did.
    invalid_from: Option<usize>,
    done: bool,
    /// Whether the object ends where its bytes stop being a call's object: where its arguments
    /// stop being a JSON prefix, or where its outline stops short of the closing brace (the brace
    /// left off, text where a member should be, another member's value that stops being JSON).
    /// The bytes from there on are then the caller's
    /// ([`Self::refused_tail`]), for a shaped call's trailer to read, rather than the object's.
    ends_where_refused: bool,
    /// Where in `text` the object ended so, until the bytes from there were handed back.
    ended_at: Option<usize>,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            text: String::new(),
            scanner: Scanner::default(),
            started: false,
            emitted: 0,
            judged: 0,
            prefix: Prefix::default(),
            invalid_from: None,
            done: false,
            ends_where_refused: false,
            ended_at: None,
        }
    }

    /// Whether the arguments stopped being a JSON prefix.
    pub fn refused(&self) -> bool {
        self.invalid_from.is_some()
    }

    /// The object ends where its bytes stop being a call's object: where its arguments stop being
    /// a JSON prefix, or where its outline stops short of the closing brace, the other members'
    /// values judged as JSON too (see [`Self::refused_tail`]).
    pub fn ending_where_refused(mut self) -> Self {
        self.ends_where_refused = true;
        self.scanner = std::mem::take(&mut self.scanner).judging_values();
        self
    }

    /// Whether the object ended where its arguments stopped being a JSON prefix: it ends there
    /// ([`Self::ending_where_refused`]), and they did. An object that ended where its outline
    /// stopped, or one that closed after its arguments broke, did not.
    pub fn ended_where_refused(&self) -> bool {
        self.ends_where_refused && self.invalid_from.is_some()
    }

    /// Where the outline stopped short of the object's close, for an object that ends there.
    fn stopped_at(&self) -> Option<usize> {
        self.scanner
            .stopped_at()
            .filter(|_| self.ends_where_refused)
    }

    /// For an object that ends where refused: the bytes from where it ended on, which the object
    /// took before that was known, handed back once with the character before them, for the
    /// caller to read as what follows the arguments. `None` otherwise.
    pub fn refused_tail(&mut self) -> Option<(String, Option<char>)> {
        let at = self.ended_at.take()?;
        let tail = self.text.split_off(at);
        let before = self.text.chars().next_back();
        Some((tail, before))
    }

    /// Whether the object has closed, as a call or as an object that named none; `feed` takes
    /// nothing after that, so the format stops here.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed. A format counts
    /// its calls and advances their index by this, not by the markers it saw.
    pub fn started(&self) -> bool {
        self.started
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
        self.scanner.advance(&self.text);
        let taken = match self.scanner.found().close {
            Some(close) => {
                // The scanner stopped at the brace; what follows it is not the object's.
                self.text.truncate(close);
                close - before
            }
            None => bytes.len(),
        };
        // Two offsets and a span: nothing of the outline is copied per piece but what this
        // piece needs, and the name only once, when the call starts.
        let close = self.scanner.found().close;
        let arguments = self.scanner.found().arguments.clone();
        if !self.started {
            let Some(name) = self.scanner.found().name.clone() else {
                // Closed, or stopped short of its close, without a name: not a call. Said now, so
                // `done` is true at the close or the stop.
                if let Some(end) = close.or_else(|| self.stopped_at()) {
                    if end > 0 {
                        out.push(Event::Malformed {
                            text: Text::uncounted(&self.text[..end]),
                            why: MalformedReason::Other("a tool call without a name".to_string()),
                        });
                    }
                    if close.is_none() {
                        // The bytes from the stop on are the caller's (`refused_tail`).
                        self.ended_at = Some(end);
                    }
                    self.done = true;
                }
                return taken;
            };
            let head_end = match (&arguments, close) {
                (Some(span), _) => span.start,
                (None, Some(close)) => close,
                // Stopped short of its close before any arguments: a call without them.
                (None, None) => match self.stopped_at() {
                    Some(at) => at,
                    None => return taken,
                },
            };
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name,
                source: Text::uncounted(&self.text[..head_end]),
            });
            self.started = true;
        }
        self.emit_new_argument_bytes(arguments.as_ref(), out);
        if self.ended_at.is_none() {
            self.ended_at = self.stopped_at();
        }
        if let Some(at) = self.ended_at {
            // The object ends short of its close, unterminated: at the arguments' break, with
            // nothing after them, or where its outline stopped, with what the outline read after
            // them. The call ends with what arrived, and the bytes from there on are the caller's
            // (`refused_tail`).
            self.send_held(arguments.as_ref(), out);
            self.no_arguments(arguments.as_ref(), out);
            let tail_start = arguments
                .as_ref()
                .and_then(|span| span.end)
                .map_or(at, |end| end.min(at));
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(&self.text[tail_start..at]),
            });
            self.done = true;
            return taken;
        }
        if let Some(close) = close {
            self.send_held(arguments.as_ref(), out);
            self.no_arguments(arguments.as_ref(), out);
            let tail_start = arguments
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

    /// The call ends without an arguments member, its key never having come: it is a call with
    /// none, `{}`, one fragment that accounts for no bytes, so that a client reads the same
    /// arguments for it as for one written `"arguments": {}`, and the other syntaxes' call with
    /// nothing in it. Pushed before the end, where the object closed or ended short of its close.
    /// A member whose key came and whose value never began is not one: the call ends with what
    /// arrived, as before, and nothing is made up for a value the model botched.
    fn no_arguments(&self, arguments: Option<&Span>, out: &mut Events) {
        if arguments.is_none() && !self.scanner.found().arguments_named {
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: "{}".to_string(),
                source: Text::default(),
            });
        }
    }

    /// No more bytes will come, and the object never closed. A call that started is closed with
    /// what arrived, the bytes after its arguments value returned as `Malformed`; an object that
    /// never named its call comes back as `Malformed` whole. Either way no byte is lost.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let found = self.scanner.found().clone();
        if self.started {
            // `feed` has emitted every argument byte that arrived, as fragments or as malformed
            // text, but for a word it still held; that goes first, as the prefix it is. The object
            // never closed, so what follows the arguments value is the tail of an unterminated
            // region: a comma cut short, a complete member, or bytes that are no member at all. A
            // closed object's tail is the source of its `ToolCallEnd` instead.
            self.send_held(found.arguments.as_ref(), out);
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

    /// The argument bytes held back ([`Self::emit_new_argument_bytes`]), sent as the prefix they
    /// are when the value can take no more bytes: at its end, or the stream's.
    fn send_held(&mut self, arguments: Option<&Span>, out: &mut Events) {
        let Some(span) = arguments else {
            return;
        };
        let bytes = span.text(&self.text);
        if self.invalid_from.is_none() && self.emitted < bytes.len() {
            let held = &bytes[self.emitted..];
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: held.to_string(),
                source: Text::uncounted(held),
            });
            self.emitted = bytes.len();
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
        // Only the bytes that arrived are judged; the prefix keeps its place between pieces.
        let verdict = self.prefix.feed(&bytes[self.judged..]);
        self.judged = bytes.len();
        let valid_end = match self.invalid_from {
            Some(from) => from,
            None => match verdict {
                None => bytes.len(),
                Some(from) => *self.invalid_from.insert(from.max(self.emitted)),
            },
        };
        // Bytes the judge may still go back to (a word being spelled, `nu`; a surrogate escape
        // awaiting its low half; the whitespace before a value at the top) are held while the
        // value is still open: a fragment passed on cannot be taken back, so the pieces would say
        // `n` where the whole says nothing. They are sent once the next bytes settle them, with
        // the value's end, or when no more bytes can come ([`Self::send_held`]), as the prefix
        // they are then. A string-valued arguments streams as written, the one shape where the
        // pieces may say more than the whole (`Prefix::held_from`).
        let valid_end = match self.prefix.held_from() {
            Some(start) if span.end.is_none() && self.invalid_from.is_none() => {
                valid_end.min(start).max(self.emitted)
            }
            _ => valid_end,
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
        if self.ends_where_refused && self.invalid_from.is_some() {
            // The object ends at the break; what follows is handed back, not reported as its
            // malformed rest.
            self.ended_at = Some(span.start + valid_end);
            return;
        }
        // Past the break the rest is malformed; before any break what is not sent is held.
        if self.invalid_from.is_some() && self.emitted < bytes.len() {
            out.push(Event::Malformed {
                text: Text::uncounted(&bytes[self.emitted..]),
                why: MalformedReason::InvalidArguments,
            });
            self.emitted = bytes.len();
        }
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, _declared: &crate::tagged::Declared, out: &mut Events) -> usize {
        self.feed(text, out)
    }

    /// The object closed the call at its brace; the region's closing marker is the region's.
    fn close(self: Box<Self>, _terminal: &str, out: &mut Events) -> bool {
        (*self).finish(out);
        false
    }

    fn finish(self: Box<Self>, out: &mut Events) {
        (*self).finish(out);
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

    /// The malformed text of the events, joined.
    fn malformed(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_top_level_string_a_surrogate_breaks_is_the_one_shape_the_pieces_say_more_of() {
        // A string-valued arguments streams as written (a stringified call object is one), and
        // the parser refuses the whole value when a surrogate escape breaks inside it: the pieces
        // have said the string's bytes before the break where the whole says nothing. Every
        // byte is accounted for either way, and nothing a piece said is taken back.
        let object = r#"{"name": "f", "arguments": "ab\udc00"}"#;
        let whole = run(&[object]);
        assert_eq!(arguments(&whole), "");
        assert_eq!(malformed(&whole), r#""ab\udc00""#);
        let per_char: Vec<&str> = object
            .char_indices()
            .map(|(i, c)| &object[i..i + c.len_utf8()])
            .collect();
        let streamed = run(&per_char);
        assert_eq!(arguments(&streamed), r#""ab\udc0"#);
        assert_eq!(malformed(&streamed), r#"0""#);
        for events in [&whole, &streamed] {
            assert_eq!(sources(events), object);
        }
    }

    #[test]
    fn a_word_under_way_is_held_so_the_pieces_say_of_it_what_the_whole_says() {
        // A word that is no literal is refused from its first letter, and nothing of it is
        // passed on as arguments however the bytes were cut; a literal, whole or cut short at
        // the end, is sent once it is spelled or once nothing more can come; a number is sent as
        // far as it is one. A surrogate escape that breaks behind its backslash takes the
        // backslash with it. Open values end with the stream, closed ones at their object's brace.
        let cases = [
            ("nosuch", "", "nosuch"),
            ("trash", "", "trash"),
            ("nul", "nul", ""),
            ("null", "null", ""),
            ("true", "true", ""),
            ("1x", "1", "x"),
            (r#"{"a": nosuch}"#, r#"{"a": "#, "nosuch}"),
            (r#"{"a": null}"#, r#"{"a": null}"#, ""),
            ("[true, trash]", "[true, ", "trash]"),
            (r#"{"e": "\ud83c\x"}"#, r#"{"e": "\ud83c"#, r#"\x"}"#),
            (r#"{"e": "\ud83c\ude00"}"#, r#"{"e": "\ud83c\ude00"}"#, ""),
            (r#""ab\ud83c\ude00 c""#, r#""ab\ud83c\ude00 c""#, ""),
            // The outline keeps the whitespace before the value out of the arguments' span.
            ("\tnosuch", "", "nosuch"),
            (" null", "null", ""),
        ];
        for (value, expected_arguments, expected_malformed) in cases {
            for closed in [false, true] {
                let brace = if closed { "}" } else { "" };
                let object = format!(r#"{{"name": "f", "arguments": {value}{brace}"#);
                let whole = run(&[&object]);
                let per_char: Vec<&str> = object
                    .char_indices()
                    .map(|(i, c)| &object[i..i + c.len_utf8()])
                    .collect();
                let streamed = run(&per_char);
                for (how, events) in [("whole", &whole), ("per character", &streamed)] {
                    assert_eq!(
                        arguments(events),
                        expected_arguments,
                        "{object:?} {how}: arguments"
                    );
                    assert_eq!(
                        malformed(events),
                        expected_malformed,
                        "{object:?} {how}: malformed"
                    );
                    assert_eq!(sources(events), object, "{object:?} {how}: every byte");
                }
            }
        }
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
    fn an_object_without_arguments_is_a_call_with_an_empty_object_that_starts_and_ends_at_its_close(
    ) {
        // The call starts at the close, since nothing before it says the object has no arguments
        // member; its arguments are `{}`, one fragment of no bytes, as a call written
        // `"arguments": {}` reads, and as every other syntax reads a call with nothing in it.
        let expected = vec![
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "f".into(),
                source: Text::uncounted(r#"{"name": "f"}"#),
            },
            Event::ToolCallArguments {
                index: 0,
                json: "{}".into(),
                source: Text::default(),
            },
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
        ];
        let text = r#"{"name": "f"}"#;
        assert_eq!(run(&[text]), expected);
        let per_char: Vec<&str> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        assert_eq!(run(&per_char), expected, "byte by byte");
        // Other members do not make arguments: the call still has `{}`; without an arguments
        // value the start's source is the whole object and the end's is empty, and the
        // arguments written `{}` read the same.
        for (text, expected, end) in [
            (r#"{"name": "f", "id": 7}"#, "{}", ""),
            (r#"{"id": 7, "name": "f"}"#, "{}", ""),
            (r#"{"name": "f", "arguments": {}}"#, "{}", "}"),
        ] {
            let events = run(&[text]);
            assert_eq!(arguments(&events), expected, "{text}");
            assert_eq!(
                events.last(),
                Some(&Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted(end),
                }),
                "{text}"
            );
            assert_eq!(sources(&events), text, "{text}");
        }
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
    fn a_word_refused_hands_the_same_tail_back_whole_and_in_pieces() {
        // The shaped calls' assembler ends where its arguments stop being a value and hands the
        // bytes from there on back for the trailer to read. A word under way is held, so the
        // bytes handed back start at the word's first letter whole and in pieces alike, and so
        // do the arguments sent before them.
        let feed = |text: &str, by_char: bool| {
            let mut assembler = Assembler::new(0, "call_0").ending_where_refused();
            let mut out = Events::new();
            let mut back = String::new();
            let pieces: Vec<&str> = if by_char {
                text.char_indices()
                    .map(|(i, c)| &text[i..i + c.len_utf8()])
                    .collect()
            } else {
                vec![text]
            };
            for piece in pieces {
                let taken = assembler.feed(piece, &mut out);
                if let Some((tail, _)) = assembler.refused_tail() {
                    back.push_str(&tail);
                }
                back.push_str(&piece[taken..]);
            }
            assembler.finish(&mut out);
            let events = out.drain();
            (arguments(&events), malformed(&events), back)
        };
        for (text, expected_arguments, expected_back) in [
            (r#"{"name": "f", "arguments": nosuch</f>"#, "", "nosuch</f>"),
            (r#"{"name": "f", "arguments": nul</f>"#, "nul", "</f>"),
            (r#"{"name": "f", "arguments": null</f>"#, "null", "</f>"),
            (
                r#"{"name": "f", "arguments": {"a": nosuch}</f>"#,
                r#"{"a": "#,
                "nosuch}</f>",
            ),
        ] {
            let whole = feed(text, false);
            let in_pieces = feed(text, true);
            assert_eq!(whole, in_pieces, "{text:?}: whole and in pieces");
            assert_eq!(whole.0, expected_arguments, "{text:?}: arguments");
            assert_eq!(whole.2, expected_back, "{text:?}: handed back");
        }
    }

    /// With `ending_where_refused`, an object whose outline stops short of its closing brace (the
    /// brace left off, text where a member should be) ends there, as one whose arguments break
    /// does: the call ends with what the outline read after the arguments, and the bytes from the
    /// stop on are handed back. Stopped before its name, the object named no call, and its bytes
    /// come back malformed.
    #[test]
    fn an_object_ending_where_refused_ends_where_its_outline_stops() {
        let feed = |text: &str, by_char: bool| {
            let mut assembler = Assembler::new(0, "call_0").ending_where_refused();
            let mut out = Events::new();
            if by_char {
                for (i, c) in text.char_indices() {
                    assembler.feed(&text[i..i + c.len_utf8()], &mut out);
                }
            } else {
                assert_eq!(assembler.feed(text, &mut out), text.len(), "{text:?}");
            }
            let tail = assembler.refused_tail();
            assert!(
                assembler.done() && !assembler.refused() && !assembler.ended_where_refused(),
                "{text:?}"
            );
            assert_eq!(assembler.feed("more", &mut out), 0, "{text:?}");
            assembler.finish(&mut out);
            (out.drain(), tail)
        };
        let start = |source: &str| Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "f".into(),
            source: Text::uncounted(source),
        };
        let end = |source: &str| Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted(source),
        };
        let head = r#"{"name": "f", "arguments": "#;
        let value = Event::ToolCallArguments {
            index: 0,
            json: r#"{"a": 1}"#.into(),
            source: Text::uncounted(r#"{"a": 1}"#),
        };

        let text = r#"{"name": "f", "arguments": {"a": 1}</f>"#;
        let (events, tail) = feed(text, false);
        assert_eq!(
            events,
            vec![start(head), value.clone(), end("")],
            "a brace left off"
        );
        assert_eq!(tail, Some(("</f>".to_string(), Some('}'))));
        let (events, tail) = feed(text, true);
        assert_eq!(events.first(), Some(&start(head)), "byte by byte");
        assert_eq!(arguments(&events), r#"{"a": 1}"#, "byte by byte");
        assert_eq!(events.last(), Some(&end("")), "byte by byte");
        assert_eq!(
            tail,
            Some(("<".to_string(), Some('}'))),
            "the tail is what came in the piece the stop came in"
        );

        let (events, tail) = feed(
            r#"{"name": "f", "arguments": {"a": 1}, "strict": true</f>"#,
            false,
        );
        assert_eq!(
            events,
            vec![start(head), value.clone(), end(r#", "strict": true"#)],
            "a literal in another member runs into the trailer"
        );
        assert_eq!(tail, Some(("</f>".to_string(), Some('e'))));

        let (events, tail) = feed(r#"{"name": "f", "arguments": {"a": 1}, junk</f>"#, false);
        assert_eq!(
            events,
            vec![start(head), value, end(", ")],
            "what the outline read after the arguments is the end's"
        );
        assert_eq!(tail, Some(("junk</f>".to_string(), Some(' '))));

        let (events, tail) = feed(r#"{"name": "f"</f>"#, false);
        assert_eq!(
            events,
            vec![
                start(r#"{"name": "f""#),
                Event::ToolCallArguments {
                    index: 0,
                    json: "{}".into(),
                    source: Text::default(),
                },
                end("")
            ],
            "a call without arguments has `{{}}`"
        );
        assert_eq!(tail, Some(("</f>".to_string(), Some('"'))));

        // An arguments member whose value never began is not a call with none: its key came, so
        // no `{}` is made up for it; the call ends with what arrived and the bytes from the stop
        // on are the caller's, as before.
        let (events, tail) = feed(r#"{"name": "f", "arguments": }</f>"#, false);
        assert_eq!(
            events,
            vec![start(r#"{"name": "f", "arguments": "#), end("")],
            "an arguments member without a value is not a call with `{{}}`"
        );
        assert_eq!(tail, Some(("}</f>".to_string(), Some(' '))));

        let (events, tail) = feed(r#"{"arguments": {"a": 1}, x</f>"#, false);
        assert_eq!(
            events,
            vec![Event::Malformed {
                text: Text::uncounted(r#"{"arguments": {"a": 1}, "#),
                why: MalformedReason::Other("a tool call without a name".into()),
            }],
            "stopped before its name: no call"
        );
        assert_eq!(tail, Some(("x</f>".to_string(), Some(' '))));

        // A plain object keeps every byte, as before: its region's close ends it.
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        assert_eq!(assembler.feed(text, &mut out), text.len());
        assert!(!assembler.done());
        assert_eq!(assembler.refused_tail(), None);
    }

    #[test]
    fn a_long_arguments_string_fed_by_the_character_costs_time_linear_in_its_length() {
        // 200 KB of arguments, the size of a code patch a model writes as one call. Read once per
        // byte, this takes well under a second in a debug build; read again on every piece, as
        // the first version did, it took minutes.
        let patch: String = (0..4_000)
            .map(|i| format!("line {i}: let value = compute(input[{i}]) + {i};\\n"))
            .collect();
        let call = format!(
            r#"{{"name": "write", "arguments": {{"path": "a.rs", "content": "{patch}"}}}}"#
        );
        assert!(call.len() > 200_000, "{} bytes", call.len());
        let started = std::time::Instant::now();
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        for (i, c) in call.char_indices() {
            assembler.feed(&call[i..i + c.len_utf8()], &mut out);
        }
        let by_char = out.drain();
        let elapsed = started.elapsed();
        let mut whole = Events::new();
        Assembler::new(0, "call_0").feed(&call, &mut whole);
        let whole = whole.drain();
        let arguments = |events: &[Event]| -> String {
            events
                .iter()
                .filter_map(|event| match event {
                    Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(arguments(&by_char), arguments(&whole));
        assert!(
            matches!(by_char.last(), Some(Event::ToolCallEnd { .. })),
            "{:?}",
            by_char.last()
        );
        assert!(
            elapsed.as_secs() < 10,
            "a character at a time took {elapsed:?}; the cost is not linear"
        );
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
