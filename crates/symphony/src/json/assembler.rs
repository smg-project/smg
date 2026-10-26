//! Events from one tool-call object as it arrives.
//!
//! [`Assembler`] takes a call's JSON object in pieces, as the format cuts it out of the model's
//! text, and turns it into the events of one call: `ToolCallStart` once the name is whole and the
//! arguments value has begun (or the object has closed without one), `ToolCallArguments` for every
//! run of new argument bytes, `ToolCallEnd` when the object closes. The fragments are the model's
//! own bytes: what a client receives, concatenated, is exactly what the model wrote as the
//! arguments value, spacing included, but for a raw control character inside a string, which JSON
//! forbids and the fragment writes as its escape (`Prefix::escaped`), and no fragment ever has to
//! be revised; arguments that stop being JSON are said to, once, by a `Malformed` event of no
//! bytes where they stop, and the call still carries every byte of them (`json::fragments`); an
//! object that ends
//! without an arguments member is a call with none, `{}`, one fragment of no bytes, as every
//! other syntax writes a call with nothing in it. The old crate had no
//! equivalent; it parsed the partial object on every chunk, re-serialized the arguments and emitted
//! the difference, which is how `{"city": "Paris"}` reached clients as `{"city":"Paris"}`.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/assemblers.md`.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::{
        fragments::Fragments,
        outline::{Scanner, Span},
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
    /// The argument bytes, sent, held or reported as they arrive.
    fragments: Fragments,
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
            fragments: Fragments::default(),
            done: false,
            ends_where_refused: false,
            ended_at: None,
        }
    }

    /// Whether the arguments stopped being a JSON prefix.
    #[cfg(test)]
    pub fn refused(&self) -> bool {
        self.fragments.refused()
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
        self.ends_where_refused && self.fragments.refused()
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
                            text: Text::uncounted(self.text[..end].to_owned()),
                            why: MalformedReason::CallWithoutName,
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
                source: Text::uncounted(self.text[..head_end].to_owned()),
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
            // An object that ended where its arguments broke said so at the break
            // (`Fragments::send`) and handed the bytes from there back; one whose outline
            // stopped with arguments that are no whole value says so here, by a `Malformed`
            // event of no bytes before the end.
            if !self.fragments.refused() && !self.arguments_whole(arguments.as_ref()) {
                out.push(Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::InvalidArguments,
                });
            }
            let tail_start = arguments
                .as_ref()
                .and_then(|span| span.end)
                .map_or(at, |end| end.min(at));
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(self.text[tail_start..at].to_owned()),
            });
            self.done = true;
            return taken;
        }
        if let Some(close) = close {
            self.send_held(arguments.as_ref(), out);
            self.no_arguments(arguments.as_ref(), out);
            // The object closed, and the arguments value with it. A value the judge refused said
            // so at the break, where the first byte a strict reader refuses stands; one the
            // outline ended whole for the partial parser and cut short for a strict reader (a
            // number or a word the object's brace stops, `12.`, `nu`) is said to be none by a
            // `Malformed` event of no bytes, as at the stream's end, so that a client is told
            // rather than handed arguments its own reader refuses.
            if !self.fragments.refused() && !self.arguments_whole(arguments.as_ref()) {
                out.push(Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::InvalidArguments,
                });
            }
            let tail_start = arguments
                .as_ref()
                .and_then(|span| span.end)
                .unwrap_or(close);
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::uncounted(self.text[tail_start..close].to_owned()),
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
    /// what arrived, the bytes after its arguments value returned as `Malformed`; an arguments
    /// value the close cut short is said to be one, by a `Malformed` event of no bytes before the
    /// end; an object that never named its call comes back as `Malformed` whole. Either way no
    /// byte is lost.
    pub fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        let found = self.scanner.found().clone();
        if self.started {
            // `feed` has emitted every argument byte that arrived as fragments, but for a word it
            // still held; that goes first, as the prefix it is. The object
            // never closed, so what follows the arguments value is the tail of an unterminated
            // region: a comma cut short, a complete member, or bytes that are no member at all. A
            // closed object's tail is the source of its `ToolCallEnd` instead.
            self.send_held(found.arguments.as_ref(), out);
            // The arguments value itself may be what the close cut: a string or a container left
            // open, a word or a number the bytes end on. The fragments sent are not revised and
            // no value is made up for one the model never finished: the call ends as it stands,
            // and a `Malformed` event of no bytes, before its end, says that its arguments are no
            // whole value, where a client would otherwise take them for one. The engine names the
            // reason for a region a marker closed.
            if !self.fragments.refused() && !self.arguments_whole(found.arguments.as_ref()) {
                out.push(Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::UnterminatedRegion,
                });
            }
            let tail_start = found
                .arguments
                .as_ref()
                .and_then(|span| span.end)
                .unwrap_or(self.text.len());
            if tail_start < self.text.len() {
                out.push(Event::Malformed {
                    text: Text::uncounted(self.text[tail_start..].to_owned()),
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

    /// Whether the arguments the call has said are one whole JSON value a strict reader takes:
    /// the value's bytes are one by the judge's account ([`Prefix::whole`]), or none came and `{}`
    /// stands for them ([`Self::no_arguments`]).
    ///
    /// [`Prefix::whole`]: crate::json::Prefix::whole
    fn arguments_whole(&self, arguments: Option<&Span>) -> bool {
        match arguments {
            Some(_) => self.fragments.whole(),
            None => !self.scanner.found().arguments_named,
        }
    }

    /// The argument bytes held back ([`Self::emit_new_argument_bytes`]), sent as the prefix they
    /// are when the value can take no more bytes: at its end, or the stream's.
    fn send_held(&mut self, arguments: Option<&Span>, out: &mut Events) {
        if let Some(span) = arguments {
            self.fragments
                .send_held(span.text(&self.text), self.index, out);
        }
    }

    /// The argument bytes that arrived since the last call: a fragment for the part that keeps the
    /// arguments a valid JSON prefix, the break said once where they stop being one, the rest as
    /// the model wrote it, and the bytes the judge may still go back to held while the value is
    /// open ([`Fragments::send`], [`Fragments::send_rest`]).
    fn emit_new_argument_bytes(&mut self, arguments: Option<&Span>, out: &mut Events) {
        let Some(span) = arguments else {
            return;
        };
        let bytes = span.text(&self.text);
        let Some(sent_to) = self
            .fragments
            .send(bytes, span.end.is_none(), self.index, out)
        else {
            return;
        };
        if self.ends_where_refused && self.fragments.refused() {
            // The object ends at the break; what follows is handed back, not sent as the rest of
            // its arguments.
            self.ended_at = Some(span.start + sent_to);
            return;
        }
        self.fragments.send_rest(bytes, self.index, out);
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, _declared: &crate::Declared, out: &mut Events) -> usize {
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
    use crate::testing::arguments;

    const CALL: &str = r#"{"name": "get_weather", "arguments": {"city": "Paris", "days": [1, 2]}}"#;
    const ARGUMENTS: &str = r#"{"city": "Paris", "days": [1, 2]}"#;

    fn run(pieces: &[&str]) -> Vec<Event<'static>> {
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
                | Event::ToolCallEnd { source, .. } => &*source.text,
                Event::Malformed { text, .. } => &*text.text,
                _ => "",
            })
            .collect()
    }

    fn starts<'a>(events: &'a [Event<'_>]) -> Vec<(&'a str, &'a str)> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallStart { name, source, .. } => Some((name.as_str(), &*source.text)),
                _ => None,
            })
            .collect()
    }

    /// The malformed text of the events, joined.
    fn malformed(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Malformed { text, .. } => Some(&*text.text),
                _ => None,
            })
            .collect()
    }

    /// The fragments sent before the first `Malformed` event, joined; `None` when none came.
    fn said_before(events: &[Event]) -> Option<String> {
        let at = events
            .iter()
            .position(|event| matches!(event, Event::Malformed { .. }))?;
        Some(arguments(&events[..at]))
    }

    #[test]
    fn a_top_level_string_a_surrogate_breaks_says_the_same_whole_and_in_pieces() {
        // A string-valued arguments streams as written (a stringified call object is one), and
        // the judge refuses the whole value when a surrogate escape breaks inside it: the pieces
        // have sent the string's bytes before the escape (held from its backslash until it is
        // decided) by the time it is decided, where the whole sends none before the break. Both
        // say the break once, with no bytes, and both carry every byte of the value as the
        // call's own, so what the call says is the same; only where the break stands among the
        // fragments differs. Nothing a piece said is taken back.
        let object = r#"{"name": "f", "arguments": "ab\udc00"}"#;
        let per_char: Vec<&str> = object
            .char_indices()
            .map(|(i, c)| &object[i..i + c.len_utf8()])
            .collect();
        for (how, events, before) in [
            ("whole", run(&[object]), ""),
            ("per character", run(&per_char), r#""ab"#),
        ] {
            assert_eq!(arguments(&events), r#""ab\udc00""#, "{how}");
            assert_eq!(said_before(&events), Some(before.to_string()), "{how}");
            assert_eq!(malformed(&events), "", "{how}: the break carries no bytes");
            assert_eq!(sources(&events), object, "{how}");
        }
    }

    #[test]
    fn a_word_under_way_is_held_so_the_pieces_say_of_it_what_the_whole_says() {
        // A word that is no literal is refused from its first letter, so the break stands before
        // it however the bytes were cut; a literal, whole or cut short at the end, is sent once
        // it is spelled or once nothing more can come; a number is sent as far as it is one. A
        // surrogate escape that breaks behind its backslash breaks from the backslash. Every byte
        // of the value is the call's own either way, and a value that broke or was cut short says
        // so once, with no bytes: `said_before` is what the call had sent by then, `None` where it
        // had nothing to say. Open values end with the stream, closed ones at their object's
        // brace.
        let cases = [
            ("nosuch", Some("")),
            ("trash", Some("")),
            ("nul", Some("nul")),
            ("null", None),
            ("true", None),
            ("1x", Some("1")),
            (r#"{"a": nosuch}"#, Some(r#"{"a": "#)),
            (r#"{"a": null}"#, None),
            ("[true, trash]", Some("[true, ")),
            (r#"{"e": "\ud83c\x"}"#, Some(r#"{"e": "\ud83c"#)),
            (r#"{"e": "\ud83c\ude00"}"#, None),
            (r#""ab\ud83c\ude00 c""#, None),
            // The outline keeps the whitespace before the value out of the arguments' span.
            ("\tnosuch", Some("")),
            (" null", None),
        ];
        for (value, expected_before) in cases {
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
                        value.trim_start(),
                        "{object:?} {how}: arguments"
                    );
                    assert_eq!(
                        said_before(events),
                        expected_before.map(str::to_string),
                        "{object:?} {how}: said before"
                    );
                    assert_eq!(malformed(events), "", "{object:?} {how}: no bytes reported");
                    assert!(
                        events
                            .iter()
                            .filter(|event| matches!(event, Event::Malformed { .. }))
                            .count()
                            <= 1,
                        "{object:?} {how}: said once"
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
                why: MalformedReason::CallWithoutName,
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
            &events[events.len() - 2..],
            [
                Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::UnterminatedRegion,
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::default(),
                },
            ],
            "the value was cut: the call says so before its end"
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

    /// A call cut inside its arguments value, by the next call's opener, the region's closing
    /// marker or the end of the stream: the fragments sent stand, nothing is made up, and a
    /// `Malformed` event of no bytes before the end says that the arguments are no whole value,
    /// whole and in pieces alike. A value that is whole when the object's brace is left off
    /// needs no word.
    #[test]
    fn a_call_cut_inside_its_arguments_says_so_before_its_end() {
        let said_cut = |events: &[Event]| {
            matches!(
                &events[events.len() - 2..],
                [
                    Event::Malformed { text, why: MalformedReason::UnterminatedRegion },
                    Event::ToolCallEnd { .. }
                ] if text.text.is_empty()
            )
        };
        let cases = [
            // The raw newlines the model wrote inside the strings reach the client escaped.
            (
                "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par\n",
                "{\"city\": \"Par\\n",
            ),
            ("{\"name\": \"ch\", \"arguments\":\"dname\n", "\"dname\\n"),
            (
                r#"{"name": "f", "arguments": {"a": [1, 2"#,
                r#"{"a": [1, 2"#,
            ),
            (r#"{"name": "f", "arguments": {"a": nul"#, r#"{"a": nul"#),
            (r#"{"name": "f", "arguments": {"a": 1.5e"#, r#"{"a": 1.5e"#),
        ];
        for (object, expected) in cases {
            let whole = run(&[object]);
            assert_eq!(arguments(&whole), expected, "{object:?}");
            assert!(said_cut(&whole), "{object:?}: {whole:?}");
            let per_char: Vec<&str> = object
                .char_indices()
                .map(|(i, c)| &object[i..i + c.len_utf8()])
                .collect();
            let streamed = run(&per_char);
            assert_eq!(arguments(&streamed), expected, "{object:?} per character");
            assert!(said_cut(&streamed), "{object:?} per character");
            assert_eq!(sources(&streamed), object);
        }
        // The arguments are whole, the object's brace is missing: a call, with no word.
        let whole = run(&[r#"{"name": "f", "arguments": {"a": 1}"#]);
        assert_eq!(arguments(&whole), r#"{"a": 1}"#);
        assert!(!whole
            .iter()
            .any(|event| matches!(event, Event::Malformed { .. })));
        // Arguments that broke have said so already, at the byte that broke them, and carry it.
        let broken = run(&[r#"{"name": "f", "arguments": {"a": 12abc"#]);
        assert_eq!(arguments(&broken), r#"{"a": 12abc"#);
        assert_eq!(said_before(&broken), Some(r#"{"a": 12"#.to_string()));
        assert_eq!(malformed(&broken), "");
        assert_eq!(
            broken
                .iter()
                .filter(|event| matches!(event, Event::Malformed { .. }))
                .count(),
            1
        );
    }

    /// A raw control character inside a string, which JSON forbids: the fragment writes it as its
    /// escape, the source keeps the model's bytes, and the arguments a client assembles are JSON a
    /// strict reader accepts, whole and in pieces alike. A newline between tokens is whitespace
    /// and stays; a call cut after one is still said to be cut.
    #[test]
    fn a_raw_control_character_inside_a_string_reaches_the_client_escaped() {
        let object = "{\"name\": \"f\", \"arguments\": {\"city\": \"Pa\nris\", \"k\u{12}\": \"\\\n\",\n\"days\": 3}}";
        let expected = "{\"city\": \"Pa\\nris\", \"k\\u0012\": \"\\n\",\n\"days\": 3}";
        let per_char: Vec<&str> = object
            .char_indices()
            .map(|(i, c)| &object[i..i + c.len_utf8()])
            .collect();
        for (how, events) in [("whole", run(&[object])), ("per character", run(&per_char))] {
            assert_eq!(arguments(&events), expected, "{how}");
            assert_eq!(sources(&events), object, "{how}");
            assert!(
                serde_json::from_str::<serde_json::Value>(&arguments(&events)).is_ok(),
                "{how}"
            );
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, Event::Malformed { .. })),
                "{how}: {events:?}"
            );
        }
        let cut = run(&["{\"name\": \"f\", \"arguments\": {\"city\": \"Pa\nris\", \"days\": 3"]);
        assert_eq!(arguments(&cut), "{\"city\": \"Pa\\nris\", \"days\": 3");
        assert!(matches!(
            &cut[cut.len() - 2..],
            [Event::Malformed { .. }, Event::ToolCallEnd { .. }]
        ));
        let whole_value = run(&["{\"name\": \"f\", \"arguments\": {\"city\": \"Pa\nris\"}"]);
        assert_eq!(arguments(&whole_value), "{\"city\": \"Pa\\nris\"}");
        assert!(
            !whole_value
                .iter()
                .any(|event| matches!(event, Event::Malformed { .. })),
            "whole arguments, the object's brace missing: no word"
        );
    }

    /// What the partial parser tolerates and a strict reader refuses (a value left out, a trailing
    /// comma after a value and after a member, list items left out, an escape JSON lacks, a `\u`
    /// escape short of four hex digits, a bracket of the wrong kind, whitespace JSON lacks, a
    /// number and a word cut short before a close) closed a call as whole with arguments no JSON
    /// reader takes. The break is said at the first byte a strict reader refuses, by one
    /// `Malformed { InvalidArguments }` event of no bytes, the fragments before it a prefix such a
    /// reader takes, the rest of the value fragments as written; the object's brace ends the
    /// call, and the cut changes nothing of it.
    #[test]
    fn arguments_a_strict_reader_refuses_are_said_to_at_the_first_such_byte_and_kept_whole() {
        let strict_prefix = |text: &str| match serde_json::from_str::<serde_json::Value>(text) {
            Ok(_) => true,
            Err(error) => error.is_eof(),
        };
        for (text, prefix, rest) in [
            (
                r#"{"name": "get_weather", "arguments": {"city": "Paris", "days": ,}}"#,
                r#"{"city": "Paris", "days": "#,
                ",}",
            ),
            (
                r#"{"name": "f", "arguments": {"city": "Paris", "days": 3, }}"#,
                r#"{"city": "Paris", "days": 3, "#,
                "}",
            ),
            (
                r#"{"name": "f", "arguments": {"city": "Paris",}}"#,
                r#"{"city": "Paris","#,
                "}",
            ),
            (
                r#"{"name": "f", "arguments": {"tags": [,,]}}"#,
                r#"{"tags": ["#,
                ",,]}",
            ),
            // An escape is held from its backslash until it is decided, so the fragments end
            // before the one the reader refuses.
            (
                r#"{"name": "f", "arguments": {"city": "Pa\yris"}}"#,
                r#"{"city": "Pa"#,
                "\\yris\"}",
            ),
            (
                r#"{"name": "f", "arguments": {"city": "\u12G4"}}"#,
                r#"{"city": ""#,
                "\\u12G4\"}",
            ),
            // The outline counts a closing bracket of either kind, so the arguments run to the
            // second brace and the third closes the object.
            (
                r#"{"name": "f", "arguments": {"a": [1}}}"#,
                r#"{"a": [1"#,
                "}}",
            ),
            (
                "{\"name\": \"f\", \"arguments\": {\"a\":\u{a0}1}}",
                "{\"a\":",
                "\u{a0}1}",
            ),
            (
                r#"{"name": "f", "arguments": {"n": 1.}}"#,
                r#"{"n": 1."#,
                "}",
            ),
            (
                r#"{"name": "f", "arguments": {"flag": nu}}"#,
                r#"{"flag": "#,
                "nu}",
            ),
        ] {
            let whole = run(&[text]);
            let pieces: Vec<&str> = text
                .char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect();
            let by_char = run(&pieces);
            for (events, how) in [(&whole, "whole"), (&by_char, "by the character")] {
                assert_eq!(
                    arguments(events),
                    format!("{prefix}{rest}"),
                    "{text:?} {how}"
                );
                assert_eq!(
                    said_before(events),
                    Some(prefix.to_string()),
                    "{text:?} {how}"
                );
                assert!(
                    strict_prefix(prefix),
                    "{text:?}: the arguments before the break are a prefix a strict reader takes"
                );
                assert_eq!(
                    malformed(events),
                    "",
                    "{text:?} {how}: the break carries no bytes"
                );
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| matches!(event, Event::Malformed { .. }))
                        .count(),
                    1,
                    "{text:?} {how}: said once"
                );
                assert!(
                    events.iter().all(|event| !matches!(
                        event,
                        Event::Malformed { why, .. } if *why != MalformedReason::InvalidArguments
                    )),
                    "{text:?} {how}: {events:#?}"
                );
                assert_eq!(
                    events.last(),
                    Some(&Event::ToolCallEnd {
                        index: 0,
                        source: Text::uncounted("}"),
                    }),
                    "{text:?} {how}"
                );
                assert_eq!(
                    sources(events),
                    text,
                    "{text:?} {how}: every byte accounted for"
                );
            }
        }
    }

    /// A value the object's own brace cuts short of whole for a strict reader (a number, a word)
    /// is said to be none before the end, by a `Malformed` event of no bytes, where the brace
    /// closed the call without a word before.
    #[test]
    fn a_value_the_objects_brace_cuts_short_is_said_to_be_none_before_the_end() {
        for (text, expected_arguments) in [
            (r#"{"name": "f", "arguments": 1.}"#, "1."),
            (r#"{"name": "f", "arguments": -}"#, "-"),
            (r#"{"name": "f", "arguments": nu}"#, "nu"),
        ] {
            let events = run(&[text]);
            assert_eq!(arguments(&events), expected_arguments, "{text:?}");
            assert_eq!(
                &events[events.len() - 2..],
                &[
                    Event::Malformed {
                        text: Text::default(),
                        why: MalformedReason::InvalidArguments,
                    },
                    Event::ToolCallEnd {
                        index: 0,
                        source: Text::uncounted("}"),
                    },
                ],
                "{text:?}: {events:#?}"
            );
            assert_eq!(sources(&events), text, "{text:?}");
        }
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
    fn arguments_that_stop_being_a_json_prefix_say_so_there_and_go_on_as_written() {
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
                    text: Text::default(),
                    why: MalformedReason::InvalidArguments,
                },
                Event::ToolCallArguments {
                    index: 0,
                    json: "abc".into(),
                    source: Text::uncounted("abc"),
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
                Event::Malformed { text, .. } => Some(&*text.text),
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
            // A word the trailer cuts short of a literal is no JSON to a strict reader: held, it
            // is handed back whole, as a word that is no literal is.
            (r#"{"name": "f", "arguments": nul</f>"#, "", "nul</f>"),
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
            source: Text::uncounted(source.to_owned()),
        };
        let end = |source: &str| Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted(source.to_owned()),
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
        // no `{}` is made up for it; the call ends with what arrived, a `Malformed` event of no
        // bytes saying that it has no whole arguments, and the bytes from the stop on are the
        // caller's, as before.
        let (events, tail) = feed(r#"{"name": "f", "arguments": }</f>"#, false);
        assert_eq!(
            events,
            vec![
                start(r#"{"name": "f", "arguments": "#),
                Event::Malformed {
                    text: Text::default(),
                    why: MalformedReason::InvalidArguments,
                },
                end("")
            ],
            "an arguments member without a value is not a call with `{{}}`"
        );
        assert_eq!(tail, Some(("}</f>".to_string(), Some(' '))));

        let (events, tail) = feed(r#"{"arguments": {"a": 1}, x</f>"#, false);
        assert_eq!(
            events,
            vec![Event::Malformed {
                text: Text::uncounted(r#"{"arguments": {"a": 1}, "#),
                why: MalformedReason::CallWithoutName,
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
