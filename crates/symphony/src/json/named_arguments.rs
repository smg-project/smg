//! A call written as its tool's name, a marker, then the arguments object alone: the events of
//! one Mistral call, from the bytes after `[TOOL_CALLS]`.
//!
//! Mistral's templates (Devstral 2, Ministral 3, Mistral Medium 3.5, Mistral Small 4) write a call
//! as
//!
//! ```text
//! [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}
//! ```
//!
//! and a turn's calls one after another, each from its own `[TOOL_CALLS]`, with nothing between
//! or after them. The engine's table owns the opener ([`formats::mistral_v13`]): it enters this
//! assembler after every `[TOOL_CALLS]`, and leaves at the next one or at the end of the turn. The
//! bytes before the marker are the tool's name, as the template writes it and the only place it
//! is written: `ToolCallStart` carries them with the marker as its source once the marker has
//! arrived. The object after the marker is the arguments, every byte of it a fragment for as long
//! as the bytes keep it a JSON prefix ([`Prefix`](super::Prefix)), and `Malformed { InvalidArguments }` from the
//! first byte that does not; its close, found by the outline's scanner ([`Scanner`]), is the call's
//! end, and bytes after it are not taken, so the engine routes what follows the object.
//!
//! What the syntax does not check: the name before the marker is taken as written, spaces
//! included, since the template writes it without any; and a value that is not an object is
//! streamed for as long as it is a JSON prefix and ends with the region, since the outline finds
//! no close for it, a word among them held until it is spelled or refused, so that the pieces say
//! of it what the whole says. The template writes neither.
//!
//! [`formats::mistral_v13`]: crate::formats::mistral_v13()

use super::header::{Found, Header};
use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::{fragments::Fragments, is_complete, outline::Scanner},
};

const NAME_WITHOUT_ARGUMENTS: &str = "a call that names a tool and writes no arguments";

/// The events of one call, from the bytes after the call's opener.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    /// The bytes before the marker, held until it arrives or the region ends.
    header: Header,
    /// The arguments, once the marker has arrived.
    arguments: Option<Arguments>,
}

/// The arguments object as it arrives.
#[derive(Clone, Debug)]
struct Arguments {
    index: u32,
    text: String,
    /// The outline of `text`, read as it grows: its close is the object's.
    scanner: Scanner,
    /// The bytes, sent, held or reported as they arrive.
    fragments: Fragments,
    done: bool,
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned just
    /// after the call's opener; `marker` is the text between the tool's name and the object.
    pub fn new(index: u32, id: impl Into<String>, marker: &'static str) -> Self {
        Self {
            index,
            id: id.into(),
            header: Header::new(marker),
            arguments: None,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.arguments.is_some()
    }

    /// Append the next bytes of the region, push the events they complete, and return how many of
    /// the bytes the call took: all of them before the object closes, up to its closing brace in
    /// the piece that carries it, none after it.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        if let Some(arguments) = &mut self.arguments {
            return arguments.feed(bytes, out);
        }
        let Some(Found { framing, at, after }) = self.header.push(bytes) else {
            return bytes.len();
        };
        out.push(Event::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: framing[..at].to_string(),
            source: Text::uncounted(framing),
        });
        let mut arguments = Arguments::new(self.index);
        let taken = arguments.feed(&after, out);
        self.arguments = Some(arguments);
        bytes.len() - (after.len() - taken)
    }

    /// The region ended at the table's terminal: the next call's opener, or the turn's end. A
    /// name with no marker after it is reported; arguments that never closed end with what
    /// arrived; an empty header, two openers in a row, leaves nothing.
    pub fn close(self, out: &mut Events) {
        match self.arguments {
            Some(arguments) => arguments.finish(out),
            None => self.header.ended(
                || MalformedReason::Other(NAME_WITHOUT_ARGUMENTS.into()),
                out,
            ),
        }
    }

    /// The stream was cut: a name held without its marker comes back as
    /// `Malformed { UnterminatedRegion }`, and arguments that never closed end with what arrived.
    pub fn finish(self, out: &mut Events) {
        match self.arguments {
            Some(arguments) => arguments.finish(out),
            None => self
                .header
                .ended(|| MalformedReason::UnterminatedRegion, out),
        }
    }
}

impl Arguments {
    fn new(index: u32) -> Self {
        Self {
            index,
            text: String::new(),
            scanner: Scanner::default(),
            fragments: Fragments::default(),
            done: false,
        }
    }

    /// Append the next bytes of the object and push the fragments and the end they complete; how
    /// many of the bytes the object took.
    fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        if self.done {
            return 0;
        }
        let before = self.text.len();
        self.text.push_str(bytes);
        self.scanner.advance(&self.text);
        let close = self.scanner.found().close;
        let taken = match close {
            Some(close) => {
                // The scanner stopped at the brace; what follows it is not the object's.
                self.text.truncate(close);
                close - before
            }
            None => bytes.len(),
        };
        self.emit_new_bytes(close.is_none(), out);
        if close.is_some() {
            out.push(Event::ToolCallEnd {
                index: self.index,
                source: Text::default(),
            });
            self.done = true;
        }
        taken
    }

    /// No more bytes will come: a closed object has said everything; one that never closed ends
    /// with what arrived, every byte of it already a fragment or malformed text but for a word it
    /// still held, which goes first, as the prefix it is, and a `Malformed` event of no bytes
    /// before the end says that the arguments are no whole value, unless the bytes happen to be
    /// one (a value that is no object, which the template never writes).
    fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.fragments.send_held(&self.text, self.index, out);
        if !self.fragments.refused() && !is_complete(&self.text) {
            out.push(Event::Malformed {
                text: Text::default(),
                why: MalformedReason::UnterminatedRegion,
            });
        }
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::default(),
        });
    }

    /// The bytes that arrived since the last call: a fragment for the part that keeps the
    /// arguments a valid JSON prefix, malformed text for the rest, once the prefix has broken,
    /// and the bytes the judge may still go back to held while the object is `open`
    /// ([`Fragments::send`]); they go with the object's close, or when no more bytes can come
    /// ([`Self::finish`]).
    fn emit_new_bytes(&mut self, open: bool, out: &mut Events) {
        if self
            .fragments
            .send(&self.text, open, self.index, out)
            .is_some()
        {
            self.fragments.send_malformed(&self.text, out);
        }
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, _declared: &crate::tagged::Declared, out: &mut Events) -> usize {
        self.feed(text, out)
    }

    /// The next call's opener is the region's: the object closed the call at its brace.
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

    const MARKER: &str = "[ARGS]";
    const CALL: &str = "get_weather[ARGS]{\"city\": \"Paris\", \"days\": [1, 2]}";
    const ARGUMENTS: &str = "{\"city\": \"Paris\", \"days\": [1, 2]}";

    fn run(pieces: &[&str], cut: bool) -> (Vec<Event<'static>>, usize) {
        let mut assembler = Assembler::new(0, "call_0", MARKER);
        let mut out = Events::new();
        let mut taken = 0;
        for piece in pieces {
            taken += assembler.feed(piece, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(&mut out);
        }
        (out.into_owned().drain(), taken)
    }

    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::Content(t) | Event::Reasoning(t) => &*t.text,
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => &*text.text,
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => &*source.text,
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
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

    #[test]
    fn a_word_under_way_is_held_so_the_pieces_say_of_it_what_the_whole_says() {
        // A word that is no literal is refused from its first letter and nothing of it is sent
        // as arguments however the bytes were cut; a literal, whole or cut short at the end, is
        // sent once it is spelled or once nothing more can come; a number streams as far as it
        // is one. The region ends with the stream (cut) or at the next opener (closed).
        let cases = [
            ("nosuch", "", "nosuch"),
            ("trash", "", "trash"),
            ("nul", "nul", ""),
            ("null", "null", ""),
            ("1x", "1", "x"),
            ("{\"a\": nosuch}", "{\"a\": ", "nosuch}"),
            ("{\"a\": null}", "{\"a\": null}", ""),
            ("{\"e\": \"\\ud83c\\x\"}", "{\"e\": \"\\ud83c", "\\x\"}"),
            ("\"ab\\ud83c\\ude00 c\"", "\"ab\\ud83c\\ude00 c\"", ""),
            ("\tnosuch", "", "\tnosuch"),
            (" null", " null", ""),
        ];
        for (value, expected_arguments, expected_malformed) in cases {
            for cut in [true, false] {
                let call = format!("f[ARGS]{value}");
                let (whole, _) = run(&[&call], cut);
                let per_char: Vec<&str> = call
                    .char_indices()
                    .map(|(i, c)| &call[i..i + c.len_utf8()])
                    .collect();
                let (streamed, _) = run(&per_char, cut);
                for (how, events) in [("whole", &whole), ("per character", &streamed)] {
                    assert_eq!(
                        arguments(events),
                        expected_arguments,
                        "{call:?} cut={cut} {how}: arguments"
                    );
                    assert_eq!(
                        malformed(events),
                        expected_malformed,
                        "{call:?} cut={cut} {how}: malformed"
                    );
                    assert_eq!(bytes(events), call, "{call:?} cut={cut} {how}: every byte");
                }
            }
        }
    }

    #[test]
    fn a_call_names_its_tool_before_the_marker_and_streams_the_object_as_its_arguments() {
        let (events, taken) = run(&[CALL], false);
        assert_eq!(taken, CALL.len());
        assert_eq!(bytes(&events), CALL);
        assert!(matches!(
            &events[0],
            Event::ToolCallStart { index: 0, id, name, source }
                if id == "call_0" && name == "get_weather" && source.text == "get_weather[ARGS]"
        ));
        assert_eq!(arguments(&events), ARGUMENTS);
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { index: 0, source }) if source.text.is_empty()
        ));
    }

    #[test]
    fn the_events_are_the_same_however_the_bytes_are_cut() {
        let (whole, _) = run(&[CALL], false);
        for cut in 1..CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let (pieces, taken) = run(&[&CALL[..cut], &CALL[cut..]], false);
            assert_eq!(taken, CALL.len(), "cut at {cut}");
            assert_eq!(bytes(&pieces), CALL, "cut at {cut}");
            assert_eq!(arguments(&pieces), arguments(&whole), "cut at {cut}");
            assert_eq!(
                pieces
                    .iter()
                    .filter(|e| matches!(e, Event::ToolCallStart { .. }))
                    .count(),
                1,
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn bytes_after_the_object_are_left_to_the_engine() {
        let text = format!("{CALL} and more");
        let (events, taken) = run(&[&text], false);
        assert_eq!(taken, CALL.len());
        assert_eq!(bytes(&events), CALL);
        assert_eq!(arguments(&events), ARGUMENTS);
    }

    #[test]
    fn a_name_without_the_marker_is_reported_at_the_regions_end_and_as_cut_when_the_stream_is() {
        let (events, taken) = run(&["get_weather"], false);
        assert_eq!(taken, "get_weather".len());
        assert_eq!(
            events,
            vec![Event::Malformed {
                text: Text::uncounted("get_weather"),
                why: MalformedReason::Other(NAME_WITHOUT_ARGUMENTS.into()),
            }]
        );
        let (events, _) = run(&["get_wea"], true);
        assert_eq!(
            events,
            vec![Event::Malformed {
                text: Text::uncounted("get_wea"),
                why: MalformedReason::UnterminatedRegion,
            }]
        );
        let (events, _) = run(&[], false);
        assert!(events.is_empty(), "two openers in a row leave nothing");
    }

    #[test]
    fn an_object_cut_short_ends_with_what_arrived() {
        let cut = "get_weather[ARGS]{\"city\": \"Par";
        let (events, taken) = run(&[cut], true);
        assert_eq!(taken, cut.len());
        assert_eq!(bytes(&events), cut);
        assert_eq!(arguments(&events), "{\"city\": \"Par");
        assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
    }

    #[test]
    fn bytes_that_break_the_json_prefix_are_malformed_from_there_whatever_the_cut() {
        // `@` is no JSON value's first byte, so the boundary does not depend on the cut (junk that
        // begins like a literal does, as the prefix judge is as lenient as the old crate's).
        let text = "get_weather[ARGS]{\"city\": @}";
        let (whole, _) = run(&[text], false);
        assert_eq!(bytes(&whole), text);
        assert_eq!(arguments(&whole), "{\"city\": ");
        assert!(whole.iter().any(|e| matches!(
            e,
            Event::Malformed { why: MalformedReason::InvalidArguments, text } if text.text == "@}"
        )));
        for cut in 1..text.len() {
            let (pieces, _) = run(&[&text[..cut], &text[cut..]], false);
            assert_eq!(arguments(&pieces), arguments(&whole), "cut at {cut}");
            assert_eq!(bytes(&pieces), text, "cut at {cut}");
        }
    }
}
