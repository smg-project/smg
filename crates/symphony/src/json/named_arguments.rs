//! The arguments object of a call named before it: the events of the object alone, from its first
//! byte, for a call that started with the name the template wrote before its marker
//! ([`named`](super::named), with [`After::Arguments`](super::named::After::Arguments)).
//!
//! Every byte of the object is a fragment for as long as the bytes keep it a JSON prefix
//! ([`Prefix`](super::Prefix)), and `Malformed { InvalidArguments }` from the first byte that does
//! not; its close, found by the outline's scanner ([`Scanner`]), is the call's end, and bytes
//! after it are not taken, so the caller routes what follows the object. The bytes are accounted
//! for through the judge the JSON assembler uses ([`Fragments`]), so that the two readers say the
//! same of the same bytes.
//!
//! What the reader leaves open: a value that is not an object is streamed for as long as it is a
//! JSON prefix and ends with the region, since the outline finds no close for it, a word among
//! them held until it is spelled or refused, so that the pieces say of it what the whole says. The
//! templates write none.

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::{fragments::Fragments, is_complete, outline::Scanner},
};

/// The arguments object as it arrives.
#[derive(Clone, Debug)]
pub(super) struct Object {
    index: u32,
    text: String,
    /// The outline of `text`, read as it grows: its close is the object's.
    scanner: Scanner,
    /// The bytes, sent, held or reported as they arrive.
    fragments: Fragments,
    done: bool,
}

impl Object {
    /// The reader of the arguments of the call at `index`, positioned at the object's first byte.
    pub(super) fn new(index: u32) -> Self {
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
    pub(super) fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
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
    pub(super) fn finish(mut self, out: &mut Events) {
        if self.done {
            return;
        }
        self.fragments.send_held(&self.text, self.index, out);
        if !self.fragments.refused() && !is_complete(&self.fragments.escaped(0, &self.text)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The object fed in `pieces` and finished, as the region's end or the stream's finishes it:
    /// its events and how many bytes it took.
    fn run(pieces: &[&str]) -> (Vec<Event>, usize) {
        let mut object = Object::new(0);
        let mut out = Events::new();
        let mut taken = 0;
        for piece in pieces {
            taken += object.feed(piece, &mut out);
        }
        object.finish(&mut out);
        (out.drain(), taken)
    }

    fn per_char(text: &str) -> Vec<&str> {
        text.char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect()
    }

    /// Every byte the events account for, in order.
    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|e| match e {
                Event::ToolCallArguments { source, .. } | Event::ToolCallEnd { source, .. } => {
                    source.text.as_str()
                }
                Event::Malformed { text, .. } => text.text.as_str(),
                other => panic!("an event the object never pushes: {other:?}"),
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
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_word_under_way_is_held_so_the_pieces_say_of_it_what_the_whole_says() {
        // A word that is no literal is refused from its first letter and nothing of it is sent
        // as arguments however the bytes were cut; a literal, whole or cut short at the end, is
        // sent once it is spelled or once nothing more can come; a number streams as far as it
        // is one. A value that is no object ends with the region.
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
            let (whole, _) = run(&[value]);
            let (streamed, _) = run(&per_char(value));
            for (how, events) in [("whole", &whole), ("per character", &streamed)] {
                assert_eq!(
                    arguments(events),
                    expected_arguments,
                    "{value:?} {how}: arguments"
                );
                assert_eq!(
                    malformed(events),
                    expected_malformed,
                    "{value:?} {how}: malformed"
                );
                assert_eq!(bytes(events), value, "{value:?} {how}: every byte");
                assert!(
                    matches!(events.last(), Some(Event::ToolCallEnd { .. })),
                    "{value:?} {how}: the end"
                );
            }
        }
    }

    #[test]
    fn a_closed_object_takes_its_bytes_up_to_the_brace_and_a_cut_one_ends_with_what_arrived() {
        let object = "{\"city\": \"Paris\", \"days\": [1, 2]}";
        let text = format!("{object} and more");
        let (events, taken) = run(&[&text]);
        assert_eq!(taken, object.len());
        assert_eq!(bytes(&events), object);
        assert_eq!(arguments(&events), object);
        assert_eq!(
            events.last(),
            Some(&Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            })
        );
        let cut = "{\"city\": \"Par";
        let (events, taken) = run(&[cut]);
        assert_eq!(taken, cut.len());
        assert_eq!(bytes(&events), cut);
        assert_eq!(arguments(&events), cut);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Malformed { text, why: MalformedReason::UnterminatedRegion } if text.text.is_empty()
        )));
        assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
    }

    #[test]
    fn bytes_that_break_the_json_prefix_are_malformed_from_there_whatever_the_cut() {
        // `@` is no JSON value's first byte, so the boundary does not depend on the cut (junk that
        // begins like a literal does, as the prefix judge is as lenient as the old crate's).
        let text = "{\"city\": @}";
        let (whole, _) = run(&[text]);
        assert_eq!(bytes(&whole), text);
        assert_eq!(arguments(&whole), "{\"city\": ");
        assert!(whole.iter().any(|e| matches!(
            e,
            Event::Malformed { why: MalformedReason::InvalidArguments, text } if text.text == "@}"
        )));
        for cut in 1..text.len() {
            let (pieces, _) = run(&[&text[..cut], &text[cut..]]);
            assert_eq!(arguments(&pieces), arguments(&whole), "cut at {cut}");
            assert_eq!(bytes(&pieces), text, "cut at {cut}");
        }
    }
}
