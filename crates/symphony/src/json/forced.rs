//! The forced call: the arguments alone, the call named by the request. What the engine writes for
//! a forced function under the JSON-schema constraint the gateway builds for it (`tool_choice`
//! naming one function takes that function's `parameters` as the schema), the one shape known from
//! the request rather than from the model. The table ([`formats::forced`]) has one state, an
//! arguments state the output starts in, and no marker, so this assembler sees every byte of the
//! output, and the stream's end closes the call.
//!
//! The value reads through [`Arguments`]: the whitespace before it is dropped, its bytes are
//! fragments while they keep it a JSON prefix, and the rest is `Malformed { InvalidArguments }`.
//! The value's first byte starts the call, named by the function the request forces
//! ([`Declared::forced`]) and with no source byte of its own, since no output byte names it; the
//! call ends where the value is whole, or with the stream. Bytes after a whole value are not
//! taken: the engine reports them as text after the object. Without a forced function the output
//! has no call to name, which the gateway never asks for: every byte comes back as
//! `Malformed { Other }`.
//!
//! [`formats::forced`]: crate::formats::forced()
//! [`Declared::forced`]: crate::tagged::Declared::forced

use crate::{
    event::{Event, Events, MalformedReason, Text},
    json::arguments::{json_space, Arguments, AsWritten},
    tagged::Declared,
};

const NO_FORCED_FUNCTION: &str = "arguments with no forced function to name them";

/// The events of the forced call, from the output's first byte.
#[derive(Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    arguments: Arguments,
    /// `ToolCallStart` was pushed.
    started: bool,
    /// `ToolCallEnd` was pushed.
    ended: bool,
    /// No function was forced: every byte is reported.
    unnamed: bool,
}

impl Assembler {
    /// An assembler positioned at the output's start, for the call at `index` with the id the
    /// format minted for it.
    pub fn new(index: u32, id: impl Into<String>) -> Self {
        Self {
            index,
            id: id.into(),
            arguments: Arguments::new(index),
            started: false,
            ended: false,
            unnamed: false,
        }
    }

    /// Append the next bytes and push the events they complete; the number of bytes taken: every
    /// byte up to the value's end, and none after it.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) -> usize {
        if self.unnamed {
            unnamed(bytes, out);
            return bytes.len();
        }
        if self.ended {
            return 0;
        }
        let mut taken = 0;
        if !self.started {
            let value = bytes.trim_start_matches(json_space);
            taken = self
                .arguments
                .feed(&bytes[..bytes.len() - value.len()], &AsWritten, out);
            if value.is_empty() {
                return taken;
            }
            let Some(name) = declared.forced() else {
                self.unnamed = true;
                unnamed(value, out);
                return bytes.len();
            };
            out.push(Event::ToolCallStart {
                index: self.index,
                id: self.id.clone(),
                name: name.to_string(),
                source: Text::default(),
            });
            self.started = true;
        }
        taken += self.arguments.feed(&bytes[taken..], &AsWritten, out);
        if self.arguments.whole() {
            self.end(out);
        }
        taken
    }

    /// The stream ended: a value still open ends with what arrived, whole when a number or a word
    /// the bytes end on completes it.
    pub fn finish(mut self, out: &mut Events) {
        if self.started && !self.ended {
            self.arguments.finish(out);
            self.end(out);
        }
    }

    fn end(&mut self, out: &mut Events) {
        self.ended = true;
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::default(),
        });
    }
}

/// Bytes with no forced function to name a call for them: reported, not read.
fn unnamed(bytes: &str, out: &mut Events) {
    out.push(Event::Malformed {
        text: Text::uncounted(bytes),
        why: MalformedReason::Other(NO_FORCED_FUNCTION.to_string()),
    });
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, declared: &Declared, out: &mut Events) -> usize {
        self.feed(text, declared, out)
    }

    fn close(self: Box<Self>, _terminal: &str, out: &mut Events) -> bool {
        // The table has no terminal that closes the region; the end of the stream does.
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

    const ARGUMENTS: &str = "{\"city\": \"Paris\", \"days\": [1, 2]}";

    fn run(pieces: &[&str], declared: &Declared) -> (Vec<Event>, usize) {
        let mut assembler = Assembler::new(0, "call_0");
        let mut out = Events::new();
        let mut taken = 0;
        for piece in pieces {
            taken += assembler.feed(piece, declared, &mut out);
        }
        assembler.finish(&mut out);
        (out.drain(), taken)
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

    fn names(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }

    fn ends(events: &[Event]) -> usize {
        events
            .iter()
            .filter(|e| matches!(e, Event::ToolCallEnd { .. }))
            .count()
    }

    #[test]
    fn the_value_is_the_call_named_by_the_request_however_it_is_cut() {
        let declared = Declared::default().forcing("get_weather");
        for pieces in [
            vec![ARGUMENTS],
            vec!["  ", "\n{\"city\": \"Pa", "ris\", \"days\": [1, 2]}"],
            vec!["{\"city\": \"Paris\", \"days\": [1, 2]", "}"],
        ] {
            let (events, taken) = run(&pieces, &declared);
            assert_eq!(names(&events), ["get_weather"], "{pieces:?}");
            assert_eq!(arguments(&events), ARGUMENTS, "{pieces:?}");
            assert_eq!(ends(&events), 1, "{pieces:?}");
            assert_eq!(taken, pieces.concat().len(), "{pieces:?}");
        }
    }

    #[test]
    fn bytes_after_the_whole_value_are_left_to_the_engine() {
        let declared = Declared::default().forcing("f");
        let (events, taken) = run(&["{} trailing"], &declared);
        assert_eq!(arguments(&events), "{}");
        assert_eq!(ends(&events), 1);
        assert_eq!(taken, 2);
        let (_, taken) = run(&["{}", " more"], &declared);
        assert_eq!(taken, 2);
    }

    #[test]
    fn a_cut_value_ends_with_the_stream_and_a_blank_output_is_no_call() {
        let declared = Declared::default().forcing("f");
        let (events, _) = run(&["{\"city\": \"Par"], &declared);
        assert_eq!(names(&events), ["f"]);
        assert_eq!(arguments(&events), "{\"city\": \"Par");
        assert_eq!(ends(&events), 1);
        let (events, _) = run(&["  \n"], &declared);
        assert!(names(&events).is_empty());
        assert_eq!(ends(&events), 0);
        assert!(events.iter().all(|e| matches!(e, Event::Dropped { .. })));
    }

    #[test]
    fn without_a_forced_function_every_byte_is_reported() {
        let (events, taken) = run(&["{\"a\": 1}", " more"], &Declared::default());
        assert!(names(&events).is_empty());
        assert_eq!(taken, "{\"a\": 1} more".len());
        let reported: String = events
            .iter()
            .filter_map(|e| match e {
                Event::Malformed {
                    text,
                    why: MalformedReason::Other(why),
                } if why == NO_FORCED_FUNCTION => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reported, "{\"a\": 1} more");
    }
}
