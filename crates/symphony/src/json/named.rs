//! A call written as its tool's name, a marker, then one JSON object: the events of one Inkling
//! call message, from the bytes after `<|message_model|>`.
//!
//! Inkling writes a call on one line, broken here after the marker, as
//!
//! ```text
//! <|message_model|>get_weather<|content_invoke_tool_json|>
//! {"name":"get_weather","args":{"city":"Paris"}}<|end_message|>
//! ```
//!
//! and a thought or an answer as `<|message_model|><|content_thinking|>…<|end_message|>` or
//! `<|message_model|><|content_text|>…<|end_message|>`. The engine's table owns the message
//! markers ([`formats::inkling`]): it enters this assembler after every `<|message_model|>`, since
//! what the message is comes only after it, and leaves at the terminal that says so. For a thought
//! or an answer nothing was fed and no call starts. For a call the bytes before the marker are the
//! tool's name as the template writes it, which repeats the object's: the assembler drops them with
//! the marker as the template's wrapping and reads the object with the JSON assembler
//! ([`Assembler`](super::Assembler)), whose arguments member is named `args` here. Bytes after the
//! object are not taken, so the engine routes what follows it; the message's closing terminal is
//! the region's, dropped by the engine as Qwen3's `</tool_call>` is.
//!
//! What the syntax does not check: the name before the marker against the object's. The object is
//! what the template writes as the call, and it names the function; a mismatch between the two
//! spellings is not reported.
//!
//! [`formats::inkling`]: crate::formats::inkling()

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    json,
};

const NAME_WITHOUT_A_CALL: &str = "a message that names a tool and writes no call";

/// The events of one call message, from the bytes after `<|message_model|>`.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    marker: &'static str,
    /// The bytes before the marker, held until it arrives or the message ends.
    header: String,
    /// The object, once the marker has arrived.
    object: Option<json::Assembler>,
}

impl Assembler {
    /// An assembler for the message at `index` with the id the format minted for it, positioned
    /// just after `<|message_model|>`; `marker` is the text between the tool's name and the object.
    pub fn new(index: u32, id: impl Into<String>, marker: &'static str) -> Self {
        Self {
            index,
            id: id.into(),
            marker,
            header: String::new(),
            object: None,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.object.as_ref().is_some_and(json::Assembler::started)
    }

    /// Append the next bytes of the message, push the events they complete, and return how many of
    /// the bytes the message took: all of them before the object closes, up to its closing brace in
    /// the piece that carries it, none after it.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        if let Some(object) = &mut self.object {
            return object.feed(bytes, out);
        }
        // The marker can only end inside the new bytes, so the search starts `marker.len() - 1`
        // bytes before them (at a character boundary): a header that never meets the marker
        // costs each piece its own length, not the header's.
        let before = self.header.len();
        self.header.push_str(bytes);
        let from = before.saturating_sub(self.marker.len() - 1);
        let from = (0..=from)
            .rev()
            .find(|&at| self.header.is_char_boundary(at))
            .unwrap_or(0);
        let Some(at) = self.header[from..].find(self.marker).map(|at| from + at) else {
            return bytes.len();
        };
        // The marker ends inside this piece, so what follows it is this piece's too.
        let after = self.header.split_off(at + self.marker.len());
        let framing = std::mem::take(&mut self.header);
        out.push(Event::Dropped {
            text: Text::uncounted(framing),
            why: DropReason::Wrapper,
        });
        let mut object = json::Assembler::new(self.index, self.id.clone());
        let taken = object.feed(&after, out);
        self.object = Some(object);
        bytes.len() - (after.len() - taken)
    }

    /// The message ended at the table's terminal. A name with no marker after it is reported; an
    /// object that never closed is finished as the JSON assembler finishes one; an empty header,
    /// a thought's or an answer's, leaves nothing.
    pub fn close(self, out: &mut Events) {
        match self.object {
            Some(object) => object.finish(out),
            None if !self.header.is_empty() => out.push(Event::Malformed {
                text: Text::uncounted(self.header),
                why: MalformedReason::Other(NAME_WITHOUT_A_CALL.to_string()),
            }),
            None => {}
        }
    }

    /// The stream was cut: what was held comes back as `Malformed { UnterminatedRegion }`, and an
    /// object that never closed is finished as the JSON assembler finishes one.
    pub fn finish(self, out: &mut Events) {
        match self.object {
            Some(object) => object.finish(out),
            None if !self.header.is_empty() => out.push(Event::Malformed {
                text: Text::uncounted(self.header),
                why: MalformedReason::UnterminatedRegion,
            }),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MARKER: &str = "<|content_invoke_tool_json|>";
    const CALL: &str = concat!(
        "get_weather<|content_invoke_tool_json|>",
        "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
    );

    fn run(pieces: &[&str], cut: bool) -> (Vec<Event>, usize) {
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
        (out.drain(), taken)
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

    fn arguments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_call_message_drops_its_framing_and_reads_its_object_with_every_byte() {
        let (events, taken) = run(&[CALL], false);
        assert_eq!(taken, CALL.len());
        assert_eq!(bytes(&events), CALL);
        assert_eq!(
            events[0],
            Event::Dropped {
                text: Text::uncounted("get_weather<|content_invoke_tool_json|>"),
                why: DropReason::Wrapper,
            }
        );
        assert!(matches!(
            &events[1],
            Event::ToolCallStart { index: 0, name, source, .. }
                if name == "get_weather" && source.text == "{\"name\":\"get_weather\",\"args\":"
        ));
        assert_eq!(arguments(&events), "{\"city\":\"Paris\"}");
        assert!(matches!(
            events.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text == "}"
        ));
    }

    #[test]
    fn the_marker_and_the_object_may_arrive_in_pieces() {
        let pieces = [
            "get_weat",
            "her<|content_inv",
            "oke_tool_json|>{\"name\":\"get_weather\",",
            "\"args\":{\"city\":\"Pa",
            "ris\"}}",
        ];
        let (events, taken) = run(&pieces, false);
        assert_eq!(taken, pieces.concat().len());
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(arguments(&events), "{\"city\":\"Paris\"}");
        assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
    }

    #[test]
    fn a_long_header_fed_a_byte_at_a_time_reads_as_it_does_whole() {
        // The search for the marker covers the new bytes and the marker's length before them, so
        // a long name costs each piece its own length; the result is the same as fed whole, the
        // marker found whether it arrives in one piece or split anywhere.
        let name = "n".repeat(4_000);
        let call = format!("{name}{MARKER}{{\"name\":\"n\",\"args\":{{}}}}");
        let (whole, taken_whole) = run(&[&call], false);
        let pieces: Vec<&str> = call
            .char_indices()
            .map(|(at, c)| &call[at..at + c.len_utf8()])
            .collect();
        let (one_at_a_time, taken) = run(&pieces, false);
        assert_eq!(bytes(&one_at_a_time), bytes(&whole));
        assert_eq!(taken, taken_whole);
        // A multibyte name with the marker split in two: the search never lands inside a character.
        let call = format!("héé{MARKER}{{\"name\":\"h\",\"args\":{{}}}}");
        let split = 3 + MARKER.len() / 2;
        let (events, _) = run(&[&call[..split], &call[split..]], false);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn bytes_after_the_object_are_not_taken() {
        let text = format!("{CALL} trailing");
        let (_, taken) = run(&[&text], false);
        assert_eq!(taken, CALL.len());
    }

    #[test]
    fn a_thought_or_an_answer_feeds_nothing_and_leaves_nothing() {
        let (events, taken) = run(&[], false);
        assert!(events.is_empty());
        assert_eq!(taken, 0);
    }

    #[test]
    fn a_name_with_no_call_after_it_is_reported_at_the_messages_end() {
        let (events, _) = run(&["get_weather"], false);
        assert_eq!(
            events,
            [Event::Malformed {
                text: Text::uncounted("get_weather"),
                why: MalformedReason::Other(NAME_WITHOUT_A_CALL.to_string()),
            }]
        );
    }

    #[test]
    fn a_cut_stream_leaves_the_name_unterminated_and_the_object_as_the_json_assembler_does() {
        let (in_name, _) = run(&["get_wea"], true);
        assert_eq!(
            in_name,
            [Event::Malformed {
                text: Text::uncounted("get_wea"),
                why: MalformedReason::UnterminatedRegion,
            }]
        );
        let (in_object, _) = run(
            &["get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"ci"],
            true,
        );
        assert!(in_object
            .iter()
            .any(|e| matches!(e, Event::ToolCallStart { name, .. } if name == "get_weather")));
        assert!(matches!(
            in_object.last(),
            Some(Event::ToolCallEnd { source, .. }) if source.text.is_empty()
        ));
    }
}
