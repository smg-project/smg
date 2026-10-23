//! A call written as its tool's name, a marker, then JSON: the events of one such call, from the
//! bytes after the table's opener.
//!
//! Two tables write a call this way and differ in what the JSON after the marker is, which the
//! assembler takes as a parameter ([`After`]):
//!
//! - **The call's object** ([`After::Object`]), which names the function again. Inkling writes a
//!   call on one line, broken here after the marker, as
//!
//!   ```text
//!   <|message_model|>get_weather<|content_invoke_tool_json|>
//!   {"name":"get_weather","args":{"city":"Paris"}}<|end_message|>
//!   ```
//!
//!   and a thought or an answer as `<|message_model|><|content_thinking|>…<|end_message|>` or
//!   `<|message_model|><|content_text|>…<|end_message|>`. The engine's table owns the message
//!   markers ([`formats::inkling`]): it enters this assembler after every `<|message_model|>`,
//!   since what the message is comes only after it, and leaves at the terminal that says so. For
//!   a thought or an answer nothing was fed and no call starts. For a call the bytes before the
//!   marker are the tool's name as the template writes it, which repeats the object's: the
//!   assembler drops them with the marker as the template's wrapping and reads the object with
//!   the JSON assembler ([`Assembler`](super::Assembler)), whose arguments member is named `args`
//!   here.
//!
//! - **The arguments object alone** ([`After::Arguments`]). Mistral's templates (Devstral 2,
//!   Ministral 3, Mistral Medium 3.5, Mistral Small 4) write a call as
//!
//!   ```text
//!   [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}
//!   ```
//!
//!   and a turn's calls one after another, each from its own `[TOOL_CALLS]`, with nothing between
//!   or after them. The engine's table owns the opener ([`formats::mistral_v13`]): it enters this
//!   assembler after every `[TOOL_CALLS]`, and leaves at the next one or at the end of the turn.
//!   The bytes before the marker are the tool's name, as the template writes it and the only
//!   place it is written: `ToolCallStart` carries them with the marker as its source once the
//!   marker has arrived, and the object after the marker is the arguments, read by the
//!   arguments-object reader (`named_arguments`), every byte of it a fragment for as long as the
//!   bytes keep it a JSON prefix.
//!
//! Either way the header (`Header`) holds the bytes before the marker until it arrives, so a
//! marker cut across pieces is found when its last byte comes; the object's close is the call's
//! end, and bytes after it are not taken, so the engine routes what follows the object (the
//! message's closing terminal is the region's, dropped by the engine as Qwen3's `</tool_call>`
//! is; the next `[TOOL_CALLS]` is the region's too). A name with no marker after it is reported
//! at the region's end by the reason its syntax names (a name without a call, a name without
//! arguments), and as `Malformed { UnterminatedRegion }` when the stream is cut; an empty header
//! (a thought's or an answer's, two openers in a row) leaves nothing.
//!
//! What the syntax does not check: the name before the marker against the object's, where the
//! object names the function too (the object is what the template writes as the call, and a
//! mismatch between the two spellings is not reported); and the name is taken as written, spaces
//! included, since the templates write it without any.
//!
//! [`formats::inkling`]: crate::formats::inkling()
//! [`formats::mistral_v13`]: crate::formats::mistral_v13()

use super::{
    header::{Found, Header},
    named_arguments,
};
use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    json,
};

/// What the marker introduces: the call's object, or its arguments alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum After {
    /// The call's object, which names the call: the header is the template's wrapping, dropped.
    Object,
    /// The arguments object alone: the header names the call.
    Arguments,
}

impl After {
    /// The marker arrived after `framing`, at `at` in it: the event the header's bytes make, and
    /// the reader of what follows for the call at `index` with `id`.
    fn open(self, index: u32, id: &str, framing: String, at: usize, out: &mut Events) -> Body {
        match self {
            Self::Object => {
                out.push(Event::Dropped {
                    text: Text::uncounted(framing),
                    why: DropReason::Wrapper,
                });
                Body::Object(json::Assembler::new(index, id))
            }
            Self::Arguments => {
                out.push(Event::ToolCallStart {
                    index,
                    id: id.to_string(),
                    name: framing[..at].to_string(),
                    source: Text::uncounted(framing),
                });
                Body::Arguments(named_arguments::Object::new(index))
            }
        }
    }

    /// The reason for a name the region ended after with no marker: the call it never came to,
    /// or the arguments.
    fn without_json(self) -> MalformedReason {
        match self {
            Self::Object => MalformedReason::NameWithoutACall,
            Self::Arguments => MalformedReason::NameWithoutArguments,
        }
    }
}

/// The events of one call, from the bytes after the table's opener.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    after: After,
    /// The bytes before the marker, held until it arrives or the region ends.
    header: Header,
    /// The JSON after the marker, once it has arrived.
    body: Option<Body>,
}

/// The JSON after the marker, read as it arrives.
#[derive(Clone, Debug)]
enum Body {
    /// The call's object, read by the JSON assembler, which starts the call at the object's name.
    Object(json::Assembler),
    /// The arguments object alone; the call started with the header.
    Arguments(named_arguments::Object),
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned just
    /// after the table's opener; `marker` is the text between the tool's name and the JSON, and
    /// `after` says what the JSON is.
    pub fn new(index: u32, id: impl Into<String>, marker: &'static str, after: After) -> Self {
        Self {
            index,
            id: id.into(),
            after,
            header: Header::new(marker),
            body: None,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    #[cfg(test)]
    pub fn started(&self) -> bool {
        self.body.as_ref().is_some_and(Body::started)
    }

    /// Append the next bytes of the region, push the events they complete, and return how many of
    /// the bytes the call took: all of them before the object closes, up to its closing brace in
    /// the piece that carries it, none after it.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        if let Some(body) = &mut self.body {
            return body.feed(bytes, out);
        }
        let Some(Found { framing, at, after }) = self.header.push(bytes) else {
            return bytes.len();
        };
        let mut body = self.after.open(self.index, &self.id, framing, at, out);
        let taken = body.feed(&after, out);
        self.body = Some(body);
        bytes.len() - (after.len() - taken)
    }

    /// The region ended at the table's terminal: the message's closing terminal, the next call's
    /// opener or the turn's end. A name with no marker after it is reported by the reason its
    /// syntax names; JSON that never closed is finished as its reader finishes it; an empty
    /// header leaves nothing.
    pub fn close(self, out: &mut Events) {
        let why = self.after.without_json();
        self.ended(why, out);
    }

    /// The stream was cut: a name held without its marker comes back as
    /// `Malformed { UnterminatedRegion }`, and JSON that never closed is finished as its reader
    /// finishes it.
    pub fn finish(self, out: &mut Events) {
        self.ended(MalformedReason::UnterminatedRegion, out);
    }

    /// No more bytes come: the header's bytes come back as `Malformed` for `why`, or the JSON
    /// after the marker ends with what arrived.
    fn ended(self, why: MalformedReason, out: &mut Events) {
        match self.body {
            Some(body) => body.finish(out),
            None => self.header.ended(why, out),
        }
    }
}

impl Body {
    #[cfg(test)]
    fn started(&self) -> bool {
        match self {
            Self::Object(object) => object.started(),
            Self::Arguments(_) => true,
        }
    }

    fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        match self {
            Self::Object(object) => object.feed(bytes, out),
            Self::Arguments(arguments) => arguments.feed(bytes, out),
        }
    }

    fn finish(self, out: &mut Events) {
        match self {
            Self::Object(object) => object.finish(out),
            Self::Arguments(arguments) => arguments.finish(out),
        }
    }
}

impl crate::engine::CallAssembler for Assembler {
    fn feed(&mut self, text: &str, _declared: &crate::Declared, out: &mut Events) -> usize {
        self.feed(text, out)
    }

    /// The region's terminal (the message's close, the next call's opener) is the region's: the
    /// object closed the call at its brace.
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
    use crate::testing::bytes;

    const OBJECT_MARKER: &str = "<|content_invoke_tool_json|>";
    const OBJECT_CALL: &str = concat!(
        "get_weather<|content_invoke_tool_json|>",
        "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\"}}"
    );
    const ARGUMENTS_MARKER: &str = "[ARGS]";
    const ARGUMENTS_CALL: &str = "get_weather[ARGS]{\"city\": \"Paris\", \"days\": [1, 2]}";
    const ARGUMENTS: &str = "{\"city\": \"Paris\", \"days\": [1, 2]}";

    fn run(
        after: After,
        marker: &'static str,
        pieces: &[&str],
        cut: bool,
    ) -> (Vec<Event<'static>>, usize) {
        let mut assembler = Assembler::new(0, "call_0", marker, after);
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

    /// A call whose object follows the marker (the Inkling spelling).
    fn run_object(pieces: &[&str], cut: bool) -> (Vec<Event<'static>>, usize) {
        run(After::Object, OBJECT_MARKER, pieces, cut)
    }

    /// A call whose arguments alone follow the marker (the Mistral spelling).
    fn run_arguments(pieces: &[&str], cut: bool) -> (Vec<Event<'static>>, usize) {
        run(After::Arguments, ARGUMENTS_MARKER, pieces, cut)
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

    fn per_char(text: &str) -> Vec<&str> {
        text.char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect()
    }

    #[test]
    fn an_object_without_its_arguments_member_is_a_call_with_an_empty_object() {
        // The object's assembler says `{}` for a call whose object lacks the member, here `args`,
        // whole and byte by byte.
        let call = "get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\"}";
        for (events, taken) in [
            run_object(&[call], false),
            run_object(&per_char(call), false),
        ] {
            assert_eq!(taken, call.len());
            assert_eq!(bytes(&events), call);
            assert_eq!(arguments(&events), "{}");
            assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
        }
    }

    #[test]
    fn a_call_message_drops_its_framing_and_reads_its_object_with_every_byte() {
        let (events, taken) = run_object(&[OBJECT_CALL], false);
        assert_eq!(taken, OBJECT_CALL.len());
        assert_eq!(bytes(&events), OBJECT_CALL);
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
    fn a_call_names_its_tool_before_the_marker_and_streams_the_object_as_its_arguments() {
        let (events, taken) = run_arguments(&[ARGUMENTS_CALL], false);
        assert_eq!(taken, ARGUMENTS_CALL.len());
        assert_eq!(bytes(&events), ARGUMENTS_CALL);
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
    fn the_marker_and_the_object_may_arrive_in_pieces() {
        let pieces = [
            "get_weat",
            "her<|content_inv",
            "oke_tool_json|>{\"name\":\"get_weather\",",
            "\"args\":{\"city\":\"Pa",
            "ris\"}}",
        ];
        let (events, taken) = run_object(&pieces, false);
        assert_eq!(taken, pieces.concat().len());
        assert_eq!(bytes(&events), pieces.concat());
        assert_eq!(arguments(&events), "{\"city\":\"Paris\"}");
        assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
    }

    #[test]
    fn the_events_are_the_same_however_the_bytes_are_cut() {
        let (whole, _) = run_arguments(&[ARGUMENTS_CALL], false);
        for cut in 1..ARGUMENTS_CALL.len() {
            if !ARGUMENTS_CALL.is_char_boundary(cut) {
                continue;
            }
            let (pieces, taken) =
                run_arguments(&[&ARGUMENTS_CALL[..cut], &ARGUMENTS_CALL[cut..]], false);
            assert_eq!(taken, ARGUMENTS_CALL.len(), "cut at {cut}");
            assert_eq!(bytes(&pieces), ARGUMENTS_CALL, "cut at {cut}");
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
    fn a_long_header_fed_a_byte_at_a_time_reads_as_it_does_whole() {
        // The search for the marker covers the new bytes and the marker's length before them, so
        // a long name costs each piece its own length; the result is the same as fed whole, the
        // marker found whether it arrives in one piece or split anywhere.
        let name = "n".repeat(4_000);
        let call = format!("{name}{OBJECT_MARKER}{{\"name\":\"n\",\"args\":{{}}}}");
        let (whole, taken_whole) = run_object(&[&call], false);
        let (one_at_a_time, taken) = run_object(&per_char(&call), false);
        assert_eq!(bytes(&one_at_a_time), bytes(&whole));
        assert_eq!(taken, taken_whole);
        // A multibyte name with the marker split in two: the search never lands inside a character.
        let call = format!("héé{OBJECT_MARKER}{{\"name\":\"h\",\"args\":{{}}}}");
        let split = 3 + OBJECT_MARKER.len() / 2;
        let (events, _) = run_object(&[&call[..split], &call[split..]], false);
        assert_eq!(bytes(&events), call);
    }

    #[test]
    fn bytes_after_the_object_are_not_taken() {
        let text = format!("{OBJECT_CALL} trailing");
        let (_, taken) = run_object(&[&text], false);
        assert_eq!(taken, OBJECT_CALL.len());
        let text = format!("{ARGUMENTS_CALL} and more");
        let (events, taken) = run_arguments(&[&text], false);
        assert_eq!(taken, ARGUMENTS_CALL.len());
        assert_eq!(bytes(&events), ARGUMENTS_CALL);
        assert_eq!(arguments(&events), ARGUMENTS);
    }

    #[test]
    fn a_thought_or_an_answer_feeds_nothing_and_leaves_nothing() {
        let (events, taken) = run_object(&[], false);
        assert!(events.is_empty());
        assert_eq!(taken, 0);
        let (events, _) = run_arguments(&[], false);
        assert!(events.is_empty(), "two openers in a row leave nothing");
    }

    #[test]
    fn a_name_with_no_call_after_it_is_reported_at_the_messages_end() {
        let (events, _) = run_object(&["get_weather"], false);
        assert_eq!(
            events,
            [Event::Malformed {
                text: Text::uncounted("get_weather"),
                why: MalformedReason::NameWithoutACall,
            }]
        );
    }

    #[test]
    fn a_name_without_the_marker_is_reported_at_the_regions_end_and_as_cut_when_the_stream_is() {
        let (events, taken) = run_arguments(&["get_weather"], false);
        assert_eq!(taken, "get_weather".len());
        assert_eq!(
            events,
            vec![Event::Malformed {
                text: Text::uncounted("get_weather"),
                why: MalformedReason::NameWithoutArguments,
            }]
        );
        let (events, _) = run_arguments(&["get_wea"], true);
        assert_eq!(
            events,
            vec![Event::Malformed {
                text: Text::uncounted("get_wea"),
                why: MalformedReason::UnterminatedRegion,
            }]
        );
    }

    #[test]
    fn a_cut_stream_leaves_the_name_unterminated_and_the_object_as_the_json_assembler_does() {
        let (in_name, _) = run_object(&["get_wea"], true);
        assert_eq!(
            in_name,
            [Event::Malformed {
                text: Text::uncounted("get_wea"),
                why: MalformedReason::UnterminatedRegion,
            }]
        );
        let (in_object, _) = run_object(
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

    #[test]
    fn arguments_cut_short_end_with_what_arrived_and_the_call_counts_as_started() {
        let cut = "get_weather[ARGS]{\"city\": \"Par";
        let mut assembler = Assembler::new(0, "call_0", ARGUMENTS_MARKER, After::Arguments);
        let mut out = Events::new();
        assert!(!assembler.started());
        let taken = assembler.feed(cut, &mut out);
        assert!(assembler.started(), "the header started the call");
        assembler.finish(&mut out);
        let events = out.drain();
        assert_eq!(taken, cut.len());
        assert_eq!(bytes(&events), cut);
        assert_eq!(arguments(&events), "{\"city\": \"Par");
        assert!(matches!(events.last(), Some(Event::ToolCallEnd { .. })));
    }
}
