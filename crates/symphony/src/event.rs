//! Typed events. Every byte of model output appears in exactly one event.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A run of text and, when the parser saw token ids, the number of engine tokens it came from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Text {
    /// The text.
    pub text: String,
    /// How many engine tokens produced it. `None` when the parser was not given token ids, as
    /// through the compatibility bridge over the old crates.
    pub tokens: Option<u32>,
}

impl Text {
    /// A text run with a known token count.
    pub fn new(text: impl Into<String>, tokens: u32) -> Self {
        Self {
            text: text.into(),
            tokens: Some(tokens),
        }
    }

    /// A text run whose token count is not known.
    pub fn uncounted(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: None,
        }
    }

    fn is_nothing(&self) -> bool {
        self.text.is_empty() && self.tokens.unwrap_or(0) == 0
    }

    fn append(&mut self, other: &Text) {
        self.text.push_str(&other.text);
        self.tokens = match (self.tokens, other.tokens) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
}

/// The runs of `text` in order, each all whitespace or none of it as `char::is_whitespace` says,
/// with whether it is whitespace. Every run ends on a character boundary, and the runs join back
/// into `text`. The parsers read the text between a format's structure by it: whitespace is the
/// template's wrapping, anything else is reported.
pub(crate) fn runs(text: &str) -> impl Iterator<Item = (&str, bool)> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let space = rest.chars().next()?.is_whitespace();
        let length = rest
            .char_indices()
            .find(|(_, c)| c.is_whitespace() != space)
            .map_or(rest.len(), |(at, _)| at);
        let (run, after) = rest.split_at(length);
        rest = after;
        Some((run, space))
    })
}

/// Why text was dropped rather than shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// A control or special token the format defines (a marker, a channel tag).
    ControlToken,
    /// Format wrapper text between regions (for example text between two tool invocations).
    Wrapper,
    /// Whitespace the format treats as structural.
    Whitespace,
}

/// Why a region could not be parsed. Its text is returned, never lost.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MalformedReason {
    /// The stream ended inside a region that never closed.
    UnterminatedRegion,
    /// Tool-call arguments did not parse under the format's argument syntax.
    InvalidArguments,
    /// A tool call named a function the request did not declare.
    UnknownTool,
    /// Anything else, described. The crate's own reasons are fixed texts and borrow; a reason
    /// built for the occasion is owned. On the wire both are the string, as before.
    Other(Cow<'static, str>),
}

/// The finish reason the parser reports after refining the engine's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop.
    Stop,
    /// Token limit.
    Length,
    /// At least one tool call was emitted and the model stopped.
    ToolCalls,
    /// The request was aborted.
    Abort,
    /// A finish reason this crate does not interpret, carried through from the engine verbatim.
    Other(String),
}

/// One parsed unit of model output, in wire order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// Visible assistant text.
    Content(Text),
    /// A reasoning region opened.
    ReasoningStart,
    /// Reasoning text.
    Reasoning(Text),
    /// The reasoning region closed.
    ReasoningEnd,
    /// A tool call began; the id is assigned by the parser so the caller never mints one.
    ToolCallStart {
        /// Position of this call among the output's calls, starting at zero.
        index: u32,
        /// The call id, in the format the model family uses.
        id: String,
        /// The function name.
        name: String,
        /// The output bytes this event accounts for: the opening marker and the name as written.
        source: Text,
    },
    /// A fragment of the call's arguments. Concatenated fragments for one index are always a
    /// valid JSON prefix and equal the final arguments when the call ends.
    ToolCallArguments {
        /// Which call the fragment belongs to.
        index: u32,
        /// The fragment, as JSON text.
        json: String,
        /// The output bytes this event accounts for, in the format's own syntax. Empty when the
        /// fragment came from bytes an earlier event already accounted for.
        source: Text,
    },
    /// The call's region closed.
    ToolCallEnd {
        /// Which call ended.
        index: u32,
        /// The output bytes this event accounts for: the closing marker, and any buffered bytes
        /// no earlier event of this call accounted for.
        source: Text,
    },
    /// Text the format consumed without showing it.
    Dropped {
        /// The text.
        text: Text,
        /// Why it was dropped.
        why: DropReason,
    },
    /// Text of a region the parser gave up on; the adapter decides what to do with it.
    Malformed {
        /// The text, returned whole.
        text: Text,
        /// Why parsing stopped.
        why: MalformedReason,
    },
    /// The stream ended.
    Finish {
        /// The refined finish reason.
        reason: FinishReason,
        /// Tool calls emitted in this output.
        tool_calls: u32,
        /// Engine tokens spent on reasoning, markers excluded.
        reasoning_tokens: u32,
    },
}

/// An ordered list of events that merges adjacent text of the same kind.
///
/// A parser pushes what it has as soon as it has it; two consecutive `Content` events become one,
/// and so do two consecutive `Reasoning` events. Nothing else is merged, and a text event with no
/// text and no tokens is not recorded.
///
/// Merging only ever touches events the caller has not taken yet. The caller takes events with
/// [`Events::drain`] after each `feed`; what it took is final, and the next push starts a new
/// event even when it is text of the same kind.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Events {
    items: Vec<Event>,
}

impl Events {
    /// An empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event, merging it into the previous one when both are text of the same kind.
    pub fn push(&mut self, event: Event) {
        match (&event, self.items.last_mut()) {
            (Event::Content(t), _) | (Event::Reasoning(t), _) if t.is_nothing() => {}
            (Event::Content(t), Some(Event::Content(prev))) => prev.append(t),
            (Event::Reasoning(t), Some(Event::Reasoning(prev))) => prev.append(t),
            _ => self.items.push(event),
        };
    }

    /// Append visible text.
    pub fn push_content(&mut self, text: Text) {
        self.push(Event::Content(text));
    }

    /// Append reasoning text.
    pub fn push_reasoning(&mut self, text: Text) {
        self.push(Event::Reasoning(text));
    }

    /// The events so far.
    pub fn as_slice(&self) -> &[Event] {
        &self.items
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether no event was recorded.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The output bytes these events account for, in order: content, reasoning, dropped and
    /// malformed text, and the source bytes of tool-call events. For a complete, well-formed
    /// parse this equals the model's output byte for byte; that is the conservation property.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for event in &self.items {
            match event {
                Event::Content(t) | Event::Reasoning(t) => out.push_str(&t.text),
                Event::Dropped { text, .. }
                | Event::Malformed { text, .. }
                | Event::ToolCallStart { source: text, .. }
                | Event::ToolCallArguments { source: text, .. }
                | Event::ToolCallEnd { source: text, .. } => out.push_str(&text.text),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => {}
            };
        }
        out
    }

    /// Take every event recorded so far, leaving the list empty. What is taken is final: later
    /// pushes never merge into it.
    pub fn drain(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.items)
    }
}

impl IntoIterator for Events {
    type Item = Event;
    type IntoIter = std::vec::IntoIter<Event>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_content_merges_and_counts_tokens() {
        let mut events = Events::new();
        events.push_content(Text::new("Hel", 1));
        events.push_content(Text::new("lo", 2));
        assert_eq!(events.as_slice(), &[Event::Content(Text::new("Hello", 3))]);
    }

    #[test]
    fn adjacent_reasoning_merges_but_not_across_kinds() {
        let mut events = Events::new();
        events.push(Event::ReasoningStart);
        events.push_reasoning(Text::new("think", 1));
        events.push_reasoning(Text::new("ing", 1));
        events.push(Event::ReasoningEnd);
        events.push_content(Text::new("answer", 1));
        events.push_reasoning(Text::new("late", 1));
        assert_eq!(
            events.as_slice(),
            &[
                Event::ReasoningStart,
                Event::Reasoning(Text::new("thinking", 2)),
                Event::ReasoningEnd,
                Event::Content(Text::new("answer", 1)),
                Event::Reasoning(Text::new("late", 1)),
            ]
        );
    }

    #[test]
    fn drained_events_are_final_and_later_text_starts_a_new_event() {
        let mut events = Events::new();
        events.push_content(Text::new("Hel", 1));
        let taken = events.drain();
        assert_eq!(taken, vec![Event::Content(Text::new("Hel", 1))]);
        assert!(events.is_empty());
        events.push_content(Text::new("lo", 1));
        assert_eq!(events.as_slice(), &[Event::Content(Text::new("lo", 1))]);
    }

    #[test]
    fn empty_text_is_not_recorded_but_tokens_without_text_are() {
        let mut events = Events::new();
        events.push_content(Text::new("", 0));
        events.push_content(Text::uncounted(""));
        assert!(events.is_empty());
        events.push_content(Text::new("", 1));
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn a_merge_with_an_uncounted_run_loses_the_count() {
        let mut events = Events::new();
        events.push_content(Text::new("a", 1));
        events.push_content(Text::uncounted("b"));
        assert_eq!(events.as_slice(), &[Event::Content(Text::uncounted("ab"))]);
    }

    #[test]
    fn text_concatenates_shown_dropped_and_malformed_in_order() {
        let mut events = Events::new();
        events.push_content(Text::new("a", 1));
        events.push(Event::Dropped {
            text: Text::new("<m>", 1),
            why: DropReason::ControlToken,
        });
        events.push(Event::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            source: Text::new("<tool_call>f", 3),
        });
        events.push(Event::ToolCallArguments {
            index: 0,
            json: "{\"a\":1".into(),
            source: Text::new("<arg a=1>", 4),
        });
        events.push(Event::Malformed {
            text: Text::new("{broken", 2),
            why: MalformedReason::InvalidArguments,
        });
        assert_eq!(events.text(), "a<m><tool_call>f<arg a=1>{broken");
    }

    #[test]
    fn events_serialize_with_a_kind_tag() {
        let event = Event::ToolCallArguments {
            index: 2,
            json: "{\"a\":".into(),
            source: Text::new("<a>", 1),
        };
        let json = serde_json::to_string(&event).expect("serializes");
        assert_eq!(
            json,
            r#"{"kind":"tool_call_arguments","index":2,"json":"{\"a\":","source":{"text":"<a>","tokens":1}}"#
        );
        let back: Event = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, event);
        let finish = serde_json::to_string(&Event::Finish {
            reason: FinishReason::ToolCalls,
            tool_calls: 1,
            reasoning_tokens: 7,
        })
        .expect("serializes");
        assert_eq!(
            finish,
            r#"{"kind":"finish","reason":"tool_calls","tool_calls":1,"reasoning_tokens":7}"#
        );
        let other = serde_json::to_string(&FinishReason::Other("content_filter".into()))
            .expect("serializes");
        assert_eq!(other, r#"{"other":"content_filter"}"#);
    }

    /// A malformed region's reason is one string on the wire, whether the crate borrowed a fixed
    /// text or built one; what is read back equals what was written.
    #[test]
    fn a_malformed_reason_is_its_text_on_the_wire_borrowed_or_owned() {
        let malformed = Event::Malformed {
            text: Text::uncounted("x"),
            why: MalformedReason::Other("text between a call's tags".into()),
        };
        let json = serde_json::to_string(&malformed).expect("serializes");
        assert_eq!(
            json,
            r#"{"kind":"malformed","text":{"text":"x","tokens":null},"why":{"other":"text between a call's tags"}}"#
        );
        let back: Event = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, malformed, "read back owned, equal by its text");
        let owned = MalformedReason::Other(String::from("built for the occasion").into());
        assert_eq!(
            serde_json::to_string(&owned).expect("serializes"),
            r#"{"other":"built for the occasion"}"#
        );
    }

    #[test]
    fn runs_alternate_whitespace_and_the_rest_on_character_boundaries_and_join_back() {
        assert_eq!(runs("").count(), 0);
        assert_eq!(runs(" \n\t").collect::<Vec<_>>(), vec![(" \n\t", true)]);
        assert_eq!(runs("ab").collect::<Vec<_>>(), vec![("ab", false)]);
        assert_eq!(
            runs(" a  bc\n").collect::<Vec<_>>(),
            vec![
                (" ", true),
                ("a", false),
                ("  ", true),
                ("bc", false),
                ("\n", true)
            ]
        );
        // A no-break space and an ideographic space are whitespace; the letters around them are
        // two, three and four bytes long.
        assert_eq!(
            runs("\u{e9}\u{a0}\u{65e5}\u{672c}\u{3000}\u{1f327}").collect::<Vec<_>>(),
            vec![
                ("\u{e9}", false),
                ("\u{a0}", true),
                ("\u{65e5}\u{672c}", false),
                ("\u{3000}", true),
                ("\u{1f327}", false)
            ]
        );
        for text in ["", " ", "a", " a b ", "\u{e9}\u{a0}\u{65e5} \u{1f327}"] {
            assert_eq!(runs(text).map(|(run, _)| run).collect::<String>(), text);
        }
    }
}
