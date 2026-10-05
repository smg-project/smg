//! Typed events. Every byte of model output appears in exactly one event.

use serde::{Deserialize, Serialize};

/// A run of text and the number of engine tokens it came from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Text {
    /// The text.
    pub s: String,
    /// How many engine tokens produced it; zero when unknown.
    pub tokens: u32,
}

impl Text {
    /// A text run.
    pub fn new(s: impl Into<String>, tokens: u32) -> Self {
        Self {
            s: s.into(),
            tokens,
        }
    }

    fn is_nothing(&self) -> bool {
        self.s.is_empty() && self.tokens == 0
    }

    fn append(&mut self, other: &Text) {
        self.s.push_str(&other.s);
        self.tokens += other.tokens;
    }
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
    /// Anything else, described.
    Other(String),
}

/// The finish reason the parser reports after refining the engine's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    },
    /// A fragment of the call's arguments. Concatenated fragments for one index are always a
    /// valid JSON prefix and equal the final arguments when the call ends.
    ToolCallArguments {
        /// Which call the fragment belongs to.
        index: u32,
        /// The fragment, as JSON text.
        json: String,
    },
    /// The call's region closed.
    ToolCallEnd {
        /// Which call ended.
        index: u32,
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
    pub fn push_content(&mut self, s: impl Into<String>, tokens: u32) {
        self.push(Event::Content(Text::new(s, tokens)));
    }

    /// Append reasoning text.
    pub fn push_reasoning(&mut self, s: impl Into<String>, tokens: u32) {
        self.push(Event::Reasoning(Text::new(s, tokens)));
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

    /// All text these events carry, in order: content, reasoning, dropped and malformed text.
    /// Tool-call arguments are not included, they are JSON derived from the source bytes rather
    /// than the bytes themselves; the conservation check covers them through their source spans.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for event in &self.items {
            match event {
                Event::Content(t) | Event::Reasoning(t) => out.push_str(&t.s),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => {
                    out.push_str(&text.s);
                }
                _ => {}
            };
        }
        out
    }

    /// Consume the list.
    pub fn into_vec(self) -> Vec<Event> {
        self.items
    }
}

impl From<Events> for Vec<Event> {
    fn from(events: Events) -> Self {
        events.items
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
        events.push_content("Hel", 1);
        events.push_content("lo", 2);
        assert_eq!(events.as_slice(), &[Event::Content(Text::new("Hello", 3))]);
    }

    #[test]
    fn adjacent_reasoning_merges_but_not_across_kinds() {
        let mut events = Events::new();
        events.push(Event::ReasoningStart);
        events.push_reasoning("think", 1);
        events.push_reasoning("ing", 1);
        events.push(Event::ReasoningEnd);
        events.push_content("answer", 1);
        events.push_reasoning("late", 1);
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
    fn empty_text_is_not_recorded_but_tokens_without_text_are() {
        let mut events = Events::new();
        events.push_content("", 0);
        assert!(events.is_empty());
        events.push_content("", 1);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn text_concatenates_shown_dropped_and_malformed_in_order() {
        let mut events = Events::new();
        events.push_content("a", 1);
        events.push(Event::Dropped {
            text: Text::new("<m>", 1),
            why: DropReason::ControlToken,
        });
        events.push(Event::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
        });
        events.push(Event::Malformed {
            text: Text::new("{broken", 2),
            why: MalformedReason::InvalidArguments,
        });
        assert_eq!(events.text(), "a<m>{broken");
    }

    #[test]
    fn events_serialize_with_a_kind_tag() {
        let event = Event::ToolCallArguments {
            index: 2,
            json: "{\"a\":".into(),
        };
        let json = serde_json::to_string(&event).expect("serializes");
        assert_eq!(
            json,
            r#"{"kind":"tool_call_arguments","index":2,"json":"{\"a\":"}"#
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
    }
}
