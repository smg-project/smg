//! Typed events. Every byte of model output appears in exactly one event.
//!
//! An event's text borrows the delta it came from where the engine held nothing back, so the
//! common content delta costs no copy; a run the engine held (whitespace waiting for the text that
//! decides whose it is, a header until its marker), a run merged with an earlier one, and the
//! events an assembler builds carry their text owned. The lifetime `'d` is the delta's: a step's
//! events live in the caller's [`Events`] until the caller has read them, [`Events::recycle`]
//! readies the buffer for the next step without giving its allocation back, and a caller that
//! keeps events past the step takes them with [`Event::into_owned`].

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A run of text and, when the parser saw token ids, the number of engine tokens it came from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Text<'d> {
    /// The text: borrowed from the delta it came from, or owned where the engine held or merged
    /// it or an assembler built it.
    pub text: Cow<'d, str>,
    /// How many engine tokens produced it. `None` when the parser was not given token ids, as
    /// through the compatibility bridge over the old crates.
    pub tokens: Option<u32>,
}

impl<'d> Text<'d> {
    /// A text run with a known token count.
    pub fn new(text: impl Into<Cow<'d, str>>, tokens: u32) -> Self {
        Self {
            text: text.into(),
            tokens: Some(tokens),
        }
    }

    /// A text run whose token count is not known.
    pub fn uncounted(text: impl Into<Cow<'d, str>>) -> Self {
        Self {
            text: text.into(),
            tokens: None,
        }
    }

    /// The same run with its text owned, free of the delta it came from.
    pub fn into_owned(self) -> Text<'static> {
        Text {
            text: Cow::Owned(self.text.into_owned()),
            tokens: self.tokens,
        }
    }

    fn is_nothing(&self) -> bool {
        self.text.is_empty() && self.tokens.unwrap_or(0) == 0
    }

    fn append(&mut self, other: &Text<'_>) {
        self.text.to_mut().push_str(&other.text);
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
    runs_at(text).map(move |(range, space)| (&text[range], space))
}

/// The runs of `text` as [`runs`] has them, each as its byte range in `text`: for a reader that
/// holds the text in a [`Cow`] and keeps a run of a borrowed delta borrowed.
pub(crate) fn runs_at(text: &str) -> impl Iterator<Item = (std::ops::Range<usize>, bool)> + '_ {
    let mut at = 0;
    std::iter::from_fn(move || {
        let rest = &text[at..];
        let space = rest.chars().next()?.is_whitespace();
        let length = rest
            .char_indices()
            .find(|(_, c)| c.is_whitespace() != space)
            .map_or(rest.len(), |(end, _)| end);
        let start = at;
        at += length;
        Some((start..at, space))
    })
}

/// Carried bytes that are the template's wrapping (the whitespace a syntax holds after a tag it
/// read or between two pieces of text, a wrapper's spacing): one `Dropped { Wrapper }` event of
/// them, so a reader that counts reasons counts them once; nothing when there are none.
pub(crate) fn drop_carried(carried: String, out: &mut Events) {
    if carried.is_empty() {
        return;
    }
    out.push(Event::Dropped {
        text: Text::uncounted(carried),
        why: DropReason::Wrapper,
    });
}

/// Text a syntax has no place for where its structure stands (a word between tags, between a
/// list's objects or between a block's calls, a tag out of its place, text after a call's object),
/// released as `Malformed` with `why`, which the chat surface shows as content. `carried` holds
/// the whitespace read since the syntax last read its structure (a tag, an object, a bracket, a
/// marker); `released` says whether text was released since then. Before any, that whitespace is
/// the template's spacing after its structure and is dropped (`Dropped { Wrapper }`), so the
/// report holds the model's text alone; after released text it is the model's, between two pieces
/// of its text, and leaves with this piece, so that a client reads the text as the model wrote
/// it, byte for byte, however the bytes were cut. One event per piece, so a reader that counts
/// the reason counts as before; a piece of no bytes releases the carried whitespace alone, for a
/// reader that must place something between two pieces of it. The reader clears `released` where
/// its syntax reads its structure again. `text` keeps its borrow when nothing is carried before
/// it, so a reader that borrows from the delta costs no copy there.
pub(crate) fn release_between<'d>(
    carried: &mut String,
    released: &mut bool,
    text: Cow<'d, str>,
    why: MalformedReason,
    out: &mut Events<'d>,
) {
    if !*released {
        drop_carried(std::mem::take(carried), out);
    }
    let text = if carried.is_empty() {
        text
    } else {
        carried.push_str(&text);
        Cow::Owned(std::mem::take(carried))
    };
    if text.is_empty() {
        return;
    }
    out.push(Event::Malformed {
        text: Text::uncounted(text),
        why,
    });
    *released = true;
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
    /// A call after the first, where the request takes one call only ([`first_call_only`]).
    LaterCall,
    /// A call the request forbade (`tool_choice: none`), read whole so its bytes are no content
    /// and delivered as none ([`forbidden_calls`]).
    ForbiddenCall,
    /// Bytes of a call block the call syntax could not read as the call, for a syntax whose block
    /// is never prose (the keyed syntax: the block's markers and its four tags are the format's
    /// own tokens, and what stands between them where the syntax has no place for text is the
    /// model's failed call; the attributed syntax inside an invoke, `tagged::attributed`, the
    /// same). A `Malformed` event of no bytes with the reason stands right before
    /// them, or, for a call the end of the stream or a marker cut, right after them and before
    /// its `ToolCallEnd` (`tagged::end_cut_call`), so the failure is counted; the bytes
    /// themselves never reach a client as content.
    MalformedCall,
}

/// The reasons, each with its doc and the sentence a log or a report shows: one list defines the
/// enum, [`MalformedReason::ALL`] and [`MalformedReason::sentence`], so a reason added here is in
/// all three at once.
macro_rules! malformed_reasons {
    ($($(#[$doc:meta])* $name:ident => $sentence:literal,)*) => {
        /// Why a region could not be parsed. Its text is returned, never lost: with the event, or,
        /// for a call syntax whose block is never prose, as `Dropped { MalformedCall }` beside an
        /// event of no bytes, right after it, or right before it and the call's end where a cut
        /// ended the call ([`DropReason::MalformedCall`]).
        ///
        /// The set is closed: every reason a parser gives is one of these, so a reader can count
        /// them and a client can match on them; a new reason is a new variant.
        /// [`Display`](std::fmt::Display) writes the sentence a log or a report shows.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        #[non_exhaustive]
        pub enum MalformedReason {
            $($(#[$doc])* $name,)*
        }

        impl MalformedReason {
            /// Every reason, in the enum's order: what a reader that labels or counts the reasons
            /// iterates. The list, the enum and the sentences come from one definition, so a reason
            /// added to the crate is in all three at once (the enum is non-exhaustive outside the
            /// crate).
            pub const ALL: &'static [Self] = &[$(Self::$name,)*];

            /// The sentence a log or a report shows for the reason.
            pub fn sentence(self) -> &'static str {
                match self {
                    $(Self::$name => $sentence,)*
                }
            }
        }
    };
}

malformed_reasons! {
    /// The stream ended inside a region that never closed.
    UnterminatedRegion => "the stream ended inside a region that never closed",
    /// Tool-call arguments did not parse under the format's argument syntax. From a JSON syntax
    /// the event carries no bytes and stands where the arguments stopped being JSON; the call's
    /// fragments carry every byte the model wrote.
    InvalidArguments => "arguments that did not parse under the call's syntax",
    /// A tool call named a function the request did not declare. The engine's own event, for
    /// every syntax: it stands right before the call's start and carries no bytes, the call goes
    /// out as the syntax read it, and a request that declares no tools is not asked.
    UnknownTool => "a call to a function the request did not declare",
    /// Text where the call's syntax has only tags: between a tag and the next, outside a value.
    TextBetweenTags => "text between a call's tags",
    /// A tag of the syntax where it has no place: a value's close with no value open, a
    /// parameter tag before the function's.
    TagOutOfPlace => "a tag where the call's syntax has none",
    /// A tag another tag, or the call's end, cut short before it closed.
    TagCutShort => "a tag that another tag or the call's end cut short",
    /// Text between a tag's name and its close, where the syntax has none.
    TagTail => "text after a tag's name",
    /// A tag with no name between its opener and its close.
    EmptyName => "a tag without a name",
    /// A call (an object, a block, an invoke) that names no function.
    CallWithoutName => "a call without a name",
    /// A call region with no function tag in it.
    CallWithoutFunctionTag => "a tool call without a function tag",
    /// A second function tag inside a call that has one.
    SecondFunctionTag => "a second function tag inside a call",
    /// A call that closed before its name's tag closed.
    ClosedBeforeTheName => "a call that closed before its name closed",
    /// Text where the call's name should be: a name holds no whitespace, quote or bracket.
    TextWhereTheNameShouldBe => "text where the call's name should be",
    /// Text where the syntax has a call or nothing: between a block's or a list's calls, or
    /// where a call should begin.
    TextBetweenCalls => "text between calls",
    /// Text where an argument should begin.
    TextWhereAnArgumentShouldBegin => "text where an argument should begin",
    /// Text where the list of calls should be.
    TextInPlaceOfTheList => "text where the list of calls should be",
    /// Text after a call's arguments and before the terminal that closes the call.
    TextAfterTheArguments => "text after a call's arguments",
    /// Text after the list of calls closed.
    TextAfterTheList => "text after the list's end",
    /// A call that names a function and writes no arguments.
    NameWithoutArguments => "a call that names a tool and writes no arguments",
    /// A name with no call after it: a message that names a tool and writes none, a Python
    /// name without its parenthesis.
    NameWithoutACall => "a name without a call",
    /// A parameter whose key the call already wrote.
    KeyWrittenTwice => "a parameter whose key the call already wrote",
    /// A key with no value after it.
    KeyWithoutAValue => "a key without a value",
    /// A key the call's end cut short.
    KeyCutShort => "a key the call's end cut short",
    /// A value the call's end cut short.
    ValueCutShort => "a value the call's end cut short",
    /// A key with no text.
    EmptyKey => "a key with no text",
    /// A tool-call block that closed without a complete call in it.
    BlockWithoutACompleteCall => "a tool-call block that closed without a complete call",
    /// Arguments with no forced function to name them.
    NoForcedFunction => "arguments with no forced function to name them",
    /// A marker of the table in a content state no row leads out of on it, kept as that state's
    /// text where the model put it (an answer's opener written again inside the answer, a close
    /// with nothing open). The event carries no bytes and stands before the marker's text: a frame
    /// the table did not read is counted without a byte moving.
    MarkerOutOfPlace => "a marker of the table where no row leads, kept as text",
    /// A call, or a string value in it, that the model left open at its own end of output (the
    /// stream's stop, not a cut), closed there by a syntax that writes the arguments' JSON for
    /// values the model wrote as text between tags: the string gets its closing quote, the
    /// object its brace, so that a call delivered under `tool_calls` reads as JSON. The event
    /// carries no bytes and stands before the fragment that closes what was open. A string the
    /// stop closed is counted under every table; the call itself only where the table has no
    /// stop row, since there the template writes the closing marker the model skipped.
    OpenAtTheStop => "a call, or a string value in it, left open at the stream's stop",
    /// A string value the model left open at the call's closing marker (the value's closing tag
    /// never came, the call's close did), closed there by a syntax that writes the arguments' JSON
    /// for values the model wrote as text between tags: the string gets its closing quote, and
    /// the object closes as at any marker, so that a call delivered under `tool_calls` reads as
    /// JSON. The marker is the model's own end of the call, so the value's bytes are complete, as
    /// at the stop. The event carries no bytes and stands before the closing quote.
    OpenAtTheMarker => "a string value left open at the call's closing marker",
}

impl std::fmt::Display for MalformedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.sentence())
    }
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

/// One parsed unit of model output, in wire order. Its text borrows the delta it came from for
/// the life of the step (`'d`), as the module doc says; [`Event::into_owned`] frees it of the
/// delta.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event<'d> {
    /// Visible assistant text.
    Content(Text<'d>),
    /// A reasoning region opened.
    ReasoningStart,
    /// Reasoning text.
    Reasoning(Text<'d>),
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
        source: Text<'d>,
    },
    /// A fragment of the call's arguments. Concatenated fragments for one index are a prefix of
    /// JSON a strict reader accepts (a number out of the reader's range excepted: the prefix judge
    /// reads a number's grammar, not its size) until a `Malformed { InvalidArguments }` event of
    /// no bytes says where they stopped being one, from which the fragments are the model's bytes
    /// as written (`json` equal to `source`), and equal the final arguments when the call ends: a
    /// call whose arguments are no whole JSON says so by a `Malformed` event before its end.
    ToolCallArguments {
        /// Which call the fragment belongs to.
        index: u32,
        /// The fragment, as JSON text: borrowed from the delta it came from where the fragment
        /// is the model's own bytes and arrived in one delta, as a text event's text is; owned
        /// where the syntax wrote it (a value given as text between tags, a raw control
        /// character written as its escape) or where it reaches back into an earlier delta.
        json: Cow<'d, str>,
        /// The output bytes this event accounts for, in the format's own syntax. Empty when the
        /// fragment came from bytes an earlier event already accounted for.
        source: Text<'d>,
    },
    /// The call's region closed.
    ToolCallEnd {
        /// Which call ended.
        index: u32,
        /// The output bytes this event accounts for: the closing marker, and any buffered bytes
        /// no earlier event of this call accounted for.
        source: Text<'d>,
    },
    /// Text the format consumed without showing it.
    Dropped {
        /// The text.
        text: Text<'d>,
        /// Why it was dropped.
        why: DropReason,
    },
    /// Text of a region the parser gave up on; the adapter decides what to do with it.
    Malformed {
        /// The text, returned whole.
        text: Text<'d>,
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

impl Event<'_> {
    /// The same event with its text owned, free of the delta it came from: for a caller that
    /// keeps events past the step that produced them.
    pub fn into_owned(self) -> Event<'static> {
        match self {
            Event::Content(text) => Event::Content(text.into_owned()),
            Event::ReasoningStart => Event::ReasoningStart,
            Event::Reasoning(text) => Event::Reasoning(text.into_owned()),
            Event::ReasoningEnd => Event::ReasoningEnd,
            Event::ToolCallStart {
                index,
                id,
                name,
                source,
            } => Event::ToolCallStart {
                index,
                id,
                name,
                source: source.into_owned(),
            },
            Event::ToolCallArguments {
                index,
                json,
                source,
            } => Event::ToolCallArguments {
                index,
                json: Cow::Owned(json.into_owned()),
                source: source.into_owned(),
            },
            Event::ToolCallEnd { index, source } => Event::ToolCallEnd {
                index,
                source: source.into_owned(),
            },
            Event::Dropped { text, why } => Event::Dropped {
                text: text.into_owned(),
                why,
            },
            Event::Malformed { text, why } => Event::Malformed {
                text: text.into_owned(),
                why,
            },
            Event::Finish {
                reason,
                tool_calls,
                reasoning_tokens,
            } => Event::Finish {
                reason,
                tool_calls,
                reasoning_tokens,
            },
        }
    }
}

/// An ordered list of events that merges adjacent text of the same kind: the buffer a parser
/// fills.
///
/// A parser pushes what it has as soon as it has it; two consecutive `Content` events become one,
/// and so do two consecutive `Reasoning` events. Nothing else is merged, and a text event with no
/// text and no tokens is not recorded.
///
/// Merging only ever touches events the caller has not taken yet. A stream's caller keeps one
/// buffer and, after each `feed`, reads the events in place and then hands the buffer to the next
/// step with [`Events::recycle`], which empties it and keeps its allocation; what was read is
/// final, and the next push starts a new event even when it is text of the same kind. A caller
/// that reads a whole output at once takes the events with [`Events::drain`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Events<'d> {
    items: Vec<Event<'d>>,
}

impl<'d> Events<'d> {
    /// An empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty list with room for `capacity` events before it grows.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Append an event, merging it into the previous one when both are text of the same kind.
    pub fn push(&mut self, event: Event<'d>) {
        match (&event, self.items.last_mut()) {
            (Event::Content(t), _) | (Event::Reasoning(t), _) if t.is_nothing() => {}
            (Event::Content(t), Some(Event::Content(prev))) => prev.append(t),
            (Event::Reasoning(t), Some(Event::Reasoning(prev))) => prev.append(t),
            _ => self.items.push(event),
        };
    }

    /// Append visible text.
    pub fn push_content(&mut self, text: Text<'d>) {
        self.push(Event::Content(text));
    }

    /// Append reasoning text.
    pub fn push_reasoning(&mut self, text: Text<'d>) {
        self.push(Event::Reasoning(text));
    }

    /// The events so far.
    pub fn as_slice(&self) -> &[Event<'d>] {
        &self.items
    }

    /// The events so far, to change in place.
    pub fn as_mut_slice(&mut self) -> &mut [Event<'d>] {
        &mut self.items
    }

    /// The events so far, in order.
    pub fn iter(&self) -> std::slice::Iter<'_, Event<'d>> {
        self.items.iter()
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether no event was recorded.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// How many events the list holds before it grows.
    pub fn capacity(&self) -> usize {
        self.items.capacity()
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
    pub fn drain(&mut self) -> Vec<Event<'d>> {
        std::mem::take(&mut self.items)
    }

    /// The events so far, moved out one by one; the list is left empty with its room kept. For a
    /// step whose events are rendered once: what an event owns (a call's id and name, a
    /// fragment's JSON) moves into the rendering, and the buffer goes on to the next step.
    pub fn consume(&mut self) -> std::vec::Drain<'_, Event<'d>> {
        self.items.drain(..)
    }

    /// The same list for the next step: empty, free of this step's delta, with its allocation
    /// kept, so a stream of many steps fills one buffer. An empty vector collected into the next
    /// lifetime's keeps its buffer (the standard library collects a vector into one of the same
    /// layout in place), which [`Events::capacity`] shows.
    pub fn recycle<'e>(mut self) -> Events<'e> {
        self.items.clear();
        Events {
            items: self.items.into_iter().map(Event::into_owned).collect(),
        }
    }

    /// The same events with their text owned, free of the delta they came from.
    pub fn into_owned(self) -> Events<'static> {
        Events {
            items: self.items.into_iter().map(Event::into_owned).collect(),
        }
    }
}

impl<'d> IntoIterator for Events<'d> {
    type Item = Event<'d>;
    type IntoIter = std::vec::IntoIter<Event<'d>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, 'd> IntoIterator for &'a Events<'d> {
    type Item = &'a Event<'d>;
    type IntoIter = std::slice::Iter<'a, Event<'d>>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.iter()
    }
}

/// The engine tokens the reasoning text among `events` cost.
pub fn reasoning_text_tokens(events: &[Event]) -> u32 {
    events
        .iter()
        .map(|event| match event {
            Event::Reasoning(text) => text.tokens.unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// One call only (`parallel_tool_calls: false`, `disable_parallel_tool_use`): the events of
/// every call after the first become its dropped text, so every byte of the output still lands
/// in one event; the finish counts at most one. A table's grammar limits a forced call; this
/// limits a free one.
pub fn first_call_only(events: &mut [Event]) {
    drop_calls_from(events, 1, DropReason::LaterCall);
}

/// No call at all (`tool_choice: none` over tools the request declares): the events of every
/// call become its dropped text, as [`first_call_only`] drops the later calls, so a call the
/// model writes anyway is neither a call nor content, as the engine's own server reads and
/// withholds it; the finish counts none, and a `tool_calls` finish is the engine's `stop` again.
pub fn forbidden_calls(events: &mut [Event]) {
    drop_calls_from(events, 0, DropReason::ForbiddenCall);
}

/// The events of every call from the index `kept` on become their dropped text under `why`, so
/// every byte of the output still lands in one event; the finish counts at most `kept` calls,
/// and one that counts none no longer says `tool_calls`.
fn drop_calls_from(events: &mut [Event], kept: u32, why: DropReason) {
    for event in events {
        match event {
            Event::ToolCallStart { index, source, .. }
            | Event::ToolCallArguments { index, source, .. }
            | Event::ToolCallEnd { index, source, .. }
                if *index >= kept =>
            {
                *event = Event::Dropped {
                    text: std::mem::take(source),
                    why,
                };
            }
            Event::Finish {
                tool_calls, reason, ..
            } => {
                *tool_calls = (*tool_calls).min(kept);
                if *tool_calls == 0 && *reason == FinishReason::ToolCalls {
                    *reason = FinishReason::Stop;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_between_drops_the_carried_whitespace_before_the_first_piece_and_keeps_it_after() {
        let mut out = Events::new();
        let (mut carried, mut released) = (String::from("\n"), false);
        let why = MalformedReason::TextBetweenTags;
        release_between(&mut carried, &mut released, "please".into(), why, &mut out);
        carried.push(' ');
        release_between(&mut carried, &mut released, "note".into(), why, &mut out);
        release_between(&mut carried, &mut released, "".into(), why, &mut out);
        // A piece of no bytes after released text: the whitespace alone, in its place.
        carried.push(' ');
        release_between(&mut carried, &mut released, "".into(), why, &mut out);
        let events = out.drain();
        assert!(matches!(
            &events[0],
            Event::Dropped { text, why: DropReason::Wrapper } if text.text == "\n"
        ));
        assert!(matches!(
            &events[1],
            Event::Malformed { text, why: MalformedReason::TextBetweenTags } if text.text == "please"
        ));
        assert!(matches!(&events[2], Event::Malformed { text, .. } if text.text == " note"));
        assert!(matches!(&events[3], Event::Malformed { text, .. } if text.text == " "));
        assert_eq!(events.len(), 4);
        assert!(released);
        assert!(carried.is_empty());
        // Nothing carried before the first piece: no event is spent on it, and the piece keeps
        // its borrow.
        let mut out = Events::new();
        let (mut carried, mut released) = (String::new(), false);
        release_between(
            &mut carried,
            &mut released,
            Cow::Borrowed("word"),
            why,
            &mut out,
        );
        let events = out.drain();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            Event::Malformed { text, .. } if matches!(text.text, Cow::Borrowed("word"))
        ));
    }

    #[test]
    fn every_reason_is_in_all_once_with_its_own_sentence_and_name() {
        // One definition writes the enum, `ALL` and the sentences, so the list cannot drift from
        // the enum; what is checked is the list itself: each reason once, a sentence of its own
        // (non-empty and shared with no other reason, so a sampled line names one reason), and a
        // serde name that reads back to it.
        let mut seen = std::collections::HashSet::new();
        let mut sentences = std::collections::HashSet::new();
        for why in MalformedReason::ALL.iter().copied() {
            assert!(seen.insert(why), "{why:?} is listed twice");
            assert!(!why.sentence().is_empty(), "{why:?}");
            assert!(
                sentences.insert(why.sentence()),
                "{why:?} shares its sentence with another reason"
            );
            let name = serde_json::to_value(why).expect("a reason serializes");
            let back: MalformedReason =
                serde_json::from_value(name.clone()).expect("the name reads back");
            assert_eq!(back, why, "{name}");
        }
        assert_eq!(seen.len(), MalformedReason::ALL.len());
    }

    #[test]
    fn reasoning_text_tokens_sum_the_reasoning_texts_alone() {
        let events = vec![
            Event::Content(Text::new("a", 3)),
            Event::Reasoning(Text::new("b", 2)),
            Event::Reasoning(Text::uncounted("c")),
            Event::Reasoning(Text::new("d", 5)),
        ];
        assert_eq!(reasoning_text_tokens(&events), 7);
        assert_eq!(reasoning_text_tokens(&[]), 0);
    }

    #[test]
    fn forbidden_calls_drops_every_call_and_the_finish_counts_none() {
        let call = |index: u32| {
            [
                Event::ToolCallStart {
                    index,
                    id: format!("call_{index}"),
                    name: "f".to_string(),
                    source: Text::uncounted("<"),
                },
                Event::ToolCallArguments {
                    index,
                    json: "{}".into(),
                    source: Text::uncounted("{}"),
                },
                Event::ToolCallEnd {
                    index,
                    source: Text::uncounted(">"),
                },
            ]
        };
        let mut events = vec![Event::Content(Text::uncounted("x"))];
        events.extend(call(0));
        events.extend(call(1));
        events.push(Event::Finish {
            reason: FinishReason::ToolCalls,
            tool_calls: 2,
            reasoning_tokens: 3,
        });

        forbidden_calls(&mut events);

        assert!(
            events.iter().all(|event| !matches!(
                event,
                Event::ToolCallStart { .. }
                    | Event::ToolCallArguments { .. }
                    | Event::ToolCallEnd { .. }
            )),
            "no call is left: {events:?}"
        );
        assert!(matches!(events.first(), Some(Event::Content(_))));
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::ForbiddenCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(
            dropped,
            ["<", "{}", ">", "<", "{}", ">"],
            "every call's bytes stay, in place, as dropped text"
        );
        assert_eq!(events.len(), 8);
        assert_eq!(
            events.last(),
            Some(&Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 0,
                reasoning_tokens: 3,
            }),
            "the finish counts no call and is the engine's stop again"
        );
        // A finish the engine gave for another reason keeps it.
        let mut events = vec![Event::Finish {
            reason: FinishReason::Length,
            tool_calls: 1,
            reasoning_tokens: 0,
        }];
        forbidden_calls(&mut events);
        assert!(matches!(
            events.last(),
            Some(Event::Finish {
                reason: FinishReason::Length,
                tool_calls: 0,
                ..
            })
        ));
    }

    #[test]
    fn first_call_only_keeps_the_first_call_and_caps_the_finish() {
        let call = |index: u32| {
            [
                Event::ToolCallStart {
                    index,
                    id: format!("call_{index}"),
                    name: "f".to_string(),
                    source: Text::uncounted("<"),
                },
                Event::ToolCallArguments {
                    index,
                    json: "{}".into(),
                    source: Text::uncounted("{}"),
                },
                Event::ToolCallEnd {
                    index,
                    source: Text::uncounted(">"),
                },
            ]
        };
        let mut events = vec![Event::Content(Text::uncounted("x"))];
        events.extend(call(0));
        events.extend(call(1));
        events.push(Event::Finish {
            reason: FinishReason::ToolCalls,
            tool_calls: 2,
            reasoning_tokens: 0,
        });

        first_call_only(&mut events);

        let indices: Vec<u32> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallStart { index, .. }
                | Event::ToolCallArguments { index, .. }
                | Event::ToolCallEnd { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, [0, 0, 0]);
        assert!(matches!(events.first(), Some(Event::Content(_))));
        let dropped: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::LaterCall,
                } => Some(&*text.text),
                _ => None,
            })
            .collect();
        assert_eq!(
            dropped,
            ["<", "{}", ">"],
            "the second call's bytes stay, in place"
        );
        assert_eq!(events.len(), 8);
        assert!(matches!(
            events.last(),
            Some(Event::Finish { tool_calls: 1, .. })
        ));
    }

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
        // A malformed region's reason is the variant's name on the wire, and a sentence in prose.
        let malformed = serde_json::to_string(&Event::Malformed {
            text: Text::uncounted("<parameter"),
            why: MalformedReason::TextBetweenTags,
        })
        .expect("serializes");
        assert_eq!(
            malformed,
            r#"{"kind":"malformed","text":{"text":"<parameter","tokens":null},"why":"text_between_tags"}"#
        );
        let back: Event = serde_json::from_str(&malformed).expect("deserializes");
        assert!(matches!(
            back,
            Event::Malformed {
                why: MalformedReason::TextBetweenTags,
                ..
            }
        ));
        assert_eq!(
            MalformedReason::TextBetweenTags.to_string(),
            "text between a call's tags"
        );
        assert_eq!(
            MalformedReason::BlockWithoutACompleteCall.sentence(),
            "a tool-call block that closed without a complete call"
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

    #[test]
    fn a_run_borrowed_from_the_delta_is_owned_once_merged_or_taken_past_the_step() {
        let delta = String::from("Hello, world");
        let mut events = Events::new();
        events.push_content(Text::new(&delta[..5], 1));
        assert!(matches!(
            events.as_slice(),
            [Event::Content(Text {
                text: Cow::Borrowed("Hello"),
                ..
            })]
        ));
        events.push_content(Text::new(&delta[5..], 2));
        let [Event::Content(merged)] = events.as_slice() else {
            panic!("one merged event, got {events:?}");
        };
        assert!(matches!(merged.text, Cow::Owned(_)), "a merge copies once");
        assert_eq!(*merged, Text::new("Hello, world", 3));
        let kept = events.drain().remove(0).into_owned();
        drop(delta);
        assert_eq!(kept, Event::Content(Text::new("Hello, world", 3)));
    }

    #[test]
    fn a_recycled_buffer_is_empty_and_keeps_its_allocation_for_the_next_step() {
        let mut spare: Events<'static> = Events::with_capacity(16);
        let buffer = spare.as_slice().as_ptr().cast::<u8>();
        for step in ["one", "two", "three"] {
            let delta = step.to_string();
            let mut events: Events<'_> = spare;
            events.push_content(Text::uncounted(delta.as_str()));
            events.push(Event::ReasoningStart);
            assert_eq!(events.len(), 2);
            spare = events.recycle();
            assert!(spare.is_empty());
            assert_eq!(spare.capacity(), 16, "the allocation outlives the step");
            assert!(
                std::ptr::eq(spare.as_slice().as_ptr().cast::<u8>(), buffer),
                "the same buffer"
            );
        }
    }
}
