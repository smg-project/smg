//! Messages: parser events as the content blocks of one assistant message.
//!
//! [`output`] folds a whole event sequence into the message's `content` blocks and its
//! `stop_reason`. It is pure like [`chat::message`](super::chat::message) and
//! [`responses::output`](super::responses::output): call ids ride on the events, the stop reason is
//! decided from the `Finish` event alone, and nothing is remembered between calls. The driver wraps
//! the blocks in the message envelope (id, model, usage) and, when its stop decoder matched a stop
//! sequence, sets `stop_sequence` and that reason, since the events carry neither. A `tool_use`
//! block's `input` is the call's argument fragments parsed as one JSON value. Fragments that are no
//! whole JSON text, which the parser said in a `Malformed` event (the output cut inside the
//! arguments, or the model wrote something else after a prefix a strict reader accepts), give the
//! value their complete members make ([`Prefix::closed`]): the prefix the readers sent, cut back to
//! the last member complete at every level and closed, which is the value a client's partial JSON
//! reader ends with from the stream's deltas; `{}` where nothing of it is complete, and for a call
//! the model closed without arguments. Fragments that are one whole JSON text no `input` value
//! holds get `{}` too, and nothing in the events says so: [`Output::refused`] names each such call
//! with the reader's reason, so that the driver decides what the client gets rather than hand over
//! `{}` for arguments the model wrote. Whole by its syntax is what a reader that keeps no value
//! takes; what no value holds is a number outside the range a reader keeps, a surrogate escape
//! without its other half, or nesting past the value reader's limit. The crate's own readers refuse
//! the last two as they arrive and say so, so from them the set is the numbers.
//!
//! [`Stream`] is the streaming half, the content-block machine: one block is open at a time, and
//! each parser event becomes the `content_block_start`, `content_block_delta` and
//! `content_block_stop` events a client expects, the block index advancing by one per block. A
//! `thinking` block opens at the first reasoning text and closes when the region ends; a `text`
//! block opens at the first content; a `tool_use` block opens at `ToolCallStart` with an empty
//! `input` and closes at `ToolCallEnd`, and its `input_json_delta`s stop at a `Malformed {
//! InvalidArguments }` event of no bytes (the fragments after it are the model's bytes as written
//! and go nowhere), so the joined `partial_json` is the prefix a strict reader accepted, which a
//! client's partial JSON reader closes into the value its complete members make, the fold's
//! `input` for the call; a block closes when text of another kind arrives.
//! Whitespace-only text that arrives with no text block open is held: it leads the text block when
//! text with substance follows, and is let go when a block of another kind starts or the stream
//! ends, so the template's separators between blocks open no block of their own. `Finish` closes
//! the open block and hands the stop reason back for the driver's `message_delta`, which also
//! carries the usage and the stop sequence the events do not have; `message_start`, `ping` and
//! `message_stop` are the driver's as well. For an output with one reasoning region, then text,
//! then calls, with nothing but calls after the first call and each call's events together, the
//! stream's blocks are the ones [`output`] folds from the same events. Otherwise the two differ:
//! the stream keeps the order the model wrote the blocks in, gives each reasoning region a thinking
//! block of its own, lets go of the whitespace between blocks, and makes text of a fragment whose
//! call's block another block has closed; the fold gathers each kind into one block, gives every
//! fragment to its call, and once its text has substance keeps every run of whitespace in it, the
//! separators after a call included.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/adapters.md`.

use std::mem;

use openai_protocol::messages::{
    ContentBlock, ContentBlockDelta, MessageStreamEvent, StopReason, ToolUseCaller,
};
use serde::de::IgnoredAny;
use serde_json::{Map, Value};

use super::fold::fold;
use crate::{
    event::{Event, FinishReason, MalformedReason},
    json::Prefix,
};

/// A whole output as the Messages API reports it: the content blocks and the stop reason, and the
/// calls whose `tool_use` block could not carry their arguments.
#[derive(Clone, Debug)]
pub struct Output {
    /// The content blocks in the Messages API's order: thinking, text, then the calls as they
    /// started.
    pub content: Vec<ContentBlock>,
    /// The stop reason decided from the `Finish` event; none when the sequence has not finished.
    pub stop_reason: Option<StopReason>,
    /// The calls whose argument fragments are one whole JSON text that no `input` value holds (a
    /// number outside the range a reader keeps; a surrogate escape without its other half or
    /// nesting past the value reader's limit, which the crate's readers refuse before they get
    /// here), in the order the calls started: their block carries `{}`, and no `Malformed` event
    /// says so, since the text is JSON by its syntax. Empty when every block carries its call's
    /// arguments, or the parser said they are none.
    pub refused: Vec<RefusedInput>,
}

/// A call whose `tool_use` block carries `{}` in place of arguments that are JSON by their syntax
/// ([`Output::refused`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefusedInput {
    /// The call's index among the output's calls.
    pub index: u32,
    /// The function's name.
    pub name: String,
    /// The arguments as the model wrote them, the fragments joined.
    pub arguments: String,
    /// Why no `input` value holds them, in the reader's words.
    pub why: String,
}

/// The message's content and stop reason, folded from a whole event sequence.
pub fn output<'a, 'd: 'a>(events: impl IntoIterator<Item = &'a Event<'d>>) -> Output {
    let folded = fold(events);
    let mut blocks = Vec::new();
    if !folded.reasoning.is_empty() {
        blocks.push(ContentBlock::Thinking {
            thinking: folded.reasoning,
            signature: String::new(),
        });
    }
    if !folded.content.trim().is_empty() {
        blocks.push(ContentBlock::Text {
            text: folded.content,
            citations: None,
        });
    }
    let mut refused = Vec::new();
    blocks.extend(folded.calls.into_iter().map(|call| {
        let input = input(&call.arguments).unwrap_or_else(|why| {
            refused.push(RefusedInput {
                index: call.index,
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                why,
            });
            Value::Object(Map::new())
        });
        ContentBlock::ToolUse {
            id: tool_use_id(&call.id),
            name: call.name,
            input,
            caller: ToolUseCaller::Direct,
        }
    }));

    Output {
        content: blocks,
        stop_reason: folded
            .finish
            .map(|finish| stop(&finish.reason, finish.tool_calls)),
        refused,
    }
}

/// The streaming half: parser events as `content_block_*` events, one block open at a time.
///
/// One per message, like the parser that feeds it. Feed every event in order; the machine opens,
/// continues and closes blocks as the module doc describes and numbers them from zero. `Finish`
/// closes the open block and returns the stop reason for the driver's `message_delta`.
#[derive(Clone, Debug, Default)]
pub struct Stream {
    /// The open block, if any.
    open: Option<Block>,
    /// The index of the open block, or of the next block to open.
    index: u32,
    /// Whitespace-only text that arrived with no text block open, waiting for text with substance.
    held: String,
    /// Whether the open `tool_use` block's arguments stopped being JSON (a `Malformed {
    /// InvalidArguments }` event of no bytes while it was open): its later fragments are not
    /// sent, so the joined `partial_json` stays the prefix the readers accepted, which a client's
    /// partial JSON reader closes into the value its complete members make, as the fold does for
    /// the call ([`input`]).
    refused: bool,
}

/// The kind of block that is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Block {
    Thinking,
    Text,
    ToolUse {
        /// The call whose arguments the block takes.
        call: u32,
    },
}

impl Stream {
    /// A machine with no block open; the first block gets index zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends the stream events `event` produces to `out`, in order. Returns the stop reason when
    /// `event` is `Finish`, after closing the open block.
    pub fn feed(
        &mut self,
        event: &Event<'_>,
        out: &mut Vec<MessageStreamEvent>,
    ) -> Option<StopReason> {
        match event {
            Event::Malformed {
                text,
                why: MalformedReason::InvalidArguments,
            } if text.text.is_empty() && matches!(self.open, Some(Block::ToolUse { .. })) => {
                self.refused = true;
            }
            Event::Content(text) | Event::Malformed { text, .. } => self.text(&text.text, out),
            Event::Reasoning(text) => self.reasoning(&text.text, out),
            Event::ReasoningEnd => {
                if self.open == Some(Block::Thinking) {
                    self.stop_open(out);
                }
            }
            Event::ToolCallStart {
                index, id, name, ..
            } => self.start(
                Block::ToolUse { call: *index },
                ContentBlock::ToolUse {
                    id: tool_use_id(id),
                    name: name.clone(),
                    input: Value::Object(Map::new()),
                    caller: ToolUseCaller::Direct,
                },
                out,
            ),
            Event::ToolCallArguments { index, json, .. } => self.arguments(*index, json, out),
            Event::ToolCallEnd { index, .. } => {
                if self.open == Some(Block::ToolUse { call: *index }) {
                    self.stop_open(out);
                }
            }
            Event::Finish {
                reason, tool_calls, ..
            } => {
                self.close(out);
                return Some(stop(reason, *tool_calls));
            }
            Event::ReasoningStart | Event::Dropped { .. } => {}
        }
        None
    }

    /// Ends the stream's blocks: stops the open block, if any, and lets held whitespace go. For a
    /// stream that ends without `Finish`; `feed` does this itself at `Finish`.
    pub fn close(&mut self, out: &mut Vec<MessageStreamEvent>) {
        self.stop_open(out);
        self.held.clear();
    }

    /// Stops the open block, if any, so that the next block gets the next index.
    fn stop_open(&mut self, out: &mut Vec<MessageStreamEvent>) {
        if self.open.take().is_some() {
            out.push(MessageStreamEvent::ContentBlockStop { index: self.index });
            self.index += 1;
            self.refused = false;
        }
    }

    /// Reasoning text into the open thinking block, opened first when another kind of block or none
    /// is open.
    fn reasoning(&mut self, text: &str, out: &mut Vec<MessageStreamEvent>) {
        if text.is_empty() {
            return;
        }
        self.start(
            Block::Thinking,
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            out,
        );
        out.push(MessageStreamEvent::ContentBlockDelta {
            index: self.index,
            delta: ContentBlockDelta::ThinkingDelta {
                thinking: text.to_string(),
            },
        });
    }

    /// A fragment of call `call`'s arguments into its open `tool_use` block; text when the call has
    /// no open block, so that no byte is lost; nothing once the block's arguments stopped being
    /// JSON (a client's partial JSON reader takes a prefix cut short, not a token that is no JSON;
    /// it closes the prefix into the value its complete members make, as the fold does for the
    /// call).
    fn arguments(&mut self, call: u32, json: &str, out: &mut Vec<MessageStreamEvent>) {
        if json.is_empty() {
            return;
        }
        if self.open != Some(Block::ToolUse { call }) {
            self.text(json, out);
            return;
        }
        if self.refused {
            return;
        }
        out.push(MessageStreamEvent::ContentBlockDelta {
            index: self.index,
            delta: ContentBlockDelta::InputJsonDelta {
                partial_json: json.to_string(),
            },
        });
    }

    /// Text into the open text block, opened first when another kind of block or none is open.
    /// Whitespace-only text with no text block open is held for the text with substance that may
    /// follow it.
    fn text(&mut self, text: &str, out: &mut Vec<MessageStreamEvent>) {
        if text.is_empty() {
            return;
        }
        if self.open != Some(Block::Text) {
            if text.trim().is_empty() {
                self.held.push_str(text);
                return;
            }
            self.start(
                Block::Text,
                ContentBlock::Text {
                    text: String::new(),
                    citations: None,
                },
                out,
            );
        }
        let mut text_with_held = mem::take(&mut self.held);
        text_with_held.push_str(text);
        out.push(MessageStreamEvent::ContentBlockDelta {
            index: self.index,
            delta: ContentBlockDelta::TextDelta {
                text: text_with_held,
            },
        });
    }

    /// Makes `block` the open block: nothing when it already is, otherwise the open block is closed
    /// and `content_block` starts at the next index. Held whitespace is let go when the block is
    /// not a text block.
    fn start(
        &mut self,
        block: Block,
        content_block: ContentBlock,
        out: &mut Vec<MessageStreamEvent>,
    ) {
        if self.open == Some(block) {
            return;
        }
        if block != Block::Text {
            self.held.clear();
        }
        self.stop_open(out);
        out.push(MessageStreamEvent::ContentBlockStart {
            index: self.index,
            content_block,
        });
        self.open = Some(block);
    }
}

/// The stop reason for the engine's reason: a truncation first, then a call made, then the turn.
fn stop(reason: &FinishReason, tool_calls: u32) -> StopReason {
    match reason {
        FinishReason::Length => StopReason::MaxTokens,
        FinishReason::ToolCalls => StopReason::ToolUse,
        FinishReason::Stop | FinishReason::Abort | FinishReason::Other(_) if tool_calls > 0 => {
            StopReason::ToolUse
        }
        FinishReason::Stop | FinishReason::Abort | FinishReason::Other(_) => StopReason::EndTurn,
    }
}

/// The `tool_use` id for a call id: `call_<suffix>` becomes `toolu_<suffix>`, anything else stays.
fn tool_use_id(id: &str) -> String {
    match id.strip_prefix("call_") {
        Some(suffix) => format!("toolu_{suffix}"),
        None => id.to_string(),
    }
}

/// The `input` for a call's joined argument fragments: the JSON value they form; when they form
/// none and the parser said so (the output cut inside the arguments, or arguments that stopped
/// being JSON, whose bytes from the break on are the model's as written), the value their complete
/// members make, the prefix a strict reader accepts cut back to the last complete member at every
/// level and closed ([`Prefix::closed`]), which is the value a client's partial JSON reader ends
/// with from the fragments the stream sent, and `{}` where nothing of it is complete (a call
/// closed without arguments, a first value cut short) or no value holds the closure (the parser's
/// word stands); and the reader's refusal, in its words, when they are one whole JSON text that no
/// value holds (a number outside the range a reader keeps, a surrogate escape without its other
/// half, nesting past the value reader's limit), which nothing else says.
fn input(arguments: &str) -> Result<Value, String> {
    match serde_json::from_str(arguments) {
        Ok(value) => Ok(value),
        Err(why) if whole_json(arguments) => Err(why.to_string()),
        Err(_) => Ok(Prefix::closed(arguments)
            .and_then(|closed| serde_json::from_str(&closed).ok())
            .unwrap_or_else(|| Value::Object(Map::new()))),
    }
}

/// Whether `text` is one whole JSON text by its syntax, whatever its values: what a reader that
/// keeps no value takes, as the Responses adapter judges a call's arguments.
fn whole_json(text: &str) -> bool {
    !text.is_empty() && serde_json::from_str::<IgnoredAny>(text).is_ok()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        event::{DropReason, MalformedReason, Text},
        formats::qwen3,
        ids::is_minted,
        CallSyntax, Declared, Engine, EngineFinish, Events, Input, Parser,
    };

    fn call(index: u32, name: &str) -> Event<'static> {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: name.into(),
            source: Text::default(),
        }
    }

    fn arguments(index: u32, json: &str) -> Event<'static> {
        Event::ToolCallArguments {
            index,
            json: json.into(),
            source: Text::default(),
        }
    }

    fn end(index: u32) -> Event<'static> {
        Event::ToolCallEnd {
            index,
            source: Text::default(),
        }
    }

    fn finish(reason: FinishReason, tool_calls: u32) -> Event<'static> {
        Event::Finish {
            reason,
            tool_calls,
            reasoning_tokens: 0,
        }
    }

    fn wire(blocks: &[ContentBlock]) -> Value {
        serde_json::to_value(blocks).expect("serializable")
    }

    #[test]
    fn a_whole_output_folds_into_the_gateway_blocks_in_order() {
        let events = [
            Event::ReasoningStart,
            Event::Reasoning(Text::uncounted("plan ")),
            Event::Reasoning(Text::uncounted("more")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("Let me check.")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            arguments(0, r#""Paris"}"#),
            end(0),
            finish(FinishReason::Stop, 1),
        ];
        let out = output(&events);
        assert_eq!(
            wire(&out.content),
            json!([
                {"type": "thinking", "thinking": "plan more", "signature": ""},
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_0",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                    "caller": {"type": "direct"},
                },
            ])
        );
        assert_eq!(out.stop_reason, Some(StopReason::ToolUse));
    }

    #[test]
    fn whitespace_only_content_and_nothing_else_give_no_blocks() {
        let events = [
            Event::Content(Text::uncounted("  \n")),
            finish(FinishReason::Stop, 0),
        ];
        let out = output(&events);
        assert!(out.content.is_empty(), "{:?}", out.content);
        assert_eq!(out.stop_reason, Some(StopReason::EndTurn));
    }

    #[test]
    fn malformed_text_and_a_stray_fragment_join_the_text_block() {
        let events = [
            Event::Content(Text::uncounted("Sure. ")),
            Event::Malformed {
                text: Text::uncounted("<tool_call>{broken"),
                why: MalformedReason::InvalidArguments,
            },
            arguments(7, r#"{"stray":true}"#),
        ];
        let out = output(&events);
        assert_eq!(
            wire(&out.content),
            json!([{"type": "text", "text": "Sure. <tool_call>{broken{\"stray\":true}"}])
        );
        assert_eq!(
            out.stop_reason, None,
            "no Finish event, so the message has no stop reason yet"
        );
    }

    #[test]
    fn calls_keep_the_order_they_started_in_and_their_own_inputs() {
        let events = [
            call(0, "get_weather"),
            call(1, "get_time"),
            arguments(1, r#"{"zone":"CET"}"#),
            arguments(0, r#"{"city":"Paris"}"#),
            end(1),
            end(0),
            finish(FinishReason::Stop, 2),
        ];
        assert_eq!(
            wire(&output(&events).content),
            json!([
                {"type": "tool_use", "id": "toolu_0", "name": "get_weather", "input": {"city": "Paris"},
                 "caller": {"type": "direct"}},
                {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {"zone": "CET"},
                 "caller": {"type": "direct"}},
            ])
        );
    }

    #[test]
    fn the_stop_reason_follows_the_finish_reason_with_truncation_first() {
        let reason = |reason: FinishReason, tool_calls: u32| {
            output(&[finish(reason, tool_calls)]).stop_reason
        };
        assert_eq!(reason(FinishReason::Stop, 0), Some(StopReason::EndTurn));
        assert_eq!(reason(FinishReason::Stop, 2), Some(StopReason::ToolUse));
        assert_eq!(
            reason(FinishReason::ToolCalls, 0),
            Some(StopReason::ToolUse)
        );
        assert_eq!(reason(FinishReason::Length, 0), Some(StopReason::MaxTokens));
        assert_eq!(
            reason(FinishReason::Length, 1),
            Some(StopReason::MaxTokens),
            "parsed calls do not override a truncation"
        );
        assert_eq!(reason(FinishReason::Abort, 0), Some(StopReason::EndTurn));
        assert_eq!(
            reason(FinishReason::Other("failed".into()), 0),
            Some(StopReason::EndTurn)
        );
        assert_eq!(
            reason(FinishReason::Abort, 1),
            Some(StopReason::ToolUse),
            "a message with tool_use blocks says so, whatever stopped the model"
        );
        assert_eq!(
            reason(FinishReason::Other("failed".into()), 1),
            Some(StopReason::ToolUse)
        );
        assert_eq!(output(&[]).stop_reason, None);
    }

    /// The output of one call of `f` with `fragments` for its arguments, ended by `length`.
    fn one_call(fragments: &[&str]) -> Output {
        let mut events = vec![call(0, "f")];
        events.extend(fragments.iter().map(|json| arguments(0, json)));
        events.push(end(0));
        events.push(finish(FinishReason::Length, 1));
        output(&events)
    }

    /// Arguments that are no whole JSON value: the `input` is the value their complete members
    /// make, the prefix a strict reader accepts cut back to the last complete member at every
    /// level and closed, as a client's partial JSON reader ends with it from the stream; `{}`
    /// where nothing of them is complete, and for a call closed without arguments. The parser's
    /// word stands: none of these is named as refused.
    #[test]
    fn arguments_that_are_no_whole_json_value_give_the_value_their_complete_members_make() {
        let inputs = |fragments: &[&str]| {
            let output = one_call(fragments);
            assert!(
                output.refused.is_empty(),
                "{fragments:?}: the parser's word stands"
            );
            wire(&output.content)[0]["input"].clone()
        };
        assert_eq!(
            inputs(&[r#"{"city":"#]),
            json!({}),
            "cut before the first value"
        );
        assert_eq!(
            inputs(&[r#"{"city": "Par"#]),
            json!({}),
            "cut inside the first value"
        );
        assert_eq!(
            inputs(&[r#"{"city": "Paris", "#, r#""days": 3"#]),
            json!({"city": "Paris", "days": 3}),
            "cut after a number whole as it stands"
        );
        assert_eq!(
            inputs(&[r#"{"city": "Paris", "days": [1, "#, "2"]),
            json!({"city": "Paris", "days": [1, 2]}),
            "cut inside a nested array: closed at every level"
        );
        assert_eq!(
            inputs(&[r#"{"city": "Paris", "days": "#]),
            json!({"city": "Paris"}),
            "cut after a key: the key leaves with the value it never got"
        );
        assert_eq!(
            inputs(&[r#"{"city": "Paris", "days": [1, "tw"#]),
            json!({"city": "Paris", "days": [1]}),
            "cut inside a string: the string leaves, the members before it stay"
        );
        assert_eq!(inputs(&["not json"]), json!({}), "not JSON at all");
        assert_eq!(inputs(&[]), json!({}), "closed without arguments");
        assert_eq!(
            inputs(&[r#"{"city":"#, r#""Paris"}"#]),
            json!({"city": "Paris"}),
            "whole across fragments"
        );
        assert_eq!(
            inputs(&["[1, 2]"]),
            json!([1, 2]),
            "a whole value of another kind"
        );
    }

    /// Arguments that are JSON by their syntax and that no `input` value holds: the block carries
    /// `{}`, as for arguments that are none, and the output names the call with the reader's
    /// reason, since no event says so; the stream carries the text as the model wrote it. Three
    /// kinds of text are such: a number outside the range a reader keeps, a surrogate escape
    /// without its other half, and nesting past the value reader's limit (the crate's readers
    /// refuse the last two as they arrive, so from them only the numbers get here).
    #[test]
    fn arguments_no_input_value_holds_are_named_as_refused_with_the_readers_reason() {
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        for (text, reason) in [
            (r#"{"days": 9.99e999999}"#, "number out of range"),
            (r#"{"n": 1e400, "city": "Paris"}"#, "number out of range"),
            ("[-1e400]", "number out of range"),
            (r#"{"q": "\ud800"}"#, "hex escape"),
            (deep.as_str(), "recursion limit exceeded"),
        ] {
            let folded = one_call(&[text]);
            assert_eq!(wire(&folded.content)[0]["input"], json!({}), "{text}");
            assert_eq!(folded.refused.len(), 1, "{text}: {:?}", folded.refused);
            let refused = &folded.refused[0];
            assert_eq!((refused.index, refused.name.as_str()), (0, "f"), "{text}");
            assert_eq!(refused.arguments, text);
            assert!(refused.why.contains(reason), "{text}: {}", refused.why);
            assert_eq!(
                folded.stop_reason,
                Some(StopReason::MaxTokens),
                "{text}: the stop reason is the finish's, as for every output"
            );
            // The same arguments in two fragments and in two calls: each call is named once, in
            // the order the calls started, with its own arguments.
            let (head, tail) = text.split_at(4);
            let mut events = vec![
                call(0, "f"),
                arguments(0, head),
                arguments(0, tail),
                end(0),
                call(1, "g"),
                arguments(1, r#"{"ok": true}"#),
                end(1),
                call(2, "h"),
                arguments(2, text),
                end(2),
                finish(FinishReason::Stop, 3),
            ];
            let folded = output(&events);
            assert_eq!(
                folded
                    .refused
                    .iter()
                    .map(|refused| (refused.index, refused.name.as_str()))
                    .collect::<Vec<_>>(),
                [(0, "f"), (2, "h")],
                "{text}"
            );
            assert_eq!(
                wire(&folded.content)[1]["input"],
                json!({"ok": true}),
                "{text}"
            );
            // The stream hands the fragments on as written, for the client's own reader.
            events.truncate(4);
            let mut stream = Stream::new();
            let mut out = Vec::new();
            for event in &events {
                stream.feed(event, &mut out);
            }
            let partial: String = stream_wire(&out)
                .as_array()
                .expect("events")
                .iter()
                .filter_map(|event| event["delta"]["partial_json"].as_str().map(String::from))
                .collect();
            assert_eq!(partial, text);
        }
    }

    /// The stream of a call whose arguments stop being JSON: the fragments before the break are
    /// `input_json_delta`s, the break (a `Malformed { InvalidArguments }` event of no bytes) ends
    /// them, and the fragments after it go nowhere, since a client's partial JSON reader takes a
    /// prefix cut short but not a token that is no JSON: it closes the prefix into the value its
    /// complete members make, which is the fold's `input` for the call, so a streaming client and
    /// a whole-message client end with the same call. The next call's block streams whole again.
    #[test]
    fn arguments_that_stop_being_json_stream_the_prefix_and_nothing_after_the_break() {
        let events = [
            call(0, "plot_sine_wave"),
            arguments(0, r#"{"start_range": 0, "end_range": 2.0 "#),
            Event::Malformed {
                text: Text::default(),
                why: MalformedReason::InvalidArguments,
            },
            arguments(0, r#"* Math.PI, "frequency": 5}"#),
            end(0),
            call(1, "g"),
            arguments(1, r#"{"ok": true}"#),
            end(1),
            finish(FinishReason::ToolCalls, 2),
        ];
        let mut stream = Stream::new();
        let mut out = Vec::new();
        for event in &events {
            stream.feed(event, &mut out);
        }
        let events_wire = stream_wire(&out);
        let events_wire = events_wire.as_array().expect("events");
        let partial = |index: u64| -> String {
            events_wire
                .iter()
                .filter(|event| event["index"] == index)
                .filter_map(|event| event["delta"]["partial_json"].as_str().map(String::from))
                .collect()
        };
        assert_eq!(partial(0), r#"{"start_range": 0, "end_range": 2.0 "#);
        assert_eq!(partial(1), r#"{"ok": true}"#);
        assert!(
            events_wire
                .iter()
                .all(|event| event["delta"]["type"] != "text_delta"),
            "nothing of the tail is text: {events_wire:?}"
        );
        assert_eq!(
            events_wire
                .iter()
                .filter(|event| event["type"] == "content_block_start")
                .count(),
            2,
            "two tool_use blocks and nothing else: {events_wire:?}"
        );
        let folded = output(&events);
        assert_eq!(
            wire(&folded.content)[0]["input"],
            json!({"start_range": 0, "end_range": 2.0}),
            "the fold: the prefix's complete members, nothing of the tail"
        );
        let closed_by_a_client: Value =
            serde_json::from_str(&format!("{}}}", partial(0).trim_end())).expect("closed");
        assert_eq!(
            wire(&folded.content)[0]["input"],
            closed_by_a_client,
            "the stream's prefix, closed as a client's partial reader closes it"
        );
        assert_eq!(wire(&folded.content)[1]["input"], json!({"ok": true}));
    }

    #[test]
    fn call_ids_take_the_tool_use_prefix_and_other_shapes_pass_through() {
        assert_eq!(tool_use_id("call_abc123"), "toolu_abc123");
        assert_eq!(tool_use_id("toolu_abc123"), "toolu_abc123");
        assert_eq!(
            tool_use_id("functions.get_weather:0"),
            "functions.get_weather:0"
        );
        assert_eq!(tool_use_id(""), "");
        let events = [
            Event::ToolCallStart {
                index: 0,
                id: "functions.get_weather:0".into(),
                name: "get_weather".into(),
                source: Text::default(),
            },
            end(0),
            finish(FinishReason::Stop, 1),
        ];
        assert_eq!(
            wire(&output(&events).content)[0]["id"],
            json!("functions.get_weather:0")
        );
    }

    #[test]
    fn events_without_a_messages_representation_shape_nothing() {
        let silent = [
            Event::ReasoningStart,
            Event::ReasoningEnd,
            end(0),
            Event::Dropped {
                text: Text::uncounted("<think>"),
                why: DropReason::Wrapper,
            },
            Event::Content(Text::uncounted("")),
            Event::Reasoning(Text::uncounted("")),
            finish(FinishReason::Stop, 0),
        ];
        let out = output(&silent);
        assert!(out.content.is_empty(), "{:?}", out.content);
        assert_eq!(out.stop_reason, Some(StopReason::EndTurn));
    }

    /// Every event through one machine; the stop reason of the last `Finish`, if any.
    fn stream(events: &[Event]) -> (Vec<MessageStreamEvent>, Option<StopReason>) {
        let mut machine = Stream::new();
        let mut out = Vec::new();
        let mut stop = None;
        for event in events {
            if let Some(reason) = machine.feed(event, &mut out) {
                stop = Some(reason);
            }
        }
        (out, stop)
    }

    fn stream_wire(events: &[MessageStreamEvent]) -> Value {
        serde_json::to_value(events).expect("serializable")
    }

    fn event_wire(event: &MessageStreamEvent) -> Value {
        serde_json::to_value(event).expect("serializable")
    }

    #[test]
    fn a_whole_output_streams_as_the_gateway_events_in_order() {
        let events = [
            Event::ReasoningStart,
            Event::Reasoning(Text::uncounted("plan ")),
            Event::Reasoning(Text::uncounted("more")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("Let me ")),
            Event::Content(Text::uncounted("check.")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            arguments(0, r#""Paris"}"#),
            end(0),
            finish(FinishReason::Stop, 1),
        ];
        let (out, stop) = stream(&events);
        assert_eq!(
            stream_wire(&out),
            json!([
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "thinking_delta", "thinking": "plan "}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "thinking_delta", "thinking": "more"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_start", "index": 1,
                 "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "text_delta", "text": "Let me "}},
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "text_delta", "text": "check."}},
                {"type": "content_block_stop", "index": 1},
                {"type": "content_block_start", "index": 2,
                 "content_block": {"type": "tool_use", "id": "toolu_0", "name": "get_weather", "input": {},
                                   "caller": {"type": "direct"}}},
                {"type": "content_block_delta", "index": 2,
                 "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}},
                {"type": "content_block_delta", "index": 2,
                 "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}},
                {"type": "content_block_stop", "index": 2},
            ])
        );
        assert_eq!(stop, Some(StopReason::ToolUse));
    }

    #[test]
    fn blocks_alternate_in_the_order_written_and_the_index_advances_per_block() {
        let events = [
            Event::Content(Text::uncounted("a")),
            Event::Reasoning(Text::uncounted("b")),
            Event::Content(Text::uncounted("c")),
            call(0, "f"),
            Event::Content(Text::uncounted("d")),
        ];
        let (out, _) = stream(&events);
        let summary: Vec<String> = out
            .iter()
            .map(|event| match event {
                MessageStreamEvent::ContentBlockStart { index, .. } => format!(
                    "start {index} {}",
                    event_wire(event)["content_block"]["type"]
                        .as_str()
                        .expect("a type")
                ),
                MessageStreamEvent::ContentBlockDelta { index, .. } => format!("delta {index}"),
                MessageStreamEvent::ContentBlockStop { index } => format!("stop {index}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            summary,
            [
                "start 0 text",
                "delta 0",
                "stop 0",
                "start 1 thinking",
                "delta 1",
                "stop 1",
                "start 2 text",
                "delta 2",
                "stop 2",
                "start 3 tool_use",
                "stop 3",
                "start 4 text",
                "delta 4",
            ]
        );
    }

    #[test]
    fn finish_closes_the_open_block_and_returns_the_stop_reason() {
        let (out, stop) = stream(&[
            Event::Content(Text::uncounted("done")),
            finish(FinishReason::Stop, 0),
        ]);
        assert_eq!(
            stream_wire(&out),
            json!([
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                {"type": "content_block_stop", "index": 0},
            ])
        );
        assert_eq!(stop, Some(StopReason::EndTurn));

        let (out, stop) = stream(&[finish(FinishReason::Length, 1)]);
        assert!(out.is_empty(), "no block was open: {out:?}");
        assert_eq!(stop, Some(StopReason::MaxTokens));
    }

    #[test]
    fn a_fragment_with_no_open_tool_use_block_is_text() {
        let never_started = [arguments(7, r#"{"stray":true}"#)];
        assert_eq!(
            stream_wire(&stream(&never_started).0),
            json!([
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": "{\"stray\":true}"}},
            ])
        );

        let closed_by_text = [
            call(0, "f"),
            arguments(0, r#"{"a":"#),
            Event::Content(Text::uncounted("between")),
            arguments(0, "1}"),
        ];
        let (out, _) = stream(&closed_by_text);
        let last = stream_wire(&out);
        let last = last
            .as_array()
            .expect("an array")
            .last()
            .expect("an event")
            .clone();
        assert_eq!(
            last,
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "1}"}})
        );
    }

    #[test]
    fn a_fragment_of_a_call_whose_block_another_call_closed_is_text() {
        let interleaved = [
            call(0, "get_weather"),
            call(1, "get_time"),
            arguments(1, r#"{"zone":"CET"}"#),
            arguments(0, r#"{"city":"Paris"}"#),
            end(1),
            end(0),
            finish(FinishReason::Stop, 2),
        ];
        let (out, _) = stream(&interleaved);
        let summary: Vec<String> = out
            .iter()
            .map(|event| match event {
                MessageStreamEvent::ContentBlockStart { index, .. } => format!(
                    "start {index} {}",
                    event_wire(event)["content_block"]["type"]
                        .as_str()
                        .expect("a type")
                ),
                MessageStreamEvent::ContentBlockDelta { index, delta } => match delta {
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        format!("input {index} {partial_json}")
                    }
                    ContentBlockDelta::TextDelta { text } => format!("text {index} {text}"),
                    other => format!("{other:?}"),
                },
                MessageStreamEvent::ContentBlockStop { index } => format!("stop {index}"),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            summary,
            [
                "start 0 tool_use",
                "stop 0",
                "start 1 tool_use",
                r#"input 1 {"zone":"CET"}"#,
                "stop 1",
                "start 2 text",
                r#"text 2 {"city":"Paris"}"#,
                "stop 2",
            ]
        );
        assert_eq!(
            wire(&output(&interleaved).content)[0]["input"],
            json!({"city": "Paris"}),
            "the fold still gives the fragment to its call"
        );
    }

    #[test]
    fn events_without_a_block_to_shape_produce_no_stream_events() {
        let (out, stop) = stream(&[
            Event::ReasoningStart,
            Event::ReasoningEnd,
            end(3),
            Event::Dropped {
                text: Text::uncounted("<think>"),
                why: DropReason::Wrapper,
            },
            Event::Content(Text::uncounted("")),
            Event::Reasoning(Text::uncounted("")),
            arguments(0, ""),
        ]);
        assert!(out.is_empty(), "{out:?}");
        assert_eq!(stop, None);
    }

    #[test]
    fn close_stops_the_open_block_once() {
        let mut machine = Stream::new();
        let mut out = Vec::new();
        machine.feed(&Event::Reasoning(Text::uncounted("x")), &mut out);
        machine.close(&mut out);
        machine.close(&mut out);
        assert_eq!(
            stream_wire(&out),
            json!([
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}},
                {"type": "content_block_stop", "index": 0},
            ])
        );
        machine.feed(&Event::Content(Text::uncounted("y")), &mut out);
        assert_eq!(
            stream_wire(&out)[3]["index"],
            json!(1),
            "the next block takes the next index"
        );
    }

    #[test]
    fn whitespace_only_text_opens_no_block_until_substance_joins_it() {
        let separators_only = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("\n\n")),
            call(0, "f"),
            end(0),
            Event::Content(Text::uncounted("\n")),
            finish(FinishReason::Stop, 1),
        ];
        let (out, _) = stream(&separators_only);
        let kinds: Vec<Value> = stream_wire(&out)
            .as_array()
            .expect("an array")
            .iter()
            .map(|event| json!([event["type"], event["index"]]))
            .collect();
        assert_eq!(
            kinds,
            [
                json!(["content_block_start", 0]),
                json!(["content_block_delta", 0]),
                json!(["content_block_stop", 0]),
                json!(["content_block_start", 1]),
                json!(["content_block_stop", 1]),
            ],
            "the separators open no text block"
        );

        let led_by_whitespace = [
            Event::Content(Text::uncounted("\n\n")),
            Event::Content(Text::uncounted("Hello")),
            Event::Content(Text::uncounted(" ")),
        ];
        assert_eq!(
            stream_wire(&stream(&led_by_whitespace).0),
            json!([
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "\n\nHello"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " "}},
            ]),
            "held whitespace leads the block; whitespace inside an open block is text at once"
        );

        let before_thinking = [
            Event::Content(Text::uncounted("\n")),
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("Hello")),
        ];
        let text: String = stream(&before_thinking)
            .0
            .iter()
            .filter_map(|event| match event {
                MessageStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::TextDelta { text },
                    ..
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            text, "Hello",
            "whitespace held before a thinking block is let go, not carried past it"
        );
    }

    /// The blocks a client assembles from the stream: the text and thinking deltas joined, and a
    /// `tool_use` block's `input` read from its joined `input_json_delta`s by a partial JSON
    /// reader that keeps complete members only ([`Prefix::closed`]), `{}` where nothing of them is
    /// complete.
    fn client_blocks(out: &[MessageStreamEvent]) -> Vec<Value> {
        let mut blocks: Vec<Value> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        for event in out {
            match event {
                MessageStreamEvent::ContentBlockStart { .. } => {
                    blocks.push(event_wire(event)["content_block"].clone());
                    inputs.push(String::new());
                }
                MessageStreamEvent::ContentBlockDelta { delta, .. } => {
                    let block = blocks.last_mut().expect("a started block");
                    let last = inputs.last_mut().expect("a started block");
                    match delta {
                        ContentBlockDelta::TextDelta { text } => {
                            let text =
                                block["text"].as_str().unwrap_or_default().to_string() + text;
                            block["text"] = json!(text);
                        }
                        ContentBlockDelta::ThinkingDelta { thinking } => {
                            let text = block["thinking"].as_str().unwrap_or_default().to_string()
                                + thinking;
                            block["thinking"] = json!(text);
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            last.push_str(partial_json);
                        }
                        other => panic!("unexpected delta {other:?}"),
                    }
                }
                MessageStreamEvent::ContentBlockStop { .. } => {}
                other => panic!("unexpected event {other:?}"),
            }
        }
        for (block, input) in blocks.iter_mut().zip(&inputs) {
            if block["type"] == "tool_use" {
                block["input"] = Prefix::closed(input)
                    .and_then(|closed| serde_json::from_str(&closed).ok())
                    .unwrap_or_else(|| json!({}));
            }
        }
        blocks
    }

    #[test]
    fn the_stream_and_the_fold_agree_on_an_output_in_block_order() {
        let events = [
            Event::ReasoningStart,
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("\n\n")),
            Event::Content(Text::uncounted("Let me ")),
            Event::Malformed {
                text: Text::uncounted("<tool_call>{broken"),
                why: MalformedReason::InvalidArguments,
            },
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            arguments(0, r#""Paris"}"#),
            end(0),
            call(1, "get_time"),
            arguments(1, r#"{"zone": "CET", "offset":"#),
            end(1),
            finish(FinishReason::Length, 2),
        ];
        let (out, stop) = stream(&events);
        // A client's partial JSON reader closes a block's joined deltas into the value their
        // complete members make: the call the output cut ends as `{"zone": "CET"}` for it, and
        // the fold gives the call the same.
        let blocks = client_blocks(&out);
        let folded = output(&events);
        assert_eq!(wire(&folded.content)[3]["input"], json!({"zone": "CET"}));
        assert_eq!(Value::Array(blocks), wire(&folded.content));
        assert_eq!(stop, folded.stop_reason);
    }

    /// The recorded output of a call whose arguments stop being JSON, through the table arriving
    /// in two pieces at every cut and ended at every byte. In two pieces: the fold gives the same
    /// blocks as for the whole, the stream's `tool_use` block carries the prefix a strict reader
    /// accepts and nothing after the break, the blocks a client assembles from the stream are the
    /// fold's, and the stop reasons agree. Ended at the byte: the `input` a client closes from the
    /// stream is the fold's, and the stop reasons agree.
    #[test]
    fn the_stream_and_the_fold_agree_on_a_call_whose_arguments_stop_being_json_at_every_cut() {
        const OUTPUT: &str = concat!(
            "<tool_call>\n{\"name\": \"plot_sine_wave\", \"arguments\": ",
            "{\"start_range\": 0, \"end_range\": 2.0 * Math.PI, \"frequency\": 5}}\n</tool_call>"
        );
        const PREFIX: &str = r#"{"start_range": 0, "end_range": 2.0 "#;
        let whole = output(&parsed(OUTPUT));
        let blocks = wire(&whole.content);
        assert_eq!(blocks.as_array().map(Vec::len), Some(1), "{blocks}");
        assert_eq!(blocks[0]["type"], "tool_use", "{blocks}");
        assert_eq!(
            blocks[0]["input"],
            json!({"start_range": 0, "end_range": 2.0}),
            "{blocks}"
        );
        assert_eq!(whole.stop_reason, Some(StopReason::ToolUse));
        // The ids are minted per reading: a comparison of two readings leaves them out.
        let without_ids = |blocks: &Value| -> Value {
            let mut blocks = blocks.clone();
            for block in blocks.as_array_mut().expect("blocks") {
                block.as_object_mut().expect("a block").remove("id");
            }
            blocks
        };
        let tool_use_input = |blocks: &[Value]| -> Option<Value> {
            blocks
                .iter()
                .find(|block| block["type"] == "tool_use")
                .map(|block| block["input"].clone())
        };
        let mut cuts = 0;
        for cut in 1..OUTPUT.len() {
            if !OUTPUT.is_char_boundary(cut) {
                continue;
            }
            cuts += 1;
            let events = parsed_in(&[&OUTPUT[..cut], &OUTPUT[cut..]], EngineFinish::Stop);
            let folded = output(&events);
            assert_eq!(
                without_ids(&wire(&folded.content)),
                without_ids(&blocks),
                "cut at {cut}"
            );
            let (out, stop) = stream(&events);
            let partial: String = out
                .iter()
                .map(event_wire)
                .filter_map(|event| event["delta"]["partial_json"].as_str().map(String::from))
                .collect();
            assert_eq!(partial, PREFIX, "cut at {cut}");
            assert_eq!(
                Value::Array(client_blocks(&out)),
                wire(&folded.content),
                "cut at {cut}"
            );
            assert_eq!(stop, folded.stop_reason, "cut at {cut}");

            let events = parsed_in(&[&OUTPUT[..cut]], EngineFinish::Length);
            let folded = wire(&output(&events).content);
            let folded = folded.as_array().expect("blocks");
            let (out, stop) = stream(&events);
            assert_eq!(
                tool_use_input(&client_blocks(&out)),
                tool_use_input(folded),
                "ended at {cut}"
            );
            assert_eq!(stop, output(&events).stop_reason, "ended at {cut}");
        }
        assert!(cuts > 100, "{cuts} cuts");
    }

    /// The events of `output` read by the Qwen3 table through an engine of its own, as the gateway
    /// reads each request.
    fn parsed(output: &str) -> Vec<Event<'static>> {
        parsed_in(&[output], EngineFinish::Stop)
    }

    /// The events of an output that arrives in `pieces`, read by the Qwen3 table through an engine
    /// of its own and ended by `finish`.
    fn parsed_in(pieces: &[&str], finish: EngineFinish) -> Vec<Event<'static>> {
        let mut engine = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        for &text in pieces {
            engine
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("a delta");
        }
        engine
            .feed(Input::End { finish }, &mut out)
            .expect("the end");
        out.into_owned().drain()
    }

    /// Two calls of Qwen3's syntax in one output.
    const TWO_CALLS: &str = concat!(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n",
        "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>"
    );

    #[test]
    fn two_turns_of_two_calls_carry_four_distinct_tool_use_ids_the_stream_and_the_fold_agree_on() {
        // A client keeps its `tool_use` blocks and `tool_result`s by id across the conversation:
        // no id of one turn may come back in the next, and the stream must name the block as the
        // folded message does.
        let mut seen: Vec<String> = Vec::new();
        for turn in 0..2 {
            let events = parsed(TWO_CALLS);
            let folded: Vec<String> = wire(&output(&events).content)
                .as_array()
                .expect("blocks")
                .iter()
                .filter(|block| block["type"] == "tool_use")
                .map(|block| block["id"].as_str().expect("an id").to_string())
                .collect();
            let (out, _) = stream(&events);
            let streamed: Vec<String> = out
                .iter()
                .map(event_wire)
                .filter(|event| event["content_block"]["type"] == "tool_use")
                .map(|event| {
                    event["content_block"]["id"]
                        .as_str()
                        .expect("an id")
                        .to_string()
                })
                .collect();
            assert_eq!(folded.len(), 2, "turn {turn}: {folded:?}");
            assert_eq!(streamed, folded, "turn {turn}");
            for id in &folded {
                let hex = id
                    .strip_prefix("toolu_")
                    .unwrap_or_else(|| panic!("turn {turn}: {id}"));
                assert!(is_minted(&format!("call_{hex}")), "turn {turn}: {id}");
            }
            seen.extend(folded);
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 4, "{seen:?}");
    }
}
