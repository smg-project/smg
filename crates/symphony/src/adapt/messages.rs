//! Messages: parser events as the content blocks of one assistant message.
//!
//! [`output`] folds a whole event sequence into the message's `content` blocks and its
//! `stop_reason`. It is pure like [`chat::message`](super::chat::message) and
//! [`responses::output`](super::responses::output): call ids ride on the events, the stop reason is
//! decided from the `Finish` event alone, and nothing is remembered between calls. The driver wraps
//! the blocks in the message envelope (id, model, usage) and, when its stop decoder matched a stop
//! sequence, sets `stop_sequence` and that reason, since the events carry neither.
//!
//! [`Stream`] is the streaming half, the content-block machine: one block is open at a time, and
//! each parser event becomes the `content_block_start`, `content_block_delta` and
//! `content_block_stop` events a client expects, the block index advancing by one per block. A
//! `thinking` block opens at the first reasoning text and closes when the region ends; a `text`
//! block opens at the first content; a `tool_use` block opens at `ToolCallStart` with an empty
//! `input` and closes at `ToolCallEnd`; a block closes when text of another kind arrives.
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
//! The blocks are the ones SMG's gateway builds today from a finished turn, in the order the
//! Messages API uses:
//!
//! - one `thinking` block when there was reasoning, with an empty `signature`: the engines sign
//!   nothing, and a signature a client could verify needs a decision of its own;
//! - one `text` block when the content is not whitespace, every content and malformed text in
//!   order;
//! - one `tool_use` block per call in the order the calls started, its `input` the joined argument
//!   fragments parsed as JSON.
//!
//! This is the same folding as `chat::message` and `responses::output`, so the three adapters agree
//! on what a turn contains.
//!
//! Ids: a call id of the form `call_<suffix>` becomes `toolu_<suffix>`, the prefix Anthropic SDKs
//! and tooling pattern-match; any other shape passes through unchanged, since a model family whose
//! id format is load-bearing in the prompt must keep it. The rule is deterministic, so a client
//! echoing the id back names the same call.
//!
//! Stop reason, from the `Finish` event: `length` is `max_tokens`, and parsed calls do not override
//! a truncation; `tool_calls`, and any other reason after at least one tool call, are `tool_use`,
//! since a message with `tool_use` blocks says so whatever stopped the model, as the API itself
//! does; `stop` is `end_turn`; `abort` and engine-specific reasons are `end_turn` too when no call
//! was made, since the Messages API has no word for them and the driver decides whether a failed
//! generation is a message at all. A sequence without `Finish` has no stop reason.
//!
//! Policy this adapter sets, where the events say more than the Messages API can:
//!
//! - `ReasoningStart` and `Dropped` shape nothing, and `ReasoningEnd` and `ToolCallEnd` only close
//!   their blocks in the stream: the markers a format consumes never reach the client.
//! - `Malformed` text is content, as in `chat`: the model wrote it, and the client sees it rather
//!   than losing it.
//! - A call whose argument fragments are not a whole JSON value, because the output was cut or the
//!   model wrote something else, has `{}` as its `input`: `input` must be a JSON value, and the
//!   fragments are not one. Chat carries the fragments as a string and Responses marks the call
//!   `incomplete`; here `max_tokens` is all a client learns. A call the model closed without
//!   arguments has `{}` too.
//! - Argument fragments for a call that never started cannot occur under the `Event` contract;
//!   should one arrive, its text joins the content so that no byte is lost. In the stream the same
//!   goes for a fragment whose call has no open `tool_use` block because other text, or another
//!   call, came between the call's start and the fragment: the Messages API streams one block at a
//!   time and cannot reopen a closed one, so the fragment is text, where the fold still gives it to
//!   its call. The formats parse a call from one contiguous region, so this does not happen for
//!   them, and nothing is dropped if it does; the old driver dropped such fragments.
//! - In the stream, the thinking block stops when the reasoning region ends rather than when the
//!   next text arrives, as the old driver did, so a client sees the block close as soon as the
//!   model stops thinking.
//! - In the stream, whitespace-only text opens no block until text with substance joins it; the old
//!   driver opened a text block for any non-empty text, so a separator between a thinking block
//!   and a call became a block holding one newline.

use std::mem;

use openai_protocol::messages::{ContentBlock, ContentBlockDelta, MessageStreamEvent, StopReason};
use serde_json::{Map, Value};

use crate::event::{Event, FinishReason};

/// A whole output as the Messages API reports it: the content blocks and the stop reason.
#[derive(Clone, Debug)]
pub struct Output {
    /// The content blocks in the Messages API's order: thinking, text, then the calls as they
    /// started.
    pub content: Vec<ContentBlock>,
    /// The stop reason decided from the `Finish` event; none when the sequence has not finished.
    pub stop_reason: Option<StopReason>,
}

/// One started call and what its events said.
struct Call {
    index: u32,
    id: String,
    name: String,
    arguments: String,
}

/// The message's content and stop reason, folded from a whole event sequence.
pub fn output<'a>(events: impl IntoIterator<Item = &'a Event>) -> Output {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<Call> = Vec::new();
    let mut stop_reason = None;
    for event in events {
        match event {
            Event::Content(text) | Event::Malformed { text, .. } => content.push_str(&text.text),
            Event::Reasoning(text) => reasoning.push_str(&text.text),
            Event::ToolCallStart {
                index, id, name, ..
            } => calls.push(Call {
                index: *index,
                id: id.clone(),
                name: name.clone(),
                arguments: String::new(),
            }),
            Event::ToolCallArguments { index, json, .. } => {
                match calls.iter_mut().find(|call| call.index == *index) {
                    Some(call) => call.arguments.push_str(json),
                    None => content.push_str(json),
                }
            }
            Event::Finish {
                reason, tool_calls, ..
            } => stop_reason = Some(stop(reason, *tool_calls)),
            Event::ReasoningStart
            | Event::ReasoningEnd
            | Event::ToolCallEnd { .. }
            | Event::Dropped { .. } => {}
        }
    }

    let mut blocks = Vec::new();
    if !reasoning.is_empty() {
        blocks.push(ContentBlock::Thinking {
            thinking: reasoning,
            signature: String::new(),
        });
    }
    if !content.trim().is_empty() {
        blocks.push(ContentBlock::Text {
            text: content,
            citations: None,
        });
    }
    blocks.extend(calls.into_iter().map(|call| ContentBlock::ToolUse {
        id: tool_use_id(&call.id),
        name: call.name,
        input: input(&call.arguments),
    }));

    Output {
        content: blocks,
        stop_reason,
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
    pub fn feed(&mut self, event: &Event, out: &mut Vec<MessageStreamEvent>) -> Option<StopReason> {
        match event {
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
    /// no open block, so that no byte is lost.
    fn arguments(&mut self, call: u32, json: &str, out: &mut Vec<MessageStreamEvent>) {
        if json.is_empty() {
            return;
        }
        if self.open != Some(Block::ToolUse { call }) {
            self.text(json, out);
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

/// The `input` for a call's joined argument fragments: the JSON value they form, or `{}` when they
/// do not form one.
fn input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::Object(Map::new()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::event::{DropReason, MalformedReason, Text};

    fn call(index: u32, name: &str) -> Event {
        Event::ToolCallStart {
            index,
            id: format!("call_{index}"),
            name: name.into(),
            source: Text::default(),
        }
    }

    fn arguments(index: u32, json: &str) -> Event {
        Event::ToolCallArguments {
            index,
            json: json.into(),
            source: Text::default(),
        }
    }

    fn end(index: u32) -> Event {
        Event::ToolCallEnd {
            index,
            source: Text::default(),
        }
    }

    fn finish(reason: FinishReason, tool_calls: u32) -> Event {
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
                {"type": "tool_use", "id": "toolu_0", "name": "get_weather", "input": {"city": "Paris"}},
                {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {"zone": "CET"}},
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

    #[test]
    fn arguments_that_are_not_a_whole_json_value_give_an_empty_input() {
        let inputs = |fragments: &[&str]| {
            let mut events = vec![call(0, "f")];
            events.extend(fragments.iter().map(|json| arguments(0, json)));
            events.push(end(0));
            events.push(finish(FinishReason::Length, 1));
            wire(&output(&events).content)[0]["input"].clone()
        };
        assert_eq!(inputs(&[r#"{"city":"#]), json!({}), "cut short");
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
                 "content_block": {"type": "tool_use", "id": "toolu_0", "name": "get_weather", "input": {}}},
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
            arguments(1, r#"{"zone":"#),
            end(1),
            finish(FinishReason::Length, 2),
        ];
        let (out, stop) = stream(&events);
        let mut blocks: Vec<Value> = Vec::new();
        let mut inputs: Vec<String> = Vec::new();
        for event in &out {
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
                block["input"] = serde_json::from_str(input).unwrap_or_else(|_| json!({}));
            }
        }
        let folded = output(&events);
        assert_eq!(Value::Array(blocks), wire(&folded.content));
        assert_eq!(stop, folded.stop_reason);
    }
}
