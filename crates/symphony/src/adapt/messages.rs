//! Messages: parser events as the content blocks of one assistant message.
//!
//! [`output`] folds a whole event sequence into the message's `content` blocks and its
//! `stop_reason`. It is pure like [`chat::message`](super::chat::message) and
//! [`responses::output`](super::responses::output): call ids ride on the events, the stop reason is
//! decided from the `Finish` event alone, and nothing is remembered between calls. The driver wraps
//! the blocks in the message envelope (id, model, usage) and, when its stop decoder matched a stop
//! sequence, sets `stop_sequence` and that reason, since the events carry neither.
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
//! - `ReasoningStart`, `ReasoningEnd`, `ToolCallEnd` and `Dropped` shape nothing; the markers a
//!   format consumes never reach the client.
//! - `Malformed` text is content, as in `chat`: the model wrote it, and the client sees it rather
//!   than losing it.
//! - A call whose argument fragments are not a whole JSON value, because the output was cut or the
//!   model wrote something else, has `{}` as its `input`: `input` must be a JSON value, and the
//!   fragments are not one. Chat carries the fragments as a string and Responses marks the call
//!   `incomplete`; here `max_tokens` is all a client learns. A call the model closed without
//!   arguments has `{}` too.
//! - Argument fragments for a call that never started cannot occur under the `Event` contract;
//!   should one arrive, its text joins the content so that no byte is lost.

use openai_protocol::messages::{ContentBlock, StopReason};
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
}
