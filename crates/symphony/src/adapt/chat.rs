//! Chat Completions: parser events as stream choices or as one message.
//!
//! [`delta`] maps one [`Event`] to at most one [`ChatStreamChoice`], [`step`] maps the events of
//! one engine step to at most one, and [`message`] folds a whole event sequence into one
//! [`ChatChoice`]. All three are pure: call ids ride on the events, the finish
//! reason is decided from the `Finish` event alone, and nothing is remembered between calls. The
//! driver wraps the choices in the response envelope (request id, model, timestamp, usage) and sets
//! `matched_stop`, and `logprobs` where it has them, since the events carry neither.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/adapters.md`.

use std::borrow::Cow;

use openai_protocol::{
    chat::{ChatChoice, ChatCompletionMessage, ChatMessageDelta, ChatStreamChoice},
    common::{FunctionCallDelta, FunctionCallResponse, ToolCall, ToolCallDelta},
};

use super::fold::fold;
use crate::event::{Event, FinishReason};

/// The stream choice one event produces for the choice at `choice`, if it produces one. The
/// choice carries a copy of the text it takes from the event; [`into_delta`] takes the event and
/// its text.
pub fn delta(choice: u32, event: &Event<'_>) -> Option<ChatStreamChoice> {
    match event {
        Event::Content(text) | Event::Malformed { text, .. } => {
            content(choice, text.text.to_string())
        }
        Event::Reasoning(text) => reasoning(choice, text.text.to_string()),
        Event::ToolCallStart {
            index, id, name, ..
        } => Some(call_start(choice, *index, id.clone(), name.clone())),
        Event::ToolCallArguments { index, json, .. } => arguments(choice, *index, json.clone()),
        Event::Finish {
            reason, tool_calls, ..
        } => Some(finish(choice, reason, *tool_calls)),
        Event::ReasoningStart
        | Event::ReasoningEnd
        | Event::ToolCallEnd { .. }
        | Event::Dropped { .. } => None,
    }
}

/// [`delta`] over an event the caller gives up: the choice takes the event's text in place of a
/// copy, which a stream that renders every delta once need not make.
pub fn into_delta(choice: u32, event: Event<'_>) -> Option<ChatStreamChoice> {
    match event {
        Event::Content(text) | Event::Malformed { text, .. } => {
            content(choice, text.text.into_owned())
        }
        Event::Reasoning(text) => reasoning(choice, text.text.into_owned()),
        Event::ToolCallStart {
            index, id, name, ..
        } => Some(call_start(choice, index, id, name)),
        Event::ToolCallArguments { index, json, .. } => arguments(choice, index, json),
        Event::Finish {
            reason, tool_calls, ..
        } => Some(finish(choice, &reason, tool_calls)),
        Event::ReasoningStart
        | Event::ReasoningEnd
        | Event::ToolCallEnd { .. }
        | Event::Dropped { .. } => None,
    }
}

/// A content delta; none for text that is empty.
fn content(choice: u32, text: String) -> Option<ChatStreamChoice> {
    (!text.is_empty()).then(|| {
        unfinished(
            choice,
            ChatMessageDelta {
                content: Some(text),
                ..assistant()
            },
        )
    })
}

/// A reasoning delta; none for text that is empty.
fn reasoning(choice: u32, text: String) -> Option<ChatStreamChoice> {
    (!text.is_empty()).then(|| {
        unfinished(
            choice,
            ChatMessageDelta {
                reasoning_content: Some(text),
                ..assistant()
            },
        )
    })
}

/// The delta that starts the call at `index`: its id and name.
fn call_start(choice: u32, index: u32, id: String, name: String) -> ChatStreamChoice {
    unfinished(
        choice,
        ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }]),
            ..assistant()
        },
    )
}

/// A delta of the call's arguments; none for an empty fragment.
fn arguments(choice: u32, index: u32, json: String) -> Option<ChatStreamChoice> {
    (!json.is_empty()).then(|| {
        unfinished(
            choice,
            ChatMessageDelta {
                tool_calls: Some(vec![ToolCallDelta {
                    index,
                    id: None,
                    tool_type: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: Some(json),
                    }),
                }]),
                ..assistant()
            },
        )
    })
}

/// The finishing choice: an empty delta without a role, and the finish reason.
fn finish(choice: u32, reason: &FinishReason, tool_calls: u32) -> ChatStreamChoice {
    ChatStreamChoice {
        index: choice,
        delta: ChatMessageDelta {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: None,
        },
        logprobs: None,
        finish_reason: Some(finish_reason(reason, tool_calls)),
        matched_stop: None,
    }
}

/// The choice at `choice` carrying `delta`, not finished.
fn unfinished(choice: u32, delta: ChatMessageDelta) -> ChatStreamChoice {
    ChatStreamChoice {
        index: choice,
        delta,
        logprobs: None,
        finish_reason: None,
        matched_stop: None,
    }
}

/// The stream choices a sequence of events produces, in order.
pub fn deltas<'a, 'd: 'a>(
    choice: u32,
    events: impl IntoIterator<Item = &'a Event<'d>>,
) -> Vec<ChatStreamChoice> {
    events
        .into_iter()
        .filter_map(|event| delta(choice, event))
        .collect()
}

/// The one stream choice a step's events make for the choice at `choice`, if they make one.
///
/// A step is what one engine delta produced. Its content and malformed text, in order, are the
/// chunk's `content`; its reasoning text is `reasoning_content`; each call the step touched is
/// one `tool_calls` entry, in the order the step first touched it, with the id and name where the
/// call started in this step and the step's argument fragments joined. The `Finish` event makes
/// no chunk here: the driver writes the finish chunk, with the matched stop. The choice carries no
/// role: on the wire only a choice's opening chunk does, and the driver writes that one. `None`
/// when the step made nothing a chunk carries. The events are taken by value: what one owns (a
/// call's id and name, a fragment's JSON) moves into the choice, a borrowed text is copied once.
pub fn step<'d>(
    choice: u32,
    events: impl IntoIterator<Item = Event<'d>>,
) -> Option<ChatStreamChoice> {
    let mut content: Option<String> = None;
    let mut reasoning: Option<String> = None;
    let mut calls: Vec<ToolCallDelta> = Vec::new();
    for event in events {
        match event {
            Event::Content(text) | Event::Malformed { text, .. } => join(&mut content, text.text),
            Event::Reasoning(text) => join(&mut reasoning, text.text),
            Event::ToolCallStart {
                index, id, name, ..
            } => calls.push(ToolCallDelta {
                index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }),
            Event::ToolCallArguments { index, json, .. } if !json.is_empty() => {
                match calls.iter_mut().find(|call| call.index == index) {
                    Some(call) => {
                        let function = call.function.get_or_insert(FunctionCallDelta {
                            name: None,
                            arguments: None,
                        });
                        join(&mut function.arguments, json);
                    }
                    None => calls.push(ToolCallDelta {
                        index,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: Some(json),
                        }),
                    }),
                }
            }
            Event::ToolCallArguments { .. }
            | Event::ReasoningStart
            | Event::ReasoningEnd
            | Event::ToolCallEnd { .. }
            | Event::Dropped { .. }
            | Event::Finish { .. } => {}
        }
    }
    if content.is_none() && reasoning.is_none() && calls.is_empty() {
        return None;
    }
    Some(unfinished(
        choice,
        ChatMessageDelta {
            role: None,
            content,
            tool_calls: (!calls.is_empty()).then_some(calls),
            reasoning_content: reasoning,
        },
    ))
}

/// Appends `text` to the delta's field, which starts at the first text that is not empty and
/// takes an owned text as it is.
fn join<'d>(field: &mut Option<String>, text: impl Into<Cow<'d, str>>) {
    let text = text.into();
    if text.is_empty() {
        return;
    }
    match field {
        Some(so_far) => so_far.push_str(&text),
        None => *field = Some(text.into_owned()),
    }
}

/// One message for the choice at `choice`, folded from a whole event sequence.
///
/// `content` is every content and malformed text in order, absent when it is only whitespace;
/// `reasoning_content` is every reasoning text, absent when there was none; `tool_calls` lists the
/// calls in the order they started, absent when there was none; `finish_reason` is set by the
/// `Finish` event, so a sequence without one gives a message that is not finished. Argument
/// fragments for a call that never started cannot occur under the `Event` contract; should one
/// arrive, its text joins the content so that no byte is lost.
pub fn message<'a, 'd: 'a>(
    choice: u32,
    events: impl IntoIterator<Item = &'a Event<'d>>,
) -> ChatChoice {
    let folded = fold(events);
    let calls: Vec<ToolCall> = folded
        .calls
        .into_iter()
        .map(|call| ToolCall {
            id: call.id,
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: call.name,
                arguments: Some(call.arguments),
            },
        })
        .collect();
    ChatChoice {
        index: choice,
        message: ChatCompletionMessage {
            role: "assistant".to_string(),
            content: (!folded.content.trim().is_empty()).then_some(folded.content),
            refusal: None,
            annotations: Vec::new(),
            tool_calls: (!calls.is_empty()).then_some(calls),
            reasoning_content: (!folded.reasoning.is_empty()).then_some(folded.reasoning),
        },
        logprobs: None,
        finish_reason: folded
            .finish
            .map(|finish| finish_reason(&finish.reason, finish.tool_calls)),
        matched_stop: None,
        hidden_states: None,
    }
}

/// The `finish_reason` string for the engine's reason: `stop` after a tool call is `tool_calls`.
fn finish_reason(reason: &FinishReason, tool_calls: u32) -> String {
    match reason {
        FinishReason::Stop if tool_calls > 0 => "tool_calls",
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Abort => "abort",
        FinishReason::Other(other) => return other.clone(),
    }
    .to_string()
}

/// An assistant delta with nothing in it yet.
fn assistant() -> ChatMessageDelta {
    ChatMessageDelta {
        role: Some("assistant".to_string()),
        content: None,
        tool_calls: None,
        reasoning_content: None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::event::{DropReason, MalformedReason, Text};

    fn wire(choice: Option<ChatStreamChoice>) -> Value {
        serde_json::to_value(choice.expect("a choice")).expect("serializable")
    }

    /// One event of every kind, with text where the kind carries it.
    fn one_of_each() -> Vec<Event<'static>> {
        vec![
            Event::Content(Text::uncounted("Hello")),
            Event::Content(Text::uncounted("")),
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningStart,
            Event::ReasoningEnd,
            Event::ToolCallStart {
                index: 1,
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                source: Text::uncounted("<tool_call>"),
            },
            Event::ToolCallArguments {
                index: 1,
                json: "{\"q\":".to_string(),
                source: Text::uncounted("{\"q\":"),
            },
            Event::ToolCallArguments {
                index: 1,
                json: String::new(),
                source: Text::uncounted(""),
            },
            Event::ToolCallEnd {
                index: 1,
                source: Text::uncounted("</tool_call>"),
            },
            Event::Dropped {
                text: Text::uncounted("\n"),
                why: DropReason::Whitespace,
            },
            Event::Malformed {
                text: Text::uncounted("junk"),
                why: MalformedReason::Other("why".into()),
            },
            Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 1,
                reasoning_tokens: 0,
            },
        ]
    }

    #[test]
    fn a_borrowed_and_an_owned_event_make_the_same_choice() {
        for event in one_of_each() {
            let borrowed =
                delta(3, &event).map(|choice| serde_json::to_value(choice).expect("wire"));
            let owned = into_delta(3, event.clone())
                .map(|choice| serde_json::to_value(choice).expect("wire"));
            assert_eq!(borrowed, owned, "{event:?}");
        }
    }

    #[test]
    fn a_step_renders_as_one_chunk_with_its_text_and_calls_together() {
        let events = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("Sure. ")),
            Event::Dropped {
                text: Text::uncounted("<tool_call>"),
                why: DropReason::Wrapper,
            },
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            arguments(0, r#""Paris"}"#),
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
        ];
        assert_eq!(
            wire(step(1, events)),
            json!({
                "index": 1,
                "delta": {
                    "content": "Sure. ",
                    "reasoning_content": "plan",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": r#"{"city":"Paris"}"#},
                    }],
                },
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn a_step_with_two_calls_keeps_one_entry_per_call_in_the_order_touched() {
        let events = [
            arguments(0, "}"),
            call(1, "get_time"),
            arguments(1, r#"{"zone":"CET"}"#),
            arguments(0, ""),
        ];
        let choice = step(0, events).expect("a chunk");
        let calls = choice.delta.tool_calls.expect("calls");
        assert_eq!(calls.len(), 2);
        assert_eq!((calls[0].index, calls[0].id.as_deref()), (0, None));
        assert_eq!(
            calls[0]
                .function
                .as_ref()
                .and_then(|f| f.arguments.as_deref()),
            Some("}")
        );
        assert_eq!(
            (calls[1].index, calls[1].id.as_deref()),
            (1, Some("call_1"))
        );
        assert_eq!(
            calls[1]
                .function
                .as_ref()
                .and_then(|f| f.arguments.as_deref()),
            Some(r#"{"zone":"CET"}"#)
        );
        assert!(choice.delta.content.is_none());
    }

    #[test]
    fn a_step_that_carries_nothing_a_chunk_shows_makes_no_chunk() {
        assert!(step(0, []).is_none());
        assert!(step(
            0,
            [
                Event::ReasoningStart,
                Event::Content(Text::uncounted("")),
                Event::Dropped {
                    text: Text::uncounted("\n"),
                    why: DropReason::Whitespace,
                },
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::default(),
                },
                finish(FinishReason::Stop, 1),
            ]
        )
        .is_none());
    }

    #[test]
    fn a_step_of_one_event_says_what_the_events_delta_says_but_for_the_role() {
        for event in one_of_each() {
            let one =
                step(3, [event.clone()]).map(|choice| serde_json::to_value(choice).expect("wire"));
            let each = delta(3, &event)
                .filter(|choice| choice.finish_reason.is_none())
                .map(|mut choice| {
                    choice.delta.role = None;
                    serde_json::to_value(choice).expect("wire")
                });
            assert_eq!(one, each, "{event:?}");
        }
    }

    #[test]
    fn content_is_an_assistant_delta_with_the_text() {
        assert_eq!(
            wire(delta(0, &Event::Content(Text::uncounted("Hello")))),
            json!({
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn reasoning_goes_to_reasoning_content() {
        assert_eq!(
            wire(delta(0, &Event::Reasoning(Text::new("plan", 1)))),
            json!({
                "index": 0,
                "delta": {"role": "assistant", "reasoning_content": "plan"},
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn a_call_begins_with_its_id_type_and_name_and_no_arguments() {
        let start = Event::ToolCallStart {
            index: 1,
            id: "call_1".into(),
            name: "get_weather".into(),
            source: Text::default(),
        };
        assert_eq!(
            wire(delta(0, &start)),
            json!({
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 1,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather"},
                    }],
                },
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn argument_fragments_carry_only_the_index_and_the_arguments() {
        let fragment = Event::ToolCallArguments {
            index: 1,
            json: r#"{"city":"#.into(),
            source: Text::default(),
        };
        assert_eq!(
            wire(delta(0, &fragment)),
            json!({
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [{"index": 1, "function": {"arguments": "{\"city\":"}}],
                },
                "logprobs": null,
                "finish_reason": null,
            })
        );
    }

    #[test]
    fn the_finishing_choice_has_an_empty_delta_and_the_reason() {
        let finish = Event::Finish {
            reason: FinishReason::Length,
            tool_calls: 0,
            reasoning_tokens: 0,
        };
        assert_eq!(
            wire(delta(3, &finish)),
            json!({
                "index": 3,
                "delta": {},
                "logprobs": null,
                "finish_reason": "length",
            })
        );
    }

    #[test]
    fn stop_after_a_tool_call_is_reported_as_tool_calls() {
        let reason = |reason: FinishReason, tool_calls: u32| {
            delta(
                0,
                &Event::Finish {
                    reason,
                    tool_calls,
                    reasoning_tokens: 0,
                },
            )
            .expect("a choice")
            .finish_reason
        };
        assert_eq!(reason(FinishReason::Stop, 0).as_deref(), Some("stop"));
        assert_eq!(reason(FinishReason::Stop, 2).as_deref(), Some("tool_calls"));
        assert_eq!(
            reason(FinishReason::ToolCalls, 0).as_deref(),
            Some("tool_calls")
        );
        assert_eq!(reason(FinishReason::Length, 1).as_deref(), Some("length"));
        assert_eq!(reason(FinishReason::Abort, 0).as_deref(), Some("abort"));
        assert_eq!(
            reason(FinishReason::Other("content_filter".into()), 0).as_deref(),
            Some("content_filter")
        );
    }

    #[test]
    fn malformed_text_reaches_the_client_as_content() {
        let malformed = Event::Malformed {
            text: Text::uncounted("<tool_call>{broken"),
            why: MalformedReason::InvalidArguments,
        };
        let choice = delta(0, &malformed).expect("a choice");
        assert_eq!(choice.delta.content.as_deref(), Some("<tool_call>{broken"));
        assert_eq!(choice.delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn events_without_a_chat_representation_produce_nothing() {
        let silent = [
            Event::ReasoningStart,
            Event::ReasoningEnd,
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
            Event::Dropped {
                text: Text::uncounted("<think>"),
                why: DropReason::Wrapper,
            },
            Event::Content(Text::uncounted("")),
            Event::Reasoning(Text::uncounted("")),
            Event::ToolCallArguments {
                index: 0,
                json: String::new(),
                source: Text::default(),
            },
        ];
        for event in &silent {
            assert!(delta(0, event).is_none(), "{event:?}");
        }
    }

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

    fn finish(reason: FinishReason, tool_calls: u32) -> Event<'static> {
        Event::Finish {
            reason,
            tool_calls,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn a_whole_output_folds_into_one_message_in_the_gateway_shape() {
        let events = [
            Event::ReasoningStart,
            Event::Reasoning(Text::uncounted("plan ")),
            Event::Reasoning(Text::uncounted("more")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("Let me check.")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            arguments(0, r#""Paris"}"#),
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
            finish(FinishReason::Stop, 1),
        ];
        assert_eq!(
            serde_json::to_value(message(1, &events)).expect("serializable"),
            json!({
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "refusal": null,
                    "annotations": [],
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                    }],
                    "reasoning_content": "plan more",
                },
                "logprobs": null,
                "finish_reason": "tool_calls",
            })
        );
    }

    #[test]
    fn whitespace_only_content_and_absent_reasoning_and_calls_are_left_out() {
        let events = [
            Event::Content(Text::uncounted("  \n")),
            finish(FinishReason::Stop, 0),
        ];
        assert_eq!(
            serde_json::to_value(message(0, &events)).expect("serializable"),
            json!({
                "index": 0,
                "message": {"role": "assistant", "content": null, "refusal": null, "annotations": [], "reasoning_content": null},
                "logprobs": null,
                "finish_reason": "stop",
            })
        );
    }

    #[test]
    fn malformed_text_joins_the_content_of_the_message() {
        let events = [
            Event::Content(Text::uncounted("Sure. ")),
            Event::Malformed {
                text: Text::uncounted("<tool_call>{broken"),
                why: MalformedReason::InvalidArguments,
            },
        ];
        let choice = message(0, &events);
        assert_eq!(
            choice.message.content.as_deref(),
            Some("Sure. <tool_call>{broken")
        );
        assert!(
            choice.finish_reason.is_none(),
            "no Finish event, so the message is not finished"
        );
    }

    #[test]
    fn calls_keep_the_order_they_started_in_and_their_own_fragments() {
        let events = [
            call(0, "get_weather"),
            call(1, "get_time"),
            arguments(1, r#"{"zone":"CET"}"#),
            arguments(0, r#"{"city":"Paris"}"#),
            finish(FinishReason::Stop, 2),
        ];
        let calls = message(0, &events).message.tool_calls.expect("two calls");
        let summary: Vec<(String, String)> = calls
            .into_iter()
            .map(|c| (c.function.name, c.function.arguments.unwrap_or_default()))
            .collect();
        assert_eq!(
            summary,
            [
                ("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string()),
                ("get_time".to_string(), r#"{"zone":"CET"}"#.to_string()),
            ]
        );
    }

    #[test]
    fn the_one_finish_rule_holds_for_the_whole_message_too() {
        let reason = |reason: FinishReason, tool_calls: u32| {
            message(0, &[call(0, "f"), finish(reason, tool_calls)]).finish_reason
        };
        assert_eq!(reason(FinishReason::Stop, 1).as_deref(), Some("tool_calls"));
        assert_eq!(reason(FinishReason::Length, 1).as_deref(), Some("length"));
        assert_eq!(reason(FinishReason::Abort, 1).as_deref(), Some("abort"));
        assert_eq!(
            reason(FinishReason::Other("content_filter".into()), 1).as_deref(),
            Some("content_filter")
        );
    }

    #[test]
    fn a_fragment_for_a_call_that_never_started_joins_the_content() {
        let events = [
            Event::Content(Text::uncounted("x")),
            arguments(7, r#"{"stray":true}"#),
        ];
        let choice = message(0, &events);
        assert_eq!(
            choice.message.content.as_deref(),
            Some(r#"x{"stray":true}"#)
        );
        assert!(choice.message.tool_calls.is_none());
    }

    #[test]
    fn a_whole_stream_renders_in_order_for_the_given_choice() {
        let events = [
            Event::ReasoningStart,
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "get_weather".into(),
                source: Text::default(),
            },
            Event::ToolCallArguments {
                index: 0,
                json: r#"{"city":"Paris"}"#.into(),
                source: Text::default(),
            },
            Event::ToolCallEnd {
                index: 0,
                source: Text::default(),
            },
            Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 1,
                reasoning_tokens: 1,
            },
        ];
        let choices = deltas(2, &events);
        let kinds: Vec<&str> = choices
            .iter()
            .map(|c| {
                match (
                    &c.delta.reasoning_content,
                    &c.delta.tool_calls,
                    &c.finish_reason,
                ) {
                    (Some(_), _, _) => "reasoning",
                    (_, Some(_), _) => "tool_call",
                    (_, _, Some(_)) => "finish",
                    _ => "content",
                }
            })
            .collect();
        assert_eq!(kinds, ["reasoning", "tool_call", "tool_call", "finish"]);
        assert!(choices.iter().all(|c| c.index == 2));
        assert_eq!(choices[3].finish_reason.as_deref(), Some("tool_calls"));
    }
}
