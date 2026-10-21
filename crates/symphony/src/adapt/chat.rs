//! Chat Completions: parser events as stream choices or as one message.
//!
//! [`delta`] maps one [`Event`] to at most one [`ChatStreamChoice`]; [`message`] folds a whole
//! event sequence into one [`ChatChoice`]. Both are pure: call ids ride on the events, the finish
//! reason is decided from the `Finish` event alone, and nothing is remembered between calls. The
//! driver wraps the choices in the response envelope (request id, model, timestamp, usage) and sets
//! `matched_stop`, and `logprobs` where it has them, since the events carry neither.
//!
//! Notes on what this module decides, and on what it leaves open: `docs/adapters.md`.

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
    delta_without_role(choice, event).map(with_role)
}

/// [`delta`] without the assistant role on the delta (see [`into_delta_without_role`]).
pub fn delta_without_role(choice: u32, event: &Event<'_>) -> Option<ChatStreamChoice> {
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
    into_delta_without_role(choice, event).map(with_role)
}

/// [`into_delta`] without the assistant role on the delta, for a stream whose opening chunk
/// carried the role and whose later deltas do not repeat it (the public API's shape).
pub fn into_delta_without_role(choice: u32, event: Event<'_>) -> Option<ChatStreamChoice> {
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
                ..bare()
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
                ..bare()
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
            ..bare()
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
                ..bare()
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

/// A delta with nothing in it yet and no role: the role goes on with
/// [`with_role`] where a caller wants it on every delta, and stays off where
/// the stream's opening chunk carried it (one allocation per delta saved,
/// measured 2026-10-21: 128 per 128-delta stream).
fn bare() -> ChatMessageDelta {
    ChatMessageDelta {
        role: None,
        content: None,
        tool_calls: None,
        reasoning_content: None,
    }
}

/// `choice` with the assistant role on its delta, as the deltas always had it;
/// the finishing choice keeps its empty delta.
fn with_role(mut choice: ChatStreamChoice) -> ChatStreamChoice {
    if choice.finish_reason.is_none() {
        choice.delta.role = Some("assistant".to_string());
    }
    choice
}

/// The byte ranges of the output that are the message's content, from the events' sources in
/// order (every byte of the output lands in one event), adjacent runs joined. Malformed text is
/// content, as [`message`] writes it; a logprobs filter keeps the tokens in these ranges. The
/// events must be the parser's own, whole; the one-call-only rule keeps the bytes of the calls it
/// drops as dropped text, so the ranges hold under it
/// ([`first_call_only`](crate::event::first_call_only)).
pub fn content_spans(events: &[Event]) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    for event in events {
        let (bytes, content) = match event {
            Event::Content(text) | Event::Malformed { text, .. } => (text.text.len(), true),
            Event::Reasoning(text) | Event::Dropped { text, .. } => (text.text.len(), false),
            Event::ToolCallStart { source, .. }
            | Event::ToolCallArguments { source, .. }
            | Event::ToolCallEnd { source, .. } => (source.text.len(), false),
            Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => (0, false),
        };
        if content && bytes > 0 {
            match spans.last_mut() {
                Some((_, end)) if *end == at => *end = at + bytes,
                _ => spans.push((at, at + bytes)),
            }
        }
        at += bytes;
    }
    spans
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::{
        event::{first_call_only, DropReason, MalformedReason, Text},
        formats::qwen3,
        ids::is_minted,
        CallSyntax, Declared, Engine, EngineFinish, Events, Input, Parser,
    };

    fn wire(choice: Option<ChatStreamChoice>) -> Value {
        serde_json::to_value(choice.expect("a choice")).expect("serializable")
    }

    #[test]
    fn content_spans_join_content_and_malformed_runs_and_skip_the_rest() {
        let events = [
            Event::Content(Text::uncounted("Sure. ")),
            Event::Malformed {
                text: Text::uncounted("{bad"),
                why: MalformedReason::InvalidArguments,
            },
            Event::Reasoning(Text::uncounted("think")),
            Event::ToolCallStart {
                index: 0,
                id: "call_0".to_string(),
                name: "f".to_string(),
                source: Text::uncounted("<call>"),
            },
            Event::Content(Text::uncounted("done")),
            Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 1,
                reasoning_tokens: 0,
            },
        ];
        assert_eq!(content_spans(&events), [(0, 10), (21, 25)]);
        assert!(content_spans(&[]).is_empty());
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
                why: MalformedReason::TextBetweenTags,
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

    /// The `_without_role` forms are the forms with the role, minus the role: for every event the
    /// same choice or none, the role on the delta exactly where the forms with it put one (every
    /// delta but the finishing choice's, which keeps its empty delta).
    #[test]
    fn a_delta_without_the_role_is_the_delta_minus_its_role() {
        let json = |choice: &ChatStreamChoice| serde_json::to_value(choice).expect("wire");
        for event in one_of_each() {
            let with = into_delta(3, event.clone());
            let without = into_delta_without_role(3, event.clone());
            assert_eq!(with.is_some(), without.is_some(), "{event:?}");
            if let (Some(with), Some(without)) = (&with, &without) {
                let expected_role = with.finish_reason.is_none().then_some("assistant");
                assert_eq!(with.delta.role.as_deref(), expected_role, "{event:?}");
                assert_eq!(without.delta.role, None, "{event:?}");
                let mut stripped = with.clone();
                stripped.delta.role = None;
                assert_eq!(json(&stripped), json(without), "{event:?}");
            }
            assert_eq!(
                delta_without_role(3, &event).map(|choice| json(&choice)),
                without.as_ref().map(json),
                "the borrowing form without the role: {event:?}"
            );
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

    /// The events of `output` read by the Qwen3 table through an engine of its own, as the gateway
    /// reads each request.
    fn parsed(output: &str) -> Vec<Event<'static>> {
        let mut engine = Engine::new(qwen3(CallSyntax::Json), Declared::default());
        let mut out = Events::new();
        engine
            .feed(
                Input::Delta {
                    token_ids: &[],
                    text: output,
                    spans: &[],
                },
                &mut out,
            )
            .expect("a delta");
        engine
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut out,
            )
            .expect("the end");
        out.into_owned().drain()
    }

    /// Two calls of Qwen3's syntax in one output.
    const TWO_CALLS: &str = concat!(
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n",
        "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"zone\": \"CET\"}}\n</tool_call>"
    );

    /// The content ranges under one call only, against the whole output's: the rule turns the
    /// calls it drops into dropped text, so the ranges after a dropped call are where the
    /// content is. Content after a dropped second call.
    #[test]
    fn content_spans_hold_after_a_call_the_one_call_only_rule_dropped() {
        let output = format!("Checking.\n{TWO_CALLS}\nDone.");
        let whole = parsed(&output);
        let mut one = whole.clone();
        first_call_only(&mut one);
        let spans = content_spans(&one);
        assert_eq!(spans, content_spans(&whole), "{one:?}");
        let kept: Vec<&str> = spans.iter().map(|&(from, to)| &output[from..to]).collect();
        assert_eq!(kept, ["Checking.\n", "\nDone."]);
        assert_eq!(one.len(), whole.len(), "one event per event: {one:?}");
    }

    /// Malformed text after a dropped second call (a third call the output cuts before its
    /// header: its opener is wrapper text the table drops, its body malformed text): the text is
    /// content, at its own offsets.
    #[test]
    fn content_spans_hold_for_malformed_text_after_a_call_the_one_call_only_rule_dropped() {
        let output = format!("{TWO_CALLS}\n<tool_call>\n{{\"name\": \"get_time\"");
        let whole = parsed(&output);
        let mut one = whole.clone();
        first_call_only(&mut one);
        let spans = content_spans(&one);
        assert_eq!(spans, content_spans(&whole), "{one:?}");
        let kept: Vec<&str> = spans.iter().map(|&(from, to)| &output[from..to]).collect();
        assert_eq!(kept, ["\n{\"name\": \"get_time\""]);
    }

    #[test]
    fn two_turns_of_two_calls_carry_four_distinct_ids_the_deltas_and_the_message_agree_on() {
        // A client pairs its tool results with its calls by id across the conversation: no id
        // of one turn may come back in the next, and the stream must name the call as the
        // message does.
        let mut seen: Vec<String> = Vec::new();
        for turn in 0..2 {
            let events = parsed(TWO_CALLS);
            let in_message: Vec<String> = message(0, &events)
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| call.id)
                .collect();
            let in_deltas: Vec<String> = deltas(0, &events)
                .into_iter()
                .filter_map(|choice| choice.delta.tool_calls)
                .flatten()
                .filter_map(|call| call.id)
                .collect();
            assert_eq!(in_message.len(), 2, "turn {turn}: {in_message:?}");
            assert_eq!(in_deltas, in_message, "turn {turn}");
            for id in &in_message {
                assert!(is_minted(id), "turn {turn}: {id}");
            }
            seen.extend(in_message);
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 4, "{seen:?}");
    }
}
