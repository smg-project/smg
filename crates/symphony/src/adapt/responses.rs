//! Responses: parser events as the output items of one response.
//!
//! [`output`] folds a whole event sequence into the response's `output` list and its status. It is
//! pure like [`chat::message`](super::chat::message): call ids ride on the events, the status is
//! decided from the `Finish` event alone, and nothing is remembered between calls. The driver wraps
//! the items in the response envelope (id, model, timestamps, usage, the request's echoed fields)
//! and sets `logprobs` where it has them, since the events carry none.
//!
//! The items are the ones SMG's gateway builds today from a finished chat message, in OpenAI's
//! order, so items replayed as the next turn's input rebuild the same turn:
//!
//! - one `reasoning` item when there was reasoning, its text as one `reasoning_text` part and an
//!   empty summary;
//! - one `message` when the content is not whitespace: `role: "assistant"`, one `output_text` part
//!   with every content and malformed text in order, and no annotations;
//! - one `function_call` per call in the order the calls started, its argument fragments joined;
//!   `output` is absent, since running the call belongs to the next turn.
//!
//! This is the same folding as `chat::message`, so the two adapters agree on what a turn contains.
//!
//! Ids: the reasoning item is `rs_<response id>` and the message `msg_<response id>`, the prefixes
//! the Responses API uses; a call's `id` and `call_id` are both the id from its event. The old
//! gateway used two schemes, `reasoning_` and `msg_` before the response id when it converted a
//! finished message and fresh `rs_`, `msg_` and `fc_` ids when it streamed; this is one.
//!
//! Status, from the `Finish` event: `stop` and `tool_calls` complete the response; `length` leaves
//! it `incomplete` with `max_output_tokens` as the reason; the engine's `failed` or `error` fails
//! it; `abort` cancels it, where the old path called an aborted response complete; any other
//! engine-specific reason completes it. A sequence without `Finish` is `in_progress`.
//!
//! Item statuses follow: `in_progress` while the response has not finished or when the engine
//! failed, `completed` otherwise, except that a call whose argument fragments never became a whole
//! JSON value is `incomplete`. Two of these differ from the old path, which called the reasoning
//! item `completed` whatever the response's state (here it follows the message) and marked a call
//! `incomplete` only after `length` (here the status describes the call, whatever stopped the
//! output). A call the model closed without arguments has empty arguments and is complete.
//!
//! Policy this adapter sets, where the events say more than the Responses API can:
//!
//! - `ReasoningStart`, `ReasoningEnd`, `ToolCallEnd` and `Dropped` shape nothing; the markers a
//!   format consumes never reach the client.
//! - `Malformed` text is content, as in `chat`: the model wrote it, and the client sees it rather
//!   than losing it.
//! - Argument fragments for a call that never started cannot occur under the `Event` contract;
//!   should one arrive, its text joins the content so that no byte is lost.
//!
//! What stays with the driver, since it depends on the request rather than the output: the `error`
//! payload of a failed response, encrypted reasoning content, and a function's namespace and the
//! `custom_tool_call` shape, both read from the request's tool list.

use openai_protocol::responses::{
    IncompleteDetails, IncompleteReason, ResponseContentPart, ResponseOutputItem,
    ResponseReasoningContent, ResponseStatus,
};
use serde::de::IgnoredAny;

use crate::event::{Event, FinishReason};

/// A whole output as the Responses API reports it: the items and the response status.
#[derive(Clone, Debug)]
pub struct Output {
    /// The output items in OpenAI's order: reasoning, message, then the calls as they started.
    pub items: Vec<ResponseOutputItem>,
    /// The response status decided from the `Finish` event.
    pub status: ResponseStatus,
    /// Why the response is incomplete, when it is.
    pub incomplete_details: Option<IncompleteDetails>,
}

/// One started call and what its events said.
struct Call {
    index: u32,
    id: String,
    name: String,
    arguments: String,
}

/// The output of the response `response_id`, folded from a whole event sequence.
pub fn output<'a>(response_id: &str, events: impl IntoIterator<Item = &'a Event>) -> Output {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<Call> = Vec::new();
    let mut finish = None;
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
            Event::Finish { reason, .. } => finish = Some(reason),
            Event::ReasoningStart
            | Event::ReasoningEnd
            | Event::ToolCallEnd { .. }
            | Event::Dropped { .. } => {}
        }
    }

    let (status, incomplete_details) = response_status(finish);
    let item_status = if matches!(status, ResponseStatus::InProgress | ResponseStatus::Failed) {
        "in_progress"
    } else {
        "completed"
    };

    let mut items = Vec::new();
    if !reasoning.is_empty() {
        items.push(ResponseOutputItem::new_reasoning(
            format!("rs_{response_id}"),
            Vec::new(),
            vec![ResponseReasoningContent::ReasoningText { text: reasoning }],
            Some(item_status.to_string()),
        ));
    }
    if !content.trim().is_empty() {
        items.push(ResponseOutputItem::Message {
            id: format!("msg_{response_id}"),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: content,
                annotations: Vec::new(),
                logprobs: None,
            }],
            status: item_status.to_string(),
            phase: None,
        });
    }
    items.extend(calls.into_iter().map(|call| {
        let status = if item_status == "completed" && !whole_arguments(&call.arguments) {
            "incomplete"
        } else {
            item_status
        };
        ResponseOutputItem::FunctionToolCall {
            id: Some(call.id.clone()),
            call_id: call.id,
            name: call.name,
            namespace: None,
            arguments: call.arguments,
            output: None,
            status: status.to_string(),
        }
    }));

    Output {
        items,
        status,
        incomplete_details,
    }
}

/// The response status for the engine's reason, and why it is incomplete when it is.
fn response_status(finish: Option<&FinishReason>) -> (ResponseStatus, Option<IncompleteDetails>) {
    match finish {
        None => (ResponseStatus::InProgress, None),
        Some(FinishReason::Stop | FinishReason::ToolCalls) => (ResponseStatus::Completed, None),
        Some(FinishReason::Length) => (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: IncompleteReason::MaxOutputTokens,
            }),
        ),
        Some(FinishReason::Abort) => (ResponseStatus::Cancelled, None),
        Some(FinishReason::Other(reason)) if reason == "failed" || reason == "error" => {
            (ResponseStatus::Failed, None)
        }
        Some(FinishReason::Other(_)) => (ResponseStatus::Completed, None),
    }
}

/// Whether the joined argument fragments are a whole JSON value. Empty arguments are a call the
/// model closed without any, so they are whole too.
fn whole_arguments(arguments: &str) -> bool {
    arguments.is_empty() || serde_json::from_str::<IgnoredAny>(arguments).is_ok()
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

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

    fn wire(items: &[ResponseOutputItem]) -> Value {
        serde_json::to_value(items).expect("serializable")
    }

    fn statuses(items: &[ResponseOutputItem]) -> Vec<String> {
        wire(items)
            .as_array()
            .expect("an array")
            .iter()
            .map(|item| item["status"].as_str().expect("a status").to_string())
            .collect()
    }

    #[test]
    fn a_whole_output_folds_into_the_gateway_items_in_openai_order() {
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
        let out = output("resp_1", &events);
        assert_eq!(
            wire(&out.items),
            json!([
                {
                    "type": "reasoning",
                    "id": "rs_resp_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan more"}],
                    "status": "completed",
                },
                {
                    "type": "message",
                    "id": "msg_resp_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check.", "annotations": []}],
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "call_0",
                    "call_id": "call_0",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}",
                    "status": "completed",
                },
            ])
        );
        assert_eq!(out.status, ResponseStatus::Completed);
        assert_eq!(out.incomplete_details, None);
    }

    #[test]
    fn whitespace_only_content_and_nothing_else_give_no_items() {
        let events = [
            Event::Content(Text::uncounted("  \n")),
            finish(FinishReason::Stop, 0),
        ];
        let out = output("resp_2", &events);
        assert!(out.items.is_empty(), "{:?}", out.items);
        assert_eq!(out.status, ResponseStatus::Completed);
    }

    #[test]
    fn malformed_text_and_a_stray_fragment_join_the_message() {
        let events = [
            Event::Content(Text::uncounted("Sure. ")),
            Event::Malformed {
                text: Text::uncounted("<tool_call>{broken"),
                why: MalformedReason::InvalidArguments,
            },
            arguments(7, r#"{"stray":true}"#),
        ];
        let out = output("resp_3", &events);
        assert_eq!(
            wire(&out.items),
            json!([{
                "type": "message",
                "id": "msg_resp_3",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Sure. <tool_call>{broken{\"stray\":true}",
                    "annotations": [],
                }],
                "status": "in_progress",
            }])
        );
        assert_eq!(
            out.status,
            ResponseStatus::InProgress,
            "no Finish event, so the response is still in progress"
        );
    }

    #[test]
    fn calls_keep_the_order_they_started_in_and_their_own_fragments() {
        let events = [
            call(0, "get_weather"),
            call(1, "get_time"),
            arguments(1, r#"{"zone":"CET"}"#),
            arguments(0, r#"{"city":"Paris"}"#),
            end(1),
            end(0),
            finish(FinishReason::Stop, 2),
        ];
        let items = wire(&output("resp_4", &events).items);
        let summary: Vec<(&str, &str, &str)> = items
            .as_array()
            .expect("an array")
            .iter()
            .map(|item| {
                (
                    item["call_id"].as_str().expect("a call id"),
                    item["name"].as_str().expect("a name"),
                    item["arguments"].as_str().expect("arguments"),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("call_0", "get_weather", r#"{"city":"Paris"}"#),
                ("call_1", "get_time", r#"{"zone":"CET"}"#),
            ]
        );
    }

    #[test]
    fn the_response_status_follows_the_finish_reason() {
        let status = |events: &[Event]| {
            let out = output("resp_5", events);
            (out.status, out.incomplete_details)
        };
        assert_eq!(status(&[]), (ResponseStatus::InProgress, None));
        assert_eq!(
            status(&[finish(FinishReason::Stop, 0)]),
            (ResponseStatus::Completed, None)
        );
        assert_eq!(
            status(&[finish(FinishReason::ToolCalls, 1)]),
            (ResponseStatus::Completed, None)
        );
        assert_eq!(
            status(&[finish(FinishReason::Length, 0)]),
            (
                ResponseStatus::Incomplete,
                Some(IncompleteDetails {
                    reason: IncompleteReason::MaxOutputTokens,
                })
            )
        );
        assert_eq!(
            status(&[finish(FinishReason::Abort, 0)]),
            (ResponseStatus::Cancelled, None)
        );
        assert_eq!(
            status(&[finish(FinishReason::Other("failed".into()), 0)]),
            (ResponseStatus::Failed, None)
        );
        assert_eq!(
            status(&[finish(FinishReason::Other("error".into()), 0)]),
            (ResponseStatus::Failed, None)
        );
        assert_eq!(
            status(&[finish(FinishReason::Other("content_filter".into()), 0)]),
            (ResponseStatus::Completed, None)
        );
    }

    #[test]
    fn a_call_cut_short_is_incomplete_and_a_whole_one_complete() {
        let cut_by_length = [
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            end(0),
            finish(FinishReason::Length, 1),
        ];
        assert_eq!(statuses(&output("r", &cut_by_length).items), ["incomplete"]);

        let whole_then_cut = [
            call(0, "get_weather"),
            arguments(0, r#"{"city":"Paris"}"#),
            end(0),
            Event::Content(Text::uncounted("and then")),
            finish(FinishReason::Length, 1),
        ];
        assert_eq!(
            statuses(&output("r", &whole_then_cut).items),
            ["completed", "completed"],
            "the call was whole; the output was cut after it"
        );

        let unfinished_at_stop = [
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            end(0),
            finish(FinishReason::Stop, 1),
        ];
        assert_eq!(
            statuses(&output("r", &unfinished_at_stop).items),
            ["incomplete"],
            "the status describes the call, whatever stopped the output"
        );

        let without_arguments = [call(0, "get_time"), end(0), finish(FinishReason::Stop, 1)];
        assert_eq!(
            statuses(&output("r", &without_arguments).items),
            ["completed"]
        );
    }

    #[test]
    fn items_stay_in_progress_until_the_response_finishes_and_when_the_engine_fails() {
        let unfinished = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::Content(Text::uncounted("text")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
        ];
        assert_eq!(
            statuses(&output("r", &unfinished).items),
            ["in_progress", "in_progress", "in_progress"]
        );

        let failed = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::Content(Text::uncounted("text")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"Paris"}"#),
            end(0),
            finish(FinishReason::Other("failed".into()), 1),
        ];
        let out = output("r", &failed);
        assert_eq!(
            statuses(&out.items),
            ["in_progress", "in_progress", "in_progress"]
        );
        assert_eq!(out.status, ResponseStatus::Failed);
    }

    #[test]
    fn events_without_a_responses_representation_shape_nothing() {
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
        let out = output("r", &silent);
        assert!(out.items.is_empty(), "{:?}", out.items);
        assert_eq!(out.status, ResponseStatus::Completed);
    }
}
