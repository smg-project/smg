//! Responses: parser events as the output items of one response.
//!
//! [`output`] folds a whole event sequence into the response's `output` list and its status. It is
//! pure like [`chat::message`](super::chat::message): call ids ride on the events, the status is
//! decided from the `Finish` event alone, and nothing is remembered between calls. The driver wraps
//! the items in the response envelope (id, model, timestamps, usage, the request's echoed fields)
//! and sets `logprobs` where it has them, since the events carry none.
//!
//! [`Stream`] is the streaming half: it opens, continues and completes output items as the events
//! arrive and produces the API's stream events, typed here as [`StreamEvent`] because the gateway
//! builds them as untyped JSON and `openai-protocol` has no type for them. Their items and parts
//! are the same `ResponseOutputItem` and `ResponseContentPart` the final response carries, so the
//! stream and the fold share one shape; they serialise as the API's `type`-tagged objects without
//! `sequence_number`, which the driver adds as it numbers every event of the response, its own
//! `response.created`, `response.in_progress` and the terminal event included. A `reasoning` item
//! opens at the first reasoning text and completes when the region ends; the `message` opens at the
//! first content with substance and stays open to the end, so content after a call continues it; a
//! `function_call` item opens at `ToolCallStart`, takes its fragments as
//! `function_call_arguments.delta` and completes at `ToolCallEnd`, so a client can act on a call
//! while the model goes on. `Finish` completes what is open and hands back the response's
//! [`Output`], items in the order they opened, for the driver's terminal event. When the reasoning
//! comes first, in one region, and the first content with substance comes before the first call,
//! those items are the fold's, statuses included. Otherwise the stream keeps the order the items
//! opened in, where the fold gathers each kind into one item in OpenAI's order: a call before any
//! content comes first, reasoning after content follows the message, and a second reasoning region
//! is a second item.
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
//! finished message and fresh `rs_`, `msg_` and `fc_` ids when it streamed; this is one. A second
//! reasoning item, which only a model that thinks again after answering produces, carries its
//! output index after the response id (`rs_<response id>_<output index>`), so ids stay unique.
//!
//! Status, from the `Finish` event: `stop` and `tool_calls` complete the response; `length` leaves
//! it `incomplete` with `max_output_tokens` as the reason; the engine's `failed` or `error` fails
//! it; `abort` cancels it, where the old path called an aborted response complete; any other
//! engine-specific reason completes it. A sequence without `Finish` is `in_progress`.
//!
//! Item statuses follow. While the response has not finished, and when the engine failed, an item
//! is `in_progress` unless its own events ended it: a reasoning item whose region closed is
//! `completed`, a call that ended is `completed`, and the message, which ends only with the
//! response, is `in_progress`. Once the response finished otherwise, every item is `completed`. In
//! either case a call whose argument fragments never became a whole JSON value is `incomplete`
//! rather than `completed`. Three of these differ from the old path, which called the reasoning
//! item `completed` whatever the response's state, left every item of a failed response
//! `in_progress`, what had finished included, and marked a call `incomplete` only after `length`;
//! here each status describes its item. A reasoning region ends at `ReasoningEnd`, or when content
//! with substance or a call comes after it, in the fold as in the stream. A call with empty
//! arguments is complete: the formats start a call only once its arguments value has begun or its
//! object has closed, so a call without arguments is one the model closed.
//!
//! Policy this adapter sets, where the events say more than the Responses API can:
//!
//! - `ReasoningStart` and `Dropped` shape nothing, and `ReasoningEnd` and `ToolCallEnd` only end
//!   their items (the statuses above, and the stream's `done` events): the markers a format
//!   consumes never reach the client.
//! - `Malformed` text is content, as in `chat`: the model wrote it, and the client sees it rather
//!   than losing it.
//! - Argument fragments for a call that never started cannot occur under the `Event` contract;
//!   should one arrive, its text joins the content so that no byte is lost. In the stream the same
//!   goes for a fragment of a call whose item is already complete.
//! - In the stream, whitespace-only content opens no message item until content with substance
//!   joins it, and leads that content when it does; held whitespace with nothing after it is let go
//!   at the end, as the fold leaves whitespace-only content out. The old emitter opened a message
//!   item for any non-empty content.
//! - In the stream, a call's item completes when the call ends and the reasoning item when its
//!   region ends; the old emitter completed every call at the end of the response and the
//!   reasoning item when the answer began. The items and parts are the typed ones, so a message's
//!   `output_item.added` carries `status: "in_progress"` and a text part its empty `annotations`,
//!   as the API's own events do; the old emitter's hand-written JSON left them out.
//!
//! What stays with the driver, since it depends on the request rather than the output: the `error`
//! payload of a failed response, encrypted reasoning content, and a function's namespace and the
//! `custom_tool_call` shape, both read from the request's tool list.

use std::mem;

use openai_protocol::responses::{
    IncompleteDetails, IncompleteReason, ResponseContentPart, ResponseOutputItem,
    ResponseReasoningContent, ResponseStatus,
};
use serde::{de::IgnoredAny, Serialize};

use crate::event::{Event, FinishReason};

/// A whole output as the Responses API reports it: the items and the response status.
#[derive(Clone, Debug)]
pub struct Output {
    /// The output items: from [`output`], in OpenAI's order (reasoning, message, then the calls as
    /// they started); from [`Stream`], in the order the items opened.
    pub items: Vec<ResponseOutputItem>,
    /// The response status decided from the `Finish` event.
    pub status: ResponseStatus,
    /// Why the response is incomplete, when it is.
    pub incomplete_details: Option<IncompleteDetails>,
}

/// One started call and what its events said.
#[derive(Clone, Debug)]
struct Call {
    index: u32,
    id: String,
    name: String,
    arguments: String,
    /// Whether `ToolCallEnd` came for it.
    ended: bool,
}

/// The output of the response `response_id`, folded from a whole event sequence.
pub fn output<'a>(response_id: &str, events: impl IntoIterator<Item = &'a Event>) -> Output {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut reasoning_ended = false;
    let mut calls: Vec<Call> = Vec::new();
    let mut finish = None;
    for event in events {
        match event {
            Event::Content(text) | Event::Malformed { text, .. } => {
                content.push_str(&text.text);
                if !text.text.trim().is_empty() {
                    reasoning_ended = true;
                }
            }
            Event::Reasoning(text) => {
                reasoning.push_str(&text.text);
                reasoning_ended = false;
            }
            Event::ReasoningEnd => reasoning_ended = true,
            Event::ToolCallStart {
                index, id, name, ..
            } => {
                reasoning_ended = true;
                calls.push(Call {
                    index: *index,
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                    ended: false,
                });
            }
            Event::ToolCallArguments { index, json, .. } => {
                match calls.iter_mut().find(|call| call.index == *index) {
                    Some(call) => call.arguments.push_str(json),
                    None => {
                        content.push_str(json);
                        if !json.trim().is_empty() {
                            reasoning_ended = true;
                        }
                    }
                }
            }
            Event::ToolCallEnd { index, .. } => {
                if let Some(call) = calls.iter_mut().find(|call| call.index == *index) {
                    call.ended = true;
                }
            }
            Event::Finish { reason, .. } => finish = Some(reason),
            Event::ReasoningStart | Event::Dropped { .. } => {}
        }
    }

    let (status, incomplete_details) = response_status(finish);

    let mut items = Vec::new();
    if !reasoning.is_empty() {
        items.push(reasoning_item(
            format!("rs_{response_id}"),
            reasoning,
            item_status(&status, reasoning_ended),
        ));
    }
    if !content.trim().is_empty() {
        items.push(message_item(
            format!("msg_{response_id}"),
            content,
            item_status(&status, false),
        ));
    }
    items.extend(calls.into_iter().map(|call| {
        let status = item_status(&status, call.ended);
        call_item(call, status)
    }));

    Output {
        items,
        status,
        incomplete_details,
    }
}

/// One event of a Responses stream, serialised as the API's `type`-tagged object without
/// `sequence_number`, which the driver adds when it numbers the response's events.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// An item opened at `output_index`, in the state it is in at that point.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item as added: a reasoning item with no content, a message with no parts, a call
        /// with no arguments.
        item: ResponseOutputItem,
    },
    /// The item at `output_index` is complete.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        /// The item's position in the response's output.
        output_index: u32,
        /// The whole item.
        item: ResponseOutputItem,
    },
    /// A content part opened inside a reasoning item or a message.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The part's position in the item's content.
        content_index: u32,
        /// The part as added, with no text yet.
        part: Part,
    },
    /// A content part is complete.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The part's position in the item's content.
        content_index: u32,
        /// The whole part.
        part: Part,
    },
    /// More text for the message's text part.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The part's position in the item's content.
        content_index: u32,
        /// The new text.
        delta: String,
    },
    /// The message's text part is complete.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The part's position in the item's content.
        content_index: u32,
        /// The whole text.
        text: String,
    },
    /// More text for the reasoning item's text part.
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The part's position in the item's content.
        content_index: u32,
        /// The new text.
        delta: String,
    },
    /// The reasoning item's text part is complete.
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The part's position in the item's content.
        content_index: u32,
        /// The whole text.
        text: String,
    },
    /// More of a call's arguments, the model's own bytes.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The new argument bytes.
        delta: String,
    },
    /// A call's arguments are complete.
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        /// The item's position in the response's output.
        output_index: u32,
        /// The item's id.
        item_id: String,
        /// The whole arguments.
        arguments: String,
    },
}

/// A content part as a stream event carries it: a message's text part or a reasoning text.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Part {
    /// A message's `output_text` part.
    Text(ResponseContentPart),
    /// A reasoning item's `reasoning_text` part.
    Reasoning(ResponseReasoningContent),
}

/// The streaming half: parser events as Responses stream events, items opened and completed as the
/// events arrive.
///
/// One per response, like the parser that feeds it. Feed every event in order. `Finish` completes
/// what is open and returns the response's [`Output`] for the driver's terminal event, the fold's
/// when the items opened in OpenAI's order (the module doc says when); [`Stream::close`] does the
/// same for a stream that ends without `Finish`.
#[derive(Clone, Debug)]
pub struct Stream {
    /// The response id the item ids are built from.
    response_id: String,
    /// Every item opened so far, in output order.
    records: Vec<Record>,
    /// The open reasoning item, as a position in `records`.
    reasoning: Option<usize>,
    /// The message item, as a position in `records`; open from its first content to the end.
    message: Option<usize>,
    /// Whitespace-only content that arrived before any content with substance.
    held: String,
    /// How many reasoning items have opened, for their ids.
    reasoning_items: u32,
}

/// One item of the stream and what its events said.
#[derive(Clone, Debug)]
enum Record {
    Reasoning {
        index: u32,
        id: String,
        text: String,
        /// Whether the region is still open.
        open: bool,
    },
    Message {
        index: u32,
        id: String,
        text: String,
    },
    Call {
        index: u32,
        call: Call,
    },
}

impl Stream {
    /// A stream for the response `response_id`, with no item open; the first item gets index zero.
    pub fn new(response_id: &str) -> Self {
        Self {
            response_id: response_id.to_string(),
            records: Vec::new(),
            reasoning: None,
            message: None,
            held: String::new(),
            reasoning_items: 0,
        }
    }

    /// Appends the stream events `event` produces to `out`, in order. Returns the response's
    /// [`Output`] when `event` is `Finish`, after completing every open item.
    pub fn feed(&mut self, event: &Event, out: &mut Vec<StreamEvent>) -> Option<Output> {
        match event {
            Event::Content(text) | Event::Malformed { text, .. } => self.content(&text.text, out),
            Event::Reasoning(text) => self.reasoning(&text.text, out),
            Event::ReasoningEnd => self.close_reasoning("completed", out),
            Event::ToolCallStart {
                index, id, name, ..
            } => self.start_call(*index, id, name, out),
            Event::ToolCallArguments { index, json, .. } => self.arguments(*index, json, out),
            Event::ToolCallEnd { index, .. } => self.end_call(*index, out),
            Event::Finish { reason, .. } => return Some(self.end(Some(reason), out)),
            Event::ReasoningStart | Event::Dropped { .. } => {}
        }
        None
    }

    /// Completes every open item and returns the [`Output`] for a stream that ends without
    /// `Finish`: the response and its items still in progress.
    pub fn close(&mut self, out: &mut Vec<StreamEvent>) -> Output {
        self.end(None, out)
    }

    /// The next item's index.
    fn next_index(&self) -> u32 {
        u32::try_from(self.records.len()).unwrap_or(u32::MAX)
    }

    /// Reasoning text into the open reasoning item, opened first when none is.
    fn reasoning(&mut self, text: &str, out: &mut Vec<StreamEvent>) {
        if text.is_empty() {
            return;
        }
        let position = match self.reasoning {
            Some(position) => position,
            None => self.open_reasoning(out),
        };
        if let Record::Reasoning {
            index,
            id,
            text: whole,
            ..
        } = &mut self.records[position]
        {
            whole.push_str(text);
            out.push(StreamEvent::ReasoningTextDelta {
                output_index: *index,
                item_id: id.clone(),
                content_index: 0,
                delta: text.to_string(),
            });
        }
    }

    /// Opens a reasoning item and its text part at the next index, and returns its position.
    fn open_reasoning(&mut self, out: &mut Vec<StreamEvent>) -> usize {
        let index = self.next_index();
        let id = if self.reasoning_items == 0 {
            format!("rs_{}", self.response_id)
        } else {
            format!("rs_{}_{index}", self.response_id)
        };
        self.reasoning_items += 1;
        out.push(StreamEvent::OutputItemAdded {
            output_index: index,
            item: ResponseOutputItem::new_reasoning(
                id.clone(),
                Vec::new(),
                Vec::new(),
                Some("in_progress".to_string()),
            ),
        });
        out.push(StreamEvent::ContentPartAdded {
            output_index: index,
            item_id: id.clone(),
            content_index: 0,
            part: Part::Reasoning(ResponseReasoningContent::ReasoningText {
                text: String::new(),
            }),
        });
        self.records.push(Record::Reasoning {
            index,
            id,
            text: String::new(),
            open: true,
        });
        let position = self.records.len() - 1;
        self.reasoning = Some(position);
        position
    }

    /// Completes the open reasoning item, if any, with `status`: `completed` when its region ended,
    /// the response's item status when the response ended first.
    fn close_reasoning(&mut self, status: &str, out: &mut Vec<StreamEvent>) {
        let Some(position) = self.reasoning.take() else {
            return;
        };
        if let Record::Reasoning {
            index,
            id,
            text,
            open,
        } = &mut self.records[position]
        {
            *open = false;
            out.push(StreamEvent::ReasoningTextDone {
                output_index: *index,
                item_id: id.clone(),
                content_index: 0,
                text: text.clone(),
            });
            out.push(StreamEvent::ContentPartDone {
                output_index: *index,
                item_id: id.clone(),
                content_index: 0,
                part: Part::Reasoning(ResponseReasoningContent::ReasoningText {
                    text: text.clone(),
                }),
            });
            out.push(StreamEvent::OutputItemDone {
                output_index: *index,
                item: reasoning_item(id.clone(), text.clone(), status),
            });
        }
    }

    /// Content into the message item, opened at the first content with substance; whitespace
    /// before that is held and leads it. Content with substance ends an open reasoning region.
    fn content(&mut self, text: &str, out: &mut Vec<StreamEvent>) {
        if text.is_empty() {
            return;
        }
        if !text.trim().is_empty() {
            self.close_reasoning("completed", out);
        }
        let position = match self.message {
            Some(position) => position,
            None if text.trim().is_empty() => {
                self.held.push_str(text);
                return;
            }
            None => self.open_message(out),
        };
        let mut delta = mem::take(&mut self.held);
        delta.push_str(text);
        if let Record::Message {
            index,
            id,
            text: whole,
        } = &mut self.records[position]
        {
            whole.push_str(&delta);
            out.push(StreamEvent::OutputTextDelta {
                output_index: *index,
                item_id: id.clone(),
                content_index: 0,
                delta,
            });
        }
    }

    /// Opens the message and its text part at the next index, and returns its position.
    fn open_message(&mut self, out: &mut Vec<StreamEvent>) -> usize {
        let index = self.next_index();
        let id = format!("msg_{}", self.response_id);
        out.push(StreamEvent::OutputItemAdded {
            output_index: index,
            item: ResponseOutputItem::Message {
                id: id.clone(),
                role: "assistant".to_string(),
                content: Vec::new(),
                status: "in_progress".to_string(),
                phase: None,
            },
        });
        out.push(StreamEvent::ContentPartAdded {
            output_index: index,
            item_id: id.clone(),
            content_index: 0,
            part: Part::Text(text_part(String::new())),
        });
        self.records.push(Record::Message {
            index,
            id,
            text: String::new(),
        });
        let position = self.records.len() - 1;
        self.message = Some(position);
        position
    }

    /// Opens a call's item.
    fn start_call(&mut self, call: u32, id: &str, name: &str, out: &mut Vec<StreamEvent>) {
        self.close_reasoning("completed", out);
        let index = self.next_index();
        let call = Call {
            index: call,
            id: id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
            ended: false,
        };
        out.push(StreamEvent::OutputItemAdded {
            output_index: index,
            item: call_item(call.clone(), "in_progress"),
        });
        self.records.push(Record::Call { index, call });
    }

    /// The position of the open item of call `call`, if it has one.
    fn open_call(&self, call: u32) -> Option<usize> {
        self.records.iter().position(|record| {
            matches!(record, Record::Call { call: started, .. } if started.index == call && !started.ended)
        })
    }

    /// A fragment into its call's open item; content when the call has no open item.
    fn arguments(&mut self, call: u32, json: &str, out: &mut Vec<StreamEvent>) {
        if json.is_empty() {
            return;
        }
        let Some(position) = self.open_call(call) else {
            self.content(json, out);
            return;
        };
        if let Record::Call { index, call, .. } = &mut self.records[position] {
            call.arguments.push_str(json);
            out.push(StreamEvent::FunctionCallArgumentsDelta {
                output_index: *index,
                item_id: call.id.clone(),
                delta: json.to_string(),
            });
        }
    }

    /// Completes a call's item, if it is open.
    fn end_call(&mut self, call: u32, out: &mut Vec<StreamEvent>) {
        if let Some(position) = self.open_call(call) {
            self.complete_call(position, "completed", out);
        }
    }

    /// Completes the call item at `position` with `status`: arguments done, then the item.
    fn complete_call(&mut self, position: usize, status: &str, out: &mut Vec<StreamEvent>) {
        if let Record::Call { index, call } = &mut self.records[position] {
            call.ended = true;
            out.push(StreamEvent::FunctionCallArgumentsDone {
                output_index: *index,
                item_id: call.id.clone(),
                arguments: call.arguments.clone(),
            });
            out.push(StreamEvent::OutputItemDone {
                output_index: *index,
                item: call_item(call.clone(), status),
            });
        }
    }

    /// Completes the message, if it opened, with `status`: text done, part done, then the item.
    fn complete_message(&mut self, status: &str, out: &mut Vec<StreamEvent>) {
        let Some(position) = self.message.take() else {
            return;
        };
        if let Record::Message { index, id, text } = &self.records[position] {
            out.push(StreamEvent::OutputTextDone {
                output_index: *index,
                item_id: id.clone(),
                content_index: 0,
                text: text.clone(),
            });
            out.push(StreamEvent::ContentPartDone {
                output_index: *index,
                item_id: id.clone(),
                content_index: 0,
                part: Part::Text(text_part(text.clone())),
            });
            out.push(StreamEvent::OutputItemDone {
                output_index: *index,
                item: message_item(id.clone(), text.clone(), status),
            });
        }
    }

    /// Completes every open item under the engine's reason and returns the response's output,
    /// each item with the status it reached: what the end closes is `in_progress` when the response
    /// did not finish or the engine failed, what ended on its own keeps its status.
    fn end(&mut self, finish: Option<&FinishReason>, out: &mut Vec<StreamEvent>) -> Output {
        let (status, incomplete_details) = response_status(finish);
        let open_status = item_status(&status, false);
        let items = self
            .records
            .iter()
            .map(|record| match record {
                Record::Reasoning { id, text, open, .. } => {
                    reasoning_item(id.clone(), text.clone(), item_status(&status, !open))
                }
                Record::Message { id, text, .. } => {
                    message_item(id.clone(), text.clone(), open_status)
                }
                Record::Call { call, .. } => {
                    let status = item_status(&status, call.ended);
                    call_item(call.clone(), status)
                }
            })
            .collect();
        self.close_reasoning(open_status, out);
        for position in 0..self.records.len() {
            if matches!(&self.records[position], Record::Call { call, .. } if !call.ended) {
                self.complete_call(position, open_status, out);
            }
        }
        self.complete_message(open_status, out);
        self.held.clear();
        Output {
            items,
            status,
            incomplete_details,
        }
    }
}

/// The status of an item under the response's status: while the response has not finished, and
/// when the engine failed, `in_progress` unless the item's own events ended it; `completed`
/// otherwise.
fn item_status(status: &ResponseStatus, ended: bool) -> &'static str {
    if !ended && matches!(status, ResponseStatus::InProgress | ResponseStatus::Failed) {
        "in_progress"
    } else {
        "completed"
    }
}

/// A reasoning item with `text` as its one `reasoning_text` part and no summary.
fn reasoning_item(id: String, text: String, status: &str) -> ResponseOutputItem {
    ResponseOutputItem::new_reasoning(
        id,
        Vec::new(),
        vec![ResponseReasoningContent::ReasoningText { text }],
        Some(status.to_string()),
    )
}

/// The assistant message with `text` as its one `output_text` part.
fn message_item(id: String, text: String, status: &str) -> ResponseOutputItem {
    ResponseOutputItem::Message {
        id,
        role: "assistant".to_string(),
        content: vec![text_part(text)],
        status: status.to_string(),
        phase: None,
    }
}

/// An `output_text` part with no annotations; `logprobs` are the driver's.
fn text_part(text: String) -> ResponseContentPart {
    ResponseContentPart::OutputText {
        text,
        annotations: Vec::new(),
        logprobs: None,
    }
}

/// A call's `function_call` item: `id` and `call_id` from the event, `output` absent, and
/// `incomplete` in place of `completed` when the arguments never became a whole JSON value.
fn call_item(call: Call, status: &str) -> ResponseOutputItem {
    let status = if status == "completed" && !whole_arguments(&call.arguments) {
        "incomplete"
    } else {
        status
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
    fn items_cut_short_stay_in_progress_and_items_that_ended_keep_their_status() {
        let unfinished = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::Content(Text::uncounted("text")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
        ];
        assert_eq!(
            statuses(&output("r", &unfinished).items),
            ["completed", "in_progress", "in_progress"],
            "the reasoning region ended when the content came; the message and the call did not end"
        );

        let failed_mid_reasoning = [
            Event::Reasoning(Text::uncounted("plan")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"Paris"}"#),
            end(0),
            Event::Reasoning(Text::uncounted("more")),
            finish(FinishReason::Other("failed".into()), 1),
        ];
        let out = output("r", &failed_mid_reasoning);
        assert_eq!(
            statuses(&out.items),
            ["in_progress", "completed"],
            "the call ended before the engine failed; the reasoning, resumed after it, did not"
        );
        assert_eq!(out.status, ResponseStatus::Failed);

        let failed_after_reasoning = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("text")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"#),
            finish(FinishReason::Other("error".into()), 1),
        ];
        assert_eq!(
            statuses(&output("r", &failed_after_reasoning).items),
            ["completed", "in_progress", "in_progress"],
            "the reasoning region closed; the message and the unended call were cut short"
        );
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

    /// Every event through one machine; the output returned by `Finish`, if any.
    fn stream(response_id: &str, events: &[Event]) -> (Vec<StreamEvent>, Option<Output>) {
        let mut machine = Stream::new(response_id);
        let mut out = Vec::new();
        let mut output = None;
        for event in events {
            if let Some(done) = machine.feed(event, &mut out) {
                output = Some(done);
            }
        }
        (out, output)
    }

    fn stream_wire(events: &[StreamEvent]) -> Value {
        serde_json::to_value(events).expect("serializable")
    }

    fn output_wire(output: &Output) -> Value {
        json!({
            "items": wire(&output.items),
            "status": output.status,
            "incomplete_details": output.incomplete_details,
        })
    }

    fn types(events: &[StreamEvent]) -> Vec<String> {
        stream_wire(events)
            .as_array()
            .expect("an array")
            .iter()
            .map(|event| event["type"].as_str().expect("a type").to_string())
            .collect()
    }

    #[test]
    fn a_whole_output_streams_as_the_api_events_in_order() {
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
        let (out, done) = stream("resp_1", &events);
        let reasoning = |text: &str| json!({"type": "reasoning_text", "text": text});
        let text = |text: &str| json!({"type": "output_text", "text": text, "annotations": []});
        assert_eq!(
            stream_wire(&out),
            json!([
                {"type": "response.output_item.added", "output_index": 0,
                 "item": {"type": "reasoning", "id": "rs_resp_1", "summary": [], "content": [], "status": "in_progress"}},
                {"type": "response.content_part.added", "output_index": 0, "item_id": "rs_resp_1",
                 "content_index": 0, "part": reasoning("")},
                {"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_resp_1",
                 "content_index": 0, "delta": "plan "},
                {"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_resp_1",
                 "content_index": 0, "delta": "more"},
                {"type": "response.reasoning_text.done", "output_index": 0, "item_id": "rs_resp_1",
                 "content_index": 0, "text": "plan more"},
                {"type": "response.content_part.done", "output_index": 0, "item_id": "rs_resp_1",
                 "content_index": 0, "part": reasoning("plan more")},
                {"type": "response.output_item.done", "output_index": 0,
                 "item": {"type": "reasoning", "id": "rs_resp_1", "summary": [],
                          "content": [reasoning("plan more")], "status": "completed"}},
                {"type": "response.output_item.added", "output_index": 1,
                 "item": {"type": "message", "id": "msg_resp_1", "role": "assistant", "content": [],
                          "status": "in_progress"}},
                {"type": "response.content_part.added", "output_index": 1, "item_id": "msg_resp_1",
                 "content_index": 0, "part": text("")},
                {"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_resp_1",
                 "content_index": 0, "delta": "Let me "},
                {"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_resp_1",
                 "content_index": 0, "delta": "check."},
                {"type": "response.output_item.added", "output_index": 2,
                 "item": {"type": "function_call", "id": "call_0", "call_id": "call_0",
                          "name": "get_weather", "arguments": "", "status": "in_progress"}},
                {"type": "response.function_call_arguments.delta", "output_index": 2,
                 "item_id": "call_0", "delta": "{\"city\":"},
                {"type": "response.function_call_arguments.delta", "output_index": 2,
                 "item_id": "call_0", "delta": "\"Paris\"}"},
                {"type": "response.function_call_arguments.done", "output_index": 2,
                 "item_id": "call_0", "arguments": "{\"city\":\"Paris\"}"},
                {"type": "response.output_item.done", "output_index": 2,
                 "item": {"type": "function_call", "id": "call_0", "call_id": "call_0",
                          "name": "get_weather", "arguments": "{\"city\":\"Paris\"}",
                          "status": "completed"}},
                {"type": "response.output_text.done", "output_index": 1, "item_id": "msg_resp_1",
                 "content_index": 0, "text": "Let me check."},
                {"type": "response.content_part.done", "output_index": 1, "item_id": "msg_resp_1",
                 "content_index": 0, "part": text("Let me check.")},
                {"type": "response.output_item.done", "output_index": 1,
                 "item": {"type": "message", "id": "msg_resp_1", "role": "assistant",
                          "content": [text("Let me check.")], "status": "completed"}},
            ])
        );
        let done = done.expect("Finish returns the output");
        assert_eq!(output_wire(&done), output_wire(&output("resp_1", &events)));
    }

    #[test]
    fn content_after_a_call_continues_the_message_and_a_call_before_text_comes_first() {
        let text_first = [
            Event::Content(Text::uncounted("a")),
            call(0, "f"),
            arguments(0, "{}"),
            end(0),
            Event::Content(Text::uncounted("b")),
            finish(FinishReason::Stop, 1),
        ];
        let (out, done) = stream("r", &text_first);
        let deltas: Vec<(u64, String)> = stream_wire(&out)
            .as_array()
            .expect("an array")
            .iter()
            .filter(|event| event["type"] == "response.output_text.delta")
            .map(|event| {
                (
                    event["output_index"].as_u64().expect("an index"),
                    event["delta"].as_str().expect("text").to_string(),
                )
            })
            .collect();
        assert_eq!(deltas, [(0, "a".to_string()), (0, "b".to_string())]);
        let done = done.expect("output");
        assert_eq!(output_wire(&done), output_wire(&output("r", &text_first)));

        let call_first = [
            call(0, "f"),
            end(0),
            Event::Content(Text::uncounted("b")),
            finish(FinishReason::Stop, 1),
        ];
        let kinds = |output: &Output| -> Vec<String> {
            wire(&output.items)
                .as_array()
                .expect("an array")
                .iter()
                .map(|item| item["type"].as_str().expect("a type").to_string())
                .collect()
        };
        let (_, done) = stream("r", &call_first);
        assert_eq!(kinds(&done.expect("output")), ["function_call", "message"]);
        assert_eq!(
            kinds(&output("r", &call_first)),
            ["message", "function_call"],
            "the fold puts the message first; the stream keeps the order written"
        );
    }

    #[test]
    fn whitespace_only_content_opens_no_message_until_substance_joins_it() {
        let separators_only = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("\n\n")),
            call(0, "f"),
            end(0),
            Event::Content(Text::uncounted("\n")),
            finish(FinishReason::Stop, 1),
        ];
        let (out, done) = stream("r", &separators_only);
        assert!(
            types(&out).iter().all(|kind| !kind.contains("output_text")),
            "{:?}",
            types(&out)
        );
        assert_eq!(
            output_wire(&done.expect("output")),
            output_wire(&output("r", &separators_only))
        );

        let led = [
            Event::Content(Text::uncounted("\n\n")),
            Event::Content(Text::uncounted("Hello")),
        ];
        let (out, _) = stream("r", &led);
        assert_eq!(
            stream_wire(&out)[2],
            json!({"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_r",
                   "content_index": 0, "delta": "\n\nHello"})
        );
    }

    #[test]
    fn a_fragment_with_no_open_call_is_content() {
        let (out, _) = stream("r", &[arguments(7, r#"{"stray":true}"#)]);
        assert_eq!(
            types(&out),
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta"
            ]
        );
        assert_eq!(stream_wire(&out)[2]["delta"], json!("{\"stray\":true}"));

        let (out, _) = stream("r", &[call(0, "f"), end(0), arguments(0, "late")]);
        let last = stream_wire(&out);
        let last = last
            .as_array()
            .expect("an array")
            .last()
            .expect("an event")
            .clone();
        assert_eq!(last["type"], "response.output_text.delta");
        assert_eq!(last["delta"], "late");
    }

    #[test]
    fn finish_completes_open_items_and_close_leaves_them_in_progress() {
        let cut = [
            call(0, "f"),
            arguments(0, r#"{"a":"#),
            finish(FinishReason::Length, 1),
        ];
        let (out, done) = stream("r", &cut);
        assert_eq!(
            types(&out)[1..],
            [
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
            ]
        );
        assert_eq!(stream_wire(&out)[3]["item"]["status"], "incomplete");
        let done = done.expect("output");
        assert_eq!(done.status, ResponseStatus::Incomplete);
        assert_eq!(statuses(&done.items), ["incomplete"]);

        let mut machine = Stream::new("r");
        let mut out = Vec::new();
        machine.feed(&Event::Content(Text::uncounted("x")), &mut out);
        let closed = machine.close(&mut out);
        assert_eq!(
            types(&out)[3..],
            [
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
            ]
        );
        assert_eq!(stream_wire(&out)[5]["item"]["status"], "in_progress");
        assert_eq!(closed.status, ResponseStatus::InProgress);
        assert_eq!(statuses(&closed.items), ["in_progress"]);

        let failed = [
            Event::Content(Text::uncounted("x")),
            finish(FinishReason::Other("failed".into()), 0),
        ];
        let (out, done) = stream("r", &failed);
        assert_eq!(stream_wire(&out)[5]["item"]["status"], "in_progress");
        assert_eq!(done.expect("output").status, ResponseStatus::Failed);

        let failed_after_a_call = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::ReasoningEnd,
            call(0, "f"),
            arguments(0, "{}"),
            end(0),
            Event::Content(Text::uncounted("x")),
            finish(FinishReason::Other("failed".into()), 1),
        ];
        let (out, done) = stream("r", &failed_after_a_call);
        let done_statuses: Vec<String> = stream_wire(&out)
            .as_array()
            .expect("an array")
            .iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .map(|event| {
                event["item"]["status"]
                    .as_str()
                    .expect("a status")
                    .to_string()
            })
            .collect();
        assert_eq!(done_statuses, ["completed", "completed", "in_progress"]);
        assert_eq!(
            statuses(&done.expect("output").items),
            ["completed", "completed", "in_progress"],
            "the terminal output keeps the status each item reached"
        );
    }

    #[test]
    fn a_region_that_ends_without_its_marker_has_the_same_status_in_stream_and_fold() {
        let unmarked = [
            Event::Reasoning(Text::uncounted("plan")),
            Event::Content(Text::uncounted("text")),
            call(0, "get_weather"),
            arguments(0, r#"{"city":"Paris"}"#),
            end(0),
            finish(FinishReason::Other("failed".into()), 1),
        ];
        let (out, done) = stream("r", &unmarked);
        let done = done.expect("output");
        assert_eq!(output_wire(&done), output_wire(&output("r", &unmarked)));
        assert_eq!(
            statuses(&done.items),
            ["completed", "in_progress", "completed"]
        );
        let reasoning_done = stream_wire(&out)
            .as_array()
            .expect("an array")
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done" && event["item"]["type"] == "reasoning"
            })
            .expect("the reasoning item completes")["item"]["status"]
            .clone();
        assert_eq!(
            reasoning_done, "completed",
            "done event and terminal item agree"
        );

        let content_then_reasoning = [
            Event::Content(Text::uncounted("a")),
            Event::Reasoning(Text::uncounted("b")),
            Event::Content(Text::uncounted("c")),
            finish(FinishReason::Other("failed".into()), 0),
        ];
        let (_, done) = stream("r", &content_then_reasoning);
        assert_eq!(
            statuses(&done.expect("output").items),
            ["in_progress", "completed"],
            "content with substance ends a reasoning region even after the message has opened"
        );

        let stray_fragment = [
            Event::Reasoning(Text::uncounted("plan")),
            arguments(7, r#"{"stray":true}"#),
            finish(FinishReason::Other("failed".into()), 0),
        ];
        let (_, done) = stream("r", &stray_fragment);
        let done = done.expect("output");
        assert_eq!(
            output_wire(&done),
            output_wire(&output("r", &stray_fragment))
        );
        assert_eq!(
            statuses(&done.items),
            ["completed", "in_progress"],
            "a fragment that falls back to content ends the region as content does"
        );
    }

    #[test]
    fn a_second_reasoning_item_carries_its_index_in_its_id() {
        let events = [
            Event::Reasoning(Text::uncounted("a")),
            Event::ReasoningEnd,
            Event::Content(Text::uncounted("x")),
            Event::Reasoning(Text::uncounted("b")),
            Event::ReasoningEnd,
            finish(FinishReason::Stop, 0),
        ];
        let (_, done) = stream("r", &events);
        let ids: Vec<String> = wire(&done.expect("output").items)
            .as_array()
            .expect("an array")
            .iter()
            .map(|item| item["id"].as_str().expect("an id").to_string())
            .collect();
        assert_eq!(ids, ["rs_r", "msg_r", "rs_r_2"]);
        let folded: Vec<String> = wire(&output("r", &events).items)
            .as_array()
            .expect("an array")
            .iter()
            .map(|item| item["id"].as_str().expect("an id").to_string())
            .collect();
        assert_eq!(
            folded,
            ["rs_r", "msg_r"],
            "the fold gathers both regions into one reasoning item"
        );
    }

    #[test]
    fn event_type_names_are_the_protocol_crate_s_constants() {
        use openai_protocol::event_types::{
            ContentPartEvent, FunctionCallEvent, OutputItemEvent, OutputTextEvent,
        };
        let events = [
            Event::Content(Text::uncounted("x")),
            call(0, "f"),
            arguments(0, "{}"),
            end(0),
            finish(FinishReason::Stop, 1),
        ];
        let (out, _) = stream("r", &events);
        let seen = types(&out);
        for expected in [
            OutputItemEvent::ADDED,
            OutputItemEvent::DONE,
            ContentPartEvent::ADDED,
            ContentPartEvent::DONE,
            OutputTextEvent::DELTA,
            OutputTextEvent::DONE,
            FunctionCallEvent::ARGUMENTS_DELTA,
            FunctionCallEvent::ARGUMENTS_DONE,
        ] {
            assert!(
                seen.iter().any(|kind| kind == expected),
                "{expected} in {seen:?}"
            );
        }
    }

    #[test]
    fn events_without_a_responses_stream_representation_produce_nothing() {
        let (out, done) = stream(
            "r",
            &[
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
            ],
        );
        assert!(out.is_empty(), "{out:?}");
        assert!(done.is_none());
    }
}
