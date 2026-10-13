//! One fold of a whole event sequence, shared by the three adapters' whole-response shapes: the
//! content, the reasoning, the calls in the order they started, and the finish. What each event
//! means for a whole response is decided here once; each adapter renders the fold in its API's
//! shape.

use crate::event::{Event, FinishReason};

/// What a whole event sequence said.
#[derive(Clone, Debug, Default)]
pub(super) struct Folded {
    /// Every content and malformed text in order, and the argument fragments of a call that never
    /// started, which cannot occur under the `Event` contract and join the content so that no byte
    /// is lost.
    pub content: String,
    /// Every reasoning text in order.
    pub reasoning: String,
    /// Whether the reasoning was over when the sequence ended: `ReasoningEnd` came, or content, a
    /// call or an orphan argument fragment followed the last reasoning text.
    pub reasoning_ended: bool,
    /// The calls in the order they started.
    pub calls: Vec<Call>,
    /// The `Finish` event's reason and tool-call count, when the sequence has one.
    pub finish: Option<Finish>,
}

/// One started call and what its events said.
#[derive(Clone, Debug)]
pub(super) struct Call {
    pub index: u32,
    pub id: String,
    pub name: String,
    /// Its argument fragments joined.
    pub arguments: String,
    /// Whether `ToolCallEnd` came for it.
    pub ended: bool,
}

/// The `Finish` event.
#[derive(Clone, Debug)]
pub(super) struct Finish {
    pub reason: FinishReason,
    pub tool_calls: u32,
}

/// The fold of `events`.
pub(super) fn fold<'a>(events: impl IntoIterator<Item = &'a Event>) -> Folded {
    let mut folded = Folded::default();
    for event in events {
        match event {
            Event::Content(text) | Event::Malformed { text, .. } => {
                folded.content.push_str(&text.text);
                if !text.text.trim().is_empty() {
                    folded.reasoning_ended = true;
                }
            }
            Event::Reasoning(text) => {
                folded.reasoning.push_str(&text.text);
                folded.reasoning_ended = false;
            }
            Event::ReasoningEnd => folded.reasoning_ended = true,
            Event::ToolCallStart {
                index, id, name, ..
            } => {
                folded.reasoning_ended = true;
                folded.calls.push(Call {
                    index: *index,
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                    ended: false,
                });
            }
            Event::ToolCallArguments { index, json, .. } => {
                match folded.calls.iter_mut().find(|call| call.index == *index) {
                    Some(call) => call.arguments.push_str(json),
                    None => {
                        folded.content.push_str(json);
                        if !json.trim().is_empty() {
                            folded.reasoning_ended = true;
                        }
                    }
                }
            }
            Event::ToolCallEnd { index, .. } => {
                if let Some(call) = folded.calls.iter_mut().find(|call| call.index == *index) {
                    call.ended = true;
                }
            }
            Event::Finish {
                reason, tool_calls, ..
            } => {
                folded.finish = Some(Finish {
                    reason: reason.clone(),
                    tool_calls: *tool_calls,
                });
            }
            Event::ReasoningStart | Event::Dropped { .. } => {}
        }
    }
    folded
}
