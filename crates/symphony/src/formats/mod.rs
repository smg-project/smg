//! The formats: how each model family writes reasoning and tool calls, as [`Parser`](crate::Parser)
//! implementations built from the scanner and the assemblers.
//!
//! [`qwen3`] is the first: `<think>` blocks and `<tool_call>` blocks holding one JSON object each.
//! [`constrained`] is the shape a grammar forces for the request's `tool_choice`: one function's
//! arguments, or a list of call objects, with no markers.

use crate::{event::FinishReason, input::EngineFinish};

pub mod constrained;
pub mod qwen3;

pub use constrained::{Choice, Constrained};
pub use qwen3::Qwen3;

/// The finish reason a format reports for the engine's: the same reason, in the event's terms. A
/// format keeps what the engine said; the adapters refine it (`stop` after a call is `tool_calls`).
pub fn finish_reason(finish: EngineFinish) -> FinishReason {
    match finish {
        EngineFinish::Stop => FinishReason::Stop,
        EngineFinish::Length => FinishReason::Length,
        EngineFinish::Abort => FinishReason::Abort,
        EngineFinish::Other(other) => FinishReason::Other(other),
    }
}
