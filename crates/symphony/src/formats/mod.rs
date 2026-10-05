//! The formats: how each model family writes reasoning and tool calls, as [`Parser`](crate::Parser)
//! implementations built from the scanner and the assemblers.
//!
//! [`qwen3`] is the first: `<think>` blocks and `<tool_call>` blocks holding one JSON object each.

pub mod qwen3;

pub use qwen3::Qwen3;
