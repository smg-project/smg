//! The formats: how each model family writes reasoning and tool calls, as [`Parser`](crate::Parser)
//! implementations built from the scanner and the assemblers.
//!
//! [`qwen3`] is the first: `<think>` blocks and `<tool_call>` blocks, each holding one call in the
//! family's call syntax, a JSON object (Qwen3) or tags (Qwen 3.5 and later, Qwen3-Coder).

pub mod qwen3;

pub use qwen3::{CallSyntax, Qwen3};
