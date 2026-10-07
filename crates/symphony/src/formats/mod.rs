//! The formats: how each model family writes reasoning and tool calls, each a [`Format`] table
//! the [`Engine`] runs.
//!
//! [`qwen3()`] is the first: `<think>` blocks and `<tool_call>` blocks, each holding one call in
//! the family's call syntax, a JSON object (Qwen3) or tags (Qwen 3.5 and later, Qwen3-Coder).
//! [`qwen2_5()`] is the same family before thinking: `<tool_call>` blocks alone, and `<think>` is
//! text. [`deepseek_v4_1()`] is DeepSeek's DSML: a `calls` block of one or several `invoke`
//! blocks, each a call whose parameter tags type their own values. [`seed_oss()`] is the Qwen
//! tagged syntax under Seed-OSS's own markers.
//!
//! [`Format`]: crate::Format
//! [`Engine`]: crate::Engine

pub mod deepseek_v4_1;
pub mod qwen2_5;
pub mod qwen3;
pub mod seed_oss;

pub use deepseek_v4_1::deepseek_v4_1;
pub use qwen2_5::qwen2_5;
pub use qwen3::qwen3;
pub use seed_oss::seed_oss;
