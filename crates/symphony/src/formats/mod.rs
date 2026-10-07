//! The formats: how each model family writes reasoning and tool calls, each a [`Format`] table
//! the [`Engine`] runs.
//!
//! [`qwen3()`] is the first: `<think>` blocks and `<tool_call>` blocks, each holding one call in
//! the family's call syntax, a JSON object (Qwen3) or tags (Qwen 3.5 and later, Qwen3-Coder).
//! [`qwen2_5()`] is the same family before thinking: `<tool_call>` blocks alone, and `<think>` is
//! text. [`deepseek_v4_1()`] is DeepSeek's DSML: a `calls` block of one or several `invoke`
//! blocks, each a call whose parameter tags type their own values. [`seed_oss()`] is the Qwen
//! tagged syntax under Seed-OSS's own markers. [`hy4()`], [`ling()`] and [`iquest()`] write a
//! call as its name and keyed arguments, each under its own markers. [`olmo3()`] and [`lfm2_5()`]
//! write their calls as Python. [`xlam()`] is a bare JSON list of calls, or content.
//! [`minimax_m3()`] is MiniMax M3: `<mm:think>` blocks, `<tool_call>` blocks of `invoke` blocks
//! whose arguments are an XML tree, and a separator token the table ignores.
//!
//! [`Format`]: crate::Format
//! [`Engine`]: crate::Engine

pub mod deepseek_v4_1;
pub mod hy4;
pub mod iquest;
pub mod lfm2_5;
pub mod ling;
pub mod minimax_m3;
pub mod olmo3;
pub mod qwen2_5;
pub mod qwen3;
pub mod seed_oss;
pub mod xlam;

pub use deepseek_v4_1::deepseek_v4_1;
pub use hy4::hy4;
pub use iquest::iquest;
pub use lfm2_5::lfm2_5;
pub use ling::ling;
pub use minimax_m3::minimax_m3;
pub use olmo3::olmo3;
pub use qwen2_5::qwen2_5;
pub use qwen3::qwen3;
pub use seed_oss::seed_oss;
pub use xlam::xlam;
