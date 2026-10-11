//! The formats: how each model family writes reasoning and tool calls, each a [`Format`] table
//! the [`Engine`] runs.
//!
//! [`qwen3()`] is the first: `<think>` blocks and `<tool_call>` blocks, each holding one call in
//! the family's call syntax, a JSON object (Qwen3) or tags (Qwen 3.5 and later, Qwen3-Coder).
//! [`qwen2_5()`] is the same family before thinking: `<tool_call>` blocks alone, and `<think>` is
//! text. [`deepseek_v4_1()`] is DeepSeek's DSML: a `calls` block of one or several `invoke`
//! blocks, each a call whose parameter tags type their own values. [`seed_oss()`] is the Qwen
//! tagged syntax under Seed-OSS's own markers. [`hy4()`], [`ling()`] and [`iquest()`] write a
//! call as its name and keyed arguments, each under its own markers; [`glm()`] is Ling's syntax
//! with nothing between the tags, under GLM's turn opener, [`glm4()`] GLM 4.5 and 4.6's newline
//! after each pair under that opener, and [`k2_horizon()`] the same newlines under `ifm|`-prefixed
//! tags in a block of calls. [`olmo3()`] and [`lfm2_5()`] write their calls as Python. [`xlam()`]
//! is a bare JSON list of calls, or content.
//! [`minimax_m2()`] is MiniMax M2: `<think>` blocks and `<minimax:tool_call>` blocks of `invoke`
//! blocks whose parameter tags name their key in an attribute, typed by the request's tools.
//! [`minimax_m3()`] is MiniMax M3: `<mm:think>` blocks, `<tool_call>` blocks of `invoke` blocks
//! whose arguments are an XML tree, and a separator token the table ignores. [`kimi_k3()`] is
//! Kimi K3: a turn of tagged regions, `think`, `response` and `tools`, the last holding `call`
//! blocks whose argument tags name each value's type.
//! [`inkling()`] is Inkling: a turn of messages, each saying what it is after its opener, a call
//! as the tool's name, a marker and one JSON object.
//! [`minicpm5()`] is MiniCPM5: `<think>` blocks and, with no block around them, `<function
//! name="…">` calls of `<param name="…">` tags, a value a CDATA section when it holds `<`, `&` or
//! a newline.
//! [`plain()`] is the table for a template that writes the message's content and nothing else: one
//! state, no marker.
//!
//! [`Format`]: crate::Format
//! [`Engine`]: crate::Engine

pub mod deepseek_v4_1;
pub mod glm;
pub mod glm4;
pub mod hy4;
pub mod inkling;
pub mod iquest;
pub mod k2_horizon;
pub mod kimi_k3;
pub mod lfm2_5;
pub mod ling;
pub mod minicpm5;
pub mod minimax_m2;
pub mod minimax_m3;
pub mod olmo3;
pub mod plain;
pub mod qwen2_5;
pub mod qwen3;
pub mod seed_oss;
pub mod xlam;

pub use deepseek_v4_1::deepseek_v4_1;
pub use glm::glm;
pub use glm4::glm4;
pub use hy4::hy4;
pub use inkling::inkling;
pub use iquest::iquest;
pub use k2_horizon::k2_horizon;
pub use kimi_k3::kimi_k3;
pub use lfm2_5::lfm2_5;
pub use ling::ling;
pub use minicpm5::minicpm5;
pub use minimax_m2::minimax_m2;
pub use minimax_m3::minimax_m3;
pub use olmo3::olmo3;
pub use plain::plain;
pub use qwen2_5::qwen2_5;
pub use qwen3::qwen3;
pub use seed_oss::seed_oss;
pub use xlam::xlam;
