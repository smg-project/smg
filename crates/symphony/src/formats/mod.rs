//! The formats: how each model family writes reasoning and tool calls, each a [`Format`] table
//! the [`Engine`] runs. One module per table, named after the family that writes it; the README's
//! models table, written by `tests/readme.rs` from the registry, lists every checkpoint each table
//! reads, and `docs/tables.md` holds the notes on each table: what its template writes, what the
//! table decides beyond the others, and what it leaves open.
//!
//! [`for_model()`] picks the table for a model id: the registry of the checkpoints bellwether has
//! recorded, each with the table that reads it and its own turn opener where the table's is not
//! its template's.
//!
//! [`Format`]: crate::Format
//! [`Engine`]: crate::Engine
pub mod deepseek_v4;
pub mod deepseek_v4_1;
pub mod dots3;
pub mod forced;
pub mod function_gemma;
pub mod gemma4;
pub mod glm;
pub mod glm4;
pub mod hy4;
pub mod inkling;
pub mod iquest;
pub mod json;
pub mod k2_horizon;
pub mod kimi_k3;
pub mod lfm2_5;
pub mod ling;
pub mod llama_nemotron;
pub mod minicpm5;
pub mod minimax_m2;
pub mod minimax_m3;
pub mod mistral_v13;
pub mod muse_spark;
pub mod olmo3;
pub mod plain;
pub mod qwen2_5;
pub mod qwen3;
pub mod registry;
pub mod seed_oss;
pub mod xlam;

pub use deepseek_v4::deepseek_v4;
pub use deepseek_v4_1::deepseek_v4_1;
pub use dots3::dots3;
pub use forced::forced;
pub use function_gemma::function_gemma;
pub use gemma4::gemma4;
pub use glm::glm;
pub use glm4::glm4;
pub use hy4::hy4;
pub use inkling::inkling;
pub use iquest::iquest;
pub use json::json;
pub use k2_horizon::k2_horizon;
pub use kimi_k3::kimi_k3;
pub use lfm2_5::lfm2_5;
pub use ling::ling;
pub use llama_nemotron::llama_nemotron;
pub use minicpm5::minicpm5;
pub use minimax_m2::minimax_m2;
pub use minimax_m3::minimax_m3;
pub use mistral_v13::mistral_v13;
pub use muse_spark::muse_spark;
pub use olmo3::olmo3;
pub use plain::plain;
pub use qwen2_5::qwen2_5;
pub use qwen3::qwen3;
pub use registry::{for_model, groups, table_for, Table};
pub use seed_oss::seed_oss;
pub use xlam::xlam;
