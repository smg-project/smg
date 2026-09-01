//! Provider-owned protocol extensions.
//!
//! Each provider contributes one module of all-`Option` extension structs that
//! are `#[serde(flatten)]`-ed into the core request types. Fields are promoted
//! here only when a provider's vendor-acceptance contract enforces behavior on
//! them; cosmetic extras stay out. Absent fields serialize to nothing, so
//! OpenAI-only traffic is wire-identical.

pub mod kimi;
pub mod minimax;
