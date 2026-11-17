//! The worker's wire types: what a worker is on the gateway's admin API and
//! in its registry (its spec, its info, its status, its load report), the
//! model card a worker serves, and the entry a card becomes in a model
//! listing.
//!
//! `openai-protocol` re-exports these, and [`UNKNOWN_MODEL_ID`], at the paths
//! they had there.

/// Default model identifier used when no model is specified.
///
/// This constant should be used instead of hardcoded "unknown" strings
/// throughout the codebase for consistency.
pub const UNKNOWN_MODEL_ID: &str = "unknown";

pub mod model_card;
pub mod model_type;
pub mod models;
pub mod worker;
