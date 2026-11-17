//! The worker's wire types: what a worker is on the gateway's admin API and
//! in its registry (its spec, its info, its status, its load report), the
//! model card a worker serves, and the entry a card becomes in a model
//! listing.
//!
//! `openai-protocol` re-exports these, and [`UNKNOWN_MODEL_ID`], at the paths
//! they had there.
//!
//! [`settings`] and [`labels`] hold the vocabulary the worker's own code
//! shares with a host's configuration. The gateway re-exports each item at
//! the path it had there.
//!
//! The `runtime` feature adds the worker's runtime, for the process that
//! hosts the workers. Its first part is `connection`: the HTTP client factory
//! and the cache of worker-directed clients.

/// Default model identifier used when no model is specified.
///
/// This constant should be used instead of hardcoded "unknown" strings
/// throughout the codebase for consistency.
pub const UNKNOWN_MODEL_ID: &str = "unknown";

#[cfg(feature = "runtime")]
pub mod connection;
pub mod labels;
pub mod model_card;
pub mod model_type;
pub mod models;
pub mod settings;
pub mod worker;
