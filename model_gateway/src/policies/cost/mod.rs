//! Cost-function worker selection.
//!
//! Cache-aware routing gathers, once per request, what it knows about every candidate worker (its
//! prefix overlap, the decayed affinity score the decision ranks on, and the placement the
//! optimistic accounting predicts for it) and hands that to a *selection policy*: a pipeline of
//! filters, additive cost scorers and one picker, registered by name.
//!
//! - [`catalog::DEFAULT_POLICY`] reproduces the pre-policy cache-aware decision exactly (an affinity
//!   group that the host resolves with its pressure gate and expected-wait selector).
//! - [`accounting::OptimisticAccounting`] closes the window between a dispatch and the engine's
//!   first event, when enabled.
//!
//! The cost of the stage itself is measured by `benches/policy_selection.rs`.

pub mod accounting;
pub mod catalog;
pub mod inputs;
pub mod policy;
pub mod softmax;

mod default;

#[cfg(test)]
mod sim_tests;

pub use accounting::OptimisticAccounting;
pub use catalog::{build, default_policy, CatalogError, DEFAULT_POLICY, POLICY_NAMES};
pub use inputs::{CandidateInputs, RequestInputs};
pub use policy::{Needs, Pick, WorkerFilter, WorkerPicker, WorkerScorer, WorkerSelectionPolicy};
