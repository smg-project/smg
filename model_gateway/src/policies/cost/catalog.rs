//! The policy catalog: names, parameters, construction.
//!
//! One selection policy ships, the cache-aware default; it takes no parameters. Any other name
//! is rejected at configuration time.

use super::{default, policy::WorkerSelectionPolicy};

/// The policy used when none is configured: the pre-policy cache-aware decision.
pub const DEFAULT_POLICY: &str = default::POLICY_NAME;

pub const POLICY_NAMES: &[&str] = &[default::POLICY_NAME];

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("unknown selection policy '{0}'; known: {known}", known = POLICY_NAMES.join(", "))]
    Unknown(String),
}

/// The default policy at the given cache-aware temperature; it takes no parameters and cannot
/// fail to build.
pub fn default_policy(selection_temperature: f32) -> WorkerSelectionPolicy {
    default::policy(selection_temperature)
}

/// Build a policy by name. `selection_temperature` is the cache-aware temperature the default
/// policy keeps using.
pub fn build(
    name: &str,
    selection_temperature: f32,
) -> Result<WorkerSelectionPolicy, CatalogError> {
    match name {
        default::POLICY_NAME => Ok(default::policy(selection_temperature)),
        other => Err(CatalogError::Unknown(other.to_string())),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/policies/cost/catalog/tests.rs"]
mod tests;
