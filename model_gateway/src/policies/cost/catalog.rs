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
mod tests {
    use super::*;

    #[test]
    fn every_name_builds_with_defaults() {
        for name in POLICY_NAMES {
            let policy = build(name, 0.0).expect("builds");
            assert_eq!(policy.name(), *name);
        }
    }

    #[test]
    fn unknown_names_are_rejected() {
        for name in ["nope", "not-a-policy"] {
            assert!(
                matches!(build(name, 0.0), Err(CatalogError::Unknown(_))),
                "{name}"
            );
        }
    }
}
