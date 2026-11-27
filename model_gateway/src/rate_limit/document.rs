//! The tenant rate-limit policy as a control-plane document: what a dry run
//! and an application say about a document against the one in force.

use std::collections::HashMap;

use serde::Serialize;

use super::config::{RateLimitYaml, TenantPolicySpec};

/// The shape of a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PolicySummary {
    /// Tenants with a policy of their own.
    pub tenants: usize,
    /// Model rules over every policy, the default's included.
    pub model_rules: usize,
    /// The default policy's token budget per minute.
    pub default_tokens_per_minute: u32,
    /// The default policy's request budget per minute.
    pub default_requests_per_minute: u32,
}

impl PolicySummary {
    /// The shape of `document`.
    pub fn of(document: &RateLimitYaml) -> Self {
        Self {
            tenants: document.tenants.len(),
            model_rules: document.default_policy.model_rules.len()
                + document
                    .tenants
                    .iter()
                    .map(|tenant| tenant.model_rules.len())
                    .sum::<usize>(),
            default_tokens_per_minute: document.default_policy.tokens_per_minute,
            default_requests_per_minute: document.default_policy.requests_per_minute,
        }
    }
}

/// What a document changes against the one in force, by tenant key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PolicyChanges {
    /// Tenants the document names and the live one does not.
    pub tenants_added: Vec<String>,
    /// Tenants the live document names and this one does not: they fall to
    /// the default policy.
    pub tenants_removed: Vec<String>,
    /// Tenants both name whose policy differs.
    pub tenants_changed: Vec<String>,
    /// Whether the default policy differs.
    pub default_changed: bool,
}

impl PolicyChanges {
    /// The changes `next` makes against `live`: the added and changed
    /// tenants in `next`'s order, the removed ones in `live`'s.
    pub fn between(live: &RateLimitYaml, next: &RateLimitYaml) -> Self {
        let before: HashMap<&str, &TenantPolicySpec> = keyed(live).collect();
        let after: HashMap<&str, &TenantPolicySpec> = keyed(next).collect();
        let mut changes = Self {
            default_changed: live.default_policy != next.default_policy,
            ..Self::default()
        };
        for (key, policy) in keyed(next) {
            match before.get(key) {
                None => changes.tenants_added.push(key.to_string()),
                Some(was) if *was != policy => changes.tenants_changed.push(key.to_string()),
                Some(_) => {}
            }
        }
        for (key, _) in keyed(live) {
            if !after.contains_key(key) {
                changes.tenants_removed.push(key.to_string());
            }
        }
        changes
    }
}

/// The tenant policies of `document` by key, in the document's order.
fn keyed(document: &RateLimitYaml) -> impl Iterator<Item = (&str, &TenantPolicySpec)> {
    document
        .tenants
        .iter()
        .filter_map(|tenant| Some((tenant.tenant_key.as_deref()?, tenant)))
}

#[cfg(test)]
#[path = "../../tests/unit/rate_limit/document/tests.rs"]
mod tests;
