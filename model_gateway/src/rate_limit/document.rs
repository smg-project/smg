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
mod tests {
    use super::{
        super::config::{ModelMatcherSpec, ModelRuleSpec},
        *,
    };

    fn policy(key: Option<&str>, tokens_per_minute: u32) -> TenantPolicySpec {
        TenantPolicySpec {
            tenant_key: key.map(str::to_string),
            tokens_per_minute,
            requests_per_minute: 10,
            model_rules: Vec::new(),
        }
    }

    fn document(default_tokens: u32, tenants: Vec<TenantPolicySpec>) -> RateLimitYaml {
        RateLimitYaml {
            default_policy: policy(None, default_tokens),
            tenants,
        }
    }

    fn rule(rule_id: &str) -> ModelRuleSpec {
        ModelRuleSpec {
            rule_id: rule_id.to_string(),
            matcher: ModelMatcherSpec::Exact {
                value: "m".to_string(),
            },
            tokens_per_minute: 5,
            requests_per_minute: 5,
        }
    }

    #[test]
    fn the_same_document_changes_nothing() {
        let live = document(100, vec![policy(Some("auth:red"), 10)]);
        assert_eq!(
            PolicyChanges::between(&live, &live.clone()),
            PolicyChanges::default()
        );
    }

    #[test]
    fn tenants_are_added_changed_and_removed_by_key_and_the_default_on_its_own() {
        let live = document(
            100,
            vec![policy(Some("auth:red"), 10), policy(Some("auth:blue"), 10)],
        );
        let next = document(
            200,
            vec![
                policy(Some("auth:blue"), 20),
                policy(Some("auth:green"), 10),
            ],
        );
        let changes = PolicyChanges::between(&live, &next);
        assert_eq!(changes.tenants_added, vec!["auth:green"]);
        assert_eq!(changes.tenants_removed, vec!["auth:red"]);
        assert_eq!(changes.tenants_changed, vec!["auth:blue"]);
        assert!(changes.default_changed);
    }

    #[test]
    fn the_summary_counts_the_tenants_and_every_model_rule() {
        let mut document = document(100, vec![policy(Some("auth:red"), 10)]);
        document.default_policy.model_rules.push(rule("d"));
        document.tenants[0].model_rules.push(rule("t"));
        assert_eq!(
            PolicySummary::of(&document),
            PolicySummary {
                tenants: 1,
                model_rules: 2,
                default_tokens_per_minute: 100,
                default_requests_per_minute: 10,
            }
        );
    }
}
