//! Compiles a validated [`RateLimitYaml`] into an immutable, cheap-to-look-up
//! policy set.

use std::collections::HashMap;

use super::config::{ModelMatcherSpec, RateLimitConfigError, RateLimitYaml, TenantPolicySpec};
use crate::tenant::TenantKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeLimits {
    pub tokens_per_minute: u32,
    pub requests_per_minute: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledModelRule {
    pub rule_id: String,
    pub limits: ScopeLimits,
}

/// A prefix rule plus the prefix it matches on, kept alongside each other
/// since [`CompiledTenantPolicy::prefix_model_rules`] is a plain sorted
/// `Vec` rather than a map (there's nothing to key a prefix scan on).
#[derive(Debug, Clone)]
pub(crate) struct CompiledPrefixRule {
    prefix: String,
    rule: CompiledModelRule,
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledTenantPolicy {
    pub limits: ScopeLimits,
    /// O(1) lookup — exact matches never need to scan.
    exact_model_rules: HashMap<String, CompiledModelRule>,
    /// Sorted longest-prefix-first once at compile time, so matching is a
    /// single linear scan that returns on the first hit rather than a
    /// scan-then-`max_by_key` at request time.
    prefix_model_rules: Vec<CompiledPrefixRule>,
}

impl CompiledTenantPolicy {
    /// The single matching rule, if any: exact wins over prefix, longest
    /// wins among prefixes. Rules never stack.
    pub(crate) fn matching_rule(&self, model_id: &str) -> Option<&CompiledModelRule> {
        if let Some(exact) = self.exact_model_rules.get(model_id) {
            return Some(exact);
        }
        self.prefix_model_rules
            .iter()
            .find(|r| model_id.starts_with(r.prefix.as_str()))
            .map(|r| &r.rule)
    }

    fn compile(spec: &TenantPolicySpec) -> Self {
        let mut exact_model_rules = HashMap::new();
        let mut prefix_model_rules = Vec::new();

        for r in &spec.model_rules {
            let rule = CompiledModelRule {
                rule_id: r.rule_id.clone(),
                limits: ScopeLimits {
                    tokens_per_minute: r.tokens_per_minute,
                    requests_per_minute: r.requests_per_minute,
                },
            };
            match &r.matcher {
                ModelMatcherSpec::Exact { value } => {
                    exact_model_rules.insert(value.clone(), rule);
                }
                ModelMatcherSpec::Prefix { value } => {
                    prefix_model_rules.push(CompiledPrefixRule {
                        prefix: value.clone(),
                        rule,
                    });
                }
            }
        }
        // Longest-first so `matching_rule`'s linear scan can return on the
        // first hit instead of comparing lengths at request time.
        prefix_model_rules.sort_by_key(|r| std::cmp::Reverse(r.prefix.len()));

        Self {
            limits: ScopeLimits {
                tokens_per_minute: spec.tokens_per_minute,
                requests_per_minute: spec.requests_per_minute,
            },
            exact_model_rules,
            prefix_model_rules,
        }
    }
}

/// Immutable, compiled rate-limit policy: a default plus per-tenant
/// overrides, resolved in O(1) by `Self::policy_for`.
#[derive(Debug, Clone)]
pub struct CompiledPolicySet {
    default: CompiledTenantPolicy,
    tenants: HashMap<TenantKey, CompiledTenantPolicy>,
}

impl CompiledPolicySet {
    pub fn compile(yaml: &RateLimitYaml) -> Result<Self, RateLimitConfigError> {
        yaml.validate()?;
        let default = CompiledTenantPolicy::compile(&yaml.default_policy);
        let tenants = yaml
            .tenants
            .iter()
            .filter_map(|spec| {
                // INVARIANT: `yaml.validate()?` above already rejected any
                // `tenants` entry with a missing `tenant_key`, so `None`
                // here is unreachable — skip rather than panic if that
                // invariant is ever violated.
                spec.tenant_key
                    .clone()
                    .map(|k| (TenantKey::from(k), CompiledTenantPolicy::compile(spec)))
            })
            .collect();
        Ok(Self { default, tenants })
    }

    pub(crate) fn policy_for(&self, tenant: &TenantKey) -> &CompiledTenantPolicy {
        self.tenants.get(tenant).unwrap_or(&self.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limit::config::{ModelRuleSpec, TenantPolicySpec};

    fn spec_with_rules(rules: Vec<ModelRuleSpec>) -> TenantPolicySpec {
        TenantPolicySpec {
            tenant_key: None,
            tokens_per_minute: 1000,
            requests_per_minute: 60,
            model_rules: rules,
        }
    }

    fn rule(rule_id: &str, matcher: ModelMatcherSpec) -> ModelRuleSpec {
        ModelRuleSpec {
            rule_id: rule_id.to_string(),
            matcher,
            tokens_per_minute: 100,
            requests_per_minute: 10,
        }
    }

    #[test]
    fn unknown_tenant_falls_back_to_default() {
        let yaml = RateLimitYaml {
            default_policy: spec_with_rules(vec![]),
            tenants: vec![],
        };
        let compiled = CompiledPolicySet::compile(&yaml).unwrap();
        let policy = compiled.policy_for(&TenantKey::new("anonymous"));
        assert_eq!(policy.limits.tokens_per_minute, 1000);
    }

    #[test]
    fn known_tenant_overrides_default() {
        let mut tenant_spec = spec_with_rules(vec![]);
        tenant_spec.tenant_key = Some("auth:team-red".to_string());
        tenant_spec.tokens_per_minute = 5000;
        let yaml = RateLimitYaml {
            default_policy: spec_with_rules(vec![]),
            tenants: vec![tenant_spec],
        };
        let compiled = CompiledPolicySet::compile(&yaml).unwrap();
        let policy = compiled.policy_for(&TenantKey::new("auth:team-red"));
        assert_eq!(policy.limits.tokens_per_minute, 5000);
    }

    #[test]
    fn exact_wins_over_prefix() {
        let spec = spec_with_rules(vec![
            rule(
                "prefix",
                ModelMatcherSpec::Prefix {
                    value: "gpt-".to_string(),
                },
            ),
            rule(
                "exact",
                ModelMatcherSpec::Exact {
                    value: "gpt-4".to_string(),
                },
            ),
        ]);
        let yaml = RateLimitYaml {
            default_policy: spec,
            tenants: vec![],
        };
        let compiled = CompiledPolicySet::compile(&yaml).unwrap();
        let policy = compiled.policy_for(&TenantKey::new("anonymous"));
        let matched = policy.matching_rule("gpt-4").unwrap();
        assert_eq!(matched.rule_id, "exact");
    }

    #[test]
    fn longest_prefix_wins() {
        let spec = spec_with_rules(vec![
            rule(
                "short",
                ModelMatcherSpec::Prefix {
                    value: "gpt-".to_string(),
                },
            ),
            rule(
                "long",
                ModelMatcherSpec::Prefix {
                    value: "gpt-4-".to_string(),
                },
            ),
        ]);
        let yaml = RateLimitYaml {
            default_policy: spec,
            tenants: vec![],
        };
        let compiled = CompiledPolicySet::compile(&yaml).unwrap();
        let policy = compiled.policy_for(&TenantKey::new("anonymous"));
        let matched = policy.matching_rule("gpt-4-turbo").unwrap();
        assert_eq!(matched.rule_id, "long");
    }

    #[test]
    fn no_match_returns_none() {
        let spec = spec_with_rules(vec![rule(
            "gpt4",
            ModelMatcherSpec::Exact {
                value: "gpt-4".to_string(),
            },
        )]);
        let yaml = RateLimitYaml {
            default_policy: spec,
            tenants: vec![],
        };
        let compiled = CompiledPolicySet::compile(&yaml).unwrap();
        let policy = compiled.policy_for(&TenantKey::new("anonymous"));
        assert!(policy.matching_rule("claude-3").is_none());
    }
}
