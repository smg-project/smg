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
    assert_eq!(&*matched.rule_id, "exact");
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
    assert_eq!(&*matched.rule_id, "long");
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
