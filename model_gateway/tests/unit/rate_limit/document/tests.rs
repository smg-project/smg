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
