use super::*;

fn policy(tenant_key: Option<&str>, tpm: u32, rpm: u32) -> TenantPolicySpec {
    TenantPolicySpec {
        tenant_key: tenant_key.map(str::to_string),
        tokens_per_minute: tpm,
        requests_per_minute: rpm,
        model_rules: Vec::new(),
    }
}

fn rule(rule_id: &str, matcher: ModelMatcherSpec, tpm: u32, rpm: u32) -> ModelRuleSpec {
    ModelRuleSpec {
        rule_id: rule_id.to_string(),
        matcher,
        tokens_per_minute: tpm,
        requests_per_minute: rpm,
    }
}

#[test]
fn valid_yaml_passes() {
    let yaml = RateLimitYaml {
        default_policy: policy(None, 1000, 60),
        tenants: vec![policy(Some("auth:team-red"), 5000, 300)],
    };
    assert!(yaml.validate().is_ok());
}

#[test]
fn the_default_policy_serializes_without_a_tenant_key() {
    let yaml = RateLimitYaml {
        default_policy: policy(None, 1000, 60),
        tenants: vec![policy(Some("auth:team-red"), 5000, 300)],
    };
    let document = serde_json::to_value(&yaml).expect("a document");
    assert!(
        document["default_policy"].get("tenant_key").is_none(),
        "the default policy carries no tenant_key on the wire"
    );
    assert_eq!(document["tenants"][0]["tenant_key"], "auth:team-red");
}

#[test]
fn misspelled_field_is_rejected_not_silently_ignored() {
    // `tenant:` typo'd for `tenants:` -- without deny_unknown_fields
    // this silently falls back to an empty tenants list.
    let raw = "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\ntenant:\n  - tenant_key: auth:team-red\n    tokens_per_minute: 5000\n    requests_per_minute: 300\n";
    assert!(serde_yaml::from_str::<RateLimitYaml>(raw).is_err());
}

#[test]
fn default_policy_with_tenant_key_rejected() {
    let yaml = RateLimitYaml {
        default_policy: policy(Some("auth:oops"), 1000, 60),
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::DefaultPolicyHasTenantKey)
    );
}

#[test]
fn tenant_missing_key_rejected() {
    let yaml = RateLimitYaml {
        default_policy: policy(None, 1000, 60),
        tenants: vec![policy(None, 5000, 300)],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::MissingTenantKey { index: 0 })
    );
}

#[test]
fn tenant_empty_or_whitespace_key_rejected() {
    for key in ["", "   "] {
        let yaml = RateLimitYaml {
            default_policy: policy(None, 1000, 60),
            tenants: vec![policy(Some(key), 5000, 300)],
        };
        assert_eq!(
            yaml.validate(),
            Err(RateLimitConfigError::MissingTenantKey { index: 0 })
        );
    }
}

#[test]
fn tenant_padded_key_rejected() {
    for tenant_key in [" auth:team-red", "auth:team-red "] {
        let yaml = RateLimitYaml {
            default_policy: policy(None, 1000, 60),
            tenants: vec![policy(Some(tenant_key), 5000, 300)],
        };
        assert_eq!(
            yaml.validate(),
            Err(RateLimitConfigError::PaddedTenantKey {
                tenant_key: tenant_key.to_string()
            })
        );
    }
}

#[test]
fn tenant_non_canonical_key_rejected() {
    // Bare tenant_id, missing the auth:/header:/ip: prefix a real
    // resolved key always has.
    for tenant_key in ["team-red", "auth:", "ip:not-an-ip", "tenant:team-red"] {
        let yaml = RateLimitYaml {
            default_policy: policy(None, 1000, 60),
            tenants: vec![policy(Some(tenant_key), 5000, 300)],
        };
        assert_eq!(
            yaml.validate(),
            Err(RateLimitConfigError::NonCanonicalTenantKey {
                tenant_key: tenant_key.to_string()
            })
        );
    }
}

#[test]
fn duplicate_tenant_key_rejected() {
    let yaml = RateLimitYaml {
        default_policy: policy(None, 1000, 60),
        tenants: vec![
            policy(Some("auth:team-red"), 5000, 300),
            policy(Some("auth:team-red"), 1000, 60),
        ],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::DuplicateTenantKey {
            tenant_key: "auth:team-red".to_string()
        })
    );
}

#[test]
fn zero_tokens_per_minute_rejected() {
    let yaml = RateLimitYaml {
        default_policy: policy(None, 0, 60),
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::ZeroTokensPerMinute {
            label: "default".to_string()
        })
    );
}

#[test]
fn zero_requests_per_minute_rejected() {
    let yaml = RateLimitYaml {
        default_policy: policy(None, 1000, 0),
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::ZeroRequestsPerMinute {
            label: "default".to_string()
        })
    );
}

#[test]
fn invalid_rule_id_rejected() {
    let mut default_policy = policy(None, 1000, 60);
    default_policy.model_rules.push(rule(
        "bad id!",
        ModelMatcherSpec::Exact {
            value: "gpt-4".to_string(),
        },
        100,
        10,
    ));
    let yaml = RateLimitYaml {
        default_policy,
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::InvalidRuleId {
            label: "default".to_string(),
            rule_id: "bad id!".to_string()
        })
    );
}

#[test]
fn duplicate_rule_id_rejected() {
    let mut default_policy = policy(None, 1000, 60);
    default_policy.model_rules.push(rule(
        "gpt4",
        ModelMatcherSpec::Exact {
            value: "gpt-4".to_string(),
        },
        100,
        10,
    ));
    default_policy.model_rules.push(rule(
        "gpt4",
        ModelMatcherSpec::Exact {
            value: "gpt-4-turbo".to_string(),
        },
        100,
        10,
    ));
    let yaml = RateLimitYaml {
        default_policy,
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::DuplicateRuleId {
            label: "default".to_string(),
            rule_id: "gpt4".to_string()
        })
    );
}

#[test]
fn zero_rule_limit_rejected() {
    let mut default_policy = policy(None, 1000, 60);
    default_policy.model_rules.push(rule(
        "gpt4",
        ModelMatcherSpec::Exact {
            value: "gpt-4".to_string(),
        },
        0,
        10,
    ));
    let yaml = RateLimitYaml {
        default_policy,
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::ZeroRuleTokensPerMinute {
            label: "default".to_string(),
            rule_id: "gpt4".to_string()
        })
    );
}

#[test]
fn empty_matcher_value_rejected() {
    let mut default_policy = policy(None, 1000, 60);
    default_policy.model_rules.push(rule(
        "catchall",
        ModelMatcherSpec::Prefix {
            value: String::new(),
        },
        100,
        10,
    ));
    let yaml = RateLimitYaml {
        default_policy,
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::EmptyMatcherValue {
            label: "default".to_string(),
            rule_id: "catchall".to_string()
        })
    );
}

#[test]
fn padded_matcher_value_rejected() {
    for value in [" gpt-4", "gpt-4 "] {
        let mut default_policy = policy(None, 1000, 60);
        default_policy.model_rules.push(rule(
            "gpt4",
            ModelMatcherSpec::Exact {
                value: value.to_string(),
            },
            100,
            10,
        ));
        let yaml = RateLimitYaml {
            default_policy,
            tenants: vec![],
        };
        assert_eq!(
            yaml.validate(),
            Err(RateLimitConfigError::PaddedMatcherValue {
                label: "default".to_string(),
                rule_id: "gpt4".to_string(),
                value: value.to_string(),
            })
        );
    }
}

#[test]
fn duplicate_exact_matcher_rejected() {
    let mut default_policy = policy(None, 1000, 60);
    default_policy.model_rules.push(rule(
        "gpt4-a",
        ModelMatcherSpec::Exact {
            value: "gpt-4".to_string(),
        },
        100,
        10,
    ));
    default_policy.model_rules.push(rule(
        "gpt4-b",
        ModelMatcherSpec::Exact {
            value: "gpt-4".to_string(),
        },
        200,
        20,
    ));
    let yaml = RateLimitYaml {
        default_policy,
        tenants: vec![],
    };
    assert_eq!(
        yaml.validate(),
        Err(RateLimitConfigError::DuplicateMatcher {
            label: "default".to_string(),
            rule_id: "gpt4-b".to_string(),
            other_rule_id: "gpt4-a".to_string()
        })
    );
}

#[test]
fn exact_and_prefix_same_value_both_allowed() {
    let mut default_policy = policy(None, 1000, 60);
    default_policy.model_rules.push(rule(
        "exact",
        ModelMatcherSpec::Exact {
            value: "gpt-4".to_string(),
        },
        100,
        10,
    ));
    default_policy.model_rules.push(rule(
        "prefix",
        ModelMatcherSpec::Prefix {
            value: "gpt-4".to_string(),
        },
        200,
        20,
    ));
    let yaml = RateLimitYaml {
        default_policy,
        tenants: vec![],
    };
    assert!(yaml.validate().is_ok());
}
