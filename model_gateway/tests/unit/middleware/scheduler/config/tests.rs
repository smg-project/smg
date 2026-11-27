use std::time::Duration;

use super::*;
use crate::middleware::scheduler::Class;

#[test]
fn test_default_for_system() {
    let cfg = ClassConfig::default_for(Class::System);
    assert_eq!(cfg.reserved_floor, 32);
    assert_eq!(cfg.queue_size, 64);
    assert_eq!(cfg.queue_timeout_secs, 30);
    assert_eq!(cfg.starvation_threshold_secs, 5);
    assert!(cfg.can_preempt);
}

#[test]
fn test_default_for_interactive() {
    let cfg = ClassConfig::default_for(Class::Interactive);
    assert_eq!(cfg.reserved_floor, 128);
    assert_eq!(cfg.queue_size, 256);
    assert_eq!(cfg.queue_timeout_secs, 30);
    assert_eq!(cfg.starvation_threshold_secs, 5);
    assert!(cfg.can_preempt);
}

#[test]
fn test_default_for_default() {
    let cfg = ClassConfig::default_for(Class::Default);
    assert_eq!(cfg.reserved_floor, 0);
    assert_eq!(cfg.queue_size, 512);
    assert_eq!(cfg.queue_timeout_secs, 60);
    assert_eq!(cfg.starvation_threshold_secs, 30);
    assert!(!cfg.can_preempt);
}

#[test]
fn test_default_for_bulk() {
    let cfg = ClassConfig::default_for(Class::Bulk);
    assert_eq!(cfg.reserved_floor, 0);
    assert_eq!(cfg.queue_size, 1024);
    assert_eq!(cfg.queue_timeout_secs, 300);
    assert_eq!(cfg.starvation_threshold_secs, 120);
    assert!(!cfg.can_preempt);
}

#[test]
fn test_runtime_config_converts_seconds_to_duration() {
    let cfg = ClassConfig::default_for(Class::Default);
    let runtime = ClassRuntimeConfig::from_class_config(&cfg);
    assert_eq!(runtime.queue_timeout, Duration::from_secs(60));
    assert_eq!(runtime.starvation_threshold, Duration::from_secs(30));
    assert!(!runtime.can_preempt);
}

#[test]
fn test_runtime_config_preserves_can_preempt_flag() {
    let interactive =
        ClassRuntimeConfig::from_class_config(&ClassConfig::default_for(Class::Interactive));
    assert!(interactive.can_preempt);
    let bulk = ClassRuntimeConfig::from_class_config(&ClassConfig::default_for(Class::Bulk));
    assert!(!bulk.can_preempt);
}

// ── PrioritySchedulerYaml serde ───────────────────────────────────

#[test]
fn test_yaml_empty_document_yields_default() {
    let parsed: PrioritySchedulerYaml = serde_yaml::from_str("").unwrap();
    assert!(parsed.classes.is_empty());
    assert!(parsed.tenant_policies.is_empty());
}

#[test]
fn test_yaml_partial_class_override_round_trips() {
    let yaml = r"
classes:
  interactive:
    reserved_floor: 200
    queue_size: 256
    queue_timeout_secs: 30
    starvation_threshold_secs: 5
    can_preempt: true
";
    let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
    let interactive = parsed
        .classes
        .get(&Class::Interactive)
        .expect("interactive present");
    assert_eq!(interactive.reserved_floor, 200);
    // Only one class entry — others are absent (settings layer fills defaults).
    assert_eq!(parsed.classes.len(), 1);
    assert!(parsed.tenant_policies.is_empty());
}

#[test]
fn test_yaml_tenant_policy_round_trips() {
    let yaml = r#"
tenant_policies:
  "auth:acme":
    max_class: interactive
  "auth:internal-cron":
    max_class: system
"#;
    let parsed: PrioritySchedulerYaml = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(parsed.tenant_policies.len(), 2);
    assert_eq!(
        parsed.tenant_policies["auth:acme"].max_class,
        Class::Interactive
    );
    assert_eq!(
        parsed.tenant_policies["auth:internal-cron"].max_class,
        Class::System
    );
}

#[test]
fn test_yaml_unknown_class_value_is_serde_error() {
    let yaml = r#"
tenant_policies:
  "auth:acme":
    max_class: garbage
"#;
    let result: Result<PrioritySchedulerYaml, _> = serde_yaml::from_str(yaml);
    assert!(result.is_err(), "expected serde error for unknown class");
}

#[test]
fn test_yaml_unknown_key_is_serde_error_naming_it() {
    // The top level, a class entry and a tenant policy each refuse a key
    // they do not know: a misspelt key must not silently drop a section.
    let documents = [
        (
            "extra_field",
            "classes: {}\ntenant_policies: {}\nextra_field: 1\n".to_owned(),
        ),
        (
            "tenant_polices",
            "tenant_polices:\n  \"auth:acme\":\n    max_class: bulk\n".to_owned(),
        ),
        (
            "queue_sized",
            "classes:\n  bulk:\n    reserved_floor: 0\n    queue_sized: 8\n    queue_size: 8\n    \
                 queue_timeout_secs: 1\n    starvation_threshold_secs: 1\n    can_preempt: false\n"
                .to_owned(),
        ),
        (
            "weight",
            "tenant_policies:\n  \"auth:acme\":\n    max_class: bulk\n    weight: 2\n".to_owned(),
        ),
    ];
    for (key, yaml) in documents {
        let err = serde_yaml::from_str::<PrioritySchedulerYaml>(&yaml)
            .expect_err(key)
            .to_string();
        assert!(
            err.contains(key),
            "the error names the unknown key `{key}`: {err}"
        );
    }
}

#[test]
fn test_yaml_class_name_serializes_as_lowercase() {
    let mut classes = HashMap::new();
    classes.insert(Class::Bulk, ClassConfig::default_for(Class::Bulk));
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: Default::default(),
    };
    let rendered = serde_yaml::to_string(&yaml).unwrap();
    assert!(
        rendered.contains("bulk:"),
        "class key should serialize as lowercase: {rendered}"
    );
}

// ── SchedulerSettings::from_cli_and_yaml ─────────────────────────

use crate::tenant::TenantKey;

#[test]
fn test_settings_no_yaml_uses_builtin_defaults() {
    let s = SchedulerSettings::from_cli_and_yaml(false, Class::Default, 32, None).unwrap();
    assert!(!s.enabled);
    assert_eq!(s.default_max_class, Class::Default);
    assert_eq!(s.tenant_metric_top_n, 32);
    for class in Class::ALL {
        assert_eq!(s.class_config(class), &ClassConfig::default_for(class));
    }
    assert!(s.tenant_policies.is_empty());
}

#[test]
fn test_settings_yaml_partial_override_merges_with_defaults() {
    let mut classes = HashMap::new();
    let mut interactive = ClassConfig::default_for(Class::Interactive);
    interactive.reserved_floor = 64;
    classes.insert(Class::Interactive, interactive);
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: Default::default(),
    };
    let s = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
    assert_eq!(s.class_config(Class::Interactive).reserved_floor, 64);
    // Bulk untouched — still equal to built-in default.
    assert_eq!(
        s.class_config(Class::Bulk),
        &ClassConfig::default_for(Class::Bulk)
    );
}

#[test]
fn test_settings_yaml_tenant_policy_propagates() {
    let mut tenant_policies = HashMap::new();
    tenant_policies.insert(
        "auth:acme".to_string(),
        TenantPolicyConfig {
            max_class: Class::Interactive,
        },
    );
    let yaml = PrioritySchedulerYaml {
        classes: Default::default(),
        tenant_policies,
    };
    let s = SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
    assert_eq!(s.tenant_policies.len(), 1);
    let key = TenantKey::new("auth:acme");
    assert_eq!(s.tenant_policies[&key].max_class, Class::Interactive);
}

#[test]
fn test_settings_rejects_zero_queue_timeout() {
    let mut classes = HashMap::new();
    let mut bulk = ClassConfig::default_for(Class::Bulk);
    bulk.queue_timeout_secs = 0;
    classes.insert(Class::Bulk, bulk);
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: Default::default(),
    };
    let err =
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap_err();
    assert!(matches!(
        err,
        SettingsValidationError::ZeroQueueTimeout { class: Class::Bulk }
    ));
}

#[test]
fn test_settings_rejects_zero_starvation_threshold() {
    let mut classes = HashMap::new();
    let mut bulk = ClassConfig::default_for(Class::Bulk);
    bulk.starvation_threshold_secs = 0;
    classes.insert(Class::Bulk, bulk);
    let yaml = PrioritySchedulerYaml {
        classes,
        tenant_policies: Default::default(),
    };
    let err =
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap_err();
    assert!(matches!(
        err,
        SettingsValidationError::ZeroStarvationThreshold { class: Class::Bulk }
    ));
}
