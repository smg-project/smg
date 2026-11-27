use super::*;
use crate::middleware::scheduler::{PrioritySchedulerYaml, TenantPolicyConfig};

#[test]
fn test_empty_resolver_returns_default() {
    let resolver = StaticTenantPolicyResolver::with_default(Class::Default);
    let policy = resolver.policy(&TenantKey::new("anonymous"));
    assert_eq!(policy.max_class, Class::Default);
}

#[test]
fn test_known_tenant_overrides_default() {
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
    let settings =
        SchedulerSettings::from_cli_and_yaml(true, Class::Default, 32, Some(&yaml)).unwrap();
    let resolver = StaticTenantPolicyResolver::from_settings(&settings);

    let known = resolver.policy(&TenantKey::new("auth:acme"));
    assert_eq!(known.max_class, Class::Interactive);

    // Unknown tenant still gets the default.
    let unknown = resolver.policy(&TenantKey::new("anonymous"));
    assert_eq!(unknown.max_class, Class::Default);
}

#[test]
fn test_resolver_trait_object_dispatches() {
    let resolver: Arc<dyn TenantPolicyResolver> =
        Arc::new(StaticTenantPolicyResolver::with_default(Class::Interactive));
    // Dispatch through the &dyn TenantPolicyResolver impl on Arc.
    let policy = resolver.policy(&TenantKey::new("anything"));
    assert_eq!(policy.max_class, Class::Interactive);
}

#[test]
fn test_clamp_via_min_with_resolved_policy() {
    // Demonstrate the admission-time clamp: effective = min(header, policy.max_class).
    let resolver = StaticTenantPolicyResolver::with_default(Class::Default);
    let header_class = Class::Interactive; // client asked for interactive
    let policy = resolver.policy(&TenantKey::new("anonymous"));
    let effective = std::cmp::min(header_class, policy.max_class);
    assert_eq!(effective, Class::Default);
}
