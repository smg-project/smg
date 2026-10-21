//! Tenant policy resolver: maps a `TenantKey` to a [`TenantPolicy`].
//!
//! Trait-shaped so a future store-backed implementation (with async lookup +
//! sync cache) can land without touching the admission middleware.

use std::{collections::HashMap, sync::Arc};

use super::{Class, SchedulerSettings};
use crate::tenant::TenantKey;

/// Per-tenant policy. The single field today is [`Self::max_class`]; future
/// non-breaking additions: `weight: u32`, `slot_quota: Option<usize>`,
/// `rps_cap: Option<u32>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantPolicy {
    pub max_class: Class,
}

/// Tenant policy lookup. Returns the [`TenantPolicy`] for a tenant,
/// with a default fallback for unknown keys. Trait-shaped so an async
/// store-backed implementation can swap in behind a sync cache without
/// changing call sites.
pub trait TenantPolicyResolver: Send + Sync {
    fn policy(&self, tenant: &TenantKey) -> TenantPolicy;
}

/// Static in-memory resolver: a `HashMap` of explicit per-tenant
/// overrides plus a single default policy applied to every other tenant.
/// Built once from [`SchedulerSettings`] at gateway startup.
pub struct StaticTenantPolicyResolver {
    policies: HashMap<TenantKey, TenantPolicy>,
    default: TenantPolicy,
}

impl StaticTenantPolicyResolver {
    /// Build from the runtime settings. Reads `tenant_policies` and
    /// `default_max_class`; never returns an error (validation happens
    /// inside [`SchedulerSettings::from_cli_and_yaml`]).
    pub fn from_settings(settings: &SchedulerSettings) -> Self {
        let policies = settings
            .tenant_policies
            .iter()
            .map(|(key, cfg)| {
                (
                    key.clone(),
                    TenantPolicy {
                        max_class: cfg.max_class,
                    },
                )
            })
            .collect();
        Self {
            policies,
            default: TenantPolicy {
                max_class: settings.default_max_class,
            },
        }
    }

    /// Test-only constructor used by call sites that don't want to build
    /// a full [`SchedulerSettings`] just to exercise the resolver.
    #[cfg(test)]
    pub(crate) fn with_default(default_max_class: Class) -> Self {
        Self {
            policies: HashMap::new(),
            default: TenantPolicy {
                max_class: default_max_class,
            },
        }
    }
}

impl TenantPolicyResolver for StaticTenantPolicyResolver {
    fn policy(&self, tenant: &TenantKey) -> TenantPolicy {
        self.policies.get(tenant).copied().unwrap_or(self.default)
    }
}

impl TenantPolicyResolver for Arc<dyn TenantPolicyResolver> {
    fn policy(&self, tenant: &TenantKey) -> TenantPolicy {
        (**self).policy(tenant)
    }
}

#[cfg(test)]
mod tests {
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
}
