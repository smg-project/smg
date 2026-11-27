//! Multi-tenant rate limit config: on-disk YAML shape + validation.
//!
//! Loaded once at startup from the file `--tenant-rate-limit-config <path>`
//! names (required with `--tenant-rate-limit-enabled`); the manager compiles
//! the parsed policies for the reserve/settle engine. Mirrors the priority
//! scheduler's `PrioritySchedulerYaml` (`middleware::scheduler::config`) —
//! kept separate from `RouterConfig` so the actual policy content isn't
//! crammed into the main config struct.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::tenant::is_canonical_serving_tenant_key;

/// A single tenant's (or the default) token/request-per-minute policy,
/// plus optional per-model rules layered on top. At most one model rule
/// matches per request — rules never stack with each other, only with
/// the tenant-global limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TenantPolicySpec {
    /// Canonical tenant key (e.g. `auth:team-red`). Must be `None` on
    /// [`RateLimitYaml::default_policy`] and `Some` on every entry in
    /// [`RateLimitYaml::tenants`]. Written only when present: the default
    /// policy carries no `tenant_key` on the wire, as the document's
    /// schema says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_key: Option<String>,
    pub tokens_per_minute: u32,
    pub requests_per_minute: u32,
    #[serde(default)]
    pub model_rules: Vec<ModelRuleSpec>,
}

/// A per-model rate-limit rule layered on top of the tenant-global limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRuleSpec {
    /// Stable identifier, unique within the tenant. `[A-Za-z0-9._-]+`.
    pub rule_id: String,
    pub matcher: ModelMatcherSpec,
    pub tokens_per_minute: u32,
    pub requests_per_minute: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelMatcherSpec {
    Exact { value: String },
    Prefix { value: String },
}

/// The YAML file `--tenant-rate-limit-config <path>` names, as read from
/// disk at startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitYaml {
    pub default_policy: TenantPolicySpec,
    #[serde(default)]
    pub tenants: Vec<TenantPolicySpec>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum RateLimitConfigError {
    #[error("default_policy must not set tenant_key")]
    DefaultPolicyHasTenantKey,
    #[error("tenants[{index}] is missing tenant_key")]
    MissingTenantKey { index: usize },
    #[error("duplicate tenant_key '{tenant_key}'")]
    DuplicateTenantKey { tenant_key: String },
    #[error("tenant_key '{tenant_key}' must not have surrounding whitespace")]
    PaddedTenantKey { tenant_key: String },
    #[error(
        "tenant_key '{tenant_key}' is not a canonical serving-path tenant key (expected auth:<id>, header:<id>, ip:<address>, or anonymous)"
    )]
    NonCanonicalTenantKey { tenant_key: String },
    #[error("policy '{label}': tokens_per_minute must be > 0")]
    ZeroTokensPerMinute { label: String },
    #[error("policy '{label}': requests_per_minute must be > 0")]
    ZeroRequestsPerMinute { label: String },
    #[error("policy '{label}': rule_id '{rule_id}' must match [A-Za-z0-9._-]+")]
    InvalidRuleId { label: String, rule_id: String },
    #[error("policy '{label}': duplicate rule_id '{rule_id}'")]
    DuplicateRuleId { label: String, rule_id: String },
    #[error("policy '{label}': rule '{rule_id}' tokens_per_minute must be > 0")]
    ZeroRuleTokensPerMinute { label: String, rule_id: String },
    #[error("policy '{label}': rule '{rule_id}' requests_per_minute must be > 0")]
    ZeroRuleRequestsPerMinute { label: String, rule_id: String },
    #[error("policy '{label}': rule '{rule_id}' matcher value must not be empty")]
    EmptyMatcherValue { label: String, rule_id: String },
    #[error("policy '{label}': rule '{rule_id}' matcher value '{value}' must not have surrounding whitespace")]
    PaddedMatcherValue {
        label: String,
        rule_id: String,
        value: String,
    },
    #[error("policy '{label}': rule '{rule_id}' matcher duplicates rule '{other_rule_id}'")]
    DuplicateMatcher {
        label: String,
        rule_id: String,
        other_rule_id: String,
    },
}

fn is_valid_rule_id(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

fn validate_policy(label: &str, spec: &TenantPolicySpec) -> Result<(), RateLimitConfigError> {
    if spec.tokens_per_minute == 0 {
        return Err(RateLimitConfigError::ZeroTokensPerMinute {
            label: label.to_string(),
        });
    }
    if spec.requests_per_minute == 0 {
        return Err(RateLimitConfigError::ZeroRequestsPerMinute {
            label: label.to_string(),
        });
    }

    let mut seen_rule_ids = HashSet::new();
    // Owned rather than borrowed: this only runs once at startup, so the
    // clones are irrelevant, and it sidesteps borrowing two different
    // locals (`seen_exact`/`seen_prefix`) through the same match arm.
    let mut seen_exact: HashMap<String, String> = HashMap::new();
    let mut seen_prefix: HashMap<String, String> = HashMap::new();

    for rule in &spec.model_rules {
        if !is_valid_rule_id(&rule.rule_id) {
            return Err(RateLimitConfigError::InvalidRuleId {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
            });
        }
        if !seen_rule_ids.insert(rule.rule_id.as_str()) {
            return Err(RateLimitConfigError::DuplicateRuleId {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
            });
        }
        if rule.tokens_per_minute == 0 {
            return Err(RateLimitConfigError::ZeroRuleTokensPerMinute {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
            });
        }
        if rule.requests_per_minute == 0 {
            return Err(RateLimitConfigError::ZeroRuleRequestsPerMinute {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
            });
        }

        let (value, table): (&str, &mut HashMap<String, String>) = match &rule.matcher {
            ModelMatcherSpec::Exact { value } => (value.as_str(), &mut seen_exact),
            ModelMatcherSpec::Prefix { value } => (value.as_str(), &mut seen_prefix),
        };
        let trimmed_value = value.trim();
        if trimmed_value.is_empty() {
            return Err(RateLimitConfigError::EmptyMatcherValue {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
            });
        }
        // A padded value (e.g. `"gpt-4 "`) would compile verbatim and never
        // match a real request's model id — `matching_rule` does raw
        // `HashMap::get`/`starts_with` checks against the model id as
        // reported by the request, which is never padded — so the rule
        // would silently never apply.
        if trimmed_value != value {
            return Err(RateLimitConfigError::PaddedMatcherValue {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
                value: value.to_string(),
            });
        }
        if let Some(other_rule_id) = table.insert(value.to_string(), rule.rule_id.clone()) {
            return Err(RateLimitConfigError::DuplicateMatcher {
                label: label.to_string(),
                rule_id: rule.rule_id.clone(),
                other_rule_id,
            });
        }
    }

    Ok(())
}

impl RateLimitYaml {
    /// Positive limits, unique tenant keys, unique+valid rule ids per
    /// tenant, no duplicate exact/prefix matcher within a tenant, no empty
    /// matcher value. Rejects at load time rather than at first use.
    pub fn validate(&self) -> Result<(), RateLimitConfigError> {
        if self.default_policy.tenant_key.is_some() {
            return Err(RateLimitConfigError::DefaultPolicyHasTenantKey);
        }
        validate_policy("default", &self.default_policy)?;

        let mut seen_tenants = HashSet::new();
        for (index, spec) in self.tenants.iter().enumerate() {
            let Some(tenant_key) = spec.tenant_key.as_deref() else {
                return Err(RateLimitConfigError::MissingTenantKey { index });
            };
            let trimmed = tenant_key.trim();
            // Empty/whitespace-only is functionally the same as missing: it
            // would compile into a `TenantKey` that can never match a real
            // resolved tenant identity (those are always `auth:`/`header:`/
            // `ip:`-prefixed or `anonymous`), so the entry would silently
            // never apply to anything.
            if trimmed.is_empty() {
                return Err(RateLimitConfigError::MissingTenantKey { index });
            }
            // A padded key (leading/trailing whitespace, no surrounding
            // quotes involved) would compile verbatim into `TenantKey` via
            // `TenantKey::from`, but resolved auth/header tenant keys are
            // always canonicalized without surrounding whitespace — the
            // override would silently never be found, and the tenant
            // falls back to the default policy instead.
            if trimmed != tenant_key {
                return Err(RateLimitConfigError::PaddedTenantKey {
                    tenant_key: tenant_key.to_string(),
                });
            }
            // E.g. a bare tenant_id copy-pasted from tenant_api_keys,
            // missing its auth: prefix — can never match a real tenant.
            if !is_canonical_serving_tenant_key(tenant_key) {
                return Err(RateLimitConfigError::NonCanonicalTenantKey {
                    tenant_key: tenant_key.to_string(),
                });
            }
            if !seen_tenants.insert(tenant_key) {
                return Err(RateLimitConfigError::DuplicateTenantKey {
                    tenant_key: tenant_key.to_string(),
                });
            }
            validate_policy(tenant_key, spec)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/rate_limit/config/tests.rs"]
mod tests;
