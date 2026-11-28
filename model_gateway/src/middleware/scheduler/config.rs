//! Scheduler config: on-disk YAML shape + runtime form.

use std::{collections::HashMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::Class;
use crate::tenant::TenantKey;

/// Per-class configuration as it appears in the optional YAML file.
///
/// Lives separately from [`ClassRuntimeConfig`] because the YAML form
/// uses primitive types friendly to serde and human editing, while the
/// runtime form pre-converts seconds into [`Duration`] so hot paths
/// don't repeat the conversion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassConfig {
    /// Absolute minimum slots reserved for this class — the value the
    /// effective reservation never drops below while the pool can hold
    /// every class's reservation. On a pool smaller than the sum of the
    /// reservations, every class keeps the same fraction of its reservation
    /// and the rest of the pool is shared (the scheduler's proportional
    /// clamp), so a small pool is never reserved whole for one class.
    /// Higher-class reservations are honored by lower-class admissions via
    /// the packed-CAS slot accounting in [`super::slots`].
    pub reserved_floor: u16,
    /// Share of backend capacity reserved for this class, on top of the
    /// floor: `effective = max(reserved_floor, ceil(reserved_per_slot ×
    /// capacity))`, recomputed as capacity changes. `0.0` (the default) is
    /// purely absolute — identical to a fixed `reserved_floor`. Must be
    /// finite and non-negative.
    #[serde(default)]
    pub reserved_per_slot: f64,
    /// Per-class queue depth limit.
    pub queue_size: u32,
    /// How long a queued waiter waits before the admission middleware
    /// sheds it with 503. Seconds at rest; converted to [`Duration`] in
    /// [`ClassRuntimeConfig`].
    pub queue_timeout_secs: u64,
    /// Head-of-queue age past which the dispatcher promotes a waiter
    /// out of normal priority order to avoid starvation. Seconds at
    /// rest; converted to [`Duration`] in [`ClassRuntimeConfig`].
    pub starvation_threshold_secs: u64,
    /// Whether admissions in this class are allowed to preempt a
    /// lower-class inflight request that has not yet emitted its first
    /// byte. Higher classes default to `true`; lower classes default
    /// to `false`.
    pub can_preempt: bool,
}

impl ClassConfig {
    /// Built-in defaults. These are what every class gets when no YAML
    /// file supplies an override.
    pub fn default_for(class: Class) -> Self {
        match class {
            Class::System => Self {
                reserved_floor: 32,
                reserved_per_slot: 0.0,
                queue_size: 64,
                queue_timeout_secs: 30,
                starvation_threshold_secs: 5,
                can_preempt: true,
            },
            Class::Interactive => Self {
                reserved_floor: 128,
                reserved_per_slot: 0.25,
                queue_size: 256,
                queue_timeout_secs: 30,
                starvation_threshold_secs: 5,
                can_preempt: true,
            },
            Class::Default => Self {
                reserved_floor: 0,
                reserved_per_slot: 0.10,
                queue_size: 512,
                queue_timeout_secs: 60,
                starvation_threshold_secs: 30,
                can_preempt: false,
            },
            Class::Bulk => Self {
                reserved_floor: 0,
                reserved_per_slot: 0.0,
                queue_size: 1024,
                queue_timeout_secs: 300,
                starvation_threshold_secs: 120,
                can_preempt: false,
            },
        }
    }
}

/// Runtime view of [`ClassConfig`] — only the fields the dispatcher
/// reads on its hot path, with seconds pre-converted to [`Duration`].
/// `reserved_floor`/`reserved_per_slot` and `queue_size` live elsewhere
/// (the packed-CAS array and the per-class queue impl), so they don't
/// appear here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassRuntimeConfig {
    pub queue_timeout: Duration,
    pub starvation_threshold: Duration,
    pub can_preempt: bool,
}

impl ClassRuntimeConfig {
    pub fn from_class_config(cfg: &ClassConfig) -> Self {
        Self {
            queue_timeout: Duration::from_secs(cfg.queue_timeout_secs),
            starvation_threshold: Duration::from_secs(cfg.starvation_threshold_secs),
            can_preempt: cfg.can_preempt,
        }
    }
}

/// Per-tenant policy entry in the YAML file.
///
/// Future fields (`weight`, `slot_quota`, `rps_cap`) are additive: adding
/// them is non-breaking because the trait
/// [`super::policy::TenantPolicyResolver`] returns the whole struct.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TenantPolicyConfig {
    pub max_class: Class,
}

/// Optional YAML config loaded via `--priority-scheduler-config <path>`.
///
/// Both maps are absent-as-empty: an empty document parses to
/// `PrioritySchedulerYaml::default()`, and downstream
/// [`SchedulerSettings::from_cli_and_yaml`] fills in built-in defaults
/// for any class that wasn't overridden.
///
/// A key none of these types know (a misspelt `tenant_polices:`, a class
/// field of another version) is a parse error naming it, as for the tenant
/// rate-limit YAML: a typo must not silently yield a scheduler without the
/// policies the operator wrote.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrioritySchedulerYaml {
    #[serde(default)]
    pub classes: HashMap<Class, ClassConfig>,
    #[serde(default)]
    pub tenant_policies: HashMap<String, TenantPolicyConfig>,
}

/// Per-field validation failures discovered while assembling
/// [`SchedulerSettings`]. Capacity-vs-reserved is not checked here: the
/// scheduler derives effective reservations from the floors + shares and
/// clamps them to the live capacity (priority-ordered), so reservations
/// can never exceed capacity and there is nothing to reject.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SettingsValidationError {
    #[error("class {class:?}: queue_timeout_secs must be > 0")]
    ZeroQueueTimeout { class: Class },
    #[error("class {class:?}: starvation_threshold_secs must be > 0")]
    ZeroStarvationThreshold { class: Class },
    #[error("class {class:?}: reserved_per_slot must be finite and >= 0")]
    InvalidReservedPerSlot { class: Class },
}

/// Runtime scheduler configuration assembled from CLI flags + the
/// optional YAML file + built-in defaults. Built once at startup and
/// then read-only.
///
/// Capacity is not stored here; the scheduler reads it live from
/// [`smg_worker::load::capacity::WorkerCapacity`] and reacts to changes via its
/// watch channel. Settings stay fixed.
#[derive(Debug, Clone)]
pub struct SchedulerSettings {
    /// Master switch. When `false`, the priority scheduler is not
    /// constructed and the gateway falls back to its legacy
    /// concurrency-limit admission path.
    pub enabled: bool,
    /// Tenant policy applied when a tenant is not in `tenant_policies`.
    pub default_max_class: Class,
    /// Per-class config, indexed by `class as usize`. Always populated
    /// for all four classes — YAML overrides land on top of
    /// [`ClassConfig::default_for`].
    classes: [ClassConfig; 4],
    /// Per-tenant clamp lookup. Keys come from
    /// [`crate::tenant::RouteRequestMeta::tenant_key`].
    pub tenant_policies: HashMap<TenantKey, TenantPolicyConfig>,
    /// Cap on the distinct tenants the scheduler's tenant-labelled counters
    /// (`smg_scheduler_clamp_total`, `smg_scheduler_unknown_priority_value_total`)
    /// keep as series of their own: the first tenant keys up to the cap keep
    /// theirs, everything past the cap is bucketed under `tenant="other"`.
    pub tenant_metric_top_n: u32,
}

impl SchedulerSettings {
    /// Indexed accessor for the per-class config.
    pub fn class_config(&self, class: Class) -> &ClassConfig {
        &self.classes[class as usize]
    }

    /// Assemble settings from CLI flags + optional YAML, validating
    /// per-field invariants. Reservations are clamped to the live capacity
    /// by the scheduler, so there is no capacity-vs-reserved check here.
    ///
    /// Merge order: built-in defaults → YAML overrides per class. CLI
    /// flags supply the top-level fields (`enabled`, `default_max_class`,
    /// `tenant_metric_top_n`) since they aren't representable in the YAML.
    pub fn from_cli_and_yaml(
        enabled: bool,
        default_max_class: Class,
        tenant_metric_top_n: u32,
        yaml: Option<&PrioritySchedulerYaml>,
    ) -> Result<Self, SettingsValidationError> {
        let mut classes: [ClassConfig; 4] = [
            ClassConfig::default_for(Class::Bulk),
            ClassConfig::default_for(Class::Default),
            ClassConfig::default_for(Class::Interactive),
            ClassConfig::default_for(Class::System),
        ];

        if let Some(yaml) = yaml {
            for (class, override_cfg) in &yaml.classes {
                classes[*class as usize] = *override_cfg;
            }
        }

        for class in Class::ALL {
            let cfg = &classes[class as usize];
            if cfg.queue_timeout_secs == 0 {
                return Err(SettingsValidationError::ZeroQueueTimeout { class });
            }
            if cfg.starvation_threshold_secs == 0 {
                return Err(SettingsValidationError::ZeroStarvationThreshold { class });
            }
            if !cfg.reserved_per_slot.is_finite() || cfg.reserved_per_slot < 0.0 {
                return Err(SettingsValidationError::InvalidReservedPerSlot { class });
            }
        }

        let tenant_policies = yaml
            .map(|y| {
                y.tenant_policies
                    .iter()
                    .map(|(k, v)| (TenantKey::new(k), *v))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            enabled,
            default_max_class,
            classes,
            tenant_policies,
            tenant_metric_top_n,
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/middleware/scheduler/config/tests.rs"]
mod tests;
