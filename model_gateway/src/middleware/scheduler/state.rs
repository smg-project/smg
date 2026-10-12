//! Startup wiring for the priority scheduler: builds the admission mode
//! the route layer branches on.

use std::{sync::Arc, time::Duration};

use tracing::{error, info};

use super::{
    Class, PriorityScheduler, SchedulerSettings, StaticTenantPolicyResolver, TenantPolicyResolver,
};
use crate::{
    config::types::RouterConfig,
    middleware::token_bucket::TokenBucket,
    worker::{CapacityTrackerSettings, WorkerCapacity, WorkerRegistry},
};

/// How often the metrics sampler refreshes the capacity / autoscaling gauges.
const SAMPLER_INTERVAL: Duration = Duration::from_secs(5);

/// State handed to `priority_admission_middleware` via `from_fn_with_state`.
/// Cheap to clone (all `Arc`).
pub struct SchedulerState {
    pub scheduler: Arc<PriorityScheduler>,
    pub resolver: Arc<dyn TenantPolicyResolver>,
    /// Per-second RPS sibling check, run before admission. Set only when an
    /// explicit `rate_limit_tokens_per_second` is configured; the bucket's
    /// concurrency-cap role is owned by the scheduler, so we must not consult
    /// it as a concurrency limiter (that would double-limit). `None` =
    /// no RPS limit.
    pub rate_limiter: Option<Arc<TokenBucket>>,
}

/// Which admission path the protected routes use. Chosen once at startup.
#[derive(Clone)]
pub enum AdmissionMode {
    /// Legacy `concurrency_limit_middleware` (default; zero behavior change).
    Legacy,
    /// Priority scheduler enabled.
    Priority(Arc<SchedulerState>),
}

impl AdmissionMode {
    /// Build the admission mode from runtime config.
    ///
    /// When `priority_scheduler_enabled` is false, returns `Legacy` without
    /// constructing anything. When true, constructs `WorkerCapacity` over
    /// the worker fleet, builds the scheduler against its current capacity,
    /// spawns the dispatcher on its watch channel, and returns
    /// `Priority(..)`.
    ///
    /// The YAML file is checked by `ConfigValidator` before start-up, so a
    /// file that cannot be read, parsed or validated is a configuration
    /// error the gateway refuses to start on. What can still fail here is
    /// the scheduler against the fleet's initial capacity (reservations
    /// exceed it): that is logged at ERROR and falls back to `Legacy`
    /// rather than aborting the whole gateway — a too-small fleet must not
    /// take the data plane down.
    pub fn from_config(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
    ) -> Self {
        if !rc.priority_scheduler_enabled {
            return Self::Legacy;
        }
        match Self::try_build_priority(rc, registry, rate_limiter) {
            Ok(mode) => {
                info!("priority scheduler enabled");
                mode
            }
            Err(e) => {
                error!(
                    error = %e,
                    "priority scheduler failed to start; falling back to legacy admission"
                );
                Self::Legacy
            }
        }
    }

    fn try_build_priority(
        rc: &RouterConfig,
        registry: Arc<WorkerRegistry>,
        rate_limiter: Option<Arc<TokenBucket>>,
    ) -> Result<Self, String> {
        // Tier-4 fallback for WorkerCapacity comes from the legacy
        // --max-concurrent-requests (clamped to u16; <=0 means "disabled",
        // for which we keep the tracker default).
        let cap_settings = CapacityTrackerSettings {
            legacy_max_concurrent_requests: u16::try_from(rc.max_concurrent_requests)
                .unwrap_or_else(|_| {
                    CapacityTrackerSettings::default().legacy_max_concurrent_requests
                }),
            ..CapacityTrackerSettings::default()
        };
        let worker_capacity = WorkerCapacity::spawn(registry, cap_settings);
        // Keep a receiver alive as soon as the tracker exists. Tokio's
        // `watch::Sender::send` does not retain a value when no receiver is
        // subscribed, so parsing the scheduler configuration must not create
        // a gap where the first fleet update is lost.
        let capacity_watch = worker_capacity.watch();

        let settings = settings_from_config(rc).map_err(|e| e.to_string())?;

        // The atomic value covers any update that won the race before the
        // receiver subscribed; subsequent updates remain queued for the
        // dispatcher through `capacity_watch`.
        let scheduler = PriorityScheduler::new(&settings, worker_capacity.current())
            .map_err(|e| e.to_string())?;
        scheduler.spawn_dispatcher_retaining_capacity(capacity_watch, worker_capacity);
        scheduler.spawn_sampler(SAMPLER_INTERVAL);

        let resolver: Arc<dyn TenantPolicyResolver> =
            Arc::new(StaticTenantPolicyResolver::from_settings(&settings));

        // The scheduler owns concurrency, so the shared bucket only survives
        // as an RPS check when an explicit per-second limit is configured.
        let rate_limiter = match rc.rate_limit_tokens_per_second {
            Some(rps) if rps > 0 => rate_limiter,
            _ => None,
        };

        Ok(Self::Priority(Arc::new(SchedulerState {
            scheduler,
            resolver,
            rate_limiter,
        })))
    }
}

/// Why a router configuration's scheduler settings are refused: the
/// configuration field at fault, its value and the reason, the shape
/// `ConfigValidator` reports a start-up error in.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field} `{value}`: {reason}")]
pub struct SchedulerConfigError {
    /// The `RouterConfig` field.
    pub field: &'static str,
    /// Its configured value (the file's path for the YAML's errors).
    pub value: String,
    pub reason: String,
}

/// The scheduler settings a router configuration describes: the CLI fields
/// plus the optional YAML file, read, parsed and validated. `ConfigValidator`
/// runs this before start-up and the admission mode builds on it, so both see
/// the same configuration the same way. The error names the field at fault
/// and what is wrong with it: a default max class that is none of the four
/// names (the header parser's fallback to `default` is for requests, not for
/// an operator's flag), a file that cannot be read or parsed (an unknown key
/// included) or a value the settings refuse.
pub fn settings_from_config(rc: &RouterConfig) -> Result<SchedulerSettings, SchedulerConfigError> {
    let default_max_class =
        Class::parse_name(&rc.priority_scheduler_default_max_class).map_err(|e| {
            SchedulerConfigError {
                field: "priority_scheduler_default_max_class",
                value: rc.priority_scheduler_default_max_class.clone(),
                reason: e.to_string(),
            }
        })?;
    let yaml = load_yaml(rc.priority_scheduler_config.as_deref())
        .map_err(|reason| file_error(rc, reason))?;
    SchedulerSettings::from_cli_and_yaml(
        true,
        default_max_class,
        rc.priority_scheduler_tenant_metric_top_n,
        yaml.as_ref(),
    )
    .map_err(|e| file_error(rc, e.to_string()))
}

/// A refusal of the YAML file (or of the settings it yields), named after
/// the path field.
fn file_error(rc: &RouterConfig, reason: String) -> SchedulerConfigError {
    SchedulerConfigError {
        field: "priority_scheduler_config",
        value: rc.priority_scheduler_config.clone().unwrap_or_default(),
        reason,
    }
}

/// Load + parse the optional priority-scheduler YAML file.
fn load_yaml(path: Option<&str>) -> Result<Option<super::PrioritySchedulerYaml>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let contents = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
    let parsed = serde_yaml::from_str(&contents).map_err(|e| format!("parsing {path}: {e}"))?;
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Instant};

    use smg_auth::RequestId;
    use tokio::time::{sleep, Duration};

    use super::*;
    use crate::worker::BasicWorkerBuilder;

    #[tokio::test]
    async fn priority_mode_applies_capacity_changes_after_startup() {
        let registry = Arc::new(WorkerRegistry::new());
        let config = RouterConfig {
            max_concurrent_requests: 256,
            priority_scheduler_enabled: true,
            ..RouterConfig::default()
        };
        let AdmissionMode::Priority(state) =
            AdmissionMode::try_build_priority(&config, Arc::clone(&registry), None).unwrap()
        else {
            panic!("priority scheduler should start");
        };

        let mut labels = HashMap::new();
        labels.insert("max_running_requests".to_string(), "1".to_string());
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://capacity-test:8000")
                .labels(labels)
                .status(openai_protocol::worker::WorkerStatus::Ready)
                .build(),
        );
        registry.register(worker).expect("worker should register");

        // The dispatcher must retain the tracker and apply its update. Once
        // capacity drops to one, a second System request cannot acquire a
        // slot while the first is still in flight.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut attempt = 0;
        loop {
            let first = state.scheduler.acquire_inflight(
                Class::System,
                RequestId(format!("capacity-first-{attempt}")),
            );
            let second = state.scheduler.acquire_inflight(
                Class::System,
                RequestId(format!("capacity-second-{attempt}")),
            );
            let updated = first.is_some() && second.is_none();
            drop(second);
            drop(first);

            if updated {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "scheduler did not apply the worker capacity update"
            );
            attempt += 1;
            sleep(Duration::from_millis(10)).await;
        }
    }
}
