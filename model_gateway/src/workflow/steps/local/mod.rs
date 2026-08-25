mod create_worker;
mod detect_backend;
mod detect_connection;
mod discover_dp;
mod discover_metadata;
mod drain_workers;
mod ensure_harmony_encoding;
mod find_worker_to_update;
mod find_workers_to_remove;
mod remove_from_policy_registry;
mod remove_from_worker_registry;
mod submit_tokenizer_job;
mod update_policies_for_worker;
mod update_remaining_policies;
mod update_worker_properties;

use std::{sync::Arc, time::Duration};

pub use create_worker::CreateLocalWorkerStep;
pub use detect_backend::DetectBackendStep;
pub use detect_connection::DetectConnectionModeStep;
pub use discover_dp::{get_dp_info, DiscoverDPInfoStep, DpInfo};
pub use discover_metadata::DiscoverMetadataStep;
pub use drain_workers::DrainWorkersStep;
pub use ensure_harmony_encoding::EnsureHarmonyEncodingStep;
pub use find_worker_to_update::FindWorkerToUpdateStep;
pub use find_workers_to_remove::{FindWorkersToRemoveStep, WorkerRemovalRequest};
use openai_protocol::worker::WorkerUpdateRequest;
pub use remove_from_policy_registry::RemoveFromPolicyRegistryStep;
pub use remove_from_worker_registry::RemoveFromWorkerRegistryStep;
pub use submit_tokenizer_job::SubmitTokenizerJobStep;
pub use update_policies_for_worker::UpdatePoliciesForWorkerStep;
pub use update_remaining_policies::UpdateRemainingPoliciesStep;
pub use update_worker_properties::UpdateWorkerPropertiesStep;
use wfaas::{BackoffStrategy, RetryPolicy, StepDefinition, WorkflowDefinition};

use crate::{
    app_context::AppContext,
    worker::{Worker, WorkerRegistry},
    workflow::data::{WorkerRemovalWorkflowData, WorkerUpdateWorkflowData},
};

/// Find workers by URL, supporting both DP-aware (prefix match) and regular (exact match) modes.
///
/// For DP-aware workers, finds all workers with URL prefix `{url}@`.
/// For regular workers, finds the single worker with exact URL match.
pub(crate) fn find_workers_by_url(
    registry: &WorkerRegistry,
    url: &str,
    dp_aware: bool,
) -> Vec<Arc<dyn Worker>> {
    if dp_aware {
        let worker_url_prefix = format!("{url}@");
        registry
            .get_all()
            .iter()
            .filter(|worker| worker.url().starts_with(&worker_url_prefix))
            .cloned()
            .collect()
    } else {
        match registry.get_by_url(url) {
            Some(worker) => vec![worker],
            None => Vec::new(),
        }
    }
}

/// Create a worker removal workflow definition.
///
/// DAG structure:
/// ```text
///     find_workers_to_remove
///              │
///         drain_workers
///              │
///     remove_from_worker_registry
///              │
///     remove_from_policy_registry
/// ```
/// Cache cleanup removes only departed tenants, so surviving workers do not
/// need the previous full-policy reinitialization pass.
pub fn create_worker_removal_workflow() -> WorkflowDefinition<WorkerRemovalWorkflowData> {
    WorkflowDefinition::new("worker_removal", "Remove worker from router")
        .add_step(
            StepDefinition::new(
                "find_workers_to_remove",
                "Find workers to remove",
                Arc::new(FindWorkersToRemoveStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "drain_workers",
                "Drain Ready workers before removal",
                Arc::new(DrainWorkersStep),
            )
            // No `with_timeout`: the step's purpose is to sleep for the
            // resolved `drain_settle_secs` (which can be set per-worker
            // via `WorkerSpec::health.drain_settle_secs`). A static
            // workflow-level timeout would be set at definition time
            // without visibility into runtime config and would
            // preemptively fail the workflow for legitimately long
            // drain windows, leaving workers stuck in `Draining`.
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["find_workers_to_remove"]),
        )
        .add_step(
            StepDefinition::new(
                "remove_from_worker_registry",
                "Remove workers from worker registry",
                Arc::new(RemoveFromWorkerRegistryStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["drain_workers"]),
        )
        .add_step(
            StepDefinition::new(
                "remove_from_policy_registry",
                "Remove workers from policy registry",
                Arc::new(RemoveFromPolicyRegistryStep),
            )
            // The old 10s override was too short for large cache cleanup.
            // Inherit the workflow default after the worker is deregistered.
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["remove_from_worker_registry"]),
        )
}

/// Create a worker update workflow definition.
///
/// DAG structure:
/// ```text
///     find_worker_to_update
///              │
///     update_worker_properties
///              │
///     update_policies_for_worker
/// ```
pub fn create_worker_update_workflow() -> WorkflowDefinition<WorkerUpdateWorkflowData> {
    WorkflowDefinition::new("worker_update", "Update worker properties")
        .add_step(
            StepDefinition::new(
                "find_worker_to_update",
                "Find worker to update",
                Arc::new(FindWorkerToUpdateStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "update_worker_properties",
                "Update worker properties",
                Arc::new(UpdateWorkerPropertiesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["find_worker_to_update"]),
        )
        .add_step(
            StepDefinition::new(
                "update_policies_for_worker",
                "Update policies for updated worker",
                Arc::new(UpdatePoliciesForWorkerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["update_worker_properties"]),
        )
}

/// Helper to create initial workflow data for worker removal
pub fn create_worker_removal_workflow_data(
    url: String,
    dp_aware: bool,
    expected_revision: Option<u64>,
    app_context: Arc<AppContext>,
) -> WorkerRemovalWorkflowData {
    WorkerRemovalWorkflowData {
        config: WorkerRemovalRequest {
            url,
            dp_aware,
            expected_revision,
        },
        workers_to_remove: None,
        worker_urls: Vec::new(),
        affected_models: std::collections::HashSet::new(),
        app_context: Some(app_context),
        actual_workers_to_remove: None,
    }
}

/// Helper to create initial workflow data for worker update
pub fn create_worker_update_workflow_data(
    worker_url: String,
    update_config: WorkerUpdateRequest,
    app_context: Arc<AppContext>,
) -> WorkerUpdateWorkflowData {
    // Determine if this is a DP-aware update based on URL pattern
    let dp_aware = worker_url.contains('@');
    WorkerUpdateWorkflowData {
        config: update_config,
        worker_url,
        dp_aware,
        app_context: Some(app_context),
        workers_to_update: None,
        updated_workers: None,
    }
}

#[cfg(test)]
mod tests {
    use wfaas::StepId;

    use super::*;

    #[test]
    fn worker_removal_deregisters_before_policy_cleanup() {
        let workflow = create_worker_removal_workflow();
        let registry_step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new("remove_from_worker_registry"))
            .unwrap();
        let policy_step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new("remove_from_policy_registry"))
            .unwrap();

        assert_eq!(registry_step.depends_on, vec![StepId::new("drain_workers")]);
        assert_eq!(
            policy_step.depends_on,
            vec![StepId::new("remove_from_worker_registry")]
        );
        assert_eq!(policy_step.timeout, None);
        assert!(workflow
            .steps
            .iter()
            .all(|step| step.id != StepId::new("update_remaining_policies")));
    }
}
