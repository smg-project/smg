mod check_local_worker;
mod ensure_harmony_encoding;
mod find_worker_to_update;
mod forget_cache_events;
mod rediscover_labels;
mod release_tokenizer;
mod remove_from_policy_registry;
pub(crate) mod submit_tokenizer_job;
#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/local/test_support.rs"]
pub(crate) mod test_support;
mod update_policies_for_worker;
mod update_remaining_policies;
mod update_worker_properties;
mod zmq_labels;

use std::{sync::Arc, time::Duration};

pub(crate) use check_local_worker::{CheckLocalWorkerStep, CHECK_LOCAL_WORKER_STEP};
pub use ensure_harmony_encoding::EnsureHarmonyEncodingStep;
pub use find_worker_to_update::FindWorkerToUpdateStep;
use openai_protocol::worker::WorkerUpdateRequest;
pub use rediscover_labels::RediscoverLabelsStep;
pub use release_tokenizer::ReleaseTokenizerStep;
pub use remove_from_policy_registry::RemoveFromPolicyRegistryStep;
use smg_worker::{
    jobs::WorkerJobAsker,
    workflow::{
        single_attempt, worker_removal_workflow, RemovalOrder, WorkerRemovalRequest,
        REMOVE_FROM_WORKER_REGISTRY_STEP,
    },
};
pub use submit_tokenizer_job::SubmitTokenizerJobStep;
pub use update_policies_for_worker::UpdatePoliciesForWorkerStep;
pub use update_remaining_policies::UpdateRemainingPoliciesStep;
pub use update_worker_properties::UpdateWorkerPropertiesStep;
use wfaas::{BackoffStrategy, RetryPolicy, StepDefinition, WorkflowDefinition};
pub(crate) use zmq_labels::{DiscoverZmqLabelsStep, ZMQ_LABELS_STEP};

use crate::{
    app_context::AppContext,
    workflow::data::{WorkerRemovalWorkflowData, WorkerUpdateWorkflowData},
};

/// Create a worker removal workflow definition: the worker crate's removal
/// ([`worker_removal_workflow`]) with this gateway's two steps after it.
///
/// DAG structure:
/// ```text
///     find_workers_to_remove
///              │
///         drain_workers
///              │
///     remove_from_worker_registry
///                  │
///       +----------+----------+
///       │                     │
///       v                     v
///  remove_from_policy_registry   release_tokenizer
/// ```
/// Cache cleanup removes only departed tenants, so surviving workers do not
/// need the previous full-policy reinitialization pass. The tokenizer release
/// waits on the registry step alone: it reads the worker registry and the
/// record of worker-loaded entries, both settled there, and a policy cleanup
/// that fails or times out must not keep a departed model's tokenizer
/// resident (a failed step's dependents are never scheduled).
pub fn create_worker_removal_workflow() -> WorkflowDefinition<WorkerRemovalWorkflowData> {
    worker_removal_workflow(RemovalOrder::Registry)
        .add_step(
            StepDefinition::new(
                "remove_from_policy_registry",
                "Remove workers from policy registry",
                Arc::new(RemoveFromPolicyRegistryStep),
            )
            // The old 10s override was too short for large cache cleanup.
            // Inherit the workflow default after the worker is deregistered.
            .with_retry(single_attempt())
            .depends_on(&[REMOVE_FROM_WORKER_REGISTRY_STEP]),
        )
        .add_step(
            StepDefinition::new(
                "release_tokenizer",
                "Release the tokenizers of the models no worker serves any more",
                Arc::new(ReleaseTokenizerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            // On the registry step, not the policy cleanup: the release reads
            // what that step settled, and a cleanup that fails or times out
            // must not hold it back (a failed step's dependents never run).
            .depends_on(&[REMOVE_FROM_WORKER_REGISTRY_STEP]),
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
            .with_retry(single_attempt()),
        )
        .add_step(
            StepDefinition::new(
                "rediscover_labels",
                "Re-read the worker's labels from its engine (label refresh only)",
                Arc::new(RediscoverLabelsStep),
            )
            .with_timeout(Duration::from_secs(10))
            // An engine that just came back may still be warming up: one more
            // try before the refresh is left to the next transition.
            .with_retry(RetryPolicy {
                max_attempts: 2,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(3)),
            })
            .depends_on(&["find_worker_to_update"]),
        )
        .add_step(
            StepDefinition::new(
                "update_worker_properties",
                "Update worker properties",
                Arc::new(UpdateWorkerPropertiesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            .depends_on(&["rediscover_labels"]),
        )
        .add_step(
            StepDefinition::new(
                "update_policies_for_worker",
                "Update policies for updated worker",
                Arc::new(UpdatePoliciesForWorkerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            .depends_on(&["update_worker_properties"]),
        )
}

/// Helper to create initial workflow data for worker removal
pub fn create_worker_removal_workflow_data(
    url: String,
    expected_revisions: Option<std::collections::HashMap<String, Option<u64>>>,
    app_context: Arc<AppContext>,
    asker: WorkerJobAsker,
) -> WorkerRemovalWorkflowData {
    WorkerRemovalWorkflowData {
        config: WorkerRemovalRequest {
            url,
            expected_revisions,
        },
        asker,
        workers_to_remove: None,
        worker_urls: Vec::new(),
        affected_models: std::collections::HashSet::new(),
        host: Some(app_context),
        actual_workers_to_remove: None,
        drained_workers: Vec::new(),
        removed_workers: Vec::new(),
    }
}

/// Helper to create initial workflow data for worker update
pub fn create_worker_update_workflow_data(
    worker_url: String,
    update_config: WorkerUpdateRequest,
    app_context: Arc<AppContext>,
    asker: WorkerJobAsker,
) -> WorkerUpdateWorkflowData {
    WorkerUpdateWorkflowData {
        config: update_config,
        asker,
        worker_url,
        app_context: Some(app_context),
        workers_to_update: None,
        updated_workers: None,
        refresh_discovered_labels: false,
        fresh_engine_labels: None,
        snapshot: None,
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/local/tests.rs"]
mod tests;
