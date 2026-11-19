//! The removal of a worker as a workflow: its steps, in this crate's order.

use std::{sync::Arc, time::Duration};

use wfaas::{StepDefinition, WorkflowDefinition};

use super::{
    data::WorkerRemovalWorkflowData, drain_workers::DrainWorkersStep,
    find_workers_to_remove::FindWorkersToRemoveStep, host::WorkflowHost,
    remove_from_worker_registry::RemoveFromWorkerRegistryStep, retry::single_attempt,
};

/// The id of the removal workflow.
pub const WORKER_REMOVAL_WORKFLOW: &str = "worker_removal";
/// The id of the step that selects the workers an address names.
pub const FIND_WORKERS_TO_REMOVE_STEP: &str = "find_workers_to_remove";
/// The id of the step that drains the selected workers.
pub const DRAIN_WORKERS_STEP: &str = "drain_workers";
/// The id of the step that takes the drained workers out of the registry.
/// What a host cleans up once a worker has left depends on this step.
pub const REMOVE_FROM_WORKER_REGISTRY_STEP: &str = "remove_from_worker_registry";

/// The worker side of a removal: the selection, the drain and the removal
/// from the registry, each tried once.
///
/// ```text
///     find_workers_to_remove
///              │
///         drain_workers
///              │
///     remove_from_worker_registry
/// ```
///
/// The order, the ids, the timeouts and the retries are this crate's, so
/// every host runs the same removal. A host adds the steps for what only it
/// keeps of a worker with [`WorkflowDefinition::add_step`], each made to
/// depend on [`REMOVE_FROM_WORKER_REGISTRY_STEP`].
pub fn worker_removal_workflow<H: WorkflowHost>() -> WorkflowDefinition<WorkerRemovalWorkflowData<H>>
{
    WorkflowDefinition::new(WORKER_REMOVAL_WORKFLOW, "Remove worker from router")
        .add_step(
            StepDefinition::new(
                FIND_WORKERS_TO_REMOVE_STEP,
                "Find workers to remove",
                Arc::new(FindWorkersToRemoveStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt()),
        )
        .add_step(
            StepDefinition::new(
                DRAIN_WORKERS_STEP,
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
            .with_retry(single_attempt())
            .depends_on(&[FIND_WORKERS_TO_REMOVE_STEP]),
        )
        .add_step(
            StepDefinition::new(
                REMOVE_FROM_WORKER_REGISTRY_STEP,
                "Remove workers from worker registry",
                Arc::new(RemoveFromWorkerRegistryStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            .depends_on(&[DRAIN_WORKERS_STEP]),
        )
}

#[cfg(test)]
mod tests {
    use wfaas::{StepId, WorkflowId};

    use super::*;
    use crate::workflow::host::testing::RegistryHost;

    /// What a host builds on: three steps in a line under the ids this crate
    /// names, the drain alone without a timeout, each step tried once.
    #[test]
    fn the_removal_is_three_steps_in_a_line_each_tried_once() {
        let mut workflow = worker_removal_workflow::<RegistryHost>();
        assert!(workflow.validate().is_ok());
        assert_eq!(workflow.id, WorkflowId::new(WORKER_REMOVAL_WORKFLOW));

        let ten_seconds = Some(Duration::from_secs(10));
        let steps: Vec<_> = workflow
            .steps
            .iter()
            .map(|step| (step.id.clone(), step.depends_on.clone(), step.timeout))
            .collect();
        assert_eq!(
            steps,
            vec![
                (
                    StepId::new(FIND_WORKERS_TO_REMOVE_STEP),
                    vec![],
                    ten_seconds
                ),
                (
                    StepId::new(DRAIN_WORKERS_STEP),
                    vec![StepId::new(FIND_WORKERS_TO_REMOVE_STEP)],
                    None,
                ),
                (
                    StepId::new(REMOVE_FROM_WORKER_REGISTRY_STEP),
                    vec![StepId::new(DRAIN_WORKERS_STEP)],
                    ten_seconds,
                ),
            ]
        );
        assert!(workflow.steps.iter().all(|step| step
            .retry_policy
            .as_ref()
            .is_some_and(|policy| policy.max_attempts == 1)));
    }
}
