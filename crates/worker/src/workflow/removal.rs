//! The worker steps for removing workers through the registry or a host's store.

use std::{sync::Arc, time::Duration};

use wfaas::{StepDefinition, WorkflowDefinition};

use super::{
    data::WorkerRemovalWorkflowData,
    drain_workers::{DrainWorkersStep, SettleDrainedWorkersStep},
    find_workers_to_remove::FindWorkersToRemoveStep,
    host::WorkflowHost,
    remove_from_worker_registry::RemoveFromWorkerRegistryStep,
    retry::single_attempt,
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
/// The id of the step that waits the settle time behind a host's drain.
pub const SETTLE_DRAINED_WORKERS_STEP: &str = "settle_drained_workers";

/// The order of a removal a host runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalOrder {
    /// Select the workers, mark them and wait for them to settle, then
    /// remove them from the registry.
    Registry,
    /// Select workers and wait for them to settle after the host's drain.
    /// The host writes membership through its own store.
    Door {
        /// The id of the host's step that marks the workers `Draining`.
        drain: &'static str,
    },
}

/// The bound of a step that waits a settle window. The window is a worker's
/// own setting, known only at run time. A step with no timeout of its own
/// runs under the workflow's default, and a window longer than that default
/// would end the removal there and leave its workers draining. This bound is
/// one day, and a longer window ends the removal at it.
///
/// A host that waits for a removal waits at least this long. A host that
/// gave up earlier would report a failure while its run went on to remove
/// the worker.
pub const SETTLE_STEP_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// Builds the worker steps of a removal. Each step has one attempt.
///
/// Use [`RemovalOrder::Registry`] when the registry owns membership. Add
/// the host's cleanup steps after [`REMOVE_FROM_WORKER_REGISTRY_STEP`].
///
/// Use [`RemovalOrder::Door`] when the host's store owns membership. The
/// worker steps select workers and wait for them to settle. The host owns
/// all writes to the registry and to worker status in this order.
///
/// Use [`StepDefinition::depends_on`] to add the host's drain after
/// [`FIND_WORKERS_TO_REMOVE_STEP`] and the host's removal after
/// [`SETTLE_DRAINED_WORKERS_STEP`]. The named drain must be present at
/// registration and must fill [`WorkerRemovalWorkflowData::drained_workers`]
/// with the workers it marked.
pub fn worker_removal_workflow<H: WorkflowHost>(
    order: RemovalOrder,
) -> WorkflowDefinition<WorkerRemovalWorkflowData<H>> {
    let workflow = WorkflowDefinition::new(WORKER_REMOVAL_WORKFLOW, "Remove worker from router")
        .add_step(
            StepDefinition::new(
                FIND_WORKERS_TO_REMOVE_STEP,
                "Find workers to remove",
                Arc::new(FindWorkersToRemoveStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt()),
        );
    // Each wait carries its own bound: the window is known only at run
    // time, and the workflow's default timeout would cut a longer one.
    match order {
        RemovalOrder::Registry => workflow
            .add_step(
                StepDefinition::new(
                    DRAIN_WORKERS_STEP,
                    "Drain Ready workers before removal",
                    Arc::new(DrainWorkersStep),
                )
                .with_timeout(SETTLE_STEP_TIMEOUT)
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
            ),
        RemovalOrder::Door { drain } => workflow.add_step(
            StepDefinition::new(
                SETTLE_DRAINED_WORKERS_STEP,
                "Wait for drained workers before removal",
                Arc::new(SettleDrainedWorkersStep),
            )
            .with_timeout(SETTLE_STEP_TIMEOUT)
            .with_retry(single_attempt())
            .depends_on(&[drain]),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use tokio::time::Instant;
    use wfaas::{
        FunctionStep, StepExecutor, StepId, StepResult, ValidationError, WorkflowContext,
        WorkflowEngine, WorkflowId, WorkflowInstanceId,
    };

    use super::*;
    use crate::{
        identity::basic::Worker,
        jobs::WorkerJobAsker,
        test_support::{BasicWorkerBuilder, WorkerRegistry},
        worker::{HealthCheckConfig, WorkerStatus},
        workflow::{find_workers_to_remove::WorkerRemovalRequest, host::testing::RegistryHost},
    };

    const HOST_DRAIN: &str = "host_drain";
    const HOST_REMOVE: &str = "host_remove";
    const URL: &str = "http://worker:8000";

    type RemovalData = WorkerRemovalWorkflowData<RegistryHost>;

    /// What a host builds on: three steps in a line under the ids this crate
    /// names, the drain under the bound of a settle, each step tried once.
    #[test]
    fn the_registry_order_is_as_it_was() {
        let mut workflow = worker_removal_workflow::<RegistryHost>(RemovalOrder::Registry);
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
                    Some(SETTLE_STEP_TIMEOUT),
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

    #[test]
    fn the_door_order_holds_the_selection_and_the_settle_alone() {
        let workflow =
            worker_removal_workflow::<RegistryHost>(RemovalOrder::Door { drain: HOST_DRAIN });

        let ids: Vec<_> = workflow.steps.iter().map(|step| step.id.clone()).collect();
        assert_eq!(
            ids,
            [
                StepId::new("find_workers_to_remove"),
                StepId::new("settle_drained_workers"),
            ]
        );
        let settle = &workflow.steps[1];
        assert_eq!(settle.depends_on, [StepId::new(HOST_DRAIN)]);
        assert!(settle.depends_on_any.is_empty());
        assert_eq!(settle.timeout, Some(SETTLE_STEP_TIMEOUT));
        assert_eq!(
            settle
                .retry_policy
                .as_ref()
                .map(|policy| policy.max_attempts),
            Some(1)
        );
    }

    #[test]
    fn the_door_order_is_refused_when_the_host_adds_no_drain_step() {
        let order = RemovalOrder::Door { drain: HOST_DRAIN };
        let engine = WorkflowEngine::<RemovalData>::new();
        let error = engine
            .register_workflow(worker_removal_workflow(order))
            .unwrap_err();
        assert!(matches!(
            error,
            ValidationError::MissingDependency { step, dependency }
                if step == StepId::new(SETTLE_DRAINED_WORKERS_STEP)
                    && dependency == StepId::new(HOST_DRAIN)
        ));
        engine
            .register_workflow(worker_removal_workflow(order).add_step(host_step(
                HOST_DRAIN,
                FIND_WORKERS_TO_REMOVE_STEP,
                |_| {},
            )))
            .unwrap();
    }

    fn host_step(
        id: &str,
        dependency: &str,
        script: impl Fn(&mut RemovalData) + Send + Sync + 'static,
    ) -> StepDefinition<RemovalData> {
        StepDefinition::new(
            id,
            id,
            Arc::new(FunctionStep::new(
                move |context: &mut WorkflowContext<RemovalData>| {
                    script(&mut context.data);
                    Box::pin(std::future::ready(Ok(StepResult::Success)))
                },
            )),
        )
        .depends_on(&[dependency])
    }

    fn worker(url: &str, drain_settle_secs: u64, status: WorkerStatus) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .health_config(HealthCheckConfig {
                drain_settle_secs,
                ..Default::default()
            })
            .build();
        worker.set_status(status);
        Arc::new(worker)
    }

    fn removal_data() -> RemovalData {
        RemovalData {
            config: WorkerRemovalRequest {
                url: URL.to_string(),
                expected_revisions: None,
            },
            asker: WorkerJobAsker::Api,
            workers_to_remove: None,
            worker_urls: Vec::new(),
            affected_models: Default::default(),
            host: None,
            actual_workers_to_remove: None,
            drained_workers: Vec::new(),
            removed_workers: Vec::new(),
        }
    }

    async fn run_door_removal(
        registry: Arc<WorkerRegistry>,
        drain: impl Fn(&mut RemovalData) + Send + Sync + 'static,
        remove: impl Fn(&mut RemovalData) + Send + Sync + 'static,
    ) {
        let workflow = worker_removal_workflow(RemovalOrder::Door { drain: HOST_DRAIN })
            .add_step(host_step(HOST_DRAIN, FIND_WORKERS_TO_REMOVE_STEP, drain))
            .add_step(host_step(HOST_REMOVE, SETTLE_DRAINED_WORKERS_STEP, remove));
        let engine = WorkflowEngine::new();
        engine.register_workflow(workflow).unwrap();
        let mut data = removal_data();
        data.host = Some(Arc::new(RegistryHost {
            worker_registry: registry,
        }));
        let run = engine
            .start_workflow(WorkflowId::new(WORKER_REMOVAL_WORKFLOW), data)
            .await
            .unwrap();
        let timeout = Duration::from_secs(600);
        tokio::time::timeout(timeout, engine.wait_for_completion(run, "removal", timeout))
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn the_settle_waits_the_largest_window_of_the_drained_workers() {
        let drained = vec![
            worker("http://worker-a:8000", 2, WorkerStatus::Draining),
            worker("http://worker-b:8000", 7, WorkerStatus::Draining),
        ];
        for (workers, seconds) in [(drained, 7), (Vec::new(), 0)] {
            let registry = Arc::new(WorkerRegistry::new());
            registry
                .register(worker(URL, 11, WorkerStatus::Ready))
                .unwrap();
            let removed_at = Arc::new(OnceLock::new());
            let record_removal = Arc::clone(&removed_at);
            let start = Instant::now();
            run_door_removal(
                registry,
                move |data| data.drained_workers.clone_from(&workers),
                move |_| record_removal.set(Instant::now()).unwrap(),
            )
            .await;

            assert_eq!(
                *removed_at.get().unwrap() - start,
                Duration::from_secs(seconds)
            );
        }
    }

    /// A drain window longer than the workflow's default step timeout
    /// still ends in the removal: the worker leaves the registry once the
    /// window is over.
    #[tokio::test(start_paused = true)]
    async fn a_drain_window_past_the_default_step_timeout_ends_in_the_removal() {
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .register(worker(URL, 400, WorkerStatus::Ready))
            .unwrap();
        let engine = WorkflowEngine::new();
        engine
            .register_workflow(worker_removal_workflow::<RegistryHost>(
                RemovalOrder::Registry,
            ))
            .unwrap();
        let mut data = removal_data();
        data.host = Some(Arc::new(RegistryHost {
            worker_registry: Arc::clone(&registry),
        }));
        let start = Instant::now();
        let run = engine
            .start_workflow(WorkflowId::new(WORKER_REMOVAL_WORKFLOW), data)
            .await
            .unwrap();
        let bound = Duration::from_secs(600);
        tokio::time::timeout(bound, engine.wait_for_completion(run, "removal", bound))
            .await
            .unwrap()
            .unwrap();

        assert!(start.elapsed() >= Duration::from_secs(400));
        assert!(registry.get_by_url(URL).is_none());
    }

    /// The same window in the door's order: the host's removal runs once
    /// the window of the workers its drain marked is over.
    #[tokio::test(start_paused = true)]
    async fn a_settle_window_past_the_default_step_timeout_reaches_the_hosts_removal() {
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .register(worker(URL, 400, WorkerStatus::Ready))
            .unwrap();
        let drained = vec![worker(URL, 400, WorkerStatus::Draining)];
        let removed_at = Arc::new(OnceLock::new());
        let record_removal = Arc::clone(&removed_at);
        let start = Instant::now();
        run_door_removal(
            registry,
            move |data| data.drained_workers.clone_from(&drained),
            move |_| record_removal.set(Instant::now()).unwrap(),
        )
        .await;

        assert_eq!(*removed_at.get().unwrap() - start, Duration::from_secs(400));
    }

    #[tokio::test(start_paused = true)]
    async fn the_door_order_writes_no_membership() {
        let worker = worker(URL, 0, WorkerStatus::Ready);
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(Arc::clone(&worker)).unwrap();

        run_door_removal(Arc::clone(&registry), |_| {}, |_| {}).await;

        let registered = registry.get_by_url(URL).unwrap();
        assert!(Arc::ptr_eq(&registered, &worker));
        assert_eq!(registered.status(), WorkerStatus::Ready);
    }

    #[tokio::test(start_paused = true)]
    async fn the_settle_step_runs_without_a_host() {
        let mut data = removal_data();
        data.drained_workers
            .push(worker(URL, 0, WorkerStatus::Draining));
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        assert_eq!(
            SettleDrainedWorkersStep
                .execute(&mut context)
                .await
                .unwrap(),
            StepResult::Success
        );
    }
}
