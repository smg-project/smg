//! Step to find a worker to update based on URL.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::find_workers_by_url;
use crate::{
    worker::{
        endpoint::Endpoint,
        registry::{WorkerId, WorkerRegistry},
    },
    workflow::data::WorkerUpdateWorkflowData,
};

/// Step to find workers to update based on URL.
///
/// A DP rank URL selects that rank. Other updates use an exact registered
/// URL lookup.
pub struct FindWorkerToUpdateStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for FindWorkerToUpdateStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let worker_url = &context.data.worker_url;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        // Whether this address has expanded ranks is read from the registry,
        // not guessed from the URL, so an update selects the same group a
        // removal would.
        let workers_to_update = find_workers_by_url(&app_context.worker_registry, worker_url);

        if workers_to_update.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_worker_to_update"),
                message: format!("Worker {worker_url} not found"),
            });
        }

        debug!(
            "Found {} worker(s) to update for {}",
            workers_to_update.len(),
            Endpoint::redacted_str(worker_url)
        );

        // The update writes back against the registration found here, by id
        // and revision captured once: a worker updated, replaced (a provider
        // registering its address included), or removed and registered again
        // at the same address in the meantime must not receive a rebuild of
        // what was found here. A refresh's engine read in between may be
        // retried, which widens the window. A worker gone before this line
        // gets an id nothing matches.
        let registry = std::sync::Arc::clone(&app_context.worker_registry);
        let snapshot: Vec<(WorkerId, u64)> = workers_to_update
            .iter()
            .map(|worker| {
                let id = registry
                    .get_id_by_url(worker.url())
                    .unwrap_or_else(|| WorkerId::from_string(String::new()));
                (id, worker.revision())
            })
            .collect();

        #[cfg(test)]
        if let Some(write) = BETWEEN_SNAPSHOT_AND_CHECK.with(|hook| hook.borrow_mut().take()) {
            write();
        }

        // The request named an id and resolved it to this URL when it was
        // accepted. A worker, or a data-parallel group, removed and
        // registered again at the address since then has an id of its own,
        // and the update accepted for the one that left does not land on
        // it. Read after the snapshot: a replacement that lands before this
        // line stops the id resolving; one that lands after it leaves the
        // snapshot on the registration that left, which the write-back's id
        // and revision check refuses.
        if !update_target_still_holds(&registry, worker_url, context.data.expected_id.as_deref()) {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_worker_to_update"),
                message: format!(
                    "Worker {} was removed or replaced since the update was accepted",
                    Endpoint::redacted_str(worker_url)
                ),
            });
        }

        context.data.snapshot = Some(snapshot);
        context.data.workers_to_update = Some(workers_to_update);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
thread_local! {
    /// Test support: a write the step runs once between its snapshot and
    /// its id check, the window a replacement of the registration can land
    /// in (the test runtime runs the step on this thread).
    static BETWEEN_SNAPSHOT_AND_CHECK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether the registration an update was accepted against still stands:
/// `expected_id`, the id the request resolved to `worker_url`, must still
/// resolve to it. A worker's id leaves with the worker and a data-parallel
/// group's with its last row, and a registration after that at the same
/// address has a fresh one. A refresh names no id and holds.
fn update_target_still_holds(
    registry: &WorkerRegistry,
    worker_url: &str,
    expected_id: Option<&str>,
) -> bool {
    expected_id.is_none_or(|id| {
        registry
            .get_url_by_id(&WorkerId::from_string(id.to_string()))
            .is_some_and(|url| url == worker_url)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::{model_card::ModelCard, worker::WorkerUpdateRequest};
    use wfaas::{StepExecutor, WorkflowContext, WorkflowError, WorkflowInstanceId};

    use super::{update_target_still_holds, FindWorkerToUpdateStep};
    use crate::{
        app_context::testing::app_context_for_tests,
        config::RouterConfig,
        worker::{BasicWorkerBuilder, Worker, WorkerRegistry},
        workflow::steps::local::create_worker_update_workflow_data,
    };

    fn worker(url: &str, model: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .build(),
        )
    }

    fn row(url: &str, rank: usize) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .dp_config(rank, 2)
                .build(),
        )
    }

    /// A PATCH accepted for a worker that is then removed and registered
    /// again at its address (a DELETE and a POST while the job waits its
    /// turn): the id it was accepted against no longer resolves to the URL,
    /// and the new registration's id does. A label refresh names no id.
    #[test]
    fn an_update_holds_while_its_id_resolves_to_the_address() {
        let registry = WorkerRegistry::new();
        let url = "http://worker:8080";
        let original = registry.register(worker(url, "original-model")).unwrap();
        assert!(update_target_still_holds(
            &registry,
            url,
            Some(original.as_str())
        ));
        assert!(
            update_target_still_holds(&registry, url, None),
            "a refresh names no id"
        );

        assert!(registry.remove(&original).is_some());
        assert!(
            !update_target_still_holds(&registry, url, Some(original.as_str())),
            "the worker left"
        );
        let recreated = registry.register(worker(url, "recreated-model")).unwrap();
        assert_ne!(recreated, original);
        assert!(!update_target_still_holds(
            &registry,
            url,
            Some(original.as_str())
        ));
        assert!(update_target_still_holds(
            &registry,
            url,
            Some(recreated.as_str())
        ));
    }

    /// The same for a data-parallel group named by its own id, the one its
    /// Location named: the id resolves to the engine's URL while a row of
    /// the group stands, and a group registered at the address after the
    /// last row left has an id of its own.
    #[test]
    fn a_groups_update_holds_while_its_id_resolves_to_the_address() {
        let registry = WorkerRegistry::new();
        let url = "http://10.0.0.20:8000";
        let group = registry
            .reserve_for_create(url)
            .expect("the address is free");
        let rows = [
            registry.register(row(url, 0)).unwrap(),
            registry.register(row(url, 1)).unwrap(),
        ];
        assert!(update_target_still_holds(
            &registry,
            url,
            Some(group.as_str())
        ));

        for id in &rows {
            assert!(registry.remove(id).is_some());
        }
        assert!(
            !update_target_still_holds(&registry, url, Some(group.as_str())),
            "the group left"
        );
        let next = registry
            .reserve_for_create(url)
            .expect("the address is free again");
        registry.register(row(url, 0)).unwrap();
        assert_ne!(next, group);
        assert!(!update_target_still_holds(
            &registry,
            url,
            Some(group.as_str())
        ));
        assert!(update_target_still_holds(
            &registry,
            url,
            Some(next.as_str())
        ));
    }

    /// The step: an update accepted against the registration that left
    /// fails before it finds the one standing at the address; the update
    /// accepted against that one finds it and snapshots it.
    #[tokio::test]
    async fn the_step_refuses_an_update_whose_registration_was_replaced() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "http://worker:8080";
        let original = registry.register(worker(url, "original-model")).unwrap();
        assert!(registry.remove(&original).is_some());
        let recreated = registry.register(worker(url, "recreated-model")).unwrap();
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .build_unchecked(),
            Arc::clone(&registry),
            None,
        );
        let data_for = |id: &str| {
            let request = WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            };
            let mut data = create_worker_update_workflow_data(
                url.to_string(),
                request,
                Arc::clone(&app_context),
            );
            data.expected_id = Some(id.to_string());
            data
        };

        let mut context =
            WorkflowContext::new(WorkflowInstanceId::new(), data_for(original.as_str()));
        let result = FindWorkerToUpdateStep.execute(&mut context).await;
        assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
        assert!(context.data.workers_to_update.is_none());

        let mut context =
            WorkflowContext::new(WorkflowInstanceId::new(), data_for(recreated.as_str()));
        FindWorkerToUpdateStep.execute(&mut context).await.unwrap();
        let found = context
            .data
            .workers_to_update
            .expect("the standing registration is found");
        assert_eq!(found.len(), 1);
        assert_eq!(
            context.data.snapshot,
            Some(vec![(recreated, found[0].revision())])
        );
    }

    /// The window between the snapshot and the id check: the registration
    /// is removed and registered again at its address right there. The
    /// snapshot holds the registration that left, so the check refuses the
    /// update. Checked before the snapshot, the new registration was
    /// snapshotted under its own ids and the write-back took the old update.
    #[tokio::test]
    async fn a_replacement_between_the_snapshot_and_the_check_is_refused() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "http://worker:8080";
        let original = registry.register(worker(url, "original-model")).unwrap();
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .build_unchecked(),
            Arc::clone(&registry),
            None,
        );
        let mut data = create_worker_update_workflow_data(
            url.to_string(),
            WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            Arc::clone(&app_context),
        );
        data.expected_id = Some(original.as_str().to_string());
        let between = Arc::clone(&registry);
        super::BETWEEN_SNAPSHOT_AND_CHECK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                assert!(between.remove(&original).is_some());
                between.register(worker(url, "recreated-model")).unwrap();
            }));
        });

        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        let result = FindWorkerToUpdateStep.execute(&mut context).await;

        assert!(
            super::BETWEEN_SNAPSHOT_AND_CHECK.with(|hook| hook.borrow().is_none()),
            "the step ran the seam"
        );
        assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
        assert!(context.data.workers_to_update.is_none());
        assert!(context.data.snapshot.is_none());
    }
}
