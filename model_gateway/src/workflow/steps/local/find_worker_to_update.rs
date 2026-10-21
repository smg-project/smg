//! Step to find a worker to update based on URL.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::find_workers_by_url;
use crate::{worker::registry::WorkerId, workflow::data::WorkerUpdateWorkflowData};

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
            worker_url
        );

        if context.data.refresh_discovered_labels {
            // A refresh writes back against the registration found here, by
            // id and revision captured once: the read that follows may be
            // retried, and a worker updated, replaced, or removed and
            // registered again at the same address in the meantime must not
            // receive the stale rebuild. A worker gone before this line gets
            // an id nothing matches.
            let registry = std::sync::Arc::clone(&app_context.worker_registry);
            context.data.snapshot = Some(
                workers_to_update
                    .iter()
                    .map(|worker| {
                        let id = registry
                            .get_id_by_url(worker.url())
                            .unwrap_or_else(|| WorkerId::from_string(String::new()));
                        (id, worker.revision())
                    })
                    .collect(),
            );
        }

        context.data.workers_to_update = Some(workers_to_update);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
