//! Step to find a worker to update based on URL.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::find_workers_by_url;
use crate::workflow::data::WorkerUpdateWorkflowData;

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
        let dp_aware = context.data.dp_aware;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let workers_to_update =
            find_workers_by_url(&app_context.worker_registry, worker_url, dp_aware);

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

        context.data.workers_to_update = Some(workers_to_update);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
