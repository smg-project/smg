//! Step to find workers to remove based on URL.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::find_workers_by_url;
use crate::workflow::data::{WorkerList, WorkerRemovalWorkflowData};

/// Request structure for worker removal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerRemovalRequest {
    pub url: String,
    pub expected_revision: Option<u64>,
}

/// Step to find workers to remove based on URL.
///
/// Uses registered URLs and DP base URLs, independent of the gateway's
/// current DP setting. Backends need not be reachable during removal.
pub struct FindWorkersToRemoveStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for FindWorkersToRemoveStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let mut workers_to_remove =
            find_workers_by_url(&app_context.worker_registry, &request.url, true);

        if let Some(expected_revision) = request.expected_revision {
            workers_to_remove.retain(|worker| worker.revision() == expected_revision);
            if workers_to_remove.is_empty() {
                debug!(
                    worker_url = %request.url,
                    expected_revision,
                    "Skipping stale worker removal job after same-URL replacement"
                );
                context.data.workers_to_remove = Some(WorkerList::new());
                context.data.actual_workers_to_remove = Some(Vec::new());
                context.data.worker_urls = Vec::new();
                context.data.affected_models = HashSet::new();
                return Ok(StepResult::Success);
            }
        } else if workers_to_remove.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_workers_to_remove"),
                message: format!("Worker {} not found", request.url),
            });
        }

        debug!(
            "Found {} worker(s) to remove for {}",
            workers_to_remove.len(),
            request.url
        );

        // Store workers and their model IDs for subsequent steps
        let worker_urls: Vec<String> = workers_to_remove
            .iter()
            .map(|w| w.url().to_string())
            .collect();

        let affected_models: HashSet<String> = workers_to_remove
            .iter()
            .map(|w| w.model_id().to_string())
            .collect();

        // Update workflow data
        context.data.workers_to_remove = Some(WorkerList::from_workers(&workers_to_remove));
        context.data.actual_workers_to_remove = Some(workers_to_remove);
        context.data.worker_urls = worker_urls;
        context.data.affected_models = affected_models;

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
