//! Step to find workers to remove based on URL.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::WorkerRemovalWorkflowData,
    host::{require_host, WorkflowHost},
    removal::FIND_WORKERS_TO_REMOVE_STEP,
    select::find_workers_by_url,
};
use crate::identity::endpoint::Endpoint;

/// Request structure for worker removal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerRemovalRequest {
    pub url: String,
    /// Revision guards keyed by registry worker id, as observed when the
    /// removal was decided.
    ///
    /// A DP-aware URL expands into one worker per rank, each with its own
    /// independent revision, and `find_workers_by_url` returns the
    /// whole group for a base URL. A single revision therefore cannot guard
    /// the group: it would retain only the ranks that happen to share it and
    /// silently leave the rest registered against a Pod that is already gone.
    ///
    /// A worker missing from the map is a different incarnation than the one
    /// observed, so it is left alone. An id mapped to `None` is removed at
    /// whatever revision it holds: a `DELETE` names the id, and a worker
    /// rebuilt under it since (a `PATCH`, a `PUT`, a label refresh) is the
    /// one the `DELETE` named. `None` for the map removes every match
    /// unguarded.
    pub expected_revisions: Option<HashMap<String, Option<u64>>>,
}

/// Step to find workers to remove based on URL.
///
/// Uses registered URLs and DP base URLs, independent of the host's
/// current DP setting. Backends need not be reachable during removal.
pub struct FindWorkersToRemoveStep;

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRemovalWorkflowData<H>> for FindWorkersToRemoveStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        let registry = require_host(context.data.host.as_ref())?.worker_registry();

        let mut workers_to_remove = find_workers_by_url(registry, &request.url);

        if let Some(expected_revisions) = &request.expected_revisions {
            // Pin each worker to its own revision. An id absent from the map
            // was registered after the snapshot; a revision that moved was
            // replaced since. Either way that worker is skipped and the next
            // reconcile pass re-evaluates it, while its siblings still drain.
            // An id pinned at no revision is removed as it stands.
            workers_to_remove.retain(|worker| {
                registry
                    .get_id_by_url(worker.url())
                    .and_then(|id| expected_revisions.get(id.as_str()).copied())
                    .is_some_and(|expected| {
                        expected.is_none_or(|revision| revision == worker.revision())
                    })
            });
            if workers_to_remove.is_empty() {
                debug!(
                    worker_url = %Endpoint::redacted_str(&request.url),
                    ?expected_revisions,
                    "Skipping stale worker removal job after same-URL replacement"
                );
                context.data.keep_only(Vec::new());
                return Ok(StepResult::Success);
            }
        } else if workers_to_remove.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new(FIND_WORKERS_TO_REMOVE_STEP),
                message: format!("Worker {} not found", Endpoint::redacted_str(&request.url)),
            });
        }

        debug!(
            "Found {} worker(s) to remove for {}",
            workers_to_remove.len(),
            Endpoint::redacted_str(&request.url)
        );

        // The workers, their URLs and their models for the steps after this one.
        context.data.keep_only(workers_to_remove);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
