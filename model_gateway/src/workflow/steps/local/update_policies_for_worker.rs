//! Step to update policies for updated workers.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    policies::as_cache_aware,
    workflow::{
        data::{require_app_context, WorkerUpdateWorkflowData},
        steps::shared::start_kv_stream,
    },
};

/// Step to update policies for updated workers.
///
/// After workers are updated, this step re-initializes cache-aware policies
/// for the affected models to reflect any priority or label changes.
pub struct UpdatePoliciesForWorkerStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdatePoliciesForWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = require_app_context(context.data.app_context.as_ref())?;
        let updated_workers =
            context.data.updated_workers.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("updated_workers".to_string())
            })?;

        // Collect affected models
        let affected_models: HashSet<String> = updated_workers
            .iter()
            .map(|w| w.model_id().to_string())
            .collect();

        debug!(
            "Updating policies for {} affected model(s) after worker update",
            affected_models.len()
        );

        for model_id in &affected_models {
            let workers = app_context.worker_registry.get_by_model(model_id);

            if let Some(policy) = app_context.policy_registry.get_policy(model_id) {
                if as_cache_aware(policy.as_ref()).is_some() && !workers.is_empty() {
                    // Re-initialize cache-aware policy with updated workers
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(model_id, &workers);

                    debug!(
                        "Updated cache-aware policy for model {} ({} workers)",
                        model_id,
                        workers.len()
                    );
                }
            }

            // Only a registration builds a model's policy: an update that
            // lands after the model's last worker was removed must not leave
            // the model one for its next registration to inherit.
            for worker in workers.iter() {
                worker.set_completion_sink(app_context.policy_registry.worker_completion_sink());
            }
        }

        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        let decode_workers = app_context.worker_registry.get_decode_workers();
        app_context
            .policy_registry
            .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        // An update that rebuilt a worker holds a new object under its URL
        // (a label refresh is the common case: an engine that came into
        // service with labels its registration did not have). The stream
        // started for the object it replaced must follow it: the load
        // snapshot fences the pushed loads of any other object out, and its
        // receipts suppress the URL's poll. The same object again changes
        // nothing on the monitor. Last in the step, after the policy work:
        // a restart waits for the old task to end. An object the registry no
        // longer holds (removed, or replaced again since the previous step
        // swapped it in, or while the restart waited) gets no stream: the
        // monitor reads the registry under its slot lock before it stops or
        // starts anything, since a stream for such an object would run until
        // the URL registers again.
        for worker in updated_workers {
            let needs_kv_events = app_context
                .policy_registry
                .worker_needs_kv_events(worker.as_ref());
            start_kv_stream(app_context, worker, needs_kv_events).await;
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/local/update_policies_for_worker/tests.rs"]
mod tests;
