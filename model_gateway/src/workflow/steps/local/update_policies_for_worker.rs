//! Step to update policies for updated workers.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::workflow::data::WorkerUpdateWorkflowData;

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
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
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
                if (policy.name() == "cache_aware" || policy.name() == "cache_aware_length")
                    && !workers.is_empty()
                {
                    // Re-initialize cache-aware (or cache_aware_length) policy
                    // with updated workers.
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(model_id, &workers);

                    debug!(
                        "Updated {} policy for model {} ({} workers)",
                        policy.name(),
                        model_id,
                        workers.len()
                    );
                }
            }

            // Notify policy registry of the update
            app_context.policy_registry.on_worker_added(model_id, None);
        }

        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        let decode_workers = app_context.worker_registry.get_decode_workers();
        app_context
            .policy_registry
            .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
