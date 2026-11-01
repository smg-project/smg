//! Step to update cache-aware policies for remaining workers after removal.

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{worker::endpoint::Endpoint, workflow::data::WorkerRemovalWorkflowData};

/// Step to update cache-aware policies for remaining workers.
///
/// After workers are removed, this step re-initializes cache-aware policies
/// for the affected models using the remaining workers.
pub struct UpdateRemainingPoliciesStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for UpdateRemainingPoliciesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let affected_models = &context.data.affected_models;
        let worker_urls = &context.data.worker_urls;

        debug!(
            "Updating cache-aware policies for {} affected model(s)",
            affected_models.len()
        );

        for model_id in affected_models {
            let remaining_workers = app_context.worker_registry.get_by_model(model_id);

            if let Some(policy) = app_context.policy_registry.get_policy(model_id) {
                if policy.name() == "cache_aware" && !remaining_workers.is_empty() {
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(model_id, &remaining_workers);

                    debug!(
                        "Updated cache-aware policy for model {} ({} remaining workers)",
                        model_id,
                        remaining_workers.len()
                    );
                }
            }
        }

        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        let decode_workers = app_context.worker_registry.get_decode_workers();
        app_context
            .policy_registry
            .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        // Log final result at info level
        if worker_urls.len() == 1 {
            info!("Removed worker {}", Endpoint::redacted_str(&worker_urls[0]));
        } else {
            info!(
                "Removed {} DP-aware workers: {:?}",
                worker_urls.len(),
                worker_urls
                    .iter()
                    .map(|url| Endpoint::redacted_str(url))
                    .collect::<Vec<_>>()
            );
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
