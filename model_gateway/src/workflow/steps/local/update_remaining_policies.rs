//! Step to update cache-aware policies for remaining workers after removal.

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    policies::as_cache_aware, worker::endpoint::Endpoint, workflow::data::WorkerRemovalWorkflowData,
};

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
                if as_cache_aware(policy.as_ref()).is_some() && !remaining_workers.is_empty() {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::{PolicyConfig, RouterConfig},
        worker::{test_support::ready_worker, WorkerRegistry},
        workflow::steps::local::create_worker_removal_workflow_data,
    };

    const MODEL: &str = "llama-3";

    /// An app context whose policy for [`MODEL`] is cache-aware with an empty
    /// tree: the registry made the policy, nothing routed a worker through it.
    fn context_with_cache_aware_model() -> Arc<AppContext> {
        let config = RouterConfig::builder()
            .policy(PolicyConfig::RoundRobin)
            .build_unchecked();
        let app_context = app_context_for_tests(config, Arc::new(WorkerRegistry::new()), None);
        app_context
            .policy_registry
            .on_worker_added(MODEL, Some("cache_aware"));
        app_context
    }

    /// The worker URLs the model's cache-aware tree routes to.
    fn tree(app_context: &AppContext) -> Vec<String> {
        let policy = app_context
            .policy_registry
            .get_policy(MODEL)
            .expect("the model's policy");
        as_cache_aware(policy.as_ref())
            .expect("a cache-aware policy")
            .tree_tenants_for_test(MODEL)
    }

    /// The step routes the model's remaining workers through its cache-aware
    /// tree: with one worker gone from the registry, the tree holds the one
    /// that stays and never learns the one that left.
    #[tokio::test]
    async fn the_step_routes_the_remaining_workers_through_the_cache_aware_tree() {
        let app_context = context_with_cache_aware_model();
        for url in ["http://w1:8000", "http://w2:8000"] {
            app_context
                .worker_registry
                .register(ready_worker(url, MODEL))
                .unwrap();
        }
        app_context
            .worker_registry
            .remove_by_url("http://w2:8000")
            .expect("the leaving worker");
        assert!(
            tree(&app_context).is_empty(),
            "nothing routed before the step"
        );

        let mut data = create_worker_removal_workflow_data(
            "http://w2:8000".to_string(),
            None,
            Arc::clone(&app_context),
        );
        data.worker_urls = vec!["http://w2:8000".to_string()];
        data.affected_models = [MODEL.to_string()].into_iter().collect();
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdateRemainingPoliciesStep
            .execute(&mut context)
            .await
            .unwrap();

        assert_eq!(tree(&app_context), ["http://w1:8000"]);
    }
}
