//! Step to update policies for updated workers.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{policies::as_cache_aware, workflow::data::WorkerUpdateWorkflowData};

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

            // Notify policy registry of the update
            app_context.policy_registry.on_worker_added(model_id, None);
            for worker in workers.iter() {
                worker.set_completion_sink(Some(app_context.policy_registry.completion_sink()));
            }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::WorkerUpdateRequest;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::{PolicyConfig, RouterConfig},
        worker::{test_support::ready_worker, WorkerRegistry},
        workflow::steps::local::create_worker_update_workflow_data,
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

    /// The step routes the updated workers' model through its cache-aware
    /// tree: a tree that held nothing holds the model's workers after it.
    #[tokio::test]
    async fn the_step_routes_the_updated_workers_through_the_cache_aware_tree() {
        let app_context = context_with_cache_aware_model();
        let workers = vec![
            ready_worker("http://w1:8000", MODEL),
            ready_worker("http://w2:8000", MODEL),
        ];
        for worker in &workers {
            app_context
                .worker_registry
                .register(Arc::clone(worker))
                .unwrap();
        }
        assert!(
            tree(&app_context).is_empty(),
            "nothing routed before the step"
        );

        let update = WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: None,
            api_key: None,
            health: None,
        };
        let mut data = create_worker_update_workflow_data(
            "http://w1:8000".to_string(),
            update,
            Arc::clone(&app_context),
        );
        data.updated_workers = Some(workers);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesForWorkerStep
            .execute(&mut context)
            .await
            .unwrap();

        assert_eq!(tree(&app_context), ["http://w1:8000", "http://w2:8000"]);
    }
}
