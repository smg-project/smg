//! Step to release the tokenizers of the models whose last worker left.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::submit_tokenizer_job::tokenizer_source;
use crate::{
    worker::{basic::merge_labels, Worker, WorkerRegistry},
    workflow::data::WorkerRemovalWorkflowData,
};

/// Step to release the tokenizer of a model no worker serves any more.
///
/// A worker's registration loads a tokenizer under its model id
/// (`SubmitTokenizerJobStep`) and nothing unloaded it: every retired model
/// id kept its tokenizer resident, tens of megabytes each, so model-id churn
/// grew the registry and the process without bound. Once the removed workers
/// have left the worker registry, every model they were registered with or
/// listed is asked after, as the policy step asks: a model no worker serves
/// any more loses the tokenizer registered under its id, with the multimodal
/// config that came with it, when that tokenizer's source is the one a
/// removed worker of the model loads it from (its labels and the router's
/// flags, as at registration), so an operator's registration of another
/// source under the same name stays. The model's next registration loads
/// the tokenizer again.
pub struct ReleaseTokenizerStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for ReleaseTokenizerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        // The workers found and the ones the registry step removed, since a
        // same-URL replacement during the drain is the one removed.
        let removed: Vec<&Arc<dyn Worker>> = context
            .data
            .actual_workers_to_remove
            .iter()
            .flatten()
            .chain(&context.data.removed_workers)
            .collect();
        let router = &app_context.router_config;
        let mut asked: HashSet<String> = HashSet::new();
        let mut released = 0usize;
        for worker in &removed {
            for model_id in WorkerRegistry::policy_model_ids(worker) {
                if !asked.insert(model_id.clone())
                    || app_context.worker_registry.contains_model(&model_id)
                {
                    continue;
                }
                let Some(entry) = app_context.tokenizer_registry.get_by_name(&model_id) else {
                    continue;
                };
                let loaded_by_a_worker = removed.iter().any(|worker| {
                    WorkerRegistry::policy_model_ids(worker).contains(&model_id) && {
                        let labels = merge_labels(
                            &worker.metadata().discovered_labels,
                            &worker.metadata().spec.labels,
                        );
                        tokenizer_source(
                            &labels,
                            router.tokenizer_path.as_deref(),
                            router.model_path.as_deref(),
                        ) == Some(entry.source.as_str())
                    }
                });
                if !loaded_by_a_worker {
                    debug!(
                        model = %model_id,
                        source = %entry.source,
                        "Tokenizer of a model no worker serves kept: not the source its workers load from"
                    );
                    continue;
                }
                if let Some(entry) = app_context.tokenizer_registry.remove(&model_id) {
                    app_context.multimodal_config_registry.remove(&entry.id);
                    released += 1;
                    info!(
                        model = %model_id,
                        tokenizer_id = %entry.id,
                        source = %entry.source,
                        "Released the tokenizer of a model no worker serves"
                    );
                }
            }
        }
        debug!(
            "Released {} tokenizer(s) of the removed workers' models",
            released
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer, TokenizerRegistry};
    use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

    use super::ReleaseTokenizerStep;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::RouterConfig,
        worker::{BasicWorkerBuilder, ModelCard, Worker, WorkerRegistry},
        workflow::steps::local::{
            create_worker_removal_workflow_data, RemoveFromPolicyRegistryStep,
            RemoveFromWorkerRegistryStep,
        },
    };

    fn app_context() -> Arc<AppContext> {
        app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        )
    }

    /// A registered worker of `model` whose `tokenizer_path` label names
    /// `source`, as a worker whose registration loaded the model's tokenizer.
    fn worker(app_context: &AppContext, url: &str, model: &str, source: &str) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .label("tokenizer_path", source)
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&worker));
        worker
    }

    /// A tokenizer registered under `name` from `source`, as a worker's
    /// registration job or the operator's `POST /v1/tokenizers` leaves it.
    async fn tokenizer(app_context: &AppContext, name: &str, source: &str) {
        app_context
            .tokenizer_registry
            .load(&TokenizerRegistry::generate_id(), name, source, || async {
                Ok(Arc::new(MockTokenizer::default()) as Arc<dyn Tokenizer>)
            })
            .await
            .unwrap();
    }

    /// The removal workflow's steps after the drain, in the workflow's order.
    async fn remove(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>) {
        let url = worker.url().to_string();
        let mut data =
            create_worker_removal_workflow_data(url.clone(), None, Arc::clone(app_context));
        data.worker_urls = vec![url];
        data.actual_workers_to_remove = Some(vec![worker]);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        RemoveFromWorkerRegistryStep
            .execute(&mut context)
            .await
            .unwrap();
        RemoveFromPolicyRegistryStep
            .execute(&mut context)
            .await
            .unwrap();
        ReleaseTokenizerStep.execute(&mut context).await.unwrap();
    }

    #[tokio::test]
    async fn the_tokenizer_leaves_with_the_models_last_worker() {
        let app_context = app_context();
        let first = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        let second = worker(&app_context, "http://w2:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/m").await;

        remove(&app_context, first).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "another worker serves the model"
        );
        remove(&app_context, second).await;
        assert!(
            !app_context.tokenizer_registry.contains("m"),
            "the model's last worker left"
        );
        assert!(app_context.tokenizer_registry.is_empty());
    }

    #[tokio::test]
    async fn an_operators_tokenizer_of_another_source_stays() {
        let app_context = app_context();
        let only = worker(&app_context, "http://w1:8000", "m", "/tok/m");
        tokenizer(&app_context, "m", "/tok/operator").await;

        remove(&app_context, only).await;
        assert!(
            app_context.tokenizer_registry.contains("m"),
            "not the source the model's workers load from"
        );
    }

    #[tokio::test]
    async fn every_model_the_worker_listed_is_asked_after() {
        let app_context = app_context();
        let endpoint: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://e:8000")
                .models(vec![ModelCard::new("a"), ModelCard::new("b")])
                .label("tokenizer_path", "/tok/shared")
                .build(),
        );
        app_context.worker_registry.register(Arc::clone(&endpoint));
        let other = worker(&app_context, "http://w2:8000", "b", "/tok/shared");
        tokenizer(&app_context, "a", "/tok/shared").await;
        tokenizer(&app_context, "b", "/tok/shared").await;

        remove(&app_context, endpoint).await;
        assert!(
            !app_context.tokenizer_registry.contains("a"),
            "no worker serves a any more"
        );
        assert!(
            app_context.tokenizer_registry.contains("b"),
            "the other worker serves b"
        );
        remove(&app_context, other).await;
        assert!(app_context.tokenizer_registry.is_empty());
    }
}
