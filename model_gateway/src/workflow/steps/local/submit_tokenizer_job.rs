//! Tokenizer registration step for local workers.
//!
//! This step submits a Job::AddTokenizer to the job queue, which triggers the
//! tokenizer_registration workflow. The workflow handles validation, deduplication,
//! and caching - this step just submits the job.

use async_trait::async_trait;
use llm_tokenizer::TokenizerRegistry;
use tracing::{debug, info, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        Job, TokenizerConfigRequest, TokenizerLoadFailure,
    },
};

/// Step: Submit tokenizer registration job for the worker's model
///
/// This step submits a Job::AddTokenizer to the job queue rather than loading
/// the tokenizer directly. This ensures tokenizer registration goes through
/// the unified tokenizer_registration workflow.
pub struct SubmitTokenizerJobStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for SubmitTokenizerJobStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        // HTTP mode has no gRPC fallback and doesn't consult the tokenizer registry.
        if context.data.connection_mode == Some(ConnectionMode::Http) {
            return Ok(StepResult::Skip);
        }

        let labels = &context.data.final_labels;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers = context
            .data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        // Check if tokenizer loading is disabled
        if app_context.router_config.disable_tokenizer_autoload {
            debug!("Tokenizer loading is disabled, skipping tokenizer registration for worker");
            return Ok(StepResult::Success);
        }

        // Get job queue
        let job_queue = match app_context.worker_job_queue.get() {
            Some(queue) => queue,
            None => {
                warn!("Job queue not available, skipping tokenizer registration");
                return Ok(StepResult::Success);
            }
        };

        // Get chat_template: first model card in config > global router config
        let chat_template = context
            .data
            .config
            .models
            .primary()
            .and_then(|m| m.chat_template.clone())
            .or_else(|| app_context.router_config.chat_template.clone());

        // Get cache config from router config
        let cache_config = app_context.router_config.tokenizer_cache.to_option();

        for worker in workers {
            let model_id = worker.model_id().to_string();

            // Get tokenizer path with fallback chain (an empty value is no path):
            // 1. Worker labels: tokenizer_path
            // 2. Worker labels: model_path
            // 3. Router config (CLI args): --tokenizer-path
            // 4. Router config (CLI args): --model-path
            let usable = |path: &&String| !path.is_empty();
            let tokenizer_path: String = if let Some(path) = labels
                .get("tokenizer_path")
                .filter(usable)
                .or_else(|| labels.get("model_path").filter(usable))
            {
                path.clone()
            } else if let Some(path) = app_context
                .router_config
                .tokenizer_path
                .as_ref()
                .filter(usable)
                .or_else(|| app_context.router_config.model_path.as_ref().filter(usable))
            {
                debug!(
                    "Using router config tokenizer path '{}' for model {}",
                    path, model_id
                );
                path.clone()
            } else {
                // Nothing of this worker's loads the model's tokenizer. Unless
                // the model has one already (another worker's or the
                // operator's registration) or a load of it is in flight (that
                // load decides), record it, so that requests are answered with
                // the fact instead of a retry hint.
                let recorded = !app_context.tokenizer_registry.contains(&model_id)
                    && app_context
                        .tokenizer_load_failures
                        .record(&model_id, TokenizerLoadFailure::NoSource);
                warn!(
                    "No tokenizer_path or model_path found for model {} (checked worker labels and router config){}",
                    model_id,
                    if recorded {
                        "; its requests answer 503 tokenizer_load_failed until a tokenizer is registered"
                    } else {
                        ""
                    }
                );
                continue;
            };

            // Note: We don't check if tokenizer already exists here.
            // The registry.load() handles deduplication gracefully (returns AlreadyExists).
            // This simplifies the code and ensures consistent behavior.

            info!(
                "Submitting tokenizer registration job for model {} from {}",
                model_id, tokenizer_path
            );

            // Create tokenizer config request
            let config = TokenizerConfigRequest {
                id: TokenizerRegistry::generate_id(),
                name: model_id.clone(),
                source: tokenizer_path,
                chat_template_path: chat_template.clone(),
                cache_config: cache_config.clone(),
                fail_on_duplicate: false,
            };

            // Submit job (fire-and-forget, don't wait for completion); the
            // queue counts the accepted load in flight for the model.
            if let Err(e) = job_queue
                .submit(Job::AddTokenizer {
                    config: Box::new(config),
                })
                .await
            {
                warn!(
                    "Failed to submit tokenizer job for model {}: {}",
                    model_id, e
                );
                app_context.tokenizer_load_failures.record(
                    &model_id,
                    TokenizerLoadFailure::Failed(format!(
                        "the registration job was not accepted: {e}"
                    )),
                );
            }
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Job submission failures are not retryable at this level
    }
}
