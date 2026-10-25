//! Tokenizer registration step for local workers.
//!
//! This step submits a Job::AddTokenizer to the job queue, which triggers the
//! tokenizer_registration workflow. The workflow handles validation, deduplication,
//! and caching - this step just submits the job.

use std::collections::HashMap;

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

            let tokenizer_path: String = if let Some(source) = tokenizer_source(
                labels,
                app_context.router_config.tokenizer_path.as_deref(),
                app_context.router_config.model_path.as_deref(),
            ) {
                source.to_string()
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
                    "No tokenizer source for model {} (checked the worker's tokenizer_path, model_path and model_id labels and the router config){}",
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

/// The source a model's tokenizer is loaded from, in order (an empty value is
/// no source): the worker's `tokenizer_path` label, its `model_path` label,
/// the router's `--tokenizer-path`, its `--model-path`, and last the worker's
/// `model_id` label. A dialect whose model info names the model by its id
/// only (a path or hub id on the worker's side) still gets a job that way:
/// for a gRPC worker a local load that fails on the id falls back to the
/// worker's own bundle (`fetch_tokenizer_from_worker`); a ZMQ worker's id has
/// to be a path or hub id the gateway itself can load. Coming last, the id
/// never displaces an operator's flag.
fn tokenizer_source<'a>(
    labels: &'a HashMap<String, String>,
    router_tokenizer_path: Option<&'a str>,
    router_model_path: Option<&'a str>,
) -> Option<&'a str> {
    let usable = |path: &&str| !path.is_empty();
    let label = |key: &str| labels.get(key).map(String::as_str).filter(usable);
    label("tokenizer_path")
        .or_else(|| label("model_path"))
        .or_else(|| router_tokenizer_path.filter(usable))
        .or_else(|| router_model_path.filter(usable))
        .or_else(|| label("model_id"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::tokenizer_source;

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    /// A worker that names its model by id only, with neither a path label
    /// nor a router flag: the id is the source (a local load that fails on
    /// it falls back to the worker's bundle); an empty id is no source.
    #[test]
    fn tokenizer_source_is_the_model_id_when_nothing_else_names_a_path() {
        let id_only = labels(&[("model_id", "/engines/served")]);
        assert_eq!(
            tokenizer_source(&id_only, None, None),
            Some("/engines/served")
        );
        assert_eq!(
            tokenizer_source(&id_only, Some(""), Some("")),
            Some("/engines/served")
        );
        assert_eq!(
            tokenizer_source(&labels(&[("model_id", "")]), None, None),
            None
        );
        assert_eq!(tokenizer_source(&labels(&[]), None, None), None);
    }

    /// The operator's flags and the worker's path labels keep their
    /// precedence over the id: `--tokenizer-path`, then `--model-path`, win
    /// over an id-only worker, and a `tokenizer_path` or `model_path` label
    /// wins over the flags as before.
    #[test]
    fn tokenizer_source_reads_the_model_id_only_after_the_flags_and_path_labels() {
        let id_only = labels(&[("model_id", "/engines/served")]);
        assert_eq!(
            tokenizer_source(&id_only, Some("/opt/tok"), None),
            Some("/opt/tok")
        );
        assert_eq!(
            tokenizer_source(&id_only, None, Some("/opt/model")),
            Some("/opt/model")
        );
        assert_eq!(
            tokenizer_source(&id_only, Some("/opt/tok"), Some("/opt/model")),
            Some("/opt/tok")
        );

        let with_paths = labels(&[
            ("model_id", "/engines/served"),
            ("model_path", "/engines/weights"),
            ("tokenizer_path", "/engines/tok"),
        ]);
        assert_eq!(
            tokenizer_source(&with_paths, Some("/opt/tok"), None),
            Some("/engines/tok")
        );
        let with_model_path = labels(&[
            ("model_id", "/engines/served"),
            ("model_path", "/engines/weights"),
        ]);
        assert_eq!(
            tokenizer_source(&with_model_path, Some("/opt/tok"), None),
            Some("/engines/weights")
        );
    }
}
