//! Prepare the gateway's tokenizer before publishing a token-inference worker.

use async_trait::async_trait;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{require_app_context, WorkerKind, WorkerWorkflowData},
        tokenizer_preparation::{prepare_worker_tokenizers, TokenizerPreparation},
    },
};

pub struct PrepareTokenizerStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for PrepareTokenizerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = require_app_context(context.data.app_context.as_ref())?.clone();
        if !app_context.router_config.wait_for_tokenizer
            || context.data.worker_kind != Some(WorkerKind::Local)
            || context.data.connection_mode == Some(ConnectionMode::Http)
            || app_context.router_config.disable_tokenizer_autoload
        {
            if let Some(preparation) = context.data.tokenizer_preparation.take() {
                preparation.close();
            }
            return Ok(StepResult::Skip);
        }
        let preparation = context
            .data
            .tokenizer_preparation
            .as_ref()
            .ok_or_else(TokenizerPreparation::closed)?;
        let workers = context
            .data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;
        prepare_worker_tokenizers(&app_context, workers, preparation).await?;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
