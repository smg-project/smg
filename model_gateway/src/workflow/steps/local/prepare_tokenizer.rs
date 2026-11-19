//! Prepare the gateway's tokenizer before publishing a token-inference worker.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepResult,
    WorkflowContext, WorkflowError, WorkflowResult,
};

use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{require_app_context, WorkerKind, WorkerRegistrationData, WorkerWorkflowData},
        tokenizer_preparation::{prepare_worker_tokenizers, TokenizerPreparation},
    },
};

pub struct PrepareTokenizerStep;

impl PrepareTokenizerStep {
    const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(300);
    const MAX_ATTEMPTS: u32 = 3;
    const RETRY_INTERVAL: Duration = Duration::from_secs(2);

    pub(crate) fn definition() -> StepDefinition<WorkerWorkflowData> {
        StepDefinition::new("prepare_tokenizer", "Prepare Tokenizer", Arc::new(Self))
            .with_timeout(Self::ATTEMPT_TIMEOUT)
            .with_retry(RetryPolicy {
                max_attempts: Self::MAX_ATTEMPTS,
                backoff: BackoffStrategy::Fixed(Self::RETRY_INTERVAL),
            })
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["ensure_harmony_encoding"])
    }

    /// The inline preparation's full attempt and backoff allowance, in
    /// addition to the registration's existing startup allowance.
    pub(crate) fn budget(data: &WorkerWorkflowData) -> Duration {
        if data.tokenizer_preparation().is_some() {
            Self::ATTEMPT_TIMEOUT * Self::MAX_ATTEMPTS
                + Self::RETRY_INTERVAL * (Self::MAX_ATTEMPTS - 1)
        } else {
            Duration::ZERO
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        config::RouterConfig,
        worker::WorkerRegistry,
        workflow::{steps::create_worker_workflow_data, WorkerRegistrationMode},
    };

    #[test]
    fn only_eligible_registrations_get_the_full_preparation_budget() {
        for wait in [false, true] {
            for disable_autoload in [false, true] {
                let app = app_context_for_tests(
                    RouterConfig::builder()
                        .wait_for_tokenizer(wait)
                        .disable_tokenizer_autoload(disable_autoload)
                        .build_unchecked(),
                    Arc::new(WorkerRegistry::new()),
                    None,
                );
                for kind in [None, Some(WorkerKind::Local), Some(WorkerKind::External)] {
                    for wire in [
                        None,
                        Some(ConnectionMode::Http),
                        Some(ConnectionMode::Grpc),
                        Some(ConnectionMode::Zmq),
                    ] {
                        let mut data = create_worker_workflow_data(
                            openai_protocol::worker::WorkerSpec::new("bare:8000"),
                            WorkerRegistrationMode::CreateOnly,
                            Arc::clone(&app),
                        );
                        data.worker_kind = kind;
                        data.connection_mode = wire;
                        let eligible = wait
                            && !disable_autoload
                            && kind == Some(WorkerKind::Local)
                            && wire.is_some_and(|wire| wire.uses_grpc_pipeline());
                        assert_eq!(
                            PrepareTokenizerStep::budget(&data),
                            if eligible {
                                Duration::from_secs(904)
                            } else {
                                Duration::ZERO
                            }
                        );
                    }
                }
            }
        }
    }
}
