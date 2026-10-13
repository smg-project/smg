//! Step to re-read a registered worker's labels from its engine.
//!
//! Runs in the update workflow when a label refresh was requested: the
//! worker's engine came back into service (a restart on the same address may
//! run with other settings), or it finished loading after the worker was
//! registered on provisional information. The labels the engine reports now
//! are handed to the update step, which rebuilds the worker only when they
//! change anything. Without a refresh request the step is skipped.

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::discover_engine_labels;
use crate::{worker::ConnectionMode, workflow::data::WorkerUpdateWorkflowData};

/// Step to re-read a worker's labels from its engine for a label refresh.
pub struct RediscoverLabelsStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for RediscoverLabelsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if !context.data.refresh_discovered_labels {
            return Ok(StepResult::Skip);
        }
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers =
            context.data.workers_to_update.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("workers_to_update".to_string())
            })?;
        // Every rank of a DP group answers for the same engine: one read.
        let Some(worker) = workers.first() else {
            return Ok(StepResult::Skip);
        };
        let connection_mode = *worker.connection_mode();
        if connection_mode == ConnectionMode::Zmq {
            // A ZMQ engine reports its geometry at the handshake, not at an
            // endpoint; there is nothing to re-read.
            return Ok(StepResult::Skip);
        }
        let spec = &worker.metadata().spec;
        // A strict read: an endpoint that fails is an error (the step retries,
        // then the next transition into service tries again), never a gap
        // that would read as labels the engine dropped.
        let (fresh, _) = discover_engine_labels(
            &app_context.client,
            worker.base_url(),
            connection_mode,
            spec.runtime_type.as_str(),
            spec.api_key.as_deref(),
            true,
        )
        .await
        .map_err(|e| WorkflowError::StepFailed {
            step_id: StepId::new("rediscover_labels"),
            message: format!("could not re-read the labels of {}: {e}", worker.url()),
        })?;
        if fresh.is_empty() {
            // Nothing to follow; a complete read that says nothing is not a
            // reason to drop what registration found.
            debug!(worker_url = %worker.url(), "The engine reports no labels; nothing to refresh");
            return Ok(StepResult::Skip);
        }
        debug!(
            worker_url = %worker.url(),
            labels = fresh.len(),
            "Re-read the engine's labels for a refresh"
        );
        context.data.fresh_engine_labels = Some(fresh);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::{WorkerStatus, WorkerUpdateRequest};
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        service_discovery::testing::create_test_app_context,
        worker::{BasicWorkerBuilder, Worker},
    };

    fn context(
        worker: Arc<dyn Worker>,
        refresh: bool,
    ) -> WorkflowContext<WorkerUpdateWorkflowData> {
        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: worker.url().to_string(),
            app_context: Some(create_test_app_context()),
            workers_to_update: Some(vec![worker]),
            updated_workers: None,
            refresh_discovered_labels: refresh,
            fresh_engine_labels: None,
            snapshot: None,
        };
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    #[tokio::test]
    async fn without_a_refresh_request_the_step_is_skipped() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
                .connection_mode(ConnectionMode::Grpc)
                .status(WorkerStatus::Ready)
                .build(),
        );
        let mut ctx = context(worker, false);
        assert_eq!(
            RediscoverLabelsStep.execute(&mut ctx).await.unwrap(),
            StepResult::Skip
        );
        assert!(ctx.data.fresh_engine_labels.is_none());
    }

    /// A read that fails leaves nothing for the update step: the worker keeps
    /// its labels, and the step reports a retryable error.
    #[tokio::test]
    async fn a_failed_read_leaves_the_worker_unchanged() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
                .connection_mode(ConnectionMode::Grpc)
                .label("max_num_seqs", "256")
                .status(WorkerStatus::Ready)
                .build(),
        );
        let mut ctx = context(Arc::clone(&worker), true);
        let err = RediscoverLabelsStep
            .execute(&mut ctx)
            .await
            .expect_err("nothing listens on the port");
        assert!(RediscoverLabelsStep.is_retryable(&err));
        assert!(ctx.data.fresh_engine_labels.is_none());
        assert_eq!(
            worker
                .metadata()
                .spec
                .labels
                .get("max_num_seqs")
                .map(String::as_str),
            Some("256")
        );
    }
}
