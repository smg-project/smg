//! Step to re-read a registered worker's labels from its engine.
//!
//! Runs in the update workflow when a label refresh was requested: the
//! worker's engine came back into service (a restart on the same address may
//! run with other settings), or it finished loading after the worker was
//! registered on provisional information. The labels the engine reports now
//! (at its metadata endpoints; at the handshake its link holds, for a ZMQ
//! engine) are handed to the update step, which rebuilds the worker only
//! when they change anything. Without a refresh request the step is skipped.

use std::collections::HashMap;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::discover_engine_labels;
use crate::{
    routers::grpc::{backend_client::BackendClient, zmq_client},
    worker::ConnectionMode,
    workflow::data::WorkerUpdateWorkflowData,
};

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
            // endpoint: the worker's own link holds it once the handshake is
            // in (the refresh runs on that connect signal). Its window joins
            // what registration read from the model's files; a link not up
            // yet leaves the labels as they are, and the next transition
            // into service refreshes again.
            let Some(link) = worker.backend_client_if_initialized() else {
                debug!(worker_url = %worker.redacted_url(), "The ZMQ link is not up; nothing to refresh");
                return Ok(StepResult::Skip);
            };
            let BackendClient::Zmq(client) = link.as_ref() else {
                return Ok(StepResult::Skip);
            };
            // The seed is the worker's record, kept in rank space
            // (`report_for_rank`); the update step passes the report
            // through that rule again, which is the identity here only
            // because the record holds neither a running-window label
            // (`WINDOW_LABELS`) nor a width (`DP_WIDTH_LABELS`): sampling
            // defaults, the grammar start and the context window, which no
            // rank shares. A record that gains a running window needs a
            // report read in engine space instead.
            let mut fresh = HashMap::clone(&worker.metadata().discovered_labels);
            fresh.extend(zmq_client::link_labels(client));
            debug!(
                worker_url = %worker.redacted_url(),
                labels = fresh.len(),
                "Read the ZMQ link's handshake window for a refresh"
            );
            context.data.fresh_engine_labels = Some(fresh);
            return Ok(StepResult::Success);
        }
        let spec = &worker.metadata().spec;
        // A strict read: an endpoint that fails is an error (the step retries,
        // then the next transition into service tries again), never a gap
        // that would read as labels the engine dropped. Through the worker's
        // own client, the one registration read with (built for its pool and
        // HTTP/2 setting); the gRPC read ignores it.
        let (fresh, _) = discover_engine_labels(
            worker.http_client(),
            worker.base_url(),
            connection_mode,
            spec.runtime_type.as_str(),
            spec.api_key.as_deref(),
            true,
        )
        .await
        .map_err(|e| WorkflowError::StepFailed {
            step_id: StepId::new("rediscover_labels"),
            message: format!(
                "could not re-read the labels of {}: {e}",
                worker.redacted_url()
            ),
        })?;
        if fresh.is_empty() {
            // Nothing to follow; a complete read that says nothing is not a
            // reason to drop what registration found.
            debug!(worker_url = %worker.redacted_url(), "The engine reports no labels; nothing to refresh");
            return Ok(StepResult::Skip);
        }
        debug!(
            worker_url = %worker.redacted_url(),
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

    /// A ZMQ worker whose link is not up has nothing to re-read yet: the
    /// labels stay, and the next transition into service refreshes again.
    #[tokio::test]
    async fn a_zmq_worker_without_its_link_is_skipped() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("ipc:///tmp/engine.sock")
                .connection_mode(ConnectionMode::Zmq)
                .status(WorkerStatus::Ready)
                .build(),
        );
        let mut ctx = context(worker, true);
        assert_eq!(
            RediscoverLabelsStep.execute(&mut ctx).await.unwrap(),
            StepResult::Skip
        );
        assert!(ctx.data.fresh_engine_labels.is_none());
    }

    /// A ZMQ worker's refresh reads the window its link learned at the
    /// handshake and keeps what registration read from the model's files.
    #[tokio::test]
    async fn a_zmq_worker_reads_its_handshake_window() {
        use std::time::Duration;

        use engine_zmq_client::{
            mock_engine::{connect_to_frontend, default_ready_response},
            protocol::handshake::EngineCoreReadyResponse,
            EngineId,
        };
        use openai_protocol::worker::RuntimeType;

        use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

        let dir = tempfile::tempdir().unwrap();
        let ep = |socket: &str| format!("ipc://{}", dir.path().join(socket).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let ready = EngineCoreReadyResponse {
            max_model_len: 4096,
            ..default_ready_response()
        };
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(&handshake, EngineId::from_engine_index(0), ready),
        );
        let _engine = engine.expect("mock engine");
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(ep("worker.ipc"))
                .connection_mode(ConnectionMode::Zmq)
                .runtime_type(RuntimeType::Vllm)
                .backend_client(BackendClient::Zmq(client.expect("adapter connect")))
                .discovered_labels(HashMap::from([(
                    "constrained_decoding_start".to_string(),
                    "first_token".to_string(),
                )]))
                .status(WorkerStatus::Ready)
                .build(),
        );
        let mut ctx = context(worker, true);
        assert_eq!(
            RediscoverLabelsStep.execute(&mut ctx).await.unwrap(),
            StepResult::Success
        );
        let fresh = ctx.data.fresh_engine_labels.expect("the link's labels");
        assert_eq!(
            fresh.get("max_context_length").map(String::as_str),
            Some("4096"),
            "the handshake's window"
        );
        assert_eq!(
            fresh.get("constrained_decoding_start").map(String::as_str),
            Some("first_token"),
            "what registration read stays"
        );
        assert_eq!(
            fresh.len(),
            2,
            "nothing else of the handshake's report: {fresh:?}"
        );
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
