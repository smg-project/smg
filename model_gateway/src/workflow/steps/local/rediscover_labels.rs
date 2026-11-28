//! Step to re-read a registered worker's labels from its engine.
//!
//! Runs in the update workflow when a label refresh was requested: the
//! worker's engine came back into service (a restart on the same address may
//! run with other settings), or it finished loading after the worker was
//! registered on provisional information. The labels the engine reports now
//! (at its metadata endpoints; at the handshake its link holds, for a ZMQ
//! engine) are handed to the update step, which rebuilds the worker only
//! when they change anything. Without a refresh request the step is skipped.

use async_trait::async_trait;
use smg_worker::{
    identity::basic::ConnectionMode,
    workflow::{discover_engine_labels, WorkflowHost},
};
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    routers::grpc::{
        backend_client::{backend_client_if_initialized, BackendClient},
        zmq_client,
    },
    workflow::data::{require_app_context, WorkerUpdateWorkflowData},
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
            let link = backend_client_if_initialized(worker.as_ref()).map_err(|e| {
                WorkflowError::StepFailed {
                    step_id: StepId::new("rediscover_labels"),
                    message: format!("could not read the link of {}: {e}", worker.redacted_url()),
                }
            })?;
            let Some(link) = link else {
                debug!(worker_url = %worker.redacted_url(), "The ZMQ link is not up; nothing to refresh");
                return Ok(StepResult::Skip);
            };
            let BackendClient::Zmq(client) = link.as_ref() else {
                return Ok(StepResult::Skip);
            };
            // The seed is the worker's record, kept in rank space
            // (`report_for_rank`); the update step passes the report
            // through that rule again. That is the identity for what the
            // record holds besides the link's own labels: sampling defaults
            // and the grammar start, which no rank shares. The link's labels
            // replace the record's in engine space: the context window, and
            // a TokenSpeed group's running window with the width its ranks
            // share it by (`refreshed_labels`).
            let fresh = zmq_client::refreshed_labels(
                &worker.metadata().discovered_labels,
                zmq_client::link_labels(client),
            );
            debug!(
                worker_url = %worker.redacted_url(),
                labels = fresh.len(),
                "Read the ZMQ link's handshake window for a refresh"
            );
            context.data.fresh_engine_labels = Some(fresh);
            return Ok(StepResult::Success);
        }
        let spec = &worker.metadata().spec;
        let app_context = require_app_context(context.data.app_context.as_ref())?;
        // A strict read: an endpoint that fails is an error (the step retries,
        // then the next transition into service tries again), never a gap
        // that would read as labels the engine dropped. Through the worker's
        // own client, the one registration read with (built for its pool and
        // HTTP/2 setting); the gRPC read ignores it.
        let (fresh, _) = discover_engine_labels(
            worker.http_client(),
            app_context.engine_connector().as_ref(),
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
}

#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/local/rediscover_labels/tests.rs"]
mod tests;
