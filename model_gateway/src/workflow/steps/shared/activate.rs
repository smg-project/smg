//! Unified worker activation step.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::info;
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    worker::{ConnectionMode, Worker},
    workflow::data::WorkerRegistrationData,
};

/// Final step in any worker registration workflow: flip Pending → Ready.
pub struct ActivateWorkersStep;

/// Decide the activation transition for a worker, or `None` to leave it as-is.
///
/// ZMQ workers stay Pending until their backend handshake completes (promotion
/// is event-driven, with the health poll as a fallback): flipping them Ready
/// here would route requests at a socket the engine has not yet dialed. Every
/// other transport activates optimistically and lets the health loop reconcile.
fn activation_status(worker: &Arc<dyn Worker>) -> Option<WorkerStatus> {
    if *worker.connection_mode() == ConnectionMode::Zmq {
        return None;
    }
    (worker.status() != WorkerStatus::Ready).then_some(WorkerStatus::Ready)
}

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let mut activated = 0;
        for worker in workers {
            if let Some(status) = activation_status(worker) {
                worker.set_status(status);
                activated += 1;
            }
        }

        if activated > 0 {
            if let Some(admission) = &app_context.prefill_admission {
                admission.notify_capacity_changed();
            }
        }

        info!(
            "Activated {activated} of {} worker(s) (others left as-is: already ready or awaiting connect signal)",
            workers.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::BasicWorkerBuilder;

    #[test]
    fn http_pending_worker_is_activated() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        assert_eq!(worker.status(), WorkerStatus::Pending);
        assert_eq!(activation_status(&worker), Some(WorkerStatus::Ready));
    }

    #[test]
    fn already_ready_worker_needs_no_transition() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .connection_mode(ConnectionMode::Http)
                .status(WorkerStatus::Ready)
                .build(),
        );
        assert_eq!(activation_status(&worker), None);
    }

    #[test]
    fn zmq_worker_is_left_pending_for_the_connect_signal() {
        // The whole point of PR B: a ZMQ worker must NOT be optimistically
        // activated — it is promoted only when its handshake lands.
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
                .connection_mode(ConnectionMode::Zmq)
                .build(),
        );
        assert_eq!(worker.status(), WorkerStatus::Pending);
        assert_eq!(
            activation_status(&worker),
            None,
            "ZMQ workers must stay Pending until connected"
        );
    }
}
