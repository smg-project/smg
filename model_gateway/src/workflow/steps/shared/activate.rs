//! Unified worker activation step.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    registry::WorkerId,
};
use tracing::{debug, info};
use wfaas::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::workflow::data::{require_app_context, WorkerRegistrationData, WorkerRegistrationMode};

/// Final step in any worker registration workflow: flip Pending → Ready.
pub struct ActivateWorkersStep;

/// The step's id in the registration DAGs.
pub(crate) const ACTIVATE_WORKERS_STEP: &str = "activate_workers";

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
        let app_context = require_app_context(context.data.get_app_context())?.clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        // The transition goes through the registry so it is published as a
        // `StatusChanged` event: the capacity tracker recomputes the backend
        // capacity on it (a worker flipped Ready directly was counted only
        // if the tracker's recompute for the `Registered` event happened to
        // run after the flip, so a second worker's slots were often never
        // added), and so does every other subscriber. A worker the registry
        // no longer holds is flipped directly, as before.
        let registry = &app_context.worker_registry;
        let mut activated = 0;
        for worker in workers {
            // A PUT's replacement shares the status of the worker it
            // replaced: a removal's Draining mark behind the PUT's write
            // stands and fails the step, decided under the worker's lock.
            if let WorkerRegistrationMode::ReplaceById { worker_id, .. } =
                context.data.registration_mode()
            {
                let worker_id = WorkerId::from_string(worker_id);
                let status = activation_status(worker);
                let decide = |held: &Arc<dyn Worker>| {
                    let mine = Arc::ptr_eq(held, worker);
                    let leaving = mine && held.status() == WorkerStatus::Draining;
                    (leaving, status.filter(|_| mine && !leaving))
                };
                // The revision is only the call's precondition (`ptr_eq` is
                // the identity test), and a mesh publication can raise it in
                // place behind the read: it is read once more.
                let outcome = registry
                    .apply_if_revision(&worker_id, worker.revision(), decide)
                    .or_else(|| registry.apply_if_revision(&worker_id, worker.revision(), decide));
                match outcome {
                    Some((true, _)) => {
                        return Err(WorkflowError::StepFailed {
                            step_id: StepId::new(ACTIVATE_WORKERS_STEP),
                            message: format!(
                                "Worker {} at {} is being removed, so its replacement was not \
                                 activated",
                                worker_id.as_str(),
                                worker.redacted_url()
                            ),
                        })
                    }
                    Some((false, Some(_))) => activated += 1,
                    // Nothing to write, or superseded since: the status is
                    // its successor's.
                    Some((false, None)) => {}
                    None => debug!(
                        worker_id = %worker_id.as_str(),
                        worker_url = worker.redacted_url(),
                        "Replacement not activated: the worker is gone or its revision moved behind both reads"
                    ),
                }
                continue;
            }
            if let Some(status) = activation_status(worker) {
                // Only the worker this workflow registered: an upsert of the
                // same URL that raced it (discovery beside an API call) must
                // not have its own worker flipped from here.
                match registry
                    .get_id_by_url(worker.url())
                    .filter(|id| registry.get(id).is_some_and(|w| Arc::ptr_eq(&w, worker)))
                {
                    Some(worker_id) => {
                        registry.transition_status(&worker_id, status);
                    }
                    None => worker.set_status(status),
                }
                activated += 1;
            }
        }

        if activated > 0 {
            if let Some(admission) = &app_context.prefill_admission {
                admission.notify_capacity_changed();
            }
        }

        info!(
            "Activated {activated} of {} worker(s) (others left as-is: already ready, awaiting connect signal or replaced since)",
            workers.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/workflow/steps/shared/activate/tests.rs"]
mod tests;
