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
    use std::sync::OnceLock;

    use llm_tokenizer::registry::TokenizerRegistry;
    use openai_protocol::worker::WorkerSpec;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::AppContext,
        config::RouterConfig,
        policies::PolicyRegistry,
        worker::{event::WorkerEvent, BasicWorkerBuilder, WorkerRegistry},
        workflow::{data::WorkerRegistrationMode, steps::create_worker_workflow_data},
    };

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

    fn test_app_context() -> Arc<AppContext> {
        let router_config = RouterConfig::default();
        Arc::new(
            AppContext::builder()
                .client(crate::http_client::client())
                .rate_limiter(None)
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(router_config.policy.clone())))
                .router_config(router_config)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .expect("app context"),
        )
    }

    /// The activation is a registry transition, published as a
    /// `StatusChanged` event: the capacity tracker recomputes the backend
    /// capacity on it, as every other subscriber reacts to it. A direct
    /// status flip left the capacity at whatever the `Registered` event's
    /// recompute had seen, which raced the activation (a second worker's
    /// slots were often never added).
    #[tokio::test]
    async fn activating_a_registered_worker_publishes_its_status_transition() {
        let app_context = test_app_context();
        let registry = Arc::clone(&app_context.worker_registry);
        let mut events = registry.subscribe_events();
        let url = "http://worker:8080";
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        // The registration's own event is not the one under test.
        while let Ok(event) = events.try_recv() {
            assert!(matches!(event, WorkerEvent::Registered { .. }));
        }
        let mut data = create_worker_workflow_data(
            WorkerSpec::new(url),
            WorkerRegistrationMode::Upsert,
            Arc::clone(&app_context),
        );
        data.actual_workers = Some(vec![Arc::clone(&worker)]);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);

        ActivateWorkersStep
            .execute(&mut context)
            .await
            .expect("activation");

        assert_eq!(worker.status(), WorkerStatus::Ready);
        let transition =
            std::iter::from_fn(|| events.try_recv().ok()).find_map(|event| match event {
                WorkerEvent::StatusChanged {
                    old_status,
                    new_status,
                    ..
                } => Some((old_status, new_status)),
                _ => None,
            });
        assert_eq!(
            transition,
            Some((WorkerStatus::Pending, WorkerStatus::Ready)),
            "the activation is published as a status transition"
        );
    }
}
