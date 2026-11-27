use openai_protocol::worker::WorkerSpec;
use smg_worker::{jobs::WorkerJobAsker, registry::event::WorkerEvent};
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    app_context::{testing::app_context_for_tests, AppContext},
    config::RouterConfig,
    routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
    workflow::steps::create_worker_workflow_data,
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
    app_context_for_tests(
        RouterConfig::default(),
        Arc::new(WorkerRegistry::new()),
        None,
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
        WorkerJobAsker::Api,
    );
    data.actual_workers = Some(vec![Arc::clone(&worker)]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);

    ActivateWorkersStep
        .execute(&mut context)
        .await
        .expect("activation");

    assert_eq!(worker.status(), WorkerStatus::Ready);
    let transition = std::iter::from_fn(|| events.try_recv().ok()).find_map(|event| match event {
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
