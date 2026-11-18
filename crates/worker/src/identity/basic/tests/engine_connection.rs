//! The worker's engine connection: made once through the connector the
//! worker was built with, refused without one, kept by a rebuild. The builder
//! named here is the one without a default connector.

use std::sync::Arc;

use crate::{
    connection::engine::{EngineConnector, NO_ENGINE_CONNECTOR},
    identity::{basic::Worker, builder::BasicWorkerBuilder, error::WorkerError},
    test_support::{no_health_check, CountingConnector, NoEngine},
    worker::ConnectionMode,
};

fn grpc_worker() -> BasicWorkerBuilder {
    BasicWorkerBuilder::new("grpc://engine:50051")
        .connection_mode(ConnectionMode::Grpc)
        .health_config(no_health_check())
}

/// A gRPC worker connects through the connector it was built with, once: a
/// second acquisition hands out the connection the first one made, and the
/// host reads its own type back as the same allocation.
#[tokio::test]
async fn a_grpc_worker_connects_once_through_its_connector() {
    let connector = Arc::new(CountingConnector::default());
    let worker = grpc_worker()
        .engine_connector(Arc::clone(&connector) as Arc<dyn EngineConnector>)
        .build();
    assert!(worker.engine_connection_if_initialized().is_none());

    let first = worker.engine_connection().await.unwrap().unwrap();
    let second = worker.engine_connection().await.unwrap().unwrap();
    assert_eq!(connector.connects(), 1);
    assert!(Arc::ptr_eq(&first, &second));

    let typed = Arc::clone(&first)
        .downcast_arc::<NoEngine>()
        .expect("the connector's own type");
    assert!(std::ptr::addr_eq(Arc::as_ptr(&typed), Arc::as_ptr(&first)));
}

/// A gRPC worker built without a connector refuses its connect and says so.
#[tokio::test]
async fn a_grpc_worker_built_without_a_connector_refuses_its_connect() {
    let worker = grpc_worker().build();

    let refusal = worker
        .engine_connection()
        .await
        .err()
        .expect("no connector, no connect");
    assert!(
        matches!(
            &refusal,
            WorkerError::ConnectionFailed { reason, .. } if reason.contains(NO_ENGINE_CONNECTOR)
        ),
        "{refusal}"
    );
    assert!(worker.engine_connection_if_initialized().is_none());
}

/// A ZMQ worker built without a connector refuses its connect as well, at
/// once and in the same words. No handshake driver starts, for an
/// acquisition or for a probe, and the slot stays empty.
#[tokio::test]
async fn a_zmq_worker_built_without_a_connector_refuses_its_connect() {
    let worker = <BasicWorkerBuilder>::new("ipc://engine.ipc")
        .connection_mode(ConnectionMode::Zmq)
        .health_config(no_health_check())
        .build();

    let refusal = worker
        .engine_connection()
        .await
        .err()
        .expect("no connector, no connect");
    assert!(
        matches!(
            &refusal,
            WorkerError::ConnectionFailed { url, reason }
                if url == "ipc://engine.ipc" && reason == NO_ENGINE_CONNECTOR
        ),
        "{refusal}"
    );
    assert!(!worker.zmq_health_check(), "a probe reads not ready");
    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "no driver starts without a connector"
    );
    assert!(worker.engine_connection_if_initialized().is_none());
}

/// A ZMQ worker built without a connector answers its health check with the
/// acquisition's refusal: the health log names the cause, where a probe
/// that only read not ready named none.
#[tokio::test]
async fn a_zmq_worker_built_without_a_connector_names_it_at_its_health_check() {
    let worker = <BasicWorkerBuilder>::new("ipc://engine.ipc")
        .connection_mode(ConnectionMode::Zmq)
        .build();

    let refusal = worker
        .check_health_async()
        .await
        .expect_err("no connector, no health");
    assert!(
        matches!(
            &refusal,
            WorkerError::ConnectionFailed { url, reason }
                if url == "ipc://engine.ipc" && reason == NO_ENGINE_CONNECTOR
        ),
        "{refusal}"
    );
    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "the check starts no driver either"
    );
}

/// A rebuild over a live worker keeps the connector the worker was built
/// with: the replacement connects the way the worker it replaces does.
#[tokio::test]
async fn a_rebuild_keeps_the_engine_connector() {
    let connector = Arc::new(CountingConnector::default());
    let worker: Arc<dyn Worker> = Arc::new(
        grpc_worker()
            .engine_connector(Arc::clone(&connector) as Arc<dyn EngineConnector>)
            .build(),
    );

    let rebuilt = <BasicWorkerBuilder>::carrying(&worker).build();
    assert!(rebuilt.engine_connection().await.unwrap().is_some());
    assert_eq!(connector.connects(), 1);
}
