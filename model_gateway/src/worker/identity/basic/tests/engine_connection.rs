//! The worker's engine connection: made once through the connector the
//! worker was built with, refused without one, kept by a rebuild. The builder
//! named here is the one without a default connector.

use std::{sync::Arc, time::Duration};

use openai_protocol::worker::{ConnectionMode, RuntimeType, WorkerType};

use crate::worker::{
    connection::engine::{EngineConnector, ZmqLink, NO_ENGINE_CONNECTOR},
    identity::{basic::connect_zmq_backend, builder::BasicWorkerBuilder},
    test_support::{no_health_check, CountingConnector, NoEngine},
    Worker, WorkerError,
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

/// A ZMQ worker built without a connector refuses its connect as well. The
/// connect is the handshake driver's: the refusal is the driver's error, the
/// caller that starts the driver reads that the handshake has not completed,
/// and the slot stays empty.
#[tokio::test]
async fn a_zmq_worker_built_without_a_connector_refuses_its_connect() {
    let link = ZmqLink {
        base_url: "ipc://engine.ipc".to_string(),
        model_id: "m".to_string(),
        runtime: RuntimeType::Vllm,
        worker_type: WorkerType::Regular,
        handshake_address: None,
        engine_count: None,
        startup: Duration::from_secs(1),
        request_timeout: None,
    };
    let refusal = connect_zmq_backend(None, link)
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

    let worker = <BasicWorkerBuilder>::new("ipc://engine.ipc")
        .connection_mode(ConnectionMode::Zmq)
        .health_config(no_health_check())
        .build();
    let answer = worker
        .engine_connection()
        .await
        .err()
        .expect("the handshake is the driver's");
    assert!(
        matches!(
            &answer,
            WorkerError::ConnectionFailed { reason, .. }
                if reason == "ZMQ backend handshake has not completed yet"
        ),
        "{answer}"
    );
    assert!(worker.engine_connection_if_initialized().is_none());
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
