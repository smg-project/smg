//! The metadata step's cases.

use std::sync::Arc;

use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    test_support::{ScriptedAnswers, ScriptedConnector},
    worker::WorkerSpec,
    workflow::host::testing::{serve_json_routes, ProbingHost},
};

fn local_context(
    connector: ScriptedConnector,
    spec: WorkerSpec,
    connection_mode: ConnectionMode,
    runtime: Option<&str>,
) -> WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>> {
    let host = ProbingHost::new(false, connector);
    let mut data = WorkerRegistrationWorkflowData::new(
        spec,
        WorkerRegistrationMode::Upsert,
        host,
        WorkerJobAsker::Api,
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(connection_mode);
    data.detected_runtime_type = runtime.map(str::to_string);
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// An HTTP worker's labels are read with the host's worker client. With no
/// runtime detected the read takes the default's, and settles no runtime.
#[tokio::test]
async fn an_http_workers_labels_are_read_with_the_hosts_client() {
    let base = serve_json_routes(vec![(
        "/server_info",
        http::StatusCode::OK,
        serde_json::json!({"model_path": "org/model-a", "tp_size": 2}),
    )])
    .await;
    let mut ctx = local_context(
        ScriptedConnector::default(),
        WorkerSpec::new(base),
        ConnectionMode::Http,
        None,
    );

    let result = DiscoverMetadataStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(
        ctx.data
            .discovered_labels
            .get("model_path")
            .map(String::as_str),
        Some("org/model-a")
    );
    assert_eq!(
        ctx.data
            .discovered_labels
            .get("tp_size")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(ctx.data.detected_runtime_type, None);
}

/// A gRPC worker's labels are read through the host's connector, as the
/// runtime the detection settled.
#[tokio::test]
async fn a_grpc_workers_labels_are_read_through_the_hosts_connector() {
    let runtime = DEFAULT_RUNTIME;
    let engine = Arc::new(ScriptedAnswers::describing(
        Ok(&[("model_path", "org/model-a")]),
        Ok(&[("version", "1.2")]),
    ));
    let mut ctx = local_context(
        ScriptedConnector::default().with_engine(runtime, engine),
        WorkerSpec::new("grpc://worker-a:9000"),
        ConnectionMode::Grpc,
        Some(runtime.as_str()),
    );

    let result = DiscoverMetadataStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(
        ctx.data
            .discovered_labels
            .get("model_path")
            .map(String::as_str),
        Some("org/model-a")
    );
    assert_eq!(
        ctx.data
            .discovered_labels
            .get("version")
            .map(String::as_str),
        Some("1.2")
    );
    assert_eq!(
        ctx.data.detected_runtime_type.as_deref(),
        Some(runtime.as_str())
    );
}

/// A read that fails leaves the labels empty and does not fail the step.
#[tokio::test]
async fn a_read_that_fails_leaves_no_labels_and_passes() {
    let mut ctx = local_context(
        ScriptedConnector::default(),
        WorkerSpec::new("grpc://worker-a:9000"),
        ConnectionMode::Grpc,
        Some(DEFAULT_RUNTIME.as_str()),
    );
    ctx.data
        .discovered_labels
        .insert("stale".to_string(), "1".to_string());

    let result = DiscoverMetadataStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert!(ctx.data.discovered_labels.is_empty());
}

/// Nothing is read for a ZMQ worker: what its host wrote stays. A worker
/// that is not local is skipped.
#[tokio::test]
async fn a_zmq_worker_is_left_to_its_host() {
    let mut ctx = local_context(
        ScriptedConnector::default(),
        WorkerSpec::new("ipc:///tmp/engine-a.sock"),
        ConnectionMode::Zmq,
        None,
    );
    ctx.data
        .discovered_labels
        .insert("of_the_host".to_string(), "1".to_string());

    let result = DiscoverMetadataStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.discovered_labels.len(), 1);

    ctx.data.worker_kind = Some(WorkerKind::External);
    assert_eq!(
        DiscoverMetadataStep.execute(&mut ctx).await.unwrap(),
        StepResult::Skip
    );
}
