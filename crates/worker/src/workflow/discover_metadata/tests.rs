//! The metadata step's cases.

use std::sync::Arc;

use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    test_support::{ScriptedAnswers, ScriptedConnector},
    worker::WorkerSpec,
    workflow::host::testing::{serve_json_routes, worker_client, ProbingHost},
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
fn label<'a>(labels: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    labels.get(key).map(String::as_str)
}

fn vllm_registration_context(
    url: String,
) -> WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>> {
    use crate::{model_card::ModelCard, worker::WorkerModels};

    let mut spec = WorkerSpec::new(url);
    spec.models = WorkerModels::Single(Box::new(ModelCard::new("configured-model")));
    spec.labels.insert("kv_connector".into(), "moriio".into());
    let mut context = local_context(
        ScriptedConnector::default(),
        spec,
        ConnectionMode::Http,
        Some("vllm"),
    );
    context.data.http2 = Some(false);
    context.data.http_client_handle = Some(worker_client());
    context
}
#[tokio::test]
async fn registration_rejects_malformed_pairing_protocol_before_worker_creation() {
    use http::StatusCode;

    use crate::workflow::host::testing::serve_json_routes;

    let mut payloads: Vec<_> = [
        serde_json::json!(4),
        serde_json::json!([]),
        serde_json::json!({}),
        serde_json::json!(false),
    ]
    .into_iter()
    .map(|value| serde_json::json!({"data": [{"id": "m"}], "pairing_protocol": value}))
    .collect();
    // A declared protocol cannot disappear because another field is invalid.
    payloads.push(serde_json::json!({"data": "invalid", "pairing_protocol": "kv-v1"}));
    for payload in payloads {
        let base = serve_json_routes(vec![("/v1/models", StatusCode::OK, payload)]).await;
        let mut context = vllm_registration_context(base);
        let result = DiscoverMetadataStep.execute(&mut context).await;
        match result {
            Err(WorkflowError::StepFailed { step_id, message }) => {
                assert_eq!(step_id, wfaas::StepId::new("discover_metadata"));
                assert!(message.contains("pairing_protocol"), "{message}");
            }
            other => panic!("explicit invalid protocol cannot become absent: {other:?}"),
        }
        assert!(context.data.discovered_labels.is_empty());
        assert!(context.data.actual_workers.is_none());
    }
}

#[tokio::test]
async fn registration_rejects_smg_invalid_metadata_before_worker_creation() {
    use http::StatusCode;

    use crate::workflow::host::testing::serve_json_routes;

    let base = serve_json_routes(vec![("/v1/models", StatusCode::BAD_GATEWAY,
        serde_json::json!({"error": {"code": "smg_http_metadata_invalid", "message": "invalid extension"}}))]).await;
    let mut context = vllm_registration_context(base);
    let result = DiscoverMetadataStep.execute(&mut context).await;
    match result {
        Err(WorkflowError::StepFailed { step_id, message }) => {
            assert_eq!(step_id, wfaas::StepId::new("discover_metadata"));
            assert!(message.contains("smg_http_metadata_invalid"), "{message}");
        }
        other => panic!("our middleware explicitly refused the metadata: {other:?}"),
    }
    assert!(context.data.discovered_labels.is_empty());
    assert!(context.data.actual_workers.is_none());
}

#[tokio::test]
async fn registration_rejects_duplicate_pairing_protocol_even_when_last_is_null() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = axum::Router::new().route(
        "/v1/models",
        axum::routing::get(|| async {
            r#"{"data":[{"id":"m"}],"pairing_protocol":"kv-v1","pairing_protocol":null}"#
        }),
    );
    let mut servers = tokio::task::JoinSet::new();
    servers.spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let mut context = vllm_registration_context(base);
    let result = DiscoverMetadataStep.execute(&mut context).await;
    servers.shutdown().await;
    match result {
        Err(WorkflowError::StepFailed { message, .. }) => {
            assert!(message.contains("pairing_protocol"), "{message}");
        }
        other => panic!("duplicate protocol must not become an absent extension: {other:?}"),
    }
    assert!(context.data.actual_workers.is_none());
}

#[tokio::test]
async fn registration_preserves_legacy_absent_valid_and_optional_failures() {
    use http::StatusCode;

    use crate::{test_support::RefusedAddress, workflow::host::testing::serve_json_routes};

    for (status, body, expected) in [
        (
            StatusCode::OK,
            serde_json::json!({"data": [{"id": "m"}]}),
            None,
        ),
        (
            StatusCode::OK,
            serde_json::json!({"data": [{"id": "m"}], "pairing_protocol": null}),
            None,
        ),
        (
            StatusCode::OK,
            serde_json::json!({"data": [{"id": "m"}], "pairing_protocol": "   "}),
            None,
        ),
        (
            StatusCode::OK,
            serde_json::json!({"data": [{"id": "m"}], "pairing_protocol": " kv-v1 "}),
            Some("kv-v1"),
        ),
        (
            StatusCode::OK,
            serde_json::json!({"data": "legacy malformed list"}),
            None,
        ),
        (StatusCode::NOT_FOUND, serde_json::json!({}), None),
        (
            StatusCode::BAD_GATEWAY,
            serde_json::json!({"error": {"code": "other_error"}}),
            None,
        ),
    ] {
        let base = serve_json_routes(vec![
            ("/v1/models", status, body),
            // The extension marker is interpreted only at its own endpoint.
            (
                "/version",
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"code": "smg_http_metadata_invalid"}}),
            ),
        ])
        .await;
        let mut context = vllm_registration_context(base);
        assert!(matches!(
            DiscoverMetadataStep.execute(&mut context).await,
            Ok(StepResult::Success)
        ));
        assert_eq!(
            label(&context.data.discovered_labels, "pairing_protocol"),
            expected
        );
    }
    let refused = RefusedAddress::reserve();
    let mut context = vllm_registration_context(refused.http_url());
    assert!(matches!(
        DiscoverMetadataStep.execute(&mut context).await,
        Ok(StepResult::Success)
    ));
    assert!(context.data.discovered_labels.is_empty());
}
