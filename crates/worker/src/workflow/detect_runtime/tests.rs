//! The runtime detection's cases.

use std::sync::Arc;

use axum::{
    http::{StatusCode, Uri},
    routing::get,
    Json, Router,
};
use serde_json::json;
use tokio::net::TcpListener;
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    connection::engine::HealthAnswer,
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    test_support::{RefusedAddress, ScriptedAnswers, ScriptedConnector},
    worker::WorkerSpec,
    workflow::host::testing::{worker_client, ProbingHost},
};

/// A runtime that answers its health RPC with `healthy: false` (its engine
/// still loading) is identified all the same, and none is asked behind it.
/// Where no runtime's service answers, the failure names every runtime's
/// reason in the order asked, and the address masked.
#[tokio::test]
async fn grpc_detection_identifies_a_runtime_that_reports_not_healthy() {
    let runtime = GRPC_DETECTION_ORDER[3];
    let engine = Arc::new(ScriptedAnswers::answering_health(Ok(HealthAnswer {
        healthy: false,
        message: "not serving".to_string(),
    })));
    let connector = ScriptedConnector::default().with_engine(runtime, engine);

    assert_eq!(
        detect_grpc_runtime(&connector, "grpc://worker-a:9000", 5).await,
        Ok(runtime)
    );
    assert_eq!(connector.connects(), 4);

    let nobody = ScriptedConnector::default();
    let err = detect_grpc_runtime(&nobody, "grpc://user:secret@worker-a:9000", 1)
        .await
        .expect_err("no runtime's service answers");
    let head = format!(
        "gRPC backend detection failed for grpc://***@worker-a:9000 (tried {}): ",
        listed(&GRPC_DETECTION_ORDER)
    );
    assert!(err.starts_with(&head), "{err}");
    let mut rest = &err[head.len()..];
    for runtime in GRPC_DETECTION_ORDER {
        let reason = format!("{runtime}: gRPC health check failed");
        let at = rest.find(&reason);
        assert!(at.is_some(), "{runtime}'s reason is not in its turn: {err}");
        rest = &rest[at.unwrap_or_default() + reason.len()..];
    }
}

async fn serve_router(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only mock backend server; handle is aborted at test end"
    )]
    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), handle)
}

fn models_route(payload: serde_json::Value) -> Router {
    Router::new().route(
        "/v1/models",
        get(move || {
            let payload = payload.clone();
            async move { Json(payload) }
        }),
    )
}

#[tokio::test]
async fn http_detection_falls_back_to_generic_for_unrecognized_owned_by() {
    // A gateway that stands in front of an engine reports an owner of its
    // own and serves no row's path.
    let (url, server) = serve_router(models_route(json!({
        "object": "list",
        "data": [{"id": "kimi-k3-256k", "object": "model", "created": 0, "owned_by": "self_hosted"}]
    })))
    .await;

    let runtime = detect_http_runtime(&url, 5, &worker_client(), None)
        .await
        .unwrap();
    server.abort();

    assert_eq!(runtime, RuntimeType::Generic);
}

#[tokio::test]
async fn http_detection_falls_back_to_generic_when_owned_by_missing() {
    let (url, server) = serve_router(models_route(json!({
        "object": "list",
        "data": [{"id": "some-model", "object": "model"}]
    })))
    .await;

    let runtime = detect_http_runtime(&url, 5, &worker_client(), None)
        .await
        .unwrap();
    server.abort();

    assert_eq!(runtime, RuntimeType::Generic);
}

#[tokio::test]
async fn http_detection_prefers_version_probe_over_generic_fallback() {
    // Unrecognized owned_by but a live /version endpoint: that row's
    // runtime, not generic.
    let router = models_route(json!({
        "object": "list",
        "data": [{"id": "m", "object": "model", "owned_by": "self_hosted"}]
    }))
    .route(
        "/version",
        get(|| async { Json(json!({"version": "0.9.0"})) }),
    );
    let (url, server) = serve_router(router).await;

    let runtime = detect_http_runtime(&url, 5, &worker_client(), None)
        .await
        .unwrap();
    server.abort();

    assert_eq!(runtime, RuntimeType::Vllm);
}

/// A worker registered with credentials at a port nobody listens on: the
/// detector's error text and every line it wrote name the address masked,
/// never the secret (the probes' failures end in the step's failure
/// message, which a log line carries). The capture is the tracing
/// subscriber itself.
#[tracing_test::traced_test]
#[tokio::test]
async fn http_detection_failures_name_the_worker_masked() {
    let refused = RefusedAddress::reserve();
    let port = refused.port();

    let err = detect_http_runtime(
        &format!("http://user:secret@127.0.0.1:{port}"),
        1,
        &worker_client(),
        None,
    )
    .await
    .unwrap_err();

    assert!(!err.contains("secret"), "{err}");
    assert!(
        err.contains(&format!("http://***@127.0.0.1:{port}/version")),
        "{err}"
    );
    assert!(!logs_contain("secret"));
    assert!(!logs_contain("user:"));
    assert!(logs_contain(&format!(
        "http://***@127.0.0.1:{port}/v1/models"
    )));
}

/// An address registered with a query, at a port nobody listens on: the
/// failure does not repeat the query. The HTTP client's own error text
/// ends with the request URL, the query kept.
#[tokio::test]
async fn http_detection_failures_go_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("{}?token=s3cret", refused.http_url());

    let err = detect_http_runtime(&url, 1, &worker_client(), None)
        .await
        .unwrap_err();

    assert!(!err.contains("s3cret"), "{err}");
}

/// An address registered with a query whose `/v1/models` answers a body
/// that is no JSON: the failure does not repeat the query. The HTTP
/// client's text of a failed body read ends with the request URL too.
#[tokio::test]
async fn an_unparsed_models_answer_goes_without_the_clients_url() {
    // The query takes in the path each probe appends, so the stub tells
    // the probes apart by the request's tail.
    let (url, server) = serve_router(Router::new().fallback(|uri: Uri| async move {
        if uri.to_string().ends_with("/v1/models") {
            (StatusCode::OK, "not json")
        } else {
            (StatusCode::NOT_FOUND, "")
        }
    }))
    .await;

    let result =
        detect_http_runtime(&format!("{url}?token=s3cret"), 1, &worker_client(), None).await;
    server.abort();

    let err = result.unwrap_err();
    assert!(err.contains("failed to parse"), "{err}");
    assert!(!err.contains("s3cret"), "{err}");
}

#[tokio::test]
async fn http_detection_fails_when_models_endpoint_unreachable() {
    // A port that refuses connections: a backend that is not up must stay
    // an error so the retryable step keeps polling.
    let refused = RefusedAddress::reserve();
    let addr = refused.addr();

    let err = detect_http_runtime(&format!("http://{addr}"), 1, &worker_client(), None)
        .await
        .unwrap_err();

    assert!(err.contains("Could not detect HTTP backend"), "got: {err}");
}

#[tokio::test]
async fn http_detection_fails_when_models_data_empty() {
    // Server is up but no model is loaded yet — keep failing so the step
    // retries instead of registering a backend with nothing to serve.
    let (url, server) = serve_router(models_route(json!({"object": "list", "data": []}))).await;

    let result = detect_http_runtime(&url, 5, &worker_client(), None).await;
    server.abort();

    result.unwrap_err();
}

#[tokio::test]
async fn http_detection_maps_a_second_owner_name_to_its_runtime() {
    let (url, server) = serve_router(models_route(json!({
        "object": "list",
        "data": [{"id": "test-model", "object": "model", "owned_by": "nvidia"}]
    })))
    .await;

    let runtime = detect_http_runtime(&url, 5, &worker_client(), None)
        .await
        .unwrap();
    server.abort();

    assert_eq!(runtime, RuntimeType::Sglang);
}

fn no_engine_host() -> Arc<ProbingHost> {
    ProbingHost::new(false, ScriptedConnector::default())
}

fn local_context(
    host: Arc<ProbingHost>,
    spec: WorkerSpec,
    connection_mode: Option<ConnectionMode>,
) -> WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>> {
    let mut data = WorkerRegistrationWorkflowData::new(
        spec,
        WorkerRegistrationMode::Upsert,
        host,
        WorkerJobAsker::Api,
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = connection_mode;
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// A runtime the spec declares is taken as it is: nothing is probed.
#[tokio::test]
async fn a_declared_runtime_is_taken_without_a_probe() {
    let declared = GRPC_DETECTION_ORDER[4];
    let host = no_engine_host();
    let mut spec = WorkerSpec::new("grpc://worker-a:9000");
    spec.runtime_type = declared;
    let mut ctx = local_context(Arc::clone(&host), spec, Some(ConnectionMode::Grpc));

    let result = DetectRuntimeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(
        ctx.data.detected_runtime_type.as_deref(),
        Some(declared.as_str())
    );
    assert_eq!(host.connects(), 0);
}

/// A gRPC worker's runtime is detected through the host's connector and
/// written by its id.
#[tokio::test]
async fn a_grpc_workers_runtime_is_written_by_its_id() {
    let runtime = GRPC_DETECTION_ORDER[1];
    let engine = Arc::new(ScriptedAnswers::answering_health(Ok(HealthAnswer {
        healthy: true,
        message: String::new(),
    })));
    let connector = ScriptedConnector::default().with_engine(runtime, engine);
    let mut ctx = local_context(
        ProbingHost::new(false, connector),
        WorkerSpec::new("grpc://worker-a:9000"),
        Some(ConnectionMode::Grpc),
    );

    let result = DetectRuntimeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(
        ctx.data.detected_runtime_type.as_deref(),
        Some(runtime.as_str())
    );
}

/// An HTTP worker's runtime is detected with the host's worker client and
/// written by its id.
#[tokio::test]
async fn an_http_workers_runtime_is_written_by_its_id() {
    let row = &HTTP_RUNTIMES[1];
    let (url, server) = serve_router(models_route(json!({
        "object": "list",
        "data": [{"id": "model-a", "object": "model", "owned_by": row.owners[0]}]
    })))
    .await;
    let mut ctx = local_context(
        no_engine_host(),
        WorkerSpec::new(url),
        Some(ConnectionMode::Http),
    );

    let result = DetectRuntimeStep.execute(&mut ctx).await.unwrap();
    server.abort();

    assert_eq!(result, StepResult::Success);
    assert_eq!(
        ctx.data.detected_runtime_type.as_deref(),
        Some(row.runtime.as_str())
    );
}

/// A failed detection fails the step under its own id, with the worker named
/// masked, and writes no runtime.
#[tokio::test]
async fn a_failed_detection_fails_the_step_naming_the_worker_masked() {
    let spec = WorkerSpec::new("grpc://user:secret@worker-a:9000");
    let mut ctx = local_context(no_engine_host(), spec, Some(ConnectionMode::Grpc));

    let err = DetectRuntimeStep.execute(&mut ctx).await.unwrap_err();

    assert!(
        matches!(&err, WorkflowError::StepFailed { step_id, .. } if *step_id == StepId::new(DETECT_RUNTIME_STEP)),
        "{err:?}"
    );
    let message = err.to_string();
    assert!(
        message.contains("gRPC backend detection failed for grpc://***@worker-a:9000"),
        "{message}"
    );
    assert!(!message.contains("secret"), "{message}");
    assert_eq!(ctx.data.detected_runtime_type, None);
}

/// A ZMQ link tells no runtime apart: a worker whose spec declares none
/// keeps none, for its host to settle.
#[tokio::test]
async fn a_zmq_worker_that_declares_no_runtime_keeps_none() {
    let mut ctx = local_context(
        no_engine_host(),
        WorkerSpec::new("ipc:///tmp/engine-a.sock"),
        Some(ConnectionMode::Zmq),
    );

    let result = DetectRuntimeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.detected_runtime_type, None);
}

/// The step reads the connection mode that the detection before it settled:
/// with none it fails naming it. A worker that is not local is skipped.
#[tokio::test]
async fn the_step_wants_a_local_worker_and_its_connection_mode() {
    let mut ctx = local_context(
        no_engine_host(),
        WorkerSpec::new("http://worker-a:8000"),
        None,
    );

    let err = DetectRuntimeStep.execute(&mut ctx).await.unwrap_err();
    assert!(matches!(err, WorkflowError::ContextValueNotFound(name) if name == "connection_mode"));

    ctx.data.worker_kind = Some(WorkerKind::External);
    assert_eq!(
        DetectRuntimeStep.execute(&mut ctx).await.unwrap(),
        StepResult::Skip
    );
}
