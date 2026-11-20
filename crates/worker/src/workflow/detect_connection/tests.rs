//! The connection detection's cases.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wfaas::WorkflowInstanceId;

use super::*;
use crate::{
    connection::engine::HealthAnswer,
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    test_support::{RefusedAddress, ScriptedAnswers, ScriptedConnector},
    worker::{HttpPoolConfig, WorkerSpec},
    workflow::{host::testing::ProbingHost, runtimes::GRPC_RUNTIMES},
};

/// A host whose connector reaches no engine.
fn host(upstream_http2: bool) -> Arc<ProbingHost> {
    ProbingHost::new(upstream_http2, ScriptedConnector::default())
}

fn local_context(
    host: Arc<ProbingHost>,
    spec: WorkerSpec,
) -> WorkflowContext<WorkerRegistrationWorkflowData<ProbingHost>> {
    let mut data = WorkerRegistrationWorkflowData::new(
        spec,
        WorkerRegistrationMode::Upsert,
        host,
        WorkerJobAsker::Api,
    );
    data.worker_kind = Some(WorkerKind::Local);
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// axum::serve answers HTTP/1.1 and prior-knowledge h2c on one listener.
async fn spawn_dual_protocol_server() -> String {
    let app = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    #[expect(
        clippy::disallowed_methods,
        reason = "test server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{addr}")
}

/// HTTP/1.1-only server: drops a connection that opens with the HTTP/2
/// preface and answers every HTTP/1.1 request 200.
async fn spawn_http1_only_server() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    #[expect(
        clippy::disallowed_methods,
        reason = "test server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.expect("accept");
            #[expect(clippy::disallowed_methods, reason = "one task per test connection")]
            tokio::spawn(async move {
                let mut buf = [0u8; 1024];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                if buf[..n].starts_with(b"PRI * HTTP/2.0") {
                    return;
                }
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                    )
                    .await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn negotiates_http2_with_a_dual_protocol_worker() {
    let url = spawn_dual_protocol_server().await;
    let host = host(true);
    let mut ctx = local_context(Arc::clone(&host), WorkerSpec::new(url));

    let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.connection_mode, Some(ConnectionMode::Http));
    assert_eq!(ctx.data.http2, Some(true));
    let expected = host
        .worker_client_cache()
        .get(&HttpPoolConfig::default(), true)
        .unwrap();
    assert!(Arc::ptr_eq(
        ctx.data.http_client_handle.as_ref().unwrap(),
        &expected
    ));
}

#[tokio::test]
async fn falls_back_to_http1_when_the_worker_rejects_the_preface() {
    let url = spawn_http1_only_server().await;
    let host = host(true);
    let mut ctx = local_context(Arc::clone(&host), WorkerSpec::new(url));

    let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.http2, Some(false));
    let expected = host
        .worker_client_cache()
        .get(&HttpPoolConfig::default(), false)
        .unwrap();
    assert!(Arc::ptr_eq(
        ctx.data.http_client_handle.as_ref().unwrap(),
        &expected
    ));
}

#[tokio::test]
async fn stays_on_http1_without_upstream_http2() {
    let url = spawn_dual_protocol_server().await;
    let mut ctx = local_context(host(false), WorkerSpec::new(url));

    let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.http2, Some(false));
}

#[tokio::test]
async fn declared_http2_false_skips_negotiation() {
    let url = spawn_dual_protocol_server().await;
    let mut spec = WorkerSpec::new(url);
    spec.http_pool.http2 = Some(false);
    let mut ctx = local_context(host(true), spec);

    let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.http2, Some(false));
}

#[tokio::test]
async fn declared_http2_true_fails_against_an_http1_worker() {
    let url = spawn_http1_only_server().await;
    let mut spec = WorkerSpec::new(url);
    spec.http_pool.http2 = Some(true);
    let mut ctx = local_context(host(false), spec);

    let err = DetectConnectionModeStep
        .execute(&mut ctx)
        .await
        .unwrap_err();

    assert!(matches!(err, WorkflowError::StepFailed { .. }), "{err:?}");
    assert_eq!(ctx.data.http2, None);
}

/// The step's lines and its failure message name the worker masked: a
/// worker registered with credentials at a port nobody listens on fails
/// its probe, and neither the failure message nor any line the step
/// wrote carries the credentials. The capture is the tracing subscriber
/// itself, so the rendering is read as a log reader would.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_failed_probe_names_the_worker_masked_in_its_lines_and_its_message() {
    let refused = RefusedAddress::reserve();
    let port = refused.port();
    let url = format!("http://user:secret@127.0.0.1:{port}");
    let mut ctx = local_context(host(false), WorkerSpec::new(url));

    let err = DetectConnectionModeStep
        .execute(&mut ctx)
        .await
        .unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains(&format!("http://***@127.0.0.1:{port}")),
        "{message}"
    );
    assert!(!message.contains("secret"), "{message}");
    assert!(!message.contains("user:"), "{message}");
    assert!(!logs_contain("secret"));
    assert!(!logs_contain("user:"));
    assert!(logs_contain(&format!("http://***@127.0.0.1:{port}")));
}

#[tokio::test]
async fn bare_host_port_negotiates_too() {
    let url = spawn_dual_protocol_server().await;
    let bare = url.trim_start_matches("http://").to_string();
    let mut ctx = local_context(host(true), WorkerSpec::new(bare));

    let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.connection_mode, Some(ConnectionMode::Http));
    assert_eq!(ctx.data.http2, Some(true));
}

/// A bare address that no HTTP server answers and a gRPC engine does is
/// detected as gRPC, through the host's connector, and settles no HTTP
/// version.
#[tokio::test]
async fn a_bare_address_only_a_grpc_engine_answers_is_detected_as_grpc() {
    let refused = RefusedAddress::reserve();
    let bare = format!("127.0.0.1:{}", refused.port());
    let engine = Arc::new(ScriptedAnswers::answering_health(Ok(HealthAnswer {
        healthy: true,
        message: String::new(),
    })));
    let connector = ScriptedConnector::default().with_engine(GRPC_RUNTIMES[0], engine);
    let mut ctx = local_context(ProbingHost::new(false, connector), WorkerSpec::new(bare));

    let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

    assert_eq!(result, StepResult::Success);
    assert_eq!(ctx.data.connection_mode, Some(ConnectionMode::Grpc));
    assert_eq!(ctx.data.http2, None);
    assert!(ctx.data.http_client_handle.is_none());
}

/// An explicit gRPC address is asked over gRPC alone. With no engine that
/// answers there the step fails, and its message names the address masked.
#[tokio::test]
async fn an_explicit_grpc_address_no_engine_answers_at_fails_the_step() {
    let spec = WorkerSpec::new("grpc://user:secret@worker-a:9000");
    let mut ctx = local_context(host(false), spec);

    let err = DetectConnectionModeStep
        .execute(&mut ctx)
        .await
        .unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains(
            "health check failed for explicitly configured worker URL grpc://***@worker-a:9000"
        ),
        "{message}"
    );
    assert!(message.contains("gRPC not reachable"), "{message}");
    assert!(!message.contains("secret"), "{message}");
    assert_eq!(ctx.data.connection_mode, None);
}
