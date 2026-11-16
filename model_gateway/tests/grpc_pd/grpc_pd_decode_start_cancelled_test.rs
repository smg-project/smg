//! A disaggregated decode call whose start the worker's side cancelled (its
//! connection closed under the call: a worker dying as the request arrived)
//! is a failure of that worker, not a rejection of the request: with another
//! decode worker available the request is retried there and answered in
//! full, the cancelled worker is charged a backend error, and nothing counts
//! as a validation error; with no other decode worker the client reads a
//! 502 under the side's own code, never a 400 it would take for its own
//! mistake and not retry.
//!
//! The mock engines speak the vLLM proto (sequential dispatch): the prefill
//! side completes, one decode worker cancels every call before its stream.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use common::{
    decode_stream::produce_after,
    engine_answers::{VllmAnswers, VllmGenerateStream},
    grpc_harness::{build_grpc_router, grpc_router_config, grpc_workers, serve_engine},
    metrics_capture::render_after,
};
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::common;

const MODEL: &str = "pd-decode-start-cancelled-test-model";
/// Tokens an engine that completes produces; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Longer than any case: a request held past it fails the case instead of
/// hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The words tonic puts on a call whose connection closed under it.
const CANCEL_MESSAGE: &str = "operation was canceled";

/// What a mock engine does with a generate call.
#[derive(Clone, Copy)]
enum Behaviour {
    /// `OUTPUT_TOKENS` tokens, then a `stop` completion.
    Completes,
    /// The call fails with `CANCELLED` before any stream: what the gateway's
    /// client sees when the worker's connection closes under the call.
    CancelsStart,
}

/// A mock engine counting its generate calls.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
}

impl Engine {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            behaviour,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[tonic::async_trait]
impl VllmAnswers for Engine {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.behaviour, Behaviour::CancelsStart) {
            return Err(Status::cancelled(CANCEL_MESSAGE));
        }
        let (tx, rx) = mpsc::channel(8);
        produce_after(tx, Duration::ZERO, OUTPUT_TOKENS);
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Serve a mock engine in-process; returns its port.
async fn start_mock_engine(engine: Engine) -> u16 {
    serve_engine(engine.into_server()).await
}

/// A PD gRPC router over the given workers (sequential dispatch).
async fn build_pd_router(workers: &[(u16, WorkerType)]) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .build_unchecked();
    build_grpc_router(
        config,
        MODEL,
        grpc_workers(workers, RuntimeType::Vllm, MODEL),
    )
    .await
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-decode-start-cancelled-test-tenant"))
}

/// One streaming chat through `router`: its status and body.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_streaming_chat(router: &dyn RouterTrait) -> (http::StatusCode, String) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": true,
    }))
    .unwrap();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect("the request must be answered within the case cap");
    let status = response.status();
    let bytes = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the response body must end within the case cap")
    .expect("response body");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("utf-8 body"),
    )
}

/// The sum of a counter's samples carrying every one of `labels`; zero when
/// none was emitted.
fn counter(rendered: &str, name: &str, labels: &[String]) -> f64 {
    let prefix = format!("{name}{{");
    rendered
        .lines()
        .filter(|line| line.starts_with(&prefix))
        .filter(|line| labels.iter().all(|label| line.contains(label.as_str())))
        .filter_map(|line| line.rsplit(' ').next())
        .filter_map(|value| value.parse::<f64>().ok())
        .sum()
}

/// `smg_worker_errors_total` for `worker_type` and `error_type`.
fn worker_errors(rendered: &str, worker_type: &str, error_type: &str) -> f64 {
    counter(
        rendered,
        "smg_worker_errors_total",
        &[
            format!("worker_type=\"{worker_type}\""),
            format!("error_type=\"{error_type}\""),
        ],
    )
}

/// `smg_router_request_errors_total` for `error_type` on the PD backend.
fn router_errors(rendered: &str, error_type: &str) -> f64 {
    counter(
        rendered,
        "smg_router_request_errors_total",
        &[
            "backend_type=\"pd\"".to_string(),
            format!("error_type=\"{error_type}\""),
        ],
    )
}

/// The first decode worker cancels the side's start; the request is retried
/// on the second and answered in full. The cancelled worker is charged one
/// backend error and never asked again; nothing counts as a validation
/// error.
#[test]
fn a_decode_start_the_worker_cancelled_is_retried_on_another_decode_worker() {
    let prefill = Engine::new(Behaviour::Completes);
    let cancelling = Engine::new(Behaviour::CancelsStart);
    let healthy = Engine::new(Behaviour::Completes);
    let (cancelling_calls, healthy_calls) = (cancelling.clone(), healthy.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let cancelling_port = start_mock_engine(cancelling).await;
        let healthy_port = start_mock_engine(healthy).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (cancelling_port, WorkerType::Decode),
            (healthy_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(!body.contains(r#""error""#), "{body}");
    });
    assert_eq!(
        cancelling_calls.calls(),
        1,
        "cancelled once, never asked again"
    );
    assert_eq!(
        healthy_calls.calls(),
        1,
        "the retry went to the other decode worker"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error"),
        1.0,
        "the cancelled worker is charged one backend error:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "validation_error"),
        0.0,
        "{rendered}"
    );
    assert_eq!(
        router_errors(&rendered, "validation_error"),
        0.0,
        "nothing about the request was wrong:\n{rendered}"
    );
}

/// With no other decode worker, the cancelled start reaches the client as
/// an upstream failure under the side's own code: a 502 it may retry, never a
/// 400 it would take for its own mistake.
#[test]
fn a_decode_start_the_worker_cancelled_with_no_other_worker_is_a_502_not_a_400() {
    let prefill = Engine::new(Behaviour::Completes);
    let cancelling = Engine::new(Behaviour::CancelsStart);
    let cancelling_calls = cancelling.clone();
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let cancelling_port = start_mock_engine(cancelling).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (cancelling_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::BAD_GATEWAY, "{body}");
        assert!(body.contains("decode_worker_failed_to_start"), "{body}");
        assert!(body.contains(CANCEL_MESSAGE), "{body}");
        assert!(
            !body.contains("invalid_request_error"),
            "not the request's fault:\n{body}"
        );
    });
    assert!(
        cancelling_calls.calls() >= 1,
        "the only decode worker was asked at least once"
    );
    assert!(
        worker_errors(&rendered, "decode", "backend_error") >= 1.0,
        "the cancelled worker is charged:\n{rendered}"
    );
    assert_eq!(
        router_errors(&rendered, "validation_error"),
        0.0,
        "{rendered}"
    );
    assert!(
        router_errors(&rendered, "backend_error") >= 1.0,
        "the request counts as a backend error on the PD backend:\n{rendered}"
    );
}
