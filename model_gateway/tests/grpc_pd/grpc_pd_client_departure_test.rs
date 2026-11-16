//! A disaggregated request whose client leaves mid-decode: the decode side's
//! abort is deferred until its first response (the proof the KV handoff
//! completed), and a non-streaming request's first response is its whole
//! answer, so that wait is bounded by the decode side's bound (three seconds
//! at the defaults) instead of the house cap. The decode engine receives the
//! abort within the bound of the client's leave on both dispatch shapes,
//! where it used to receive it with the answer (the cap at most). A streaming
//! request whose client leaves before its first token keeps the deferral to
//! that token, as before: the transfer may still be in flight.
//!
//! The parallel shape's mock engines speak the TokenSpeed proto (the prefill
//! side is the mock-worker crate's canned worker, answering at once), the
//! sequential shape's the vLLM proto (a prefill engine that completes at
//! once); each decode engine holds its one response, or its first token, for
//! a chosen time and records when the abort for the request reaches it.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use common::{
    decode_stream::{chunk, complete, complete_with},
    engine_answers::{
        TokenSpeedAnswers, TokenSpeedGenerateStream, VllmAnswers, VllmGenerateStream,
    },
    grpc_harness::{
        build_grpc_router, grpc_router_config, grpc_worker, serve_canned_worker, serve_engine,
        Canned,
    },
};
use openai_protocol::model_card::ModelCard;
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, Worker, WorkerType},
};
use smg_grpc_client::{tokenspeed_scheduler::tokenspeed_proto as ts, vllm_proto as proto};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::common;

const MODEL: &str = "pd-client-departure-test-model";
const OUTPUT_TOKENS: u32 = 3;
/// The decode side's bound at the defaults (the wedge threshold) for these
/// short prompts.
const BOUND: Duration = Duration::from_secs(3);
/// What the deferred abort's wait may add past the bound before the case
/// calls it held to the answer.
const SLACK: Duration = Duration::from_secs(1);
/// When the client leaves, counted from its request.
const LEAVE_AFTER: Duration = Duration::from_millis(500);
/// When a plain decode side's one response comes: well past the bound, so an
/// abort that waits for it arrives late, and well within the case.
const ANSWER_AFTER: Duration = Duration::from_secs(6);
/// When a streaming decode side's first token comes: before the bound.
const FIRST_TOKEN_AFTER: Duration = Duration::from_millis(1_500);
const CASE_CAP: Duration = Duration::from_secs(20);

/// What a decode engine does with a generate call's stream.
#[derive(Clone, Copy)]
enum Answer {
    /// One `Complete` after [`ANSWER_AFTER`] (a non-streaming generation).
    WholeAnswerLate,
    /// The first token after [`FIRST_TOKEN_AFTER`], the rest at once.
    FirstTokenLate,
}

/// The instants the engine was asked to abort a request, and its calls.
#[derive(Clone, Default)]
struct Aborts {
    at: Arc<Mutex<Vec<Instant>>>,
    calls: Arc<AtomicUsize>,
}

impl Aborts {
    #[expect(clippy::expect_used, reason = "test helper")]
    fn note(&self) {
        self.at.lock().expect("aborts lock").push(Instant::now());
    }

    /// The first abort's instant, waited for up to `within`.
    #[expect(clippy::expect_used, reason = "test helper")]
    async fn first_within(&self, within: Duration) -> Option<Instant> {
        let started = Instant::now();
        while started.elapsed() < within {
            if let Some(at) = self.at.lock().expect("aborts lock").first().copied() {
                return Some(at);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }
}

/// A parallel-shape decode engine (the TokenSpeed proto).
#[derive(Clone)]
struct TokenSpeedDecode {
    answer: Answer,
    aborts: Aborts,
}

fn ts_chunk(request_id: &str, id: u32) -> ts::GenerateResponse {
    ts::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(ts::generate_response::Response::Chunk(
            ts::GenerateStreamChunk {
                token_ids: vec![id],
                prompt_tokens: 1,
                completion_tokens: 1,
                index: 0,
                ..Default::default()
            },
        )),
    }
}

fn ts_complete(request_id: &str) -> ts::GenerateResponse {
    ts::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(ts::generate_response::Response::Complete(
            ts::GenerateComplete {
                output_ids: (1..=OUTPUT_TOKENS).collect(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: OUTPUT_TOKENS,
                index: 0,
                ..Default::default()
            },
        )),
    }
}

/// Feed one TokenSpeed generate stream as `answer` says; the producer task
/// ends with its stream when the reader is gone.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_ts(
    tx: mpsc::Sender<Result<ts::GenerateResponse, Status>>,
    request_id: String,
    answer: Answer,
) {
    tokio::spawn(async move {
        match answer {
            Answer::WholeAnswerLate => {
                tokio::time::sleep(ANSWER_AFTER).await;
                let _ = tx.send(Ok(ts_complete(&request_id))).await;
            }
            Answer::FirstTokenLate => {
                tokio::time::sleep(FIRST_TOKEN_AFTER).await;
                for id in 1..=OUTPUT_TOKENS {
                    if tx.send(Ok(ts_chunk(&request_id, id))).await.is_err() {
                        return;
                    }
                }
                let _ = tx.send(Ok(ts_complete(&request_id))).await;
            }
        }
    });
}

#[tonic::async_trait]
impl TokenSpeedAnswers for TokenSpeedDecode {
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        self.aborts.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(8);
        produce_ts(tx, request.into_inner().request_id, self.answer);
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn abort(
        &self,
        _request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        self.aborts.note();
        Ok(Response::new(ts::AbortResponse::default()))
    }
}

/// A sequential-shape prefill engine (the vLLM proto): one `Complete` at once.
#[derive(Clone)]
struct VllmPrefill;

#[tonic::async_trait]
impl VllmAnswers for VllmPrefill {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(2);
        let _ = tx.try_send(Ok(complete_with("stop", Vec::new())));
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Ok(Response::new(proto::AbortResponse::default()))
    }
}

/// A sequential-shape decode engine (the vLLM proto).
#[derive(Clone)]
struct VllmDecode {
    answer: Answer,
    aborts: Aborts,
}

/// Feed one vLLM generate stream as `answer` says.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_vllm(tx: mpsc::Sender<Result<proto::GenerateResponse, Status>>, answer: Answer) {
    tokio::spawn(async move {
        match answer {
            Answer::WholeAnswerLate => {
                tokio::time::sleep(ANSWER_AFTER).await;
                let _ = tx.send(Ok(complete("stop", OUTPUT_TOKENS))).await;
            }
            Answer::FirstTokenLate => {
                tokio::time::sleep(FIRST_TOKEN_AFTER).await;
                for id in 1..=OUTPUT_TOKENS {
                    if tx.send(Ok(chunk(id))).await.is_err() {
                        return;
                    }
                }
                let _ = tx.send(Ok(complete("stop", OUTPUT_TOKENS))).await;
            }
        }
    });
}

#[tonic::async_trait]
impl VllmAnswers for VllmDecode {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        self.aborts.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(8);
        produce_vllm(tx, self.answer);
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        self.aborts.note();
        Ok(Response::new(proto::AbortResponse::default()))
    }
}

/// A PD gRPC router over the given workers, registered as `runtime` (which
/// picks the dispatch shape).
async fn build_pd_router(
    endpoints: &[(u16, WorkerType)],
    runtime: RuntimeType,
) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .build_unchecked();
    let workers: Vec<Arc<dyn Worker>> = endpoints
        .iter()
        .map(|&(port, worker_type)| {
            Arc::new(grpc_worker(port, worker_type, runtime, ModelCard::new(MODEL)).build())
                as Arc<dyn Worker>
        })
        .collect();
    build_grpc_router(config, MODEL, workers).await
}

/// The parallel shape: the canned worker as the prefill side, `answer` as the
/// decode side; the decode engine's abort record comes back.
async fn parallel_pair(answer: Answer) -> (Box<dyn RouterTrait>, Aborts) {
    let prefill_port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: OUTPUT_TOKENS,
    })
    .await;
    let decode = TokenSpeedDecode {
        answer,
        aborts: Aborts::default(),
    };
    let aborts = decode.aborts.clone();
    let decode_port = serve_engine(decode.into_server()).await;
    let router = build_pd_router(
        &[
            (prefill_port, WorkerType::Prefill),
            (decode_port, WorkerType::Decode),
        ],
        RuntimeType::TokenSpeed,
    )
    .await;
    (router, aborts)
}

/// The sequential shape: a prefill engine completing at once, `answer` as the
/// decode side; the decode engine's abort record comes back.
async fn sequential_pair(answer: Answer) -> (Box<dyn RouterTrait>, Aborts) {
    let prefill_port = serve_engine(VllmPrefill.into_server()).await;
    let decode = VllmDecode {
        answer,
        aborts: Aborts::default(),
    };
    let aborts = decode.aborts.clone();
    let decode_port = serve_engine(decode.into_server()).await;
    let router = build_pd_router(
        &[
            (prefill_port, WorkerType::Prefill),
            (decode_port, WorkerType::Decode),
        ],
        RuntimeType::Vllm,
    )
    .await;
    (router, aborts)
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-client-departure-test-tenant"))
}

/// One chat through `router` whose client leaves [`LEAVE_AFTER`] in: a
/// non-streaming request's future is dropped then (the whole answer was
/// still being collected); a streaming request's response body is dropped
/// then if its headers were in (its first token not yet), else its future
/// is, as a connection's close drops the handler (the sequential shape opens
/// the stream with the decode side's first token). The request's start and
/// the instant the client left come back.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn chat_whose_client_leaves(router: &dyn RouterTrait, streaming: bool) -> (Instant, Instant) {
    let started_at = Instant::now();
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": streaming,
    }))
    .unwrap();
    let routed = tokio::time::timeout(
        LEAVE_AFTER,
        router.route_chat(None, &tenant(), request, MODEL),
    )
    .await;
    match routed {
        // The streamed answer's headers came within the leave: its body is
        // dropped at the leave, unread.
        Ok(response) => {
            assert!(
                streaming,
                "a non-streaming answer must not be in before its decode side's one response"
            );
            assert_eq!(response.status(), http::StatusCode::OK);
            tokio::time::sleep(LEAVE_AFTER).await;
            drop(response);
        }
        // The answer was still being collected (a non-streaming one), or
        // the stream had not opened yet (one that opens with the first
        // token): the request future is dropped at the leave.
        Err(_elapsed) => {}
    }
    (started_at, Instant::now())
}

/// The abort reached the decode engine within the bound of the client's
/// leave (plus the slack), not with the answer.
#[expect(clippy::expect_used, reason = "test assertion")]
async fn assert_aborted_within_the_bound(aborts: &Aborts, left_at: Instant) {
    let aborted_at = aborts
        .first_within(CASE_CAP)
        .await
        .expect("the decode engine must be asked to abort once the client left");
    let after_leave = aborted_at.saturating_duration_since(left_at);
    assert!(
        after_leave >= BOUND.saturating_sub(Duration::from_millis(200)),
        "the abort must still wait out the decode bound for the handoff: it came {after_leave:?} after the leave"
    );
    assert!(
        after_leave <= BOUND + SLACK,
        "the abort must reach the decode engine within the decode bound of the client's leave \
         ({BOUND:?} and {SLACK:?} of slack), not with the answer: it came {after_leave:?} after"
    );
    assert_eq!(aborts.calls.load(Ordering::SeqCst), 1, "one decode call");
}

/// The abort reached the decode engine after the first token (due
/// [`FIRST_TOKEN_AFTER`] from the request's start; the client left before it,
/// so the deferral held the abort for the handoff) and within the bound of
/// the client's leave. The token is expected from the start, not from the
/// leave: a streamed answer whose headers come in early is left
/// [`LEAVE_AFTER`] after them, later than a request whose stream has not
/// opened yet.
#[expect(clippy::expect_used, reason = "test assertion")]
async fn assert_aborted_after_the_first_token(
    aborts: &Aborts,
    started_at: Instant,
    left_at: Instant,
) {
    let aborted_at = aborts
        .first_within(CASE_CAP)
        .await
        .expect("the decode engine must be asked to abort once the client left");
    let token_at = started_at + FIRST_TOKEN_AFTER;
    let after_leave = aborted_at.saturating_duration_since(left_at);
    assert!(
        left_at < token_at,
        "the client must leave before the first token: it left {:?} after the start",
        left_at.saturating_duration_since(started_at)
    );
    assert!(
        aborted_at + Duration::from_millis(200) >= token_at,
        "the abort must wait for the first token (due {FIRST_TOKEN_AFTER:?} after the start): \
         it came {:?} after the start, {after_leave:?} after the leave",
        aborted_at.saturating_duration_since(started_at)
    );
    assert!(
        after_leave <= BOUND + SLACK,
        "the abort must follow the first token closely: it came {after_leave:?} after the leave"
    );
}

/// The parallel shape: a non-streaming request's decode side, holding its one
/// response past the bound, is asked to abort within the bound of the client's
/// leave.
#[tokio::test]
async fn a_parallel_plain_decode_is_aborted_within_the_bound_of_the_clients_leave() {
    let (router, aborts) = parallel_pair(Answer::WholeAnswerLate).await;
    let (_started_at, left_at) = chat_whose_client_leaves(router.as_ref(), false).await;
    assert_aborted_within_the_bound(&aborts, left_at).await;
}

/// The sequential shape: the same.
#[tokio::test]
async fn a_sequential_plain_decode_is_aborted_within_the_bound_of_the_clients_leave() {
    let (router, aborts) = sequential_pair(Answer::WholeAnswerLate).await;
    let (_started_at, left_at) = chat_whose_client_leaves(router.as_ref(), false).await;
    assert_aborted_within_the_bound(&aborts, left_at).await;
}

/// A streaming request whose client leaves before its first token keeps the
/// deferral to that token: the abort reaches the decode engine after the
/// token came (the transfer behind it may have been in flight), and within
/// the bound, on the parallel shape.
#[tokio::test]
async fn a_parallel_streamed_decode_left_before_its_first_token_is_aborted_after_that_token() {
    let (router, aborts) = parallel_pair(Answer::FirstTokenLate).await;
    let (started_at, left_at) = chat_whose_client_leaves(router.as_ref(), true).await;
    assert_aborted_after_the_first_token(&aborts, started_at, left_at).await;
}

/// The same on the sequential shape.
#[tokio::test]
async fn a_sequential_streamed_decode_left_before_its_first_token_is_aborted_after_that_token() {
    let (router, aborts) = sequential_pair(Answer::FirstTokenLate).await;
    let (started_at, left_at) = chat_whose_client_leaves(router.as_ref(), true).await;
    assert_aborted_after_the_first_token(&aborts, started_at, left_at).await;
}
