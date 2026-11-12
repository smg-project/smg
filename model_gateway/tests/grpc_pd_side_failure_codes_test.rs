//! A prefill or decode side that fails is named to the client in the one
//! vocabulary both routers share (`routers::common::pd_failure`): the side,
//! the phase it failed in and the failure's own status class. These are the
//! gRPC prefill-decode path's rows; the HTTP prefill-decode router runs the
//! same rows against its own stubs in its unit tests
//! (`a_side_failure_is_named_by_the_shared_vocabulary_on_both_dispatch_shapes`),
//! so a code that drifts on either router fails one table or the other.
//!
//! The sequential shape's mock engines speak the vLLM proto: one side fails
//! per row (its call refused before any stream, or its stream failing at its
//! first item or after its tokens), its partner completes. A side whose call
//! is refused never started; a side whose stream fails, at its first item or
//! later, failed its stream (the gateway's wait for a side's first response
//! judges silence alone and leaves a failure on the stream to its reader):
//! on a collected answer the side's stream code, on a streamed answer the
//! typed `stream_error` event, since the response has begun. The parallel
//! shape's prefill speaks the TokenSpeed proto and fails its stream after a
//! chunk, for the collected answer that drains it.
//!
//! A worker's own `INTERNAL` before the side's first byte is the worker's
//! definitive answer for the payload, and it meets the retry layer as that
//! worker's alone: the rows with the retry loop at its defaults read the
//! worker's 500 under the side's start code when the worker is the only one
//! of its kind, a fail-over to a free worker of the kind when there is one,
//! and the last worker's 500 once every worker of the kind has answered,
//! never a `no_available_workers` refusal, which names an outage the healthy
//! pair does not have. Both shapes run the rows, on the proto their runtime
//! speaks.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use common::{
    decode_stream::{chunk, complete, GenerateTx},
    engine_answers::{
        TokenSpeedAnswers, TokenSpeedGenerateStream, VllmAnswers, VllmGenerateStream,
    },
    grpc_harness::{
        build_grpc_router, grpc_router_config, grpc_workers, serve_canned_worker, serve_engine,
        Canned,
    },
};
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::{tokenspeed_scheduler::tokenspeed_proto as ts, vllm_proto as proto};
use smg_test_harness as common;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response, Status};

const MODEL: &str = "pd-side-failure-codes-test-model";
/// Tokens an engine that completes produces; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Longer than any case: a request held past it fails the case instead of
/// hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The words the failing engines put on their status.
const REASON: &str = "the engine says no";

/// What a mock engine does with a generate call.
#[derive(Clone, Copy, Debug)]
enum Behaviour {
    /// `OUTPUT_TOKENS` tokens, then a `stop` completion.
    Completes,
    /// The call fails with the code before any stream: a worker that refuses
    /// the side, or whose connection closed under the call (`Cancelled`).
    RefusesStart(Code),
    /// The stream's first item is the failure: a side admitted, then dropped
    /// before its first token. A stream failure like the next, met by the
    /// stream's reader.
    FailsFirst(Code),
    /// `OUTPUT_TOKENS` tokens, then the failure in the completion's place: a
    /// side cut after it began.
    FailsAfterTokens(Code),
}

/// A sequential-shape mock engine (the vLLM proto) with one behaviour,
/// counting its generate calls.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
}

/// Feed one generate stream as `behaviour` says.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce(tx: GenerateTx, behaviour: Behaviour) {
    tokio::spawn(async move {
        match behaviour {
            Behaviour::FailsFirst(code) => {
                let _ = tx.send(Err(Status::new(code, REASON))).await;
            }
            Behaviour::Completes | Behaviour::FailsAfterTokens(_) => {
                for id in 1..=OUTPUT_TOKENS {
                    if tx.send(Ok(chunk(id))).await.is_err() {
                        return;
                    }
                }
                let last = match behaviour {
                    Behaviour::FailsAfterTokens(code) => Err(Status::new(code, REASON)),
                    _ => Ok(complete("stop", OUTPUT_TOKENS)),
                };
                let _ = tx.send(last).await;
            }
            // Refused before any stream: nothing to feed, the sender drops.
            Behaviour::RefusesStart(_) => {}
        }
    });
}

#[tonic::async_trait]
impl VllmAnswers for Engine {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Behaviour::RefusesStart(code) = self.behaviour {
            return Err(Status::new(code, REASON));
        }
        let (tx, rx) = mpsc::channel(8);
        produce(tx, self.behaviour);
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// A parallel-shape mock engine (the TokenSpeed proto) with one behaviour,
/// counting its generate calls: it completes (`OUTPUT_TOKENS` chunks, then
/// the completion), or refuses its call with the code before any stream.
#[derive(Clone)]
struct TokenSpeedEngine {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
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

#[tonic::async_trait]
impl TokenSpeedAnswers for TokenSpeedEngine {
    #[expect(
        clippy::disallowed_methods,
        reason = "test mock; the producer task ends with its stream"
    )]
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.behaviour {
            Behaviour::Completes => {}
            Behaviour::RefusesStart(code) => return Err(Status::new(code, REASON)),
            Behaviour::FailsFirst(_) | Behaviour::FailsAfterTokens(_) => {
                return Err(Status::unimplemented(
                    "a stream failure is not scripted on the TokenSpeed proto here",
                ))
            }
        }
        let request_id = request.into_inner().request_id;
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            for id in 1..=OUTPUT_TOKENS {
                if tx.send(Ok(ts_chunk(&request_id, id))).await.is_err() {
                    return;
                }
            }
            let _ = tx.send(Ok(ts_complete(&request_id))).await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// A parallel-shape prefill engine (the TokenSpeed proto) that streams a
/// chunk and then fails its stream: a prefill side cut after it began, read
/// into a collected answer.
#[derive(Clone)]
struct CutPrefill {
    code: Code,
}

#[tonic::async_trait]
impl TokenSpeedAnswers for CutPrefill {
    #[expect(
        clippy::disallowed_methods,
        reason = "test mock; the producer task ends with its two items"
    )]
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        let request_id = request.into_inner().request_id;
        let code = self.code;
        let (tx, rx) = mpsc::channel(2);
        tokio::spawn(async move {
            let chunk = ts::GenerateResponse {
                request_id,
                response: Some(ts::generate_response::Response::Chunk(
                    ts::GenerateStreamChunk {
                        token_ids: vec![1],
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        index: 0,
                        ..Default::default()
                    },
                )),
            };
            if tx.send(Ok(chunk)).await.is_err() {
                return;
            }
            let _ = tx.send(Err(Status::new(code, REASON))).await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// Serve a sequential-shape mock engine in-process; returns its port.
async fn start_engine(behaviour: Behaviour) -> u16 {
    start_engine_of(RuntimeType::Vllm, behaviour).await.0
}

/// Serve a mock engine with `behaviour` in-process, on the proto `runtime`'s
/// dispatch shape speaks (the vLLM proto for the sequential shape, the
/// TokenSpeed proto for the parallel one); returns its port and its count of
/// generate calls.
async fn start_engine_of(runtime: RuntimeType, behaviour: Behaviour) -> (u16, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let port = match runtime {
        RuntimeType::TokenSpeed => {
            serve_engine(
                TokenSpeedEngine {
                    behaviour,
                    calls: Arc::clone(&calls),
                }
                .into_server(),
            )
            .await
        }
        _ => {
            serve_engine(
                Engine {
                    behaviour,
                    calls: Arc::clone(&calls),
                }
                .into_server(),
            )
            .await
        }
    };
    (port, calls)
}

/// A PD gRPC router over `workers`, registered as `runtime` (which picks the
/// dispatch shape), with the retry loop at its defaults and round-robin role
/// policies, so the first request goes to the first registered worker of
/// each role: the rows that read how a worker's own answer meets the retry
/// layer.
async fn build_retrying_pd_router(
    workers: &[(u16, WorkerType)],
    runtime: RuntimeType,
) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .build_unchecked();
    build_grpc_router(config, MODEL, grpc_workers(workers, runtime, MODEL)).await
}

/// A PD gRPC router over `prefill_port` and `decode_port`, registered as
/// `runtime` (which picks the dispatch shape), retries off so the first
/// attempt's answer is the client's.
async fn build_pd_router(
    prefill_port: u16,
    decode_port: u16,
    runtime: RuntimeType,
) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .disable_retries()
        .build_unchecked();
    build_grpc_router(
        config,
        MODEL,
        grpc_workers(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            runtime,
            MODEL,
        ),
    )
    .await
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-side-failure-codes-test-tenant"))
}

/// One chat through `router`, streamed or not: its status and body.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_chat(router: &dyn RouterTrait, streamed: bool) -> (http::StatusCode, String) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": streamed,
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

/// The envelope's code and message, from a JSON error body; `none` and an
/// empty message for a body that is not the envelope.
fn envelope(body: &str) -> (String, String) {
    let json = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    (
        json["error"]["code"].as_str().unwrap_or("none").to_string(),
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    )
}

/// Which side fails in a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Prefill,
    Decode,
}

/// The sequential shape, row by row: the failing side's behaviour, whether
/// the request is streamed, and the status, code and words the client reads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_side_failure_is_named_by_the_shared_vocabulary_on_the_sequential_shape() {
    use Behaviour::{FailsAfterTokens, FailsFirst, RefusesStart};
    use Side::{Decode, Prefill};

    let rows = [
        (
            Prefill,
            RefusesStart(Code::Unavailable),
            false,
            503,
            "prefill_worker_failed_to_start",
        ),
        (
            Prefill,
            RefusesStart(Code::Cancelled),
            false,
            502,
            "prefill_worker_failed_to_start",
        ),
        (
            Prefill,
            RefusesStart(Code::Unavailable),
            true,
            503,
            "prefill_worker_failed_to_start",
        ),
        (
            Prefill,
            FailsFirst(Code::Internal),
            false,
            500,
            "prefill_stream_error",
        ),
        (
            Prefill,
            FailsFirst(Code::Internal),
            true,
            500,
            "prefill_stream_error",
        ),
        (
            Prefill,
            FailsAfterTokens(Code::Internal),
            false,
            500,
            "prefill_stream_error",
        ),
        (
            Decode,
            RefusesStart(Code::Unavailable),
            false,
            503,
            "decode_worker_failed_to_start",
        ),
        (
            Decode,
            RefusesStart(Code::Cancelled),
            true,
            502,
            "decode_worker_failed_to_start",
        ),
        (
            Decode,
            FailsFirst(Code::Unavailable),
            false,
            503,
            "decode_stream_error",
        ),
        (
            Decode,
            FailsAfterTokens(Code::Unavailable),
            false,
            503,
            "decode_stream_error",
        ),
    ];
    let total = rows.len();
    let mut mismatches = Vec::new();
    for (side, behaviour, streamed, status, code) in rows {
        let case = format!("{side:?} {behaviour:?} streamed={streamed}");
        let (prefill, decode) = match side {
            Prefill => (behaviour, Behaviour::Completes),
            Decode => (Behaviour::Completes, behaviour),
        };
        let prefill_port = start_engine(prefill).await;
        let decode_port = start_engine(decode).await;
        let router = build_pd_router(prefill_port, decode_port, RuntimeType::Vllm).await;

        let (answered, body) = one_chat(router.as_ref(), streamed).await;

        // Every row is read to its end, so one run names every row that
        // answers something else.
        let (answered_code, message) = envelope(&body);
        let words = match code {
            "prefill_worker_failed_to_start" => "Prefill worker failed to start: ",
            "decode_worker_failed_to_start" => "Decode worker failed to start: ",
            "prefill_stream_error" => "Prefill stream error: ",
            _ => "Decode stream error: ",
        };
        let right = answered.as_u16() == status
            && answered_code == code
            && message.starts_with(words)
            && message.ends_with(REASON);
        if !right {
            mismatches.push(format!(
                "{case}: expected {status} {code} {words:?}..., got {} {answered_code} {message:?}",
                answered.as_u16()
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {total} rows do not answer the shared vocabulary:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Both sides complete: the answer's usage carries the token count the
/// decode's completion scripted, so the count the rows' engines send is read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completing_pair_answers_with_the_scripted_completion_count() {
    let prefill_port = start_engine(Behaviour::Completes).await;
    let decode_port = start_engine(Behaviour::Completes).await;
    let router = build_pd_router(prefill_port, decode_port, RuntimeType::Vllm).await;

    let (status, body) = one_chat(router.as_ref(), false).await;

    assert_eq!(status, http::StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).expect("a JSON answer");
    assert_eq!(
        json["usage"]["completion_tokens"].as_u64(),
        Some(u64::from(OUTPUT_TOKENS)),
        "the usage carries the decode's completion count: {body}"
    );
}

/// A decode stream that fails once the streamed response began, at its
/// first item or after its tokens, cannot change the status: the typed
/// `stream_error` event ends the stream, with the engine's words, as the HTTP
/// router's relay ends a cut decode stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_decode_stream_that_fails_after_the_response_began_ends_with_the_typed_event() {
    for behaviour in [
        Behaviour::FailsFirst(Code::Unavailable),
        Behaviour::FailsAfterTokens(Code::Unavailable),
    ] {
        let prefill_port = start_engine(Behaviour::Completes).await;
        let decode_port = start_engine(behaviour).await;
        let router = build_pd_router(prefill_port, decode_port, RuntimeType::Vllm).await;

        let (status, body) = one_chat(router.as_ref(), true).await;

        assert_eq!(status, http::StatusCode::OK, "{behaviour:?}: {body}");
        assert!(
            body.contains(r#""type":"stream_error""#),
            "{behaviour:?}: {body}"
        );
        assert!(
            body.contains(&format!("Stream error: {REASON}")),
            "{behaviour:?}: {body}"
        );
        assert!(body.ends_with("data: [DONE]\n\n"), "{behaviour:?}: {body}");
    }
}

/// The parallel shape's prefill side, cut after it began and read into a
/// collected answer, is its stream's failure under the same code as the
/// sequential shape's drain gives it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_parallel_prefill_cut_after_it_began_is_its_stream_failure_on_a_collected_answer() {
    let prefill_port = serve_engine(
        CutPrefill {
            code: Code::Internal,
        }
        .into_server(),
    )
    .await;
    let decode_port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: OUTPUT_TOKENS,
    })
    .await;
    let router = build_pd_router(prefill_port, decode_port, RuntimeType::TokenSpeed).await;

    let (status, body) = one_chat(router.as_ref(), false).await;

    assert_eq!(status, http::StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    let (code, message) = envelope(&body);
    assert_eq!(code, "prefill_stream_error", "{body}");
    assert_eq!(message, format!("Prefill stream error: {REASON}"), "{body}");
}

/// The two dispatch shapes the rows below run on, by the runtime that picks
/// each.
const SHAPES: [RuntimeType; 2] = [RuntimeType::Vllm, RuntimeType::TokenSpeed];

/// The sole worker of a side's kind answers its own `INTERNAL` before the
/// side's first byte, with the retry loop at its defaults: the worker's 500
/// under the side's start code is the client's answer on both shapes, with
/// the worker's words. No worker of the kind is left to take the request, so
/// the answer stands; it is not replaced by a `no_available_workers` refusal,
/// which would name an outage the healthy pair does not have.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sole_workers_own_500_before_its_first_byte_is_the_clients_answer_on_both_shapes() {
    use Behaviour::{Completes, RefusesStart};
    use Side::{Decode, Prefill};

    let mut mismatches = Vec::new();
    for runtime in SHAPES {
        for (side, code) in [
            (Prefill, "prefill_worker_failed_to_start"),
            (Decode, "decode_worker_failed_to_start"),
        ] {
            let (prefill, decode) = match side {
                Prefill => (RefusesStart(Code::Internal), Completes),
                Decode => (Completes, RefusesStart(Code::Internal)),
            };
            let (prefill_port, prefill_calls) = start_engine_of(runtime, prefill).await;
            let (decode_port, decode_calls) = start_engine_of(runtime, decode).await;
            let router = build_retrying_pd_router(
                &[
                    (prefill_port, WorkerType::Prefill),
                    (decode_port, WorkerType::Decode),
                ],
                runtime,
            )
            .await;

            let (answered, body) = one_chat(router.as_ref(), false).await;

            let (answered_code, message) = envelope(&body);
            let right = answered == http::StatusCode::INTERNAL_SERVER_ERROR
                && answered_code == code
                && message.ends_with(REASON);
            if !right {
                mismatches.push(format!(
                    "{runtime:?} {side:?}: expected 500 {code} ...{REASON:?}, got {} {answered_code} {message:?}",
                    answered.as_u16()
                ));
            }
            // The side that answered was asked once: a terminal first answer,
            // not a replay that ran the retries out. The other side's count
            // stays out of it: on the sequential shape a refused prefill means
            // the decode is never called.
            let answering_calls = match side {
                Prefill => &prefill_calls,
                Decode => &decode_calls,
            };
            if answering_calls.load(Ordering::SeqCst) != 1 {
                mismatches.push(format!(
                    "{runtime:?} {side:?}: the worker that answered was asked {} times, not once",
                    answering_calls.load(Ordering::SeqCst)
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "the worker's own answer is not the client's:\n{}",
        mismatches.join("\n")
    );
}

/// One decode worker answers its own `INTERNAL` before the side's first byte
/// while another decode worker is free: the retry window closes on the worker
/// that answered, not on the pair, so the request moves to the free decode
/// worker with the same prefill and is answered, on both shapes; the worker
/// that answered is not asked again for the request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_decode_workers_own_500_fails_over_to_the_free_decode_worker_on_both_shapes() {
    use Behaviour::{Completes, RefusesStart};

    for runtime in SHAPES {
        let (prefill_port, _) = start_engine_of(runtime, Completes).await;
        let (answering_port, answering_calls) =
            start_engine_of(runtime, RefusesStart(Code::Internal)).await;
        let (free_port, free_calls) = start_engine_of(runtime, Completes).await;
        // Round robin: the first request's decode side is the first decode
        // worker registered, the one that answers 500.
        let router = build_retrying_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (answering_port, WorkerType::Decode),
                (free_port, WorkerType::Decode),
            ],
            runtime,
        )
        .await;

        let (status, body) = one_chat(router.as_ref(), false).await;

        assert_eq!(status, http::StatusCode::OK, "{runtime:?}: {body}");
        assert_eq!(
            answering_calls.load(Ordering::SeqCst),
            1,
            "{runtime:?}: the decode worker that answered is asked once, never again for the request"
        );
        assert_eq!(
            free_calls.load(Ordering::SeqCst),
            1,
            "{runtime:?}: the free decode worker takes the request"
        );
    }
}

/// Every decode worker answered its own `INTERNAL` before the side's first
/// byte: no worker of the kind is left, and the last worker's 500 under the
/// side's start code stands as the client's answer on both shapes, each
/// worker asked once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn once_every_decode_worker_answered_its_own_500_the_last_answer_stands_on_both_shapes() {
    use Behaviour::{Completes, RefusesStart};

    for runtime in SHAPES {
        let (prefill_port, _) = start_engine_of(runtime, Completes).await;
        let (first_port, first_calls) =
            start_engine_of(runtime, RefusesStart(Code::Internal)).await;
        let (second_port, second_calls) =
            start_engine_of(runtime, RefusesStart(Code::Internal)).await;
        let router = build_retrying_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (first_port, WorkerType::Decode),
                (second_port, WorkerType::Decode),
            ],
            runtime,
        )
        .await;

        let (status, body) = one_chat(router.as_ref(), false).await;

        assert_eq!(
            status,
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "{runtime:?}: {body}"
        );
        let (code, message) = envelope(&body);
        assert_eq!(code, "decode_worker_failed_to_start", "{runtime:?}: {body}");
        assert!(message.ends_with(REASON), "{runtime:?}: {body}");
        assert_eq!(
            (
                first_calls.load(Ordering::SeqCst),
                second_calls.load(Ordering::SeqCst)
            ),
            (1, 1),
            "{runtime:?}: each decode worker is asked once"
        );
    }
}
