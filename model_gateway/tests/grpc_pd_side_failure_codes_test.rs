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

use std::time::Duration;

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_workers, serve_canned_worker, serve_engine, Canned,
};
use proto::generate_response::Response as GenResp;
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::{
    common_proto as grpc_common, tokenspeed_scheduler::tokenspeed_proto as ts, vllm_proto as proto,
};
use smg_test_harness as common;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Response, Status};
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

const MODEL: &str = "pd-side-failure-codes-test-model";
/// Tokens an engine that completes produces; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Longer than any case: a request held past it fails the case instead of
/// hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The words the failing engines put on their status.
const REASON: &str = "the engine says no";

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

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

/// A sequential-shape mock engine (the vLLM proto) with one behaviour.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
}

fn chunk(id: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Chunk(proto::GenerateStreamChunk {
            token_ids: vec![id],
            prompt_tokens: 1,
            completion_tokens: 1,
            index: 0,
            ..Default::default()
        })),
    }
}

fn complete(tokens: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Complete(proto::GenerateComplete {
            output_ids: (1..=tokens).collect(),
            finish_reason: "stop".to_string(),
            prompt_tokens: 1,
            completion_tokens: tokens,
            index: 0,
            ..Default::default()
        })),
    }
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
                    _ => Ok(complete(OUTPUT_TOKENS)),
                };
                let _ = tx.send(last).await;
            }
            // Refused before any stream: nothing to feed, the sender drops.
            Behaviour::RefusesStart(_) => {}
        }
    });
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for Engine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        if let Behaviour::RefusesStart(code) = self.behaviour {
            return Err(Status::new(code, REASON));
        }
        let (tx, rx) = mpsc::channel(8);
        produce(tx, self.behaviour);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Ok(Response::new(proto::AbortResponse::default()))
    }

    async fn flush_cache(
        &self,
        _request: Request<grpc_common::FlushCacheRequest>,
    ) -> Result<Response<grpc_common::FlushCacheResponse>, Status> {
        Ok(Response::new(grpc_common::FlushCacheResponse::default()))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Ok(Response::new(proto::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Ok(Response::new(proto::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Ok(Response::new(proto::GetLoadsResponse::default()))
    }

    type GetTokenizerStream = ReceiverStream<Result<grpc_common::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<grpc_common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type SubscribeKvEventsStream = ReceiverStream<Result<grpc_common::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<grpc_common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
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
impl TokenSpeedScheduler for CutPrefill {
    type GenerateStream = ReceiverStream<Result<ts::GenerateResponse, Status>>;
    type GetTokenizerStream = ReceiverStream<Result<grpc_common::GetTokenizerChunk, Status>>;
    type SubscribeKvEventsStream = ReceiverStream<Result<grpc_common::KvEventBatch, Status>>;

    #[expect(
        clippy::disallowed_methods,
        reason = "test mock; the producer task ends with its two items"
    )]
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
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
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Ok(Response::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Ok(Response::new(ts::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(ts::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Ok(Response::new(ts::GetLoadsResponse::default()))
    }

    async fn flush_cache(
        &self,
        _request: Request<grpc_common::FlushCacheRequest>,
    ) -> Result<Response<grpc_common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn start_profile(
        &self,
        _request: Request<grpc_common::StartProfileRequest>,
    ) -> Result<Response<grpc_common::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn stop_profile(
        &self,
        _request: Request<grpc_common::StopProfileRequest>,
    ) -> Result<Response<grpc_common::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<grpc_common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<grpc_common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }
}

/// Serve a sequential-shape mock engine in-process; returns its port.
async fn start_engine(behaviour: Behaviour) -> u16 {
    serve_engine(proto::vllm_engine_server::VllmEngineServer::new(Engine {
        behaviour,
    }))
    .await
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
    let prefill_port = serve_engine(TokenSpeedSchedulerServer::new(CutPrefill {
        code: Code::Internal,
    }))
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
