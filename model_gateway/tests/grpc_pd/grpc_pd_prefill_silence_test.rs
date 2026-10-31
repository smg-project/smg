//! A disaggregated prefill side that produces nothing is given up on within
//! its silence bound, on both dispatch shapes, instead of holding the request
//! (and the decode room admission claimed for it) until the request's own
//! timeout: the attempt answers 504 `prefill_worker_silent`, the prefill
//! worker is charged one `timeout` on the worker-error family, the decode
//! side is never asked to serve what the prefill never produced, and with
//! another prefill worker available the request is retried there, never on
//! the silent one again.
//!
//! The mock engines of the sequential shape speak the vLLM proto: one holds
//! its response headers for good (a servicer that flushes headers with the
//! first message, in front of a frozen engine), one returns a stream at once
//! and never writes to it (a frozen engine behind a servicer that flushes
//! headers eagerly), one answers after a delay. The parallel shape uses the
//! mock-worker crate's canned worker, whose delay runs before its stream is
//! returned, set longer than the test.
//!
//! The bound is the default wedge threshold (three seconds) for these short
//! prompts; every case is capped well below the request timeout, so a
//! gateway without the bound fails the case instead of hanging it.
//!
//! A prefill that answers late is another matter: on the parallel shape a
//! prefill side's first response also waits for the decode side to admit it, so a
//! worker that keeps answering calls one by one is backed up, not frozen.
//! Two cases queue the calls at the mock engine's gate (one every 1.2 s,
//! five at once: the last answers well past the bound) and expect every
//! request answered and nothing charged to the prefill worker. One more
//! lets a call given up on fail late, and expects the worker
//! charged once, at the verdict. One more sends a burst of long prompts to
//! a prefill worker that has answered nothing yet and whose first answer
//! comes past the first side's bound (a cold engine, or a decode side that
//! admits the burst late), then one every turn: the worker's silence counts
//! from the burst, which has the time its backlog takes, so every request
//! is answered.
//!
//! A prefill engine may also answer an engine step after its handoff (a
//! TokenSpeed prefill worker answers once it has polled the transfer's
//! success, after its next forward), and the decode side's first output is
//! the handoff. Two cases send a burst, after a warm-up, to a prefill worker
//! that answers its first burst call past the bound and then one every turn,
//! while each decode side shows its handoff before the bound: in the
//! dispatch phase (the calls return with their first message, as the Python
//! servicers send them) and in the first-response phase (the streams return
//! at once, as the Rust servicers send them). Every request is answered in
//! full (the decode side's first chunk, read ahead as the handoff, reaches
//! its client once) and nothing is charged. Two more freeze the prefill after
//! its handoff, or let its engine step after the handoff outlast the bound,
//! and expect the side given up on a bound after the handoff: a handoff
//! stands in for an answer for one bound, as an answer does.
//!
//! The decode side of the sequential shape waits on a KV transfer once the
//! prefill side has answered, and the wedge threshold never judges that
//! wait: its bound is the KV-transfer timeout its decode worker declares
//! (the `kv_transfer_timeout_secs` label), else the request's own timeout. Two
//! cases hold the decode call's headers or return an empty stream after a
//! prefill that answered at once, on a decode worker that declares a
//! three-second transfer timeout, and expect 504 `decode_worker_silent`
//! within that bound, one `timeout` on the decode worker and nothing on the
//! prefill; a third, on a decode worker that declares none, answers past the
//! wedge threshold and expects the request answered in full with nothing
//! charged (a handoff that waits on a prefill engine that stalled right after
//! answering, or a decode engine that never schedules the side, is the
//! request timeout's to end there).

use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use common::grpc_harness::{
    build_grpc_router, grpc_router_config, grpc_worker, serve_canned_worker, serve_engine, Canned,
};
use futures::future::join_all;
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::model_card::ModelCard;
use proto::generate_response::Response as GenResp;
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        RuntimeType, Worker, WorkerType,
    },
};
use smg_grpc_client::{
    common_proto as grpc_common, tokenspeed_scheduler::tokenspeed_proto as ts, vllm_proto as proto,
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

use crate::common;

const MODEL: &str = "pd-prefill-silence-test-model";
/// Tokens every answering mock produces; ids 1..=3 are in the mock tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// The default wedge threshold: the silence bound of a short prompt on an
/// idle prefill worker.
const BOUND: Duration = Duration::from_secs(3);
/// Scheduling slack for a loaded machine, on top of the bound.
const SLACK: Duration = Duration::from_secs(3);
/// Longer than any case: a gateway that does not bound the side's silence
/// fails the case here instead of hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The delay of a mock that must never answer within a case.
const FOREVER: Duration = Duration::from_secs(3_600);
/// How often a queued engine's gate lets a call through: five calls at once
/// put the last one well past the bound, with an answer every turn.
const TURN: Duration = Duration::from_millis(1_200);
/// How many prefill calls queue at once in the queued cases.
const QUEUED_CALLS: usize = 5;
/// How long after the bound a held call fails, in the late-failure case.
const LATE: Duration = Duration::from_secs(2);
/// How long after its first call a cold engine answers it, in the
/// cold-burst case: past the first side's bound, inside the time the
/// burst's backlog takes at the cold prior.
const COLD_START: Duration = Duration::from_secs(5);
/// The prompt of each cold-burst request, in tokens (the mock tokenizer's
/// `Hello` is one): the five of them are ten seconds of prefill at the
/// cold prior of 10k tokens/s.
const LONG_PROMPT_TOKENS: usize = 20_000;
/// How long after its dispatch a decode side shows the prefill's handoff,
/// in the handoff cases: inside the bound.
const HANDOFF: Duration = Duration::from_millis(2_500);
/// How long after a burst the prefill answers its first call, in the
/// handoff cases: past the bound, an engine step after the handoff.
const LATE_ANSWER: Duration = Duration::from_secs(4);

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

/// What a sequential-dispatch mock engine does with a generate call.
#[derive(Clone, Copy)]
enum Behaviour {
    /// The call never returns its stream.
    HoldHeaders,
    /// The stream is returned at once and nothing is ever written to it.
    EmptyStream,
    /// Tokens and a completion after the delay, the stream returned at once.
    Answer(Duration),
    /// [`Behaviour::Answer`] whose completion carries KV-transfer params, as
    /// a prefill side's does when its decode side pulls the prompt KV from
    /// it: the handoff that makes the decode side a pulled one.
    AnswerWithHandoff(Duration),
    /// The call returns its stream when its turn comes at the engine's
    /// gate (one call every `TURN`), the tokens at once after that.
    Queued,
    /// The call holds its headers for the delay, then fails.
    HoldHeadersThenFail(Duration),
}

/// A gate that lets one call through every turn: calls queue in arrival order,
/// as they do in front of a decode side that pre-allocates for one at a time.
#[derive(Clone, Default)]
struct Gate {
    next: Arc<tokio::sync::Mutex<Option<tokio::time::Instant>>>,
    /// How long after the first call its turn comes: zero for an engine that
    /// answers its first call at once.
    first_turn_after: Duration,
}

impl Gate {
    /// Wait for this call's turn.
    async fn turn(&self) {
        let at = {
            let mut next = self.next.lock().await;
            let now = tokio::time::Instant::now();
            let at = next.map_or(now + self.first_turn_after, |next| next.max(now));
            *next = Some(at + TURN);
            at
        };
        tokio::time::sleep_until(at).await;
    }
}

/// A sequential-dispatch mock engine, counting its generate calls.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
    gate: Gate,
    calls: Arc<AtomicUsize>,
}

impl Engine {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            behaviour,
            gate: Gate::default(),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
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

/// The completion of a stream; with `handoff` it is a prefill side's whose
/// decode side pulls the prompt KV from it: the connector's transfer params
/// ride the completion.
fn complete_with_handoff(handoff: bool) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Complete(proto::GenerateComplete {
            output_ids: (1..=OUTPUT_TOKENS).collect(),
            finish_reason: "stop".to_string(),
            prompt_tokens: 1,
            completion_tokens: OUTPUT_TOKENS,
            index: 0,
            kv_transfer_params_json: handoff.then(|| {
                r#"{"do_remote_prefill":true,"remote_block_ids":[1],"remote_engine_id":"prefill","remote_host":"127.0.0.1","remote_port":1}"#.to_string()
            }),
            ..Default::default()
        })),
    }
}

/// Feed one generate stream: the tokens after `delay`, then the completion.
/// With `delay` at [`FOREVER`] the sender is held and nothing is written.
fn produce_after(tx: GenerateTx, delay: Duration) {
    produce_after_with_handoff(tx, delay, false);
}

/// [`produce_after`] whose completion hands off when `handoff` is set.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_after_with_handoff(tx: GenerateTx, delay: Duration, handoff: bool) {
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        for id in 1..=OUTPUT_TOKENS {
            if tx.send(Ok(chunk(id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(complete_with_handoff(handoff))).await;
    });
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for Engine {
    async fn start_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StartProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn stop_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StopProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(8);
        match self.behaviour {
            Behaviour::HoldHeaders => {
                tokio::time::sleep(FOREVER).await;
            }
            Behaviour::EmptyStream => produce_after(tx, FOREVER),
            Behaviour::Answer(delay) => produce_after(tx, delay),
            Behaviour::AnswerWithHandoff(delay) => produce_after_with_handoff(tx, delay, true),
            Behaviour::Queued => {
                self.gate.turn().await;
                produce_after(tx, Duration::ZERO);
            }
            Behaviour::HoldHeadersThenFail(delay) => {
                tokio::time::sleep(delay).await;
                return Err(Status::internal("the engine gave up on the request"));
            }
        }
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
        _request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        Ok(Response::new(
            smg_grpc_client::common_proto::FlushCacheResponse::default(),
        ))
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

    type GetTokenizerStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<smg_grpc_client::common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type SubscribeKvEventsStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<smg_grpc_client::common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }
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

/// Feed one parallel-dispatch generate stream once the call's turn at `gate`
/// comes: the tokens, then the completion.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_at_turn(
    tx: mpsc::Sender<Result<ts::GenerateResponse, Status>>,
    gate: Gate,
    request_id: String,
) {
    tokio::spawn(async move {
        gate.turn().await;
        for id in 1..=OUTPUT_TOKENS {
            if tx.send(Ok(ts_chunk(&request_id, id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(ts_complete(&request_id))).await;
    });
}

/// Hold a parallel-dispatch generate stream open with nothing on it: a call
/// whose first response never comes.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn hold_open(tx: mpsc::Sender<Result<ts::GenerateResponse, Status>>) {
    tokio::spawn(async move {
        tokio::time::sleep(FOREVER).await;
        drop(tx);
    });
}

/// Feed a parallel-dispatch generate stream its tokens at once and then hold
/// it open without a completion: a decode engine that stalled mid-stream.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_then_stall(tx: mpsc::Sender<Result<ts::GenerateResponse, Status>>, request_id: String) {
    tokio::spawn(async move {
        for id in 1..=OUTPUT_TOKENS {
            if tx.send(Ok(ts_chunk(&request_id, id))).await.is_err() {
                return;
            }
        }
        tokio::time::sleep(FOREVER).await;
        drop(tx);
    });
}

/// What a parallel-dispatch mock engine does with its generate calls.
#[derive(Clone, Copy)]
enum ParallelShape {
    /// Every call's first chunk waits for its turn at the engine's gate.
    Queued,
    /// The first call is answered at once; every later call's stream stays
    /// empty for good (a prefill worker whose calls wait on a stalled decode side).
    AnswerFirstThenHold,
    /// The first call is answered at once; every later call never returns
    /// its stream (a servicer that sends its headers with the first message,
    /// in front of a prefill worker whose calls wait on a stalled decode side).
    AnswerFirstThenHoldHeaders,
    /// Every call streams its tokens at once and then stalls without a
    /// completion (a decode engine that stopped mid-stream).
    StreamThenStall,
    /// The first call is answered at once; every later call's first chunk
    /// comes at its turn at the engine's gate (`queued`) or at the gate's
    /// first turn after the call, with its headers when `headers_held` (a
    /// servicer that sends them with the first message).
    AnswerFirstThenLate { queued: bool, headers_held: bool },
}

/// A parallel-dispatch mock engine (the TokenSpeed proto, as the mock-worker
/// crate's worker speaks it): the generate call returns its stream at once,
/// and what follows on it is the engine's [`ParallelShape`]: queued, the
/// call's first chunk waits for its turn at the engine's gate (one call every
/// `TURN`: the decode side admitting the calls one by one), then the
/// tokens and the completion follow. Counts its generate calls.
#[derive(Clone)]
struct QueuedEngine {
    gate: Gate,
    shape: ParallelShape,
    calls: Arc<AtomicUsize>,
}

impl QueuedEngine {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[tonic::async_trait]
impl TokenSpeedScheduler for QueuedEngine {
    type GenerateStream = ReceiverStream<Result<ts::GenerateResponse, Status>>;
    type GetTokenizerStream = ReceiverStream<Result<grpc_common::GetTokenizerChunk, Status>>;
    type SubscribeKvEventsStream = ReceiverStream<Result<grpc_common::KvEventBatch, Status>>;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = mpsc::channel(8);
        let request_id = request.into_inner().request_id;
        match self.shape {
            ParallelShape::Queued => produce_at_turn(tx, self.gate.clone(), request_id),
            ParallelShape::AnswerFirstThenHold if call == 1 => {
                produce_at_turn(tx, self.gate.clone(), request_id);
            }
            ParallelShape::AnswerFirstThenHold => hold_open(tx),
            ParallelShape::AnswerFirstThenHoldHeaders if call == 1 => {
                produce_at_turn(tx, self.gate.clone(), request_id);
            }
            ParallelShape::AnswerFirstThenHoldHeaders => {
                tokio::time::sleep(FOREVER).await;
                drop(tx);
            }
            ParallelShape::StreamThenStall => produce_then_stall(tx, request_id),
            ParallelShape::AnswerFirstThenLate { .. } if call == 1 => {
                produce_at_turn(tx, Gate::default(), request_id);
            }
            ParallelShape::AnswerFirstThenLate {
                queued,
                headers_held,
            } => {
                let gate = if queued {
                    self.gate.clone()
                } else {
                    Gate {
                        first_turn_after: self.gate.first_turn_after,
                        ..Gate::default()
                    }
                };
                if headers_held {
                    gate.turn().await;
                    produce_at_turn(tx, Gate::default(), request_id);
                } else {
                    produce_at_turn(tx, gate, request_id);
                }
            }
        }
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

/// Serve a queued parallel-dispatch mock engine in-process; returns its port
/// and the engine (for its call count).
async fn start_queued_grpc_worker(shape: ParallelShape) -> (u16, QueuedEngine) {
    let engine = QueuedEngine {
        gate: Gate::default(),
        shape,
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let port = serve_engine(TokenSpeedSchedulerServer::new(engine.clone())).await;
    (port, engine)
}

/// [`start_queued_grpc_worker`] for an engine that answers its first call
/// `cold` after it arrives, and one call every turn after that.
async fn start_cold_queued_grpc_worker(cold: Duration) -> (u16, QueuedEngine) {
    let engine = QueuedEngine {
        gate: Gate {
            first_turn_after: cold,
            ..Gate::default()
        },
        shape: ParallelShape::Queued,
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let port = serve_engine(TokenSpeedSchedulerServer::new(engine.clone())).await;
    (port, engine)
}

/// [`start_queued_grpc_worker`] for [`ParallelShape::AnswerFirstThenLate`],
/// whose gate's first turn comes `first_turn_after` the call that waits on
/// it first.
async fn start_late_grpc_worker(
    queued: bool,
    headers_held: bool,
    first_turn_after: Duration,
) -> (u16, QueuedEngine) {
    let engine = QueuedEngine {
        gate: Gate {
            first_turn_after,
            ..Gate::default()
        },
        shape: ParallelShape::AnswerFirstThenLate {
            queued,
            headers_held,
        },
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let port = serve_engine(TokenSpeedSchedulerServer::new(engine.clone())).await;
    (port, engine)
}

/// Serve a sequential-dispatch mock engine in-process; returns its port.
async fn start_mock_engine(engine: Engine) -> u16 {
    serve_engine(proto::vllm_engine_server::VllmEngineServer::new(engine)).await
}

/// Serve the mock-worker crate's canned worker in-process (a parallel-dispatch
/// runtime once registered as one): one up-front `delay` before its stream is
/// returned, then `OUTPUT_TOKENS` tokens; returns its port.
async fn start_mock_grpc_worker(delay: Duration) -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: delay,
        output_tokens: OUTPUT_TOKENS,
    })
    .await
}

/// A parallel-dispatch mock engine (the TokenSpeed proto) that accepts every
/// generate call, returns its stream at once and fails it with the first item
/// (`UNAVAILABLE`): a decode engine that admits a request and then drops it
/// before any token.
#[derive(Clone, Default)]
struct CutEngine;

#[tonic::async_trait]
impl TokenSpeedScheduler for CutEngine {
    type GenerateStream = ReceiverStream<Result<ts::GenerateResponse, Status>>;
    type GetTokenizerStream = ReceiverStream<Result<grpc_common::GetTokenizerChunk, Status>>;
    type SubscribeKvEventsStream = ReceiverStream<Result<grpc_common::KvEventBatch, Status>>;

    #[expect(
        clippy::disallowed_methods,
        reason = "test mock; the producer task ends with its one item"
    )]
    async fn generate(
        &self,
        _request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let _ = tx
                .send(Err(Status::unavailable("the engine dropped the request")))
                .await;
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

/// Serve a [`CutEngine`] in-process; returns its port.
async fn start_cut_grpc_worker() -> u16 {
    serve_engine(TokenSpeedSchedulerServer::new(CutEngine)).await
}

/// A PD gRPC router over the given workers, registered as `runtime` (which picks
/// the dispatch shape), with round-robin role policies so the first request
/// goes to the first registered worker of each role.
async fn build_pd_router(
    workers: &[(u16, WorkerType)],
    runtime: RuntimeType,
) -> Box<dyn RouterTrait> {
    build_pd_router_with(workers, runtime, CircuitBreakerConfig::default())
        .await
        .0
}

/// [`build_pd_router`] with the decode workers declaring a KV-transfer
/// timeout of `secs` (the `kv_transfer_timeout_secs` worker label): the bound
/// on a pulled decode side's first token, counted from its dispatch.
async fn build_pd_router_with_decode_transfer_timeout(
    endpoints: &[(u16, WorkerType)],
    runtime: RuntimeType,
    secs: u64,
) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .build_unchecked();
    let workers: Vec<Arc<dyn Worker>> = endpoints
        .iter()
        .map(|&(port, worker_type)| {
            let builder = grpc_worker(port, worker_type, runtime, ModelCard::new(MODEL));
            let builder = if worker_type == WorkerType::Decode {
                builder.label("kv_transfer_timeout_secs", secs.to_string())
            } else {
                builder
            };
            Arc::new(builder.build()) as Arc<dyn Worker>
        })
        .collect();
    build_grpc_router(config, MODEL, workers).await
}

/// [`build_pd_router`] with every worker's circuit breaker configured as
/// `breaker`; the registered workers come back with the router.
async fn build_pd_router_with(
    endpoints: &[(u16, WorkerType)],
    runtime: RuntimeType,
    breaker: CircuitBreakerConfig,
) -> (Box<dyn RouterTrait>, Vec<Arc<dyn Worker>>) {
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .build_unchecked();
    let workers: Vec<Arc<dyn Worker>> = endpoints
        .iter()
        .map(|&(port, worker_type)| {
            Arc::new(
                grpc_worker(port, worker_type, runtime, ModelCard::new(MODEL))
                    .circuit_breaker_config(breaker.clone())
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    let router = build_grpc_router(config, MODEL, workers.clone()).await;
    (router, workers)
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-prefill-silence-test-tenant"))
}

/// One streaming chat through `router`: its status, body and duration.
async fn one_streaming_chat(router: &dyn RouterTrait) -> (http::StatusCode, String, Duration) {
    one_chat(router, true).await
}

/// One chat through `router`, streaming or not: its status, body and duration.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_chat(
    router: &dyn RouterTrait,
    streaming: bool,
) -> (http::StatusCode, String, Duration) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": streaming,
    }))
    .unwrap();
    let started = Instant::now();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect(
                "the request must be answered within the case cap, not held to the request timeout",
            );
    let status = response.status();
    let bytes = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the response body must end within the case cap")
    .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    (status, body, started.elapsed())
}

/// One streaming chat through `router` whose prompt is
/// [`LONG_PROMPT_TOKENS`] long: its status, body and duration.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_long_streaming_chat(router: &dyn RouterTrait) -> (http::StatusCode, String, Duration) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello ".repeat(LONG_PROMPT_TOKENS)}],
        "max_tokens": 8,
        "stream": true,
    }))
    .unwrap();
    let started = Instant::now();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect(
                "the request must be answered within the case cap, not held to the request timeout",
            );
    let status = response.status();
    let bytes = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the response body must end within the case cap")
    .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    (status, body, started.elapsed())
}

/// `n` streaming chats through `router` at once.
async fn concurrent_streaming_chats(
    router: &dyn RouterTrait,
    n: usize,
) -> Vec<(http::StatusCode, String, Duration)> {
    join_all((0..n).map(|_| one_streaming_chat(router))).await
}

/// The text a streamed chat's content deltas add up to; an event that does
/// not parse, or reports an error, fails the case.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn streamed_text(body: &str) -> String {
    let events = body.lines().filter_map(|line| line.strip_prefix("data: "));
    events
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str::<serde_json::Value>(data).unwrap())
        .inspect(|event| assert!(event.get("error").is_none(), "{event}"))
        .filter_map(|event| {
            event["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

/// Every chat answered in full, and the slowest of them later than the
/// bound (a case whose calls never waited past the bound proves nothing),
/// though within the turns its queue needed.
fn assert_all_answered_past_the_bound(answers: &[(http::StatusCode, String, Duration)]) {
    for (status, body, elapsed) in answers {
        assert_eq!(*status, http::StatusCode::OK, "{body} ({elapsed:?})");
        assert!(body.contains("data: [DONE]"), "{body}");
    }
    let slowest = answers
        .iter()
        .map(|(_, _, elapsed)| *elapsed)
        .max()
        .unwrap_or_default();
    assert!(
        slowest >= BOUND,
        "the last call's turn comes past the bound: {slowest:?}"
    );
    assert!(
        slowest < TURN * u32::try_from(QUEUED_CALLS).unwrap_or(u32::MAX) + SLACK,
        "answered once its turn came: {slowest:?}"
    );
}

/// Run `case` on a current-thread runtime under a thread-local recorder and
/// return the rendered `/metrics` text.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn render_after<F, Fut>(case: F) -> String
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        rt.block_on(case());
    });
    handle.render()
}

/// The value of the `smg_worker_errors_total` sample for `worker_type` and
/// `error_type` (any connection), zero when it was not emitted.
fn worker_errors(rendered: &str, worker_type: &str, error_type: &str) -> f64 {
    let labels = [
        format!("worker_type=\"{worker_type}\""),
        format!("error_type=\"{error_type}\""),
    ];
    rendered
        .lines()
        .filter(|line| line.starts_with("smg_worker_errors_total{"))
        .filter(|line| labels.iter().all(|label| line.contains(label.as_str())))
        .filter_map(|line| line.rsplit(' ').next())
        .filter_map(|value| value.parse::<f64>().ok())
        .sum()
}

/// What a given-up-on request looks like: a 504 under the side's own code,
/// answered within the bound plus slack.
fn assert_given_up_on(status: http::StatusCode, body: &str, elapsed: Duration) {
    assert_eq!(status, http::StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(body.contains("prefill_worker_silent"), "{body}");
    assert!(
        elapsed >= BOUND && elapsed < BOUND + SLACK,
        "given up on at the bound ({BOUND:?}), not before and not much after: {elapsed:?}"
    );
}

/// What a request given up on at its decode side looks like: a 504 under the
/// decode side's own code, answered within the bound plus slack.
fn assert_decode_given_up_on(status: http::StatusCode, body: &str, elapsed: Duration) {
    assert_eq!(status, http::StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(body.contains("decode_worker_silent"), "{body}");
    assert!(
        body.contains("declared KV-transfer timeout"),
        "the verdict names the declared bound: {body}"
    );
    assert!(
        elapsed >= BOUND && elapsed < BOUND + SLACK,
        "given up on at the bound ({BOUND:?}), not before and not much after: {elapsed:?}"
    );
}

/// Sequential dispatch, the prefill answers at once and the decode call
/// never returns its stream (a servicer that flushes headers with the first
/// message, in front of a handoff that waits on a prefill engine that
/// stalled right after answering), on a decode worker that declares a
/// three-second KV-transfer timeout: given up on at that bound, counted as
/// the decode worker's timeout, nothing charged to the prefill.
#[test]
fn a_sequential_decode_that_returns_no_stream_after_the_prefill_answered_is_given_up_on_within_the_bound(
) {
    let prefill = Engine::new(Behaviour::AnswerWithHandoff(Duration::ZERO));
    let decode = Engine::new(Behaviour::HoldHeaders);
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router_with_decode_transfer_timeout(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
            3,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_decode_given_up_on(status, &body, elapsed);
    });
    assert_eq!(prefill_calls.calls(), 1, "the prefill answered once");
    assert_eq!(decode_calls.calls(), 1, "not replayed on the silent decode");
    assert_eq!(
        worker_errors(&rendered, "decode", "timeout"),
        1.0,
        "one timeout on the decode worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error"),
        0.0,
        "counted once, under one class:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "the prefill, which answered, is not charged:\n{rendered}"
    );
}

/// Sequential dispatch, the prefill answers at once and the decode returns
/// its stream at once and never writes to it, the same declared timeout: the
/// same verdict, from the first-response phase.
#[test]
fn a_sequential_decode_whose_stream_stays_empty_after_the_prefill_answered_is_given_up_on_within_the_bound(
) {
    let prefill = Engine::new(Behaviour::AnswerWithHandoff(Duration::ZERO));
    let decode = Engine::new(Behaviour::EmptyStream);
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router_with_decode_transfer_timeout(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
            3,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_decode_given_up_on(status, &body, elapsed);
    });
    assert_eq!(prefill_calls.calls(), 1, "the prefill answered once");
    assert_eq!(decode_calls.calls(), 1, "not replayed on the silent decode");
    assert_eq!(
        worker_errors(&rendered, "decode", "timeout"),
        1.0,
        "one timeout on the decode worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "the prefill, which answered, is not charged:\n{rendered}"
    );
}

/// Sequential dispatch on a decode worker that declares no KV-transfer
/// timeout: the wedge threshold does not judge the decode side (its wait is
/// a transfer, load on the decode leg, not silence), so a decode that answers
/// a second past the threshold still answers the request in full, and
/// nothing is charged to either worker; the request's own timeout is what
/// would end a handoff that never completes.
#[test]
fn a_sequential_decode_whose_worker_declares_no_transfer_timeout_is_not_judged_by_the_wedge_threshold(
) {
    let prefill = Engine::new(Behaviour::AnswerWithHandoff(Duration::ZERO));
    let decode = Engine::new(Behaviour::Answer(BOUND + Duration::from_secs(1)));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(
            elapsed >= BOUND + Duration::from_secs(1) && elapsed < CASE_CAP,
            "answered when the decode answered, past the wedge threshold: {elapsed:?}"
        );
    });
    assert_eq!(prefill_calls.calls(), 1, "the prefill answered once");
    assert_eq!(decode_calls.calls(), 1, "not replayed");
    assert_eq!(
        worker_errors(&rendered, "decode", "timeout"),
        0.0,
        "no verdict on a side waiting for its transfer:\n{rendered}"
    );
    assert_eq!(worker_errors(&rendered, "prefill", "timeout"), 0.0);
}

/// Sequential dispatch, a non-streaming request: the decode side's first item
/// is its whole generation (the servicers send nothing of a non-streaming
/// request before it finishes), no first-token signal, so the bound does not
/// apply; a decode that answers past the bound still completes, and nothing
/// is charged to either worker.
#[test]
fn a_non_streaming_request_is_not_bound_by_the_decode_sides_first_token() {
    let prefill = Engine::new(Behaviour::Answer(Duration::ZERO));
    let decode = Engine::new(Behaviour::Answer(BOUND + Duration::from_secs(1)));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_chat(router.as_ref(), false).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(
            elapsed >= BOUND + Duration::from_secs(1) && elapsed < CASE_CAP,
            "answered when the decode answered, past the bound: {elapsed:?}"
        );
    });
    assert_eq!(prefill_calls.calls(), 1);
    assert_eq!(decode_calls.calls(), 1, "not replayed");
    assert_eq!(
        worker_errors(&rendered, "decode", "timeout"),
        0.0,
        "no verdict on a non-streaming side:\n{rendered}"
    );
    assert_eq!(worker_errors(&rendered, "prefill", "timeout"), 0.0);
}

/// Sequential dispatch, the prefill call never returns its stream: given up
/// on at the bound, counted as the prefill worker's timeout, the decode never
/// dispatched.
#[test]
fn a_sequential_prefill_that_returns_no_stream_is_given_up_on_within_the_bound() {
    let prefill = Engine::new(Behaviour::HoldHeaders);
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_given_up_on(status, &body, elapsed);
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "not replayed on the silent prefill"
    );
    assert_eq!(
        decode_calls.calls(),
        0,
        "the decode side was never dispatched"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "backend_error"),
        0.0,
        "counted once, under one class:\n{rendered}"
    );
}

/// Sequential dispatch, the prefill returns its stream at once and never
/// writes to it: the same verdict, from the first-response phase.
#[test]
fn a_sequential_prefill_whose_stream_stays_empty_is_given_up_on_within_the_bound() {
    let prefill = Engine::new(Behaviour::EmptyStream);
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_given_up_on(status, &body, elapsed);
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "not replayed on the silent prefill"
    );
    assert_eq!(
        decode_calls.calls(),
        0,
        "the decode side was never dispatched"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the prefill worker:\n{rendered}"
    );
}

/// Parallel dispatch (both sides sent together), the prefill never returns
/// its stream: given up on at the bound; the decode side, dispatched with it,
/// is dropped with the attempt. Two such requests are two consecutive
/// failures of the prefill worker's circuit breaker (its dispatch is not
/// sampled as a success first), so a breaker set to two opens; the decode
/// worker's stays closed.
#[test]
fn a_parallel_prefill_that_returns_no_stream_is_given_up_on_within_the_bound() {
    let rendered = render_after(|| async {
        let prefill_port = start_mock_grpc_worker(FOREVER).await;
        let decode_port = start_mock_grpc_worker(Duration::ZERO).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        for _ in 0..2 {
            let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
            assert_given_up_on(status, &body, elapsed);
        }
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Open,
            "two silences are two consecutive failures of the prefill worker"
        );
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Closed,
            "the decode worker did nothing wrong"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        2.0,
        "one timeout per request on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error"),
        0.0,
        "the decode side did nothing wrong:\n{rendered}"
    );
}

/// Parallel dispatch, a prefill worker whose calls queue at the engine's gate (the
/// decode side admitting them one by one): the later calls' first responses
/// come well past the bound, but the worker answers a call every turn, so
/// none is given up on. Every request is answered, nothing is charged to the
/// prefill worker, and its breaker (set to two) stays closed.
#[test]
fn a_parallel_prefill_whose_calls_queue_behind_the_decode_side_is_not_given_up_on() {
    let rendered = render_after(|| async {
        let (prefill_port, prefill) = start_queued_grpc_worker(ParallelShape::Queued).await;
        let decode_port = start_mock_grpc_worker(Duration::ZERO).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        let answers = concurrent_streaming_chats(router.as_ref(), QUEUED_CALLS).await;
        assert_all_answered_past_the_bound(&answers);
        assert_eq!(prefill.calls(), QUEUED_CALLS, "each call dispatched once");
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "a backed-up prefill is not a failing one"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "no call was given up on:\n{rendered}"
    );
}

/// Parallel dispatch, a burst of long prompts on a prefill worker that has
/// answered nothing yet, whose engine answers its first call past the first
/// side's bound (a cold engine, or a decode side admitting the burst late)
/// and then one call every turn. The worker's silence counts from the burst,
/// and the burst's backlog (ten seconds at the cold prior) is the time it
/// may take, so no side is given up on: every request is answered, nothing
/// is charged to the prefill worker, and its breaker stays closed. Judged
/// by each side's own bound and a worker that "never answered", the sides
/// would be given up on one after another, each before its call's turn.
#[test]
fn a_cold_burst_of_long_prompts_answered_past_the_bound_is_not_given_up_on() {
    let rendered = render_after(|| async {
        let (prefill_port, prefill) = start_cold_queued_grpc_worker(COLD_START).await;
        let decode_port = start_mock_grpc_worker(Duration::ZERO).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        let answers =
            join_all((0..QUEUED_CALLS).map(|_| one_long_streaming_chat(router.as_ref()))).await;
        for (status, body, elapsed) in &answers {
            assert_eq!(*status, http::StatusCode::OK, "{body} ({elapsed:?})");
            assert!(body.contains("data: [DONE]"), "{body}");
        }
        let first = answers
            .iter()
            .map(|(_, _, elapsed)| *elapsed)
            .min()
            .unwrap_or_default();
        assert!(
            first >= COLD_START,
            "the first answer comes past the bound: {first:?}"
        );
        assert_eq!(prefill.calls(), QUEUED_CALLS, "each call dispatched once");
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "a cold, backed-up prefill is not a failing one"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "no call was given up on:\n{rendered}"
    );
}

/// The PD router of the handoff cases over a prefill and a decode worker,
/// every breaker set to one; the workers come back with it.
async fn build_handoff_router(
    prefill: u16,
    decode: u16,
) -> (Box<dyn RouterTrait>, Vec<Arc<dyn Worker>>) {
    let breaker = CircuitBreakerConfig {
        failure_threshold: 1,
        ..Default::default()
    };
    let endpoints = [(prefill, WorkerType::Prefill), (decode, WorkerType::Decode)];
    let (router, workers) =
        build_pd_router_with(&endpoints, RuntimeType::TokenSpeed, breaker).await;
    let (status, body, _) = one_streaming_chat(router.as_ref()).await;
    assert_eq!(status, http::StatusCode::OK, "the warm-up answers: {body}");
    (router, workers)
}

/// Parallel dispatch, the incident's shape: after a warm-up, a burst of
/// short prompts on a prefill worker that answers its first burst call at
/// [`LATE_ANSWER`], past the bound, and one every turn after it, while each
/// decode side shows the handoff at [`HANDOFF`], inside it. The handoff is
/// the worker's sign of life: every request is answered, nothing is charged
/// to the prefill worker and its breaker stays closed. Without it, the
/// worker answered nothing for the bound and the burst's run nothing for its
/// small backlog, and every side was given up on at the bound.
/// `headers_held` picks the phase the sides wait in: their calls return with
/// the first message, or their streams at once. Each decode side's first
/// chunk, read ahead as the handoff, reaches its client once, in its place.
fn handoff_burst_case(headers_held: bool) {
    let rendered = render_after(|| async {
        let (prefill_port, prefill) = start_late_grpc_worker(true, headers_held, LATE_ANSWER).await;
        let (decode_port, _) = start_late_grpc_worker(false, headers_held, HANDOFF).await;
        let (router, workers) = build_handoff_router(prefill_port, decode_port).await;
        let answers = concurrent_streaming_chats(router.as_ref(), QUEUED_CALLS).await;
        for (status, body, elapsed) in &answers {
            assert_eq!(*status, http::StatusCode::OK, "{body} ({elapsed:?})");
            assert!(body.contains("data: [DONE]"), "{body}");
            assert_eq!(streamed_text(body), "Hello world test", "{body}");
        }
        let slowest = answers
            .iter()
            .map(|(_, _, elapsed)| *elapsed)
            .max()
            .unwrap_or_default();
        let turns = TURN * u32::try_from(QUEUED_CALLS - 1).unwrap_or(u32::MAX);
        assert!(
            slowest >= LATE_ANSWER && slowest < LATE_ANSWER + turns + SLACK,
            "answered at its turn: {slowest:?}"
        );
        assert_eq!(
            prefill.calls(),
            1 + QUEUED_CALLS,
            "each call dispatched once"
        );
        assert_eq!(workers[0].circuit_breaker_state(), CircuitState::Closed);
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "no side was given up on:\n{rendered}"
    );
}

#[test]
fn a_burst_on_a_parallel_prefill_that_answers_an_engine_step_after_its_handoff_is_not_given_up_on()
{
    handoff_burst_case(true);
}

#[test]
fn a_burst_on_a_parallel_prefill_whose_streams_return_at_once_is_not_given_up_on_either() {
    handoff_burst_case(false);
}

/// Parallel dispatch, a prefill worker silent after its handoff: the second
/// request's decode side shows the handoff at [`HANDOFF`] and its prefill
/// call returns `answer_after` its dispatch, a bound or more after the
/// handoff. A handoff stands in for an answer for one bound, as an answer
/// does: the side is given up on a bound after it, 504
/// `prefill_worker_silent`, one `timeout` on the prefill worker and nothing
/// on the decode worker.
fn silent_after_its_handoff_case(answer_after: Duration) {
    let rendered = render_after(|| async {
        let (prefill_port, _) = start_late_grpc_worker(true, true, answer_after).await;
        let (decode_port, _) = start_late_grpc_worker(false, false, HANDOFF).await;
        let (router, workers) = build_handoff_router(prefill_port, decode_port).await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert!(body.contains("prefill_worker_silent"), "{body}");
        assert!(
            elapsed >= HANDOFF + BOUND && elapsed < HANDOFF + BOUND + SLACK,
            "given up on a bound after the handoff: {elapsed:?}"
        );
        assert_eq!(workers[1].circuit_breaker_state(), CircuitState::Closed);
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error"),
        0.0,
        "{rendered}"
    );
}

#[test]
fn a_parallel_prefill_frozen_after_its_handoff_is_given_up_on_a_bound_after_it() {
    silent_after_its_handoff_case(FOREVER);
}

/// The limit of the handoff's credit: an engine step after the handoff that
/// outlasts the bound is given up on as a freeze is (the answer comes only
/// past the verdict's window), as one before the handoff is.
#[test]
fn a_parallel_prefill_whose_step_after_its_handoff_outlasts_the_bound_is_given_up_on_too() {
    silent_after_its_handoff_case(HANDOFF + BOUND + SLACK);
}

/// Parallel dispatch, the decode worker stalls mid-stream: the first
/// request's decode side streams its tokens and then stops without a
/// completion (its stream stays open); the second request's prefill side,
/// dispatched while the decode holds that stream, produces nothing (on this
/// shape a prefill side's first response waits for the decode side). At the
/// prefill's bound the gateway reads the decode partner: a worker holding a
/// stream it had been producing for and quiet for the bound takes the
/// verdict. The second request is answered 504 `decode_worker_silent` at
/// the bound, the decode worker is charged the `timeout` and a breaker
/// failure (its breaker, set to one, opens), the prefill worker nothing (its
/// breaker stays closed).
#[test]
fn a_parallel_prefill_waiting_on_a_decode_that_stalled_mid_stream_is_charged_to_the_decode() {
    let rendered = render_after(|| async {
        let (prefill_port, prefill) =
            start_queued_grpc_worker(ParallelShape::AnswerFirstThenHold).await;
        let (decode_port, _decode) = start_queued_grpc_worker(ParallelShape::StreamThenStall).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            },
        )
        .await;
        let (done_tx, done_rx) = oneshot::channel::<()>();
        // The first request streams its tokens and then hangs on the stalled
        // decode: it is let go of once the second request has its answer.
        let first = async {
            tokio::select! {
                answer = one_streaming_chat(router.as_ref()) => {
                    panic!("the stalled stream must not end on its own: {answer:?}")
                }
                _ = done_rx => {}
            }
        };
        let second = async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let answer = one_streaming_chat(router.as_ref()).await;
            let _ = done_tx.send(());
            answer
        };
        let ((), (status, body, elapsed)) = tokio::join!(first, second);
        assert_eq!(status, http::StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert!(body.contains("decode_worker_silent"), "{body}");
        assert!(!body.contains("prefill_worker_silent"), "{body}");
        assert!(
            elapsed >= BOUND && elapsed < BOUND + SLACK,
            "given up on at the bound ({BOUND:?}), not before and not much after: {elapsed:?}"
        );
        assert_eq!(prefill.calls(), 2, "both prefill calls were dispatched");
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "the prefill worker was waiting on its partner, not failing"
        );
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Open,
            "the stalled decode worker is the one charged"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "decode", "timeout"),
        1.0,
        "one timeout on the stalled decode worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "nothing on the prefill worker:\n{rendered}"
    );
}

/// The same stall, met in the prefill's dispatch phase: the prefill's second
/// call never returns its stream (its servicer sends the headers with the
/// first message), while the decode's call had returned at once, so both
/// calls' outcomes are judged together once the prefill's bound passes. The
/// verdict is still the decode's: 504 `decode_worker_silent`, the decode's
/// breaker (set to one) open, the prefill's closed, the timeout on the
/// decode.
#[test]
fn a_parallel_prefill_held_in_its_dispatch_by_a_stalled_decode_is_charged_to_the_decode() {
    let rendered = render_after(|| async {
        let (prefill_port, prefill) =
            start_queued_grpc_worker(ParallelShape::AnswerFirstThenHoldHeaders).await;
        let (decode_port, _decode) = start_queued_grpc_worker(ParallelShape::StreamThenStall).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            },
        )
        .await;
        let (done_tx, done_rx) = oneshot::channel::<()>();
        let first = async {
            tokio::select! {
                answer = one_streaming_chat(router.as_ref()) => {
                    panic!("the stalled stream must not end on its own: {answer:?}")
                }
                _ = done_rx => {}
            }
        };
        let second = async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let answer = one_streaming_chat(router.as_ref()).await;
            let _ = done_tx.send(());
            answer
        };
        let ((), (status, body, elapsed)) = tokio::join!(first, second);
        assert_eq!(status, http::StatusCode::GATEWAY_TIMEOUT, "{body}");
        assert!(body.contains("decode_worker_silent"), "{body}");
        assert!(
            elapsed >= BOUND && elapsed < BOUND + SLACK,
            "given up on at the bound: {elapsed:?}"
        );
        assert_eq!(prefill.calls(), 2, "both prefill calls were dispatched");
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Open,
            "the stalled decode worker is the one charged, in the dispatch phase too"
        );
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "the prefill worker was waiting on its partner"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "decode", "timeout"),
        1.0,
        "one timeout on the stalled decode worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "nothing on the prefill worker:\n{rendered}"
    );
}

/// Sequential dispatch, the same queue in front of the prefill call (its
/// headers held until the call's turn): no call is given up on, every request
/// is answered by the decode, nothing is charged to the prefill worker.
#[test]
fn a_sequential_prefill_whose_calls_queue_is_not_given_up_on() {
    let prefill = Engine::new(Behaviour::Queued);
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        // The requests are booked on the prefill worker at their dispatch, while
        // their calls are still held at the gate: sampled through the run,
        // the backlog shows them waiting (a booking at the call's return
        // would show nothing: the span from the call's return to the first
        // message, which is the same message, is too short to be sampled or
        // to teach a rate).
        let done = Arc::new(AtomicBool::new(false));
        let sampler = {
            let done = Arc::clone(&done);
            let prefill = Arc::clone(&workers[0]);
            async move {
                let mut largest = Duration::ZERO;
                while !done.load(Ordering::Acquire) {
                    largest = largest.max(prefill.prefill_backlog());
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                largest
            }
        };
        let chats = async {
            let answers = concurrent_streaming_chats(router.as_ref(), QUEUED_CALLS).await;
            done.store(true, Ordering::Release);
            answers
        };
        let (answers, largest_backlog) = tokio::join!(chats, sampler);
        assert!(
            largest_backlog > Duration::ZERO,
            "the held requests are pending prefill on their worker from the dispatch"
        );
        assert_all_answered_past_the_bound(&answers);
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "a backed-up prefill is not a failing one"
        );
    });
    assert_eq!(
        prefill_calls.calls(),
        QUEUED_CALLS,
        "each call dispatched once"
    );
    assert_eq!(
        decode_calls.calls(),
        QUEUED_CALLS,
        "every request reached its decode"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "no call was given up on:\n{rendered}"
    );
}

/// Sequential dispatch, the prefill holds its headers past the bound and
/// then fails: the request charged the worker once, at the verdict. The late
/// failure of the call it gave up on is not a second breaker failure (the
/// breaker, set to two, stays closed) and not a `backend_error`.
#[test]
fn the_late_failure_of_a_call_given_up_on_is_not_charged_again() {
    let prefill = Engine::new(Behaviour::HoldHeadersThenFail(BOUND + LATE));
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let prefill_calls = prefill.clone();
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_given_up_on(status, &body, elapsed);
        // The retired call lands, and fails, after the verdict.
        tokio::time::sleep(LATE + SLACK).await;
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "one failure, the verdict's; the late one is not a second"
        );
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "not replayed on the silent prefill"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "backend_error"),
        0.0,
        "the late failure is not counted again:\n{rendered}"
    );
}

/// With another prefill worker available the request is retried there and
/// answered; the silent prefill is never asked again, and its decode partner
/// is still the one that serves the retry.
#[test]
fn the_request_is_retried_on_another_prefill_worker_and_never_on_the_silent_one_again() {
    let silent = Engine::new(Behaviour::HoldHeaders);
    let healthy = Engine::new(Behaviour::Answer(Duration::from_millis(50)));
    let decode = Engine::new(Behaviour::Answer(Duration::from_millis(50)));
    let (silent_calls, healthy_calls, decode_calls) =
        (silent.clone(), healthy.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let silent_port = start_mock_engine(silent).await;
        let healthy_port = start_mock_engine(healthy).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (silent_port, WorkerType::Prefill),
                (healthy_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(
            elapsed < BOUND + SLACK,
            "one bound on the silent prefill, then the answer: {elapsed:?}"
        );
    });
    assert_eq!(
        silent_calls.calls(),
        1,
        "given up on once, never asked again"
    );
    assert_eq!(
        healthy_calls.calls(),
        1,
        "the retry went to the other prefill"
    );
    assert_eq!(
        decode_calls.calls(),
        1,
        "the decode partner served the retry"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the silent prefill worker:\n{rendered}"
    );
}

/// Parallel dispatch, a decode worker that accepts every call and fails its
/// stream before any token: each request is one failure of that worker, so
/// two failed requests open its breaker (set to two), as they do on any
/// other worker. The call's dispatch used to be sampled as a success the
/// moment its stream was returned, a success the call had not earned, and the
/// failure that followed never built a streak: a decode worker that failed
/// every request kept its breaker closed. The prefill worker, which answered
/// every call, keeps its breaker closed.
#[test]
fn a_decode_worker_that_fails_every_stream_it_accepted_opens_its_breaker() {
    let rendered = render_after(|| async {
        let prefill_port = start_mock_grpc_worker(Duration::ZERO).await;
        let decode_port = start_cut_grpc_worker().await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        for _ in 0..2 {
            let (status, body, _) = one_streaming_chat(router.as_ref()).await;
            assert!(
                status.is_server_error() || body.contains("error"),
                "the request fails on its decode side: {status} {body}"
            );
        }
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Open,
            "two failed requests are two consecutive failures of the decode worker"
        );
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "the prefill worker answered every call"
        );
    });
    assert!(
        worker_errors(&rendered, "decode", "backend_error") >= 2.0,
        "each failed request counts on the decode worker:\n{rendered}"
    );
}

/// The success a decode side does earn is still sampled, at its first
/// response: a failure before it is forgiven by a request the decode served,
/// so the next failure leaves a breaker set to two closed.
#[test]
fn a_decode_side_that_answers_earns_its_workers_breaker_success_at_its_first_response() {
    render_after(|| async {
        let prefill_port = start_mock_grpc_worker(Duration::ZERO).await;
        let decode_port = start_mock_grpc_worker(Duration::ZERO).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        workers[1].record_outcome(504);
        let (status, body, _) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
        workers[1].record_outcome(504);
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Closed,
            "the served request's first response reset the streak"
        );
    });
}
