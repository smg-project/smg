//! Mock gRPC worker implementing the TokenSpeed scheduler service. The gateway
//! tokenizes and sends token ids; this service streams back canned token ids.

use std::{
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::{common_proto as common, tokenspeed_scheduler::tokenspeed_proto as ts};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::mpsc,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use crate::{
    config::Config,
    engine::{self, Engine, KvEventStream, LoadsLike, NewRequest},
    replay::Capture,
};

/// Serve the mock TokenSpeed gRPC service on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let ip = match host.parse::<IpAddr>() {
        Ok(ip) => ip,
        Err(e) => {
            tracing::error!("grpc worker host {host} invalid: {e}");
            return;
        }
    };
    let listener = match TcpListener::bind(SocketAddr::new(ip, port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("grpc worker {port} failed to bind: {e}");
            return;
        }
    };
    serve_with_listener(cfg, listener).await;
}

/// Serve on an already-bound listener. Callers that need a free port bind
/// port 0 and read it back instead of picking one and binding it later.
pub async fn serve_with_listener(cfg: Arc<Config>, listener: TcpListener) {
    let addr = listener.local_addr().ok();
    let capture = match &cfg.replay.capture {
        Some(path) => match Capture::open(path) {
            Ok(capture) => Some(Arc::new(capture)),
            Err(e) => {
                tracing::error!(
                    "grpc worker {addr:?} cannot open capture file {}: {e}",
                    path.display()
                );
                return;
            }
        },
        None => None,
    };
    // One simulated engine per listener (i.e. per virtual worker), registered
    // in the process fleet under `grpc:<port>` for the oracle and admin API.
    let name = format!("grpc:{}", addr.map(|a| a.port()).unwrap_or(0));
    let engine = cfg
        .realistic
        .then(|| Engine::spawn_named(cfg.engine.clone(), name, true));
    // The worker's vLLM-wire KV-event publisher, numbered by its port offset.
    let publisher = engine.as_ref().and_then(|engine| {
        let port = addr.map(|a| a.port())?;
        let index = port.checked_sub(cfg.grpc_base_port)?;
        // A gRPC worker is a single-rank engine: it publishes as rank 0.
        cfg.kv_zmq_for(index, 0)
            .map(|kv| crate::kv_zmq::serve(engine.clone(), kv))
    });
    let max_message_bytes = cfg.grpc_max_message_bytes;
    let service = MockScheduler {
        cfg,
        engine,
        capture,
    };
    let server = async {
        // tonic's 4 MiB default would refuse the Generate of a long prompt (a
        // million token ids are a few megabytes as varints, more with the
        // prompt text alongside); the limit follows the config: none by
        // default, as the engine servicers run.
        if let Err(e) = Server::builder()
            .add_service(
                TokenSpeedSchedulerServer::new(service)
                    .max_decoding_message_size(max_message_bytes)
                    .max_encoding_message_size(max_message_bytes),
            )
            .serve_with_incoming(connections(listener))
            .await
        {
            tracing::error!("grpc worker {addr:?} stopped: {e}");
        }
    };
    match publisher {
        Some(publisher) => {
            tokio::join!(server, publisher);
        }
        None => server.await,
    }
}

/// The listener's connections with `TCP_NODELAY` set, as the engine
/// servicers serve theirs. tonic ignores its own `tcp_nodelay` setting for a
/// caller-provided incoming stream, and with Nagle on the response HEADERS
/// frame goes out alone while the first DATA frame (the first token) waits
/// for its acknowledgement, which the gateway's delayed ACK holds for ~40 ms
/// on every other request.
fn connections(listener: TcpListener) -> impl Stream<Item = std::io::Result<TcpStream>> {
    TcpListenerStream::new(listener).map(|conn| {
        if let Ok(stream) = &conn {
            if let Err(e) = stream.set_nodelay(true) {
                tracing::warn!("grpc worker: set_nodelay failed: {e}");
            }
        }
        conn
    })
}

#[derive(Clone)]
struct MockScheduler {
    cfg: Arc<Config>,
    /// Present iff the worker runs the realistic engine simulator.
    engine: Option<Engine>,
    /// Present iff `--capture` is set: every `Generate` request is recorded.
    capture: Option<Arc<Capture>>,
}

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for MockScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        // Record before anything else: the line is in the file before the
        // stream is returned, so before the gateway can read any frame.
        if let Some(capture) = &self.capture {
            if let Err(e) = capture.record(&req) {
                tracing::error!("capturing request {} failed: {e}", req.request_id);
                return Err(Status::internal(format!("mock-worker capture failed: {e}")));
            }
        }

        // Realistic mode: submit to the engine simulator and stream its output.
        if let Some(engine) = &self.engine {
            let request_id = req.request_id;
            let prompt_token_ids = req.tokenized.map(|t| t.input_ids).unwrap_or_default();
            // Omitted limit falls back to the worker default, matching the HTTP
            // path; `unwrap_or(0)` here would make unbounded requests generate
            // nothing (zero tokens), starving the routing signals.
            let max_new = req
                .sampling_params
                .and_then(|s| s.max_new_tokens)
                .unwrap_or(self.cfg.output_tokens);
            let stream_chunks = req.stream;
            let (tx, rx) = mpsc::unbounded_channel();
            engine.submit(NewRequest {
                request_id: request_id.clone(),
                prompt_token_ids,
                max_new,
                events: tx,
            });
            return Ok(Response::new(generate_stream(
                rx,
                stream_chunks,
                request_id,
            )));
        }

        // Canned mode: a single up-front delay, then synthetic token ids.
        let request_id = req.request_id;
        if !self.cfg.gen_delay.is_zero() {
            tokio::time::sleep(self.cfg.gen_delay).await;
        }
        let ids: Vec<u32> = (0..self.cfg.output_tokens).map(|i| 100 + i).collect();

        let mut items: Vec<Result<ts::GenerateResponse, Status>> = Vec::new();
        for id in &ids {
            items.push(Ok(ts::GenerateResponse {
                request_id: request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: vec![*id],
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    output_logprobs: None,
                    index: 0,
                    weight_version: None,
                })),
            }));
        }
        items.push(Ok(ts::GenerateResponse {
            request_id,
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: ids,
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: self.cfg.output_tokens,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
                ..Default::default()
            })),
        }));

        Ok(Response::new(Box::pin(stream::iter(items))))
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
        Ok(Response::new(ts::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            served_model_name: self.cfg.model_id.clone(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            max_context_length: self.cfg.context_length.min(i32::MAX as u32) as i32,
            max_req_input_len: self.cfg.context_length.min(i32::MAX as u32) as i32,
            vocab_size: 32000,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            weight_version: "mock".to_string(),
            default_sampling_params_json: String::new(),
            supports_vision: false,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(ts::GetServerInfoResponse {
            server_args: None,
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            start_time: None,
        }))
    }

    async fn get_loads(
        &self,
        _request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        let load = match &self.engine {
            Some(engine) => {
                snapshot_to_scheduler_load(&engine.load().as_reported_by(self.cfg.loads_like))
            }
            None => ts::SchedulerLoad {
                active_token_usage: None,
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_waiting_uncached_tokens: 0,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: 1_000_000,
                max_running_requests: 0,
                token_usage: 0.0,
                gen_throughput: 0.0,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                memory: None,
                queues: None,
            },
        };
        Ok(Response::new(ts::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: 1,
            loads: vec![load],
            aggregate: None,
        }))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        match &self.engine {
            // Realistic mode with prefix caching: stream the engine's KV events,
            // each carrying the load record a servicer attaches.
            Some(engine) if engine.kv_enabled() => {
                let start = request.into_inner().start_sequence_number;
                Ok(Response::new(with_load_records(
                    engine.clone(),
                    self.cfg.loads_like,
                    engine.subscribe_kv(start),
                )))
            }
            // Otherwise Unimplemented, on which the gateway's KvEventMonitor gives
            // up cleanly instead of keeping an idle per-worker task.
            _ => Err(Status::unimplemented(
                "mock-worker (KV events require --engine realistic with --prefix-cache true)",
            )),
        }
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        // The mock has no tokenizer artifacts to serve; the gateway's
        // remote-tokenizer fallback treats Unimplemented as "not supported".
        Err(Status::unimplemented("mock-worker"))
    }
}

/// Map the engine's [`engine::GenEvent`] channel to the gRPC generate stream.
/// In streaming mode each token becomes a `Chunk`; otherwise tokens are
/// accumulated and only the final `Complete` is sent. After `Complete` the
/// engine has dropped the sender, so the next `recv()` yields `None` and the
/// stream ends.
fn generate_stream(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    stream_chunks: bool,
    request_id: String,
) -> GenStream {
    let init = (rx, Vec::<u32>::new(), stream_chunks, request_id);
    Box::pin(stream::unfold(
        init,
        |(mut rx, mut output_ids, stream_chunks, request_id)| async move {
            loop {
                match rx.recv().await {
                    Some(engine::GenEvent::Token {
                        token_id,
                        prompt_tokens,
                        cached_tokens,
                    }) => {
                        output_ids.push(token_id);
                        if stream_chunks {
                            let resp = ts::GenerateResponse {
                                request_id: request_id.clone(),
                                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                                    token_ids: vec![token_id],
                                    prompt_tokens,
                                    completion_tokens: output_ids.len() as u32,
                                    cached_tokens,
                                    output_logprobs: None,
                                    index: 0,
                                    weight_version: None,
                                })),
                            };
                            return Some((Ok(resp), (rx, output_ids, stream_chunks, request_id)));
                        }
                        // Non-streaming: keep accumulating until Done.
                    }
                    Some(engine::GenEvent::Done {
                        finish_reason,
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                    }) => {
                        let resp = ts::GenerateResponse {
                            request_id: request_id.clone(),
                            response: Some(GenResp::Complete(ts::GenerateComplete {
                                output_ids: std::mem::take(&mut output_ids),
                                finish_reason: finish_reason.to_string(),
                                prompt_tokens,
                                completion_tokens,
                                cached_tokens,
                                output_logprobs: None,
                                matched_stop: None,
                                index: 0,
                                ..Default::default()
                            })),
                        };
                        return Some((Ok(resp), (rx, output_ids, stream_chunks, request_id)));
                    }
                    None => return None,
                }
            }
        },
    ))
}

/// Map an engine load snapshot to the TokenSpeed `SchedulerLoad` wire type.
/// The load record the engine's `GetLoads` figures make, as reported for
/// `loads_like` (the vLLM shape has no queued token-work).
fn load_record(
    snapshot: &engine::LoadSnapshot,
    like: LoadsLike,
    sample: u64,
) -> common::EngineLoad {
    let unsigned = |value: i32| u32::try_from(value).unwrap_or(0);
    common::EngineLoad {
        running_requests: unsigned(snapshot.num_running_reqs),
        waiting_requests: unsigned(snapshot.num_waiting_reqs),
        waiting_uncached_tokens: matches!(like, LoadsLike::Mock)
            .then(|| unsigned(snapshot.num_waiting_uncached_tokens)),
        token_usage: snapshot.token_usage,
        gen_throughput: snapshot.gen_throughput,
        max_running_requests: unsigned(snapshot.max_running_requests),
        age_ms: 0,
        sample,
        load_only: false,
        // What both reports carry beyond the core (the mock has no
        // memory, queue, speculative, LoRA or disaggregation sections).
        cache_hit_rate: Some(snapshot.cache_hit_rate),
        num_used_tokens: Some(snapshot.num_used_tokens),
        max_total_num_tokens: Some(snapshot.max_total_num_tokens),
        ..Default::default()
    }
}

/// Whether a record moved enough from `last` to be worth a `load_only`
/// batch: the Rust relay's rule (any queue, running or window change, KV
/// usage by half a percent, the rate by 5 % or 50 tokens per second).
fn load_changed(last: &common::EngineLoad, current: &common::EngineLoad) -> bool {
    last.running_requests != current.running_requests
        || last.waiting_requests != current.waiting_requests
        || last.waiting_uncached_tokens != current.waiting_uncached_tokens
        || last.max_running_requests != current.max_running_requests
        || (last.token_usage - current.token_usage).abs() > 0.005
        || {
            let delta = (last.gen_throughput - current.gen_throughput).abs();
            delta > 50.0 || delta > 0.05 * last.gen_throughput.max(current.gen_throughput)
        }
}

/// How often the stream checks the record while no batch flows, the silence
/// after which a heartbeat goes out, and the heartbeat interval once the
/// engine has been idle for two of them: the Rust relay's figures.
const LOAD_TICK: Duration = Duration::from_millis(100);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);
const HEARTBEAT_BACKOFF: Duration = Duration::from_secs(5);

struct LoadRecords {
    stream: KvEventStream,
    engine: engine::Engine,
    like: LoadsLike,
    last_seq: u64,
    last_rank: Option<i32>,
    last_record: Option<common::EngineLoad>,
    last_sent_at: Instant,
    idle_heartbeats: u32,
    sample: u64,
}

#[expect(
    clippy::large_enum_variant,
    reason = "the item is matched and moved out at once; boxing it would allocate per batch"
)]
enum Waited {
    Item(Option<Result<common::KvEventBatch, Status>>),
    Tick,
}

impl LoadRecords {
    fn record(&mut self, load_only: bool) -> common::EngineLoad {
        self.sample += 1;
        let snapshot = self.engine.load().as_reported_by(self.like);
        let mut record = load_record(&snapshot, self.like, self.sample);
        record.load_only = load_only;
        // As the Rust relay: telemetry on heartbeats and the first record.
        if !load_only && self.sample > 1 {
            smg_grpc_client::engine_load::core_only(&mut record);
        }
        self.last_record = Some(record.clone());
        record
    }

    /// A `load_only` batch when the record moved since the last one sent, or
    /// after the heartbeat interval of silence (the backoff interval after two
    /// unchanged heartbeats); it repeats the last sequence sent, as the
    /// gateway expects.
    fn load_only_batch(&mut self) -> Option<common::KvEventBatch> {
        let current = load_record(&self.engine.load().as_reported_by(self.like), self.like, 0);
        let changed = self
            .last_record
            .as_ref()
            .is_none_or(|last| load_changed(last, &current));
        let due_after = if self.idle_heartbeats >= 2 {
            HEARTBEAT_BACKOFF
        } else {
            HEARTBEAT_INTERVAL
        };
        if !changed && self.last_sent_at.elapsed() < due_after {
            return None;
        }
        let record = self.record(true);
        self.last_sent_at = Instant::now();
        self.idle_heartbeats = if changed {
            0
        } else {
            self.idle_heartbeats.saturating_add(1)
        };
        Some(common::KvEventBatch {
            sequence_number: self.last_seq,
            timestamp: engine::unix_seconds(),
            events: Vec::new(),
            dp_rank: self.last_rank,
            snapshot: None,
            load: Some(record),
        })
    }
}

/// `stream` with the engine's load record on every batch and `load_only`
/// batches while the engine is quiet, as the Rust servicer's relay sends
/// them (`crates/engine_servicer/src/kv_events.rs`).
fn with_load_records(
    engine: engine::Engine,
    like: LoadsLike,
    stream: KvEventStream,
) -> KvEventStream {
    let records = LoadRecords {
        stream,
        engine,
        like,
        last_seq: 0,
        last_rank: Some(0),
        last_record: None,
        last_sent_at: Instant::now(),
        idle_heartbeats: 0,
        sample: 0,
    };
    Box::pin(stream::unfold(records, |mut records| async move {
        loop {
            let waited = tokio::select! {
                item = records.stream.next() => Waited::Item(item),
                () = tokio::time::sleep(LOAD_TICK) => Waited::Tick,
            };
            match waited {
                Waited::Item(None) => return None,
                Waited::Item(Some(Err(status))) => return Some((Err(status), records)),
                Waited::Item(Some(Ok(mut batch))) => {
                    records.last_seq = batch.sequence_number;
                    records.last_rank = batch.dp_rank;
                    records.last_sent_at = Instant::now();
                    records.idle_heartbeats = 0;
                    batch.load = Some(records.record(false));
                    return Some((Ok(batch), records));
                }
                Waited::Tick => {
                    if let Some(batch) = records.load_only_batch() {
                        return Some((Ok(batch), records));
                    }
                }
            }
        }
    }))
}

fn snapshot_to_scheduler_load(s: &engine::LoadSnapshot) -> ts::SchedulerLoad {
    ts::SchedulerLoad {
        active_token_usage: None,
        dp_rank: 0,
        num_running_reqs: s.num_running_reqs,
        num_waiting_reqs: s.num_waiting_reqs,
        num_waiting_uncached_tokens: s.num_waiting_uncached_tokens,
        num_total_reqs: s.num_running_reqs + s.num_waiting_reqs,
        num_used_tokens: s.num_used_tokens,
        max_total_num_tokens: s.max_total_num_tokens,
        max_running_requests: s.max_running_requests,
        token_usage: s.token_usage,
        gen_throughput: s.gen_throughput,
        cache_hit_rate: s.cache_hit_rate,
        utilization: s.token_usage,
        memory: None,
        queues: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sockets the gRPC worker serves carry `TCP_NODELAY`, so a token
    /// frame is never held back for the acknowledgement of the frame before.
    #[tokio::test]
    async fn served_connections_have_nodelay_set() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let mut incoming = std::pin::pin!(connections(listener));
        let client = TcpStream::connect(addr).await.expect("connect");
        let accepted = incoming
            .next()
            .await
            .expect("a connection")
            .expect("accepted");
        assert!(
            accepted.nodelay().expect("nodelay"),
            "the accepted socket runs with Nagle on"
        );
        drop(client);
    }
}
