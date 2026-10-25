//! Mock gRPC worker implementing the TokenSpeed scheduler service. The gateway
//! tokenizes and sends token ids; this service streams back canned token ids.
//! Beside it the server hosts the standard `grpc.health.v1` service, which
//! answers as the scheduler's own `HealthCheck` does.

use std::{
    future::{pending, Future},
    pin::{pin, Pin},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::{
    common_proto as common, constraint_failed_status, tokenspeed_scheduler::tokenspeed_proto as ts,
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{mpsc, watch},
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{server::NamedService, transport::Server, Request, Response, Status};
use tonic_health::pb::{
    health_check_response::ServingStatus,
    health_server::{Health, HealthServer},
    HealthCheckRequest, HealthCheckResponse,
};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use crate::{
    config::Config,
    drain::InFlight,
    engine::{self, Engine, KvEventStream, LoadsLike, NewRequest},
    replay::Capture,
    script::{self, Answer, MatchedStop},
};

/// Serve the mock TokenSpeed gRPC service on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    serve_until(cfg, host, port, pending()).await;
}

/// Serve on `port` until `shutdown` resolves, then stop accepting connections
/// and let the requests in flight finish (the drain of a termination signal).
pub async fn serve_until(
    cfg: Arc<Config>,
    host: String,
    port: u16,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
    // What the HTTP listener accepts: a bare IP of either family or a name.
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("grpc worker bind {host}:{port} failed: {e}");
            return;
        }
    };
    serve_with_listener_until(cfg, listener, shutdown).await;
}

/// Serve on an already-bound listener. Callers that need a free port bind
/// port 0 and read it back instead of picking one and binding it later.
pub async fn serve_with_listener(cfg: Arc<Config>, listener: TcpListener) {
    serve_with_listener_until(cfg, listener, pending()).await;
}

/// [`serve_with_listener`] with the shutdown of [`serve_until`].
pub async fn serve_with_listener_until(
    cfg: Arc<Config>,
    listener: TcpListener,
    shutdown: impl Future<Output = ()> + Send + 'static,
) {
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
    if let Err(e) = script::ensure_loaded(&cfg) {
        tracing::error!("grpc worker {addr:?} cannot load script: {e}");
        return;
    }
    // One simulated engine per listener (i.e. per virtual worker), registered
    // in the process fleet under `grpc:<port>` for the oracle and admin API.
    let name = format!("grpc:{}", addr.map(|a| a.port()).unwrap_or(0));
    script::register(&name);
    let ranks = Rank::spawn_all(&cfg, &name);
    // The worker's vLLM-wire KV-event publisher, numbered by its port offset:
    // its first rank's (the publisher ports are per worker, not per rank).
    let publisher = ranks.first().and_then(|rank| {
        let port = addr.map(|a| a.port())?;
        let index = port.checked_sub(cfg.grpc_base_port)?;
        cfg.kv_zmq_for(index, 0)
            .map(|kv| crate::kv_zmq::serve(rank.engine.clone(), kv))
    });
    let max_message_bytes = cfg.grpc_max_message_bytes;
    // The stop reaches the service's streams that have no end of their own
    // (a KV-event subscription) before the server's graceful shutdown
    // begins, which waits for every open stream.
    let (stop, stopped) = watch::channel(false);
    let shutdown = async move {
        shutdown.await;
        let _ = stop.send(true);
    };
    let overrides = crate::overrides::for_port(addr.map(|a| a.port()).unwrap_or(0));
    let health = MockHealth {
        overrides: Arc::clone(&overrides),
        stopped: stopped.clone(),
    };
    let service = MockScheduler {
        cfg,
        name,
        ranks,
        scan_start: Arc::new(AtomicUsize::new(0)),
        capture,
        stopped,
        overrides,
    };
    let server = async {
        // tonic drains on incoming EOF when graceful shutdown is enabled,
        // but retains the stream while waiting for existing connections.
        // Drop the listener at EOF so the port closes before that wait.
        let incoming = stream::unfold(
            Box::pin(connections(listener).take_until(shutdown)),
            |mut incoming| async move {
                incoming
                    .next()
                    .await
                    .map(|connection| (connection, incoming))
            },
        );
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
            .add_service(HealthServer::new(health))
            .serve_with_incoming_shutdown(incoming, pending())
            .await
        {
            tracing::error!("grpc worker {addr:?} stopped: {e}");
        }
    };
    match publisher {
        Some(publisher) => {
            // When the server stops (the shutdown, or an error) the publisher
            // goes with it; a publisher that stops on its own leaves the
            // server serving, as before.
            let mut server = pin!(server);
            let mut publisher = pin!(publisher);
            tokio::select! {
                () = &mut server => {}
                () = &mut publisher => server.await,
            }
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
    /// The worker's name in the process (`grpc:<port>`): what the admin API
    /// and the script store address it by.
    name: String,
    /// The worker's data-parallel ranks, one realistic engine each (empty for
    /// a canned worker). A pinned request runs on its rank, an unpinned one
    /// on the rank holding the fewest of this worker's requests.
    ranks: Vec<Rank>,
    /// Where the unpinned scan starts, one rank further each time, so
    /// all-zero ties do not always resolve to rank 0.
    scan_start: Arc<AtomicUsize>,
    /// Present iff `--capture` is set: every `Generate` request is recorded.
    capture: Option<Arc<Capture>>,
    /// Flips when the worker stops: ends the streams that have no end of
    /// their own.
    stopped: watch::Receiver<bool>,
    /// What the admin API made this worker report about itself (serving,
    /// server args, model capabilities).
    overrides: Arc<crate::overrides::WorkerOverrides>,
}

/// One data-parallel rank of a gRPC worker: its engine, the prefill work in
/// hand on it (prompt tokens handed to the engine with no token yet, as a
/// servicer reports them on the rank's load record), and the requests this
/// worker placed on it whose streams have not ended.
#[derive(Clone)]
struct Rank {
    engine: Engine,
    prefill_pending: Arc<AtomicU64>,
    in_flight: Arc<AtomicUsize>,
}

impl Rank {
    /// The worker's ranks: `--grpc-dp-size` realistic engines, rank `r` named
    /// `<worker>:r<r>` in the process fleet (a single rank keeps the worker's
    /// name) and stamping `r` on its KV-event batches; none for a canned
    /// worker.
    fn spawn_all(cfg: &Config, worker: &str) -> Vec<Rank> {
        if !cfg.realistic {
            return Vec::new();
        }
        let size = usize::from(cfg.grpc_dp_size.max(1));
        (0..size)
            .map(|rank| {
                let mut params = cfg.engine.clone();
                params.dp_rank = i32::try_from(rank).unwrap_or(i32::MAX);
                let name = if size > 1 {
                    format!("{worker}:r{rank}")
                } else {
                    worker.to_string()
                };
                Rank {
                    engine: Engine::spawn_named(params, name, true),
                    prefill_pending: Arc::new(AtomicU64::new(0)),
                    in_flight: Arc::new(AtomicUsize::new(0)),
                }
            })
            .collect()
    }
}

/// A request placed on a rank: the rank's count comes down when this is
/// dropped, with the stream that carries it (however the stream ends).
struct Placed(Arc<AtomicUsize>);

impl Drop for Placed {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl MockScheduler {
    /// The rank a request runs on: its `data_parallel_rank` when pinned (a
    /// rank past the worker's is the caller's error, as the servicers answer
    /// it), else the rank holding the fewest of this worker's requests, the
    /// scan starting one rank further each time. The rank's count is taken
    /// here, under no lock: two unpinned requests placed at once may read the
    /// same counts, as a servicer's own balancer may.
    fn place(&self, pin: Option<i32>) -> Result<(usize, Placed), Status> {
        let count = self.ranks.len();
        let rank = match pin {
            Some(rank) => usize::try_from(rank)
                .ok()
                .filter(|rank| *rank < count)
                .ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "data_parallel_rank {rank} out of range [0, {count})"
                    ))
                })?,
            None => {
                let start = self.scan_start.fetch_add(1, Ordering::Relaxed) % count.max(1);
                (0..count)
                    .map(|offset| (start + offset) % count)
                    .min_by_key(|&rank| self.ranks[rank].in_flight.load(Ordering::Relaxed))
                    .unwrap_or(0)
            }
        };
        let in_flight = Arc::clone(&self.ranks[rank].in_flight);
        in_flight.fetch_add(1, Ordering::Relaxed);
        Ok((rank, Placed(in_flight)))
    }
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
        // In flight from here until the stream is done (or dropped by the
        // gateway), so a shutdown knows what it waits for.
        let in_flight = InFlight::enter();
        let req = request.into_inner();
        // Record before anything else: the line is in the file before the
        // stream is returned, so before the gateway can read any frame.
        if let Some(capture) = &self.capture {
            if let Err(e) = capture.record(&req) {
                tracing::error!("capturing request {} failed: {e}", req.request_id);
                return Err(Status::internal(format!("mock-worker capture failed: {e}")));
            }
        }

        // A scripted answer, ahead of either engine mode: the worker's queue
        // or the prompt's key decides ([`crate::script`]).
        let prompt_ids: &[u32] = req
            .tokenized
            .as_ref()
            .map(|tokenized| tokenized.input_ids.as_slice())
            .unwrap_or(&[]);
        let scripted = script::take_ids(&self.cfg, &self.name, prompt_ids)
            .map_err(Status::failed_precondition)?;
        if let Some(answer) = scripted {
            let prompt_tokens = answer.prompt_tokens(prompt_ids.len());
            let stream = scripted_stream(
                answer,
                req.request_id.clone(),
                prompt_tokens,
                req.stream,
                req.return_logprob,
                self.cfg.weight_version.clone(),
            );
            return Ok(Response::new(Box::pin(in_flight.until_done(stream))));
        }

        // Realistic mode: place the request on a rank, submit it to that
        // rank's engine simulator and stream its output.
        if !self.ranks.is_empty() {
            let (rank, placed) = self.place(req.data_parallel_rank)?;
            let rank = &self.ranks[rank];
            let engine = &rank.engine;
            // The fault hook's answer for this request, when one is armed: a
            // stall, a status before admission, or a cut after some output.
            let injected = engine.inject();
            if let Some(fault) = injected {
                if !fault.stall.is_zero() {
                    tokio::time::sleep(fault.stall).await;
                }
                if fault.status != 0 && fault.after_tokens.is_none() {
                    engine.record_failure();
                    return Err(injected_status(fault.status));
                }
            }
            let cut = injected.and_then(|fault| {
                let end = if fault.finish_error {
                    CutEnd::EngineError
                } else {
                    CutEnd::Status(injected_status(fault.status))
                };
                fault
                    .after_tokens
                    .map(|after| (after, end, engine.cut_counter()))
            });
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
            // The admission hooks: the request is held here, between the
            // gateway and the engine, as a backlog in transit would hold it;
            // the load record does not see it until it lands.
            engine.admit().await;
            let weight_version = self.cfg.weight_version.clone();
            let (tx, rx) = mpsc::unbounded_channel();
            let pending = (
                Arc::clone(&rank.prefill_pending),
                prompt_token_ids.len() as u64,
            );
            pending.0.fetch_add(pending.1, Ordering::Relaxed);
            engine.submit(NewRequest {
                request_id: request_id.clone(),
                prompt_token_ids,
                max_new,
                events: tx,
            });
            let stream = generate_stream(
                rx,
                stream_chunks,
                request_id,
                Some(pending),
                cut,
                weight_version,
            )
            // The rank's count rides with the stream: down when it is dropped.
            .map(move |item| {
                let _placed = &placed;
                item
            });
            return Ok(Response::new(Box::pin(in_flight.until_done(stream))));
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
                    weight_version: self.cfg.weight_version.clone(),
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
                weight_version: self.cfg.weight_version.clone(),
                ..Default::default()
            })),
        }));

        Ok(Response::new(Box::pin(
            in_flight.until_done(stream::iter(items)),
        )))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        let serving = self.overrides.serving();
        Ok(Response::new(ts::HealthCheckResponse {
            healthy: serving,
            message: if serving { "ok" } else { "not serving" }.to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        // The abort does not name the request's rank: every rank is asked, an
        // unknown id being harmless.
        let request_id = request.into_inner().request_id;
        for rank in &self.ranks {
            rank.engine.abort(request_id.clone());
        }
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
            supports_vision: self.overrides.info_bool("supports_vision").unwrap_or(false),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        // The configured server args, with the admin API's overrides on top:
        // what the gateway flattens into this worker's labels.
        let mut fields: std::collections::BTreeMap<String, prost_types::Value> = self
            .cfg
            .server_args
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue(v.clone())),
                    },
                )
            })
            .collect();
        // A multi-rank worker advertises its width, as an engine's server args
        // carry it; the gateway expands it into one worker per rank under
        // `--dp-aware`. The admin API's info patch may still override it.
        if self.ranks.len() > 1 {
            fields.insert(
                "dp_size".to_string(),
                prost_types::Value {
                    kind: Some(prost_types::value::Kind::StringValue(
                        self.ranks.len().to_string(),
                    )),
                },
            );
        }
        fields.extend(crate::overrides::server_args(&self.overrides.info()).fields);
        let server_args = (!fields.is_empty()).then_some(prost_types::Struct { fields });
        Ok(Response::new(ts::GetServerInfoResponse {
            server_args,
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            start_time: None,
            constrained_decoding: String::new(),
        }))
    }

    async fn get_loads(
        &self,
        request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        // One entry per rank (a canned worker is one idle rank), the request's
        // `dp_rank` keeping that rank's alone, as the servicers answer.
        let wanted = request.into_inner().dp_rank;
        let mut loads: Vec<ts::SchedulerLoad> = if self.ranks.is_empty() {
            vec![ts::SchedulerLoad {
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
            }]
        } else {
            self.ranks
                .iter()
                .enumerate()
                .map(|(index, rank)| {
                    let mut load = snapshot_to_scheduler_load(
                        &rank.engine.load().as_reported_by(self.cfg.loads_like),
                    );
                    load.dp_rank = i32::try_from(index).unwrap_or(i32::MAX);
                    load
                })
                .collect()
        };
        if let Some(wanted) = wanted {
            loads.retain(|load| load.dp_rank == wanted);
        }
        Ok(Response::new(ts::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            aggregate: None,
        }))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        // Realistic mode with prefix caching: the ranks' KV events on one
        // stream, each batch stamped with its rank and carrying the load
        // record a servicer attaches for that rank; one record state over
        // the merged stream, as the Rust relay keeps one per subscription
        // (its heartbeat names the last batch's rank).
        let request = request.into_inner();
        let sources: Vec<RankLoadSource> = self
            .ranks
            .iter()
            .filter(|rank| rank.engine.kv_enabled())
            .map(|rank| RankLoadSource {
                engine: rank.engine.clone(),
                prefill_pending: Arc::clone(&rank.prefill_pending),
            })
            .collect();
        if sources.is_empty() {
            // Unimplemented, on which the gateway's KvEventMonitor gives up
            // cleanly instead of keeping an idle per-worker task.
            return Err(Status::unimplemented(
                "mock-worker (KV events require --engine realistic with --prefix-cache true)",
            ));
        }
        let streams: Vec<KvEventStream> = sources
            .iter()
            .map(|source| source.engine.subscribe_kv(request.start_sequence_number))
            .collect();
        let events = with_load_records(
            sources,
            self.cfg.loads_like,
            self.cfg.step_reports,
            request.load_records_before_first_batch,
            Box::pin(stream::select_all(streams)),
        );
        // A subscription ends with the worker's stop, as a serving engine's
        // ends with its process; nothing else ends it, and the server's
        // shutdown waits for it.
        let mut stopped = self.stopped.clone();
        let stop = async move {
            let _ = stopped.wait_for(|stop| *stop).await;
        };
        Ok(Response::new(Box::pin(events.take_until(stop))))
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

/// `Health.Watch` poll interval: the override is read, not signalled.
const HEALTH_WATCH_INTERVAL: Duration = Duration::from_millis(250);

type HealthWatchStream = Pin<Box<dyn Stream<Item = Result<HealthCheckResponse, Status>> + Send>>;

/// The standard `grpc.health.v1` service of a gRPC worker: the whole server
/// (the empty name) and the scheduler service are SERVING exactly when the
/// scheduler's own `HealthCheck` answers healthy, i.e. unless the admin API
/// set the worker not serving (`/admin/serving/{port}`); any other name is
/// unknown. It is the service a harness or a probe asks before it asks the
/// engine's own protocol, as every engine servicer hosts it.
#[derive(Clone)]
struct MockHealth {
    overrides: Arc<crate::overrides::WorkerOverrides>,
    /// Flips when the worker stops: ends the watch streams, which have no
    /// end of their own, so the server's drain can finish.
    stopped: watch::Receiver<bool>,
}

impl MockHealth {
    /// The serving status of `service`, read from the override per probe.
    fn status(&self, service: &str) -> ServingStatus {
        if !service.is_empty()
            && service != <TokenSpeedSchedulerServer<MockScheduler> as NamedService>::NAME
        {
            ServingStatus::ServiceUnknown
        } else if self.overrides.serving() {
            ServingStatus::Serving
        } else {
            ServingStatus::NotServing
        }
    }
}

#[tonic::async_trait]
impl Health for MockHealth {
    type WatchStream = HealthWatchStream;

    /// An unknown service is NOT_FOUND, as the health protocol specifies.
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let service = request.into_inner().service;
        match self.status(&service) {
            ServingStatus::ServiceUnknown => {
                Err(Status::not_found(format!("unknown service {service:?}")))
            }
            status => Ok(Response::new(HealthCheckResponse {
                status: status as i32,
            })),
        }
    }

    /// The current status first, then every change; an unknown service is
    /// SERVICE_UNKNOWN on the stream rather than an error, as the health
    /// protocol specifies. The stream ends when the worker stops (a known
    /// name reads NOT_SERVING once more first): the server's drain waits for
    /// every open stream, and a client keeps a watch open for as long as it
    /// lives.
    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let service = request.into_inner().service;
        let known = self.status(&service) != ServingStatus::ServiceUnknown;
        let stream = stream::unfold(
            (self.clone(), service, None, self.stopped.clone(), false),
            move |(health, service, last, mut stopped, ended)| async move {
                if ended {
                    return None;
                }
                loop {
                    let status = health.status(&service) as i32;
                    if last != Some(status) {
                        return Some((
                            Ok(HealthCheckResponse { status }),
                            (health, service, Some(status), stopped, false),
                        ));
                    }
                    let stopping = tokio::select! {
                        _ = tokio::time::sleep(HEALTH_WATCH_INTERVAL) => false,
                        _ = stopped.wait_for(|stop| *stop) => true,
                    };
                    if stopping {
                        let not_serving = ServingStatus::NotServing as i32;
                        if known && last != Some(not_serving) {
                            return Some((
                                Ok(HealthCheckResponse {
                                    status: not_serving,
                                }),
                                (health, service, Some(not_serving), stopped, true),
                            ));
                        }
                        return None;
                    }
                }
            },
        );
        Ok(Response::new(Box::pin(stream)))
    }
}

/// The gRPC status code closest to the HTTP status a request fault answers,
/// chosen so the gateway's mapping of codes to HTTP statuses gives that status
/// back where it can: 408 reads back as 504 and 502 as 503 (both still
/// retryable), 499 as 502 (`cancelled`: the worker's side ending the call,
/// retryable and charged to the worker), 422 as the engine's report that it
/// could not continue the request under its constraint (a 422
/// `constraint_failed` at the gateway, never replayed), any other 4xx as 400
/// (not retryable, as on the HTTP worker), any other 5xx as 500.
fn injected_status(status: u16) -> Status {
    const MESSAGE: &str = "injected";
    match status {
        400 => Status::invalid_argument(MESSAGE),
        401 => Status::unauthenticated(MESSAGE),
        403 => Status::permission_denied(MESSAGE),
        404 => Status::not_found(MESSAGE),
        408 | 504 => Status::deadline_exceeded(MESSAGE),
        409 => Status::aborted(MESSAGE),
        422 => constraint_failed_status(MESSAGE),
        429 => Status::resource_exhausted(MESSAGE),
        // The call the worker's side ended: what a connection that closed
        // under it produces at the gateway.
        499 => Status::cancelled(MESSAGE),
        501 => Status::unimplemented(MESSAGE),
        502 | 503 => Status::unavailable(MESSAGE),
        other if (400..=499).contains(&other) => Status::invalid_argument(MESSAGE),
        _ => Status::internal(MESSAGE),
    }
}

/// How the fault hook's cut ends a stream.
#[derive(Clone, Debug)]
enum CutEnd {
    /// The gRPC status the stream ends with.
    Status(Status),
    /// A `Complete` whose finish reason is `error`, as an engine that failed
    /// the request sends it: the output so far, no status.
    EngineError,
}

/// The generate stream's state between items.
struct Generating {
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    output_ids: Vec<u32>,
    stream_chunks: bool,
    request_id: String,
    /// The fault hook's cut: the stream ends with the given end once the
    /// engine has produced more than this many tokens; the counter records
    /// the cut.
    cut: Option<(u32, CutEnd, Arc<AtomicU64>)>,
    /// The worker's pending-prefill counter and this request's share of it,
    /// settled at the first token (or at the end, if none came).
    pending: Option<(Arc<AtomicU64>, u64)>,
    /// Stamped on every chunk and the completion; `None` leaves the field
    /// unset, like an engine that predates it.
    weight_version: Option<String>,
    ended: bool,
}

impl Generating {
    /// The prompt is prefilled (a first token, or the stream's end): it no
    /// longer counts as pending prefill.
    fn settle_prefill(&mut self) {
        if let Some((counter, tokens)) = self.pending.take() {
            counter.fetch_sub(tokens, Ordering::Relaxed);
        }
    }
}

impl Drop for Generating {
    fn drop(&mut self) {
        self.settle_prefill();
    }
}

/// Map the engine's [`engine::GenEvent`] channel to the gRPC generate stream.
/// In streaming mode each token becomes a `Chunk`; otherwise tokens are
/// accumulated and only the final `Complete` is sent. After `Complete` the
/// engine has dropped the sender, so the next `recv()` yields `None` and the
/// stream ends. With `cut`, the stream ends with the given status or
/// engine-failure `Complete` instead once the engine has produced more than
/// the given number of tokens (the fault hook's `after_tokens`); an output no
/// longer than that completes as usual. Every chunk and completion carries
/// `weight_version`, as an engine stamps them.
fn generate_stream(
    rx: mpsc::UnboundedReceiver<engine::GenEvent>,
    stream_chunks: bool,
    request_id: String,
    pending: Option<(Arc<AtomicU64>, u64)>,
    cut: Option<(u32, CutEnd, Arc<AtomicU64>)>,
    weight_version: Option<String>,
) -> GenStream {
    let init = Generating {
        rx,
        output_ids: Vec::new(),
        stream_chunks,
        request_id,
        pending,
        cut,
        weight_version,
        ended: false,
    };
    Box::pin(stream::unfold(init, |mut st| async move {
        if st.ended {
            return None;
        }
        loop {
            match st.rx.recv().await {
                Some(engine::GenEvent::Token {
                    token_id,
                    prompt_tokens,
                    cached_tokens,
                }) => {
                    st.settle_prefill();
                    st.output_ids.push(token_id);
                    if let Some((after, end, cut_total)) = &st.cut {
                        if st.output_ids.len() > *after as usize {
                            let item = match end {
                                CutEnd::Status(status) => Err(status.clone()),
                                CutEnd::EngineError => {
                                    // The output the engine had produced
                                    // before it failed: the tokens it sent.
                                    st.output_ids.truncate(*after as usize);
                                    Ok(ts::GenerateResponse {
                                        request_id: st.request_id.clone(),
                                        response: Some(GenResp::Complete(ts::GenerateComplete {
                                            output_ids: std::mem::take(&mut st.output_ids),
                                            finish_reason: "error".to_string(),
                                            prompt_tokens,
                                            completion_tokens: *after,
                                            cached_tokens,
                                            output_logprobs: None,
                                            matched_stop: None,
                                            index: 0,
                                            weight_version: st.weight_version.clone(),
                                            ..Default::default()
                                        })),
                                    })
                                }
                            };
                            cut_total.fetch_add(1, Ordering::Relaxed);
                            st.ended = true;
                            return Some((item, st));
                        }
                    }
                    if st.stream_chunks {
                        let resp = ts::GenerateResponse {
                            request_id: st.request_id.clone(),
                            response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                                token_ids: vec![token_id],
                                prompt_tokens,
                                completion_tokens: st.output_ids.len() as u32,
                                cached_tokens,
                                output_logprobs: None,
                                index: 0,
                                weight_version: st.weight_version.clone(),
                            })),
                        };
                        return Some((Ok(resp), st));
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
                        request_id: st.request_id.clone(),
                        response: Some(GenResp::Complete(ts::GenerateComplete {
                            output_ids: std::mem::take(&mut st.output_ids),
                            finish_reason: finish_reason.to_string(),
                            prompt_tokens,
                            completion_tokens,
                            cached_tokens,
                            output_logprobs: None,
                            matched_stop: None,
                            index: 0,
                            weight_version: st.weight_version.clone(),
                            ..Default::default()
                        })),
                    };
                    return Some((Ok(resp), st));
                }
                None => return None,
            }
        }
    }))
}

/// The generate stream of a scripted answer: a `Chunk` per frame when the
/// request streams, then the `Complete`, with the answer's pause before every
/// frame but the first; the logprobs ride along when the request asked for
/// them. A `Complete` that follows no chunk is what a non-streaming request
/// reads, as it reads the engine's.
fn scripted_stream(
    answer: Answer,
    request_id: String,
    prompt_tokens: u32,
    stream_chunks: bool,
    return_logprob: bool,
    weight_version: Option<String>,
) -> GenStream {
    struct Scripted {
        answer: Answer,
        frames: std::collections::VecDeque<script::Frame>,
        request_id: String,
        prompt_tokens: u32,
        stream_chunks: bool,
        return_logprob: bool,
        weight_version: Option<String>,
        /// Output tokens sent so far (the chunks' running `completion_tokens`).
        sent: u32,
        started: bool,
        done: bool,
    }
    let frames = answer.token_frames().into();
    let init = Scripted {
        answer,
        frames,
        request_id,
        prompt_tokens,
        stream_chunks,
        return_logprob,
        weight_version,
        sent: 0,
        started: false,
        done: false,
    };
    Box::pin(stream::unfold(init, |mut st| async move {
        loop {
            if st.done {
                return None;
            }
            let Some(frame) = st.frames.pop_front() else {
                st.done = true;
                let answer = &st.answer;
                let output_logprobs = answer
                    .logprobs
                    .as_ref()
                    .filter(|logprobs| {
                        st.return_logprob && logprobs.len() == answer.output_ids.len()
                    })
                    .map(|logprobs| ts::OutputLogProbs {
                        token_logprobs: logprobs.clone(),
                        token_ids: answer.output_ids.clone(),
                        top_logprobs: Vec::new(),
                    });
                let matched_stop = answer.matched_stop.as_ref().map(|stop| match stop {
                    MatchedStop::Token(id) => {
                        ts::generate_complete::MatchedStop::MatchedTokenId(*id)
                    }
                    MatchedStop::Text(text) => {
                        ts::generate_complete::MatchedStop::MatchedStopStr(text.clone())
                    }
                });
                let complete = ts::GenerateResponse {
                    request_id: st.request_id.clone(),
                    response: Some(GenResp::Complete(ts::GenerateComplete {
                        output_ids: answer.output_ids.clone(),
                        finish_reason: answer.finish_reason.clone(),
                        prompt_tokens: st.prompt_tokens,
                        completion_tokens: answer.completion_tokens(),
                        cached_tokens: answer.cached_tokens,
                        output_logprobs,
                        matched_stop,
                        index: 0,
                        weight_version: st.weight_version.clone(),
                        ..Default::default()
                    })),
                };
                return Some((Ok(complete), st));
            };
            if st.started && !st.answer.delay.is_zero() {
                tokio::time::sleep(st.answer.delay).await;
            }
            st.started = true;
            st.sent += u32::try_from(frame.ids.len()).unwrap_or(u32::MAX);
            if !st.stream_chunks {
                continue;
            }
            let output_logprobs = (st.return_logprob && frame.logprobs.len() == frame.ids.len())
                .then(|| ts::OutputLogProbs {
                    token_logprobs: frame.logprobs.clone(),
                    token_ids: frame.ids.clone(),
                    top_logprobs: Vec::new(),
                });
            let chunk = ts::GenerateResponse {
                request_id: st.request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: frame.ids,
                    prompt_tokens: st.prompt_tokens,
                    completion_tokens: st.sent,
                    cached_tokens: st.answer.cached_tokens,
                    output_logprobs,
                    index: 0,
                    weight_version: st.weight_version.clone(),
                })),
            };
            return Some((Ok(chunk), st));
        }
    }))
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

/// The worker's pending prefill as the record carries it.
fn pending_u32(counter: &AtomicU64) -> u32 {
    u32::try_from(counter.load(Ordering::Relaxed)).unwrap_or(u32::MAX)
}

/// The engine's silence on its in-flight requests, as the Rust servicer's
/// record carries it (`EngineLoad.engine_silence_ms`).
fn silence_ms(engine: &engine::Engine) -> Option<u32> {
    engine
        .output_silence()
        .map(|silence| u32::try_from(silence.as_millis()).unwrap_or(u32::MAX))
}

/// Whether a record moved enough from `last` to be worth a `load_only`
/// batch: the Rust relay's rule (any queue, running or window change, KV
/// usage by half a percent, the rate by 5 % or 50 tokens per second, the
/// engine's silence by a whole second).
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
        || last.engine_silence_ms.map(|ms| ms / 1000)
            != current.engine_silence_ms.map(|ms| ms / 1000)
        || last.prefill_pending_tokens != current.prefill_pending_tokens
}

/// How often the stream checks the record while no batch flows, the silence
/// after which a heartbeat goes out, and the heartbeat interval once the
/// engine has been idle for two of them: the Rust relay's figures.
const LOAD_TICK: Duration = Duration::from_millis(100);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);
const HEARTBEAT_BACKOFF: Duration = Duration::from_secs(5);

/// Where a rank's load record comes from: its engine and the prompt tokens
/// handed to it with no token yet.
struct RankLoadSource {
    engine: engine::Engine,
    prefill_pending: Arc<AtomicU64>,
}

struct LoadRecords {
    stream: KvEventStream,
    /// The ranks' sources, by rank; a batch's record is its rank's.
    ranks: Vec<RankLoadSource>,
    like: LoadsLike,
    /// What the record says of the wire (`--step-reports`).
    step_reports: bool,
    /// The subscriber takes `load_only` batches before the stream's first
    /// batch, numbered 0 (`SubscribeKvEventsRequest.
    /// load_records_before_first_batch`); unset, it waits for that batch.
    records_before_first_batch: bool,
    /// The last sequence sent, which a `load_only` batch repeats; `None`
    /// before the stream's first batch.
    last_seq: Option<u64>,
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
    /// The source of `rank`'s record: rank 0's for a batch that names none
    /// or one past the ranks.
    fn source(&self, rank: Option<i32>) -> &RankLoadSource {
        rank.and_then(|rank| usize::try_from(rank).ok())
            .and_then(|rank| self.ranks.get(rank))
            .unwrap_or(&self.ranks[0])
    }

    /// The current record of `rank`, numbered `sample`.
    fn current(&self, rank: Option<i32>, sample: u64) -> common::EngineLoad {
        let source = self.source(rank);
        let snapshot = source.engine.load().as_reported_by(self.like);
        let mut record = load_record(&snapshot, self.like, sample);
        record.engine_silence_ms = silence_ms(&source.engine);
        record.engine_reports_steps = Some(self.step_reports);
        record.prefill_pending_tokens = Some(pending_u32(&source.prefill_pending));
        record
    }

    fn record(&mut self, rank: Option<i32>, load_only: bool) -> common::EngineLoad {
        self.sample += 1;
        let mut record = self.current(rank, self.sample);
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
    /// gateway expects. Before the stream's first batch there is none to
    /// repeat: as the Rust relay, the mock sends one numbered 0 only to a
    /// subscriber that asked for records before the first batch, and keeps
    /// an older subscriber waiting for that batch.
    fn load_only_batch(&mut self) -> Option<common::KvEventBatch> {
        let sequence_number = match self.last_seq {
            Some(sequence) => sequence,
            None if self.records_before_first_batch => 0,
            None => return None,
        };
        // The heartbeat is the last batch's rank's, as the relay's.
        let current = self.current(self.last_rank, 0);
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
        let record = self.record(self.last_rank, true);
        self.last_sent_at = Instant::now();
        self.idle_heartbeats = if changed {
            0
        } else {
            self.idle_heartbeats.saturating_add(1)
        };
        Some(common::KvEventBatch {
            sequence_number,
            timestamp: engine::unix_seconds(),
            events: Vec::new(),
            dp_rank: self.last_rank,
            snapshot: None,
            load: Some(record),
        })
    }
}

/// `stream` (the ranks' batches, each stamped with its rank) with the
/// rank's load record on every batch and `load_only` batches while the
/// stream is quiet, as the Rust servicer's relay sends them
/// (`crates/engine_servicer/src/kv_events.rs`): one record state per
/// subscription, the heartbeat naming the last batch's rank (so a quiet
/// rank's records stop while another rank is active, as on the servicer);
/// from the subscription on for a subscriber that takes records before the
/// first batch (`records_before_first_batch`, rank 0's record then), from
/// the stream's first batch otherwise.
fn with_load_records(
    ranks: Vec<RankLoadSource>,
    like: LoadsLike,
    step_reports: bool,
    records_before_first_batch: bool,
    stream: KvEventStream,
) -> KvEventStream {
    assert!(!ranks.is_empty(), "a load record needs a rank's source");
    let records = LoadRecords {
        stream,
        ranks,
        like,
        step_reports,
        records_before_first_batch,
        last_seq: None,
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
                    records.last_seq = Some(batch.sequence_number);
                    records.last_rank = batch.dp_rank;
                    records.last_sent_at = Instant::now();
                    records.idle_heartbeats = 0;
                    batch.load = Some(records.record(batch.dp_rank, false));
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
    use tonic::{transport::Channel, Code};
    use tonic_health::pb::health_client::HealthClient;
    use ts::token_speed_scheduler_client::TokenSpeedSchedulerClient;

    use super::*;

    /// The next item of a health watch, within the test's patience.
    async fn next(watch: &mut tonic::Streaming<HealthCheckResponse>) -> ServingStatus {
        tokio::time::timeout(Duration::from_secs(5), watch.message())
            .await
            .expect("a watch item in time")
            .expect("watch item")
            .expect("a stream item")
            .status()
    }

    /// A watch has no end of its own, so the worker's stop ends it: the
    /// server's drain finishes with a watch open, and the watcher reads
    /// NOT_SERVING once more, then the stream's end.
    #[tokio::test]
    async fn an_open_watch_ends_when_the_worker_stops() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test joins the server's task: its drain is what it checks"
        )]
        let server = tokio::spawn(serve_with_listener_until(
            Arc::new(Config::default()),
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let mut health = HealthClient::new(channel);
        let mut watch = health
            .watch(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .expect("watch")
            .into_inner();
        assert_eq!(next(&mut watch).await, ServingStatus::Serving);

        stop.send(()).expect("the server waits for the stop");
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the drain finishes with a watch open")
            .expect("the server task");
        assert_eq!(next(&mut watch).await, ServingStatus::NotServing);
        let end = tokio::time::timeout(Duration::from_secs(5), watch.message())
            .await
            .expect("the watch ends in time")
            .expect("a clean end of the stream");
        assert!(end.is_none(), "the watch did not end: {end:?}");
    }

    /// The standard health service answers as the scheduler's own
    /// `HealthCheck`: SERVING for the whole server and for the scheduler
    /// service, NOT_FOUND for another name, NOT_SERVING on both once the admin
    /// override flips, and `Watch` reports the flip.
    #[tokio::test]
    async fn the_standard_health_service_follows_the_serving_override() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "the server serves until the test process exits"
        )]
        let _server = tokio::spawn(serve_with_listener(Arc::new(Config::default()), listener));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let mut health = HealthClient::new(channel.clone());
        let mut scheduler = TokenSpeedSchedulerClient::new(channel);
        let scheduler_service = <TokenSpeedSchedulerServer<MockScheduler> as NamedService>::NAME;

        // One `Check` on a clone of the client, so `health` stays free to watch.
        let probe = health.clone();
        let check = |service: &str| {
            let request = HealthCheckRequest {
                service: service.to_string(),
            };
            let mut probe = probe.clone();
            async move {
                probe
                    .check(request)
                    .await
                    .map(|response| response.into_inner().status())
            }
        };
        assert_eq!(check("").await.expect("check"), ServingStatus::Serving);
        assert_eq!(
            check(scheduler_service).await.expect("check"),
            ServingStatus::Serving
        );
        assert_eq!(
            check("not.a.Service").await.expect_err("unknown").code(),
            Code::NotFound
        );

        let mut watch = health
            .watch(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .expect("watch")
            .into_inner();
        assert_eq!(next(&mut watch).await, ServingStatus::Serving);

        // The override map is process-wide and keyed by port: put the entry
        // back on the way out (a failed assertion included), or a later test
        // that draws the same ephemeral port would start out not serving.
        struct Restore(Arc<crate::overrides::WorkerOverrides>);
        impl Drop for Restore {
            fn drop(&mut self) {
                self.0.set_serving(true);
            }
        }
        let overrides = crate::overrides::for_port(addr.port());
        let _restore = Restore(Arc::clone(&overrides));
        overrides.set_serving(false);
        assert_eq!(check("").await.expect("check"), ServingStatus::NotServing);
        assert_eq!(
            check(scheduler_service).await.expect("check"),
            ServingStatus::NotServing
        );
        assert_eq!(next(&mut watch).await, ServingStatus::NotServing);
        let own = scheduler
            .health_check(ts::HealthCheckRequest::default())
            .await
            .expect("the scheduler's own health check")
            .into_inner();
        assert!(!own.healthy, "the two health answers disagree: {own:?}");

        let mut unknown = health
            .watch(HealthCheckRequest {
                service: "not.a.Service".to_string(),
            })
            .await
            .expect("watch")
            .into_inner();
        assert_eq!(next(&mut unknown).await, ServingStatus::ServiceUnknown);
    }

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

    fn token(id: u32) -> engine::GenEvent {
        engine::GenEvent::Token {
            token_id: id,
            prompt_tokens: 4,
            cached_tokens: 0,
        }
    }

    fn done(tokens: u32) -> engine::GenEvent {
        engine::GenEvent::Done {
            finish_reason: "stop",
            prompt_tokens: 4,
            completion_tokens: tokens,
            cached_tokens: 0,
        }
    }

    fn chunk_ids(item: Option<Result<ts::GenerateResponse, Status>>) -> Vec<u32> {
        match item.expect("an item").expect("ok").response {
            Some(GenResp::Chunk(chunk)) => chunk.token_ids,
            other => panic!("not a chunk: {other:?}"),
        }
    }

    /// A realistic scheduler of `ranks` ranks, as `serve_with_listener_until`
    /// builds one, with no listener behind it; the stop sender keeps the
    /// worker running.
    fn scheduler_with_ranks(ranks: u16, port: u16) -> (MockScheduler, watch::Sender<bool>) {
        let cfg = Arc::new(Config {
            realistic: true,
            grpc_dp_size: ranks,
            engine: engine::EngineParams {
                timing: engine::TimingModel::linear(),
                ..Default::default()
            },
            ..Default::default()
        });
        let name = format!("grpc:{port}");
        let (stop, stopped) = watch::channel(false);
        let service = MockScheduler {
            ranks: Rank::spawn_all(&cfg, &name),
            scan_start: Arc::new(AtomicUsize::new(0)),
            capture: None,
            stopped,
            overrides: crate::overrides::for_port(port),
            name,
            cfg,
        };
        (service, stop)
    }

    fn generate_request(id: &str, pin: Option<i32>, max_new: u32) -> ts::GenerateRequest {
        ts::GenerateRequest {
            request_id: id.to_string(),
            tokenized: Some(ts::TokenizedInput {
                input_ids: vec![11, 12, 13, 14],
                ..Default::default()
            }),
            sampling_params: Some(ts::SamplingParams {
                max_new_tokens: Some(max_new),
                ..Default::default()
            }),
            stream: true,
            data_parallel_rank: pin,
            ..Default::default()
        }
    }

    /// The engine the request was admitted on, from the process records.
    fn admitted_on(request_id: &str) -> String {
        engine::records_since(0, usize::MAX)
            .into_iter()
            .find(|record| record.request_id == request_id)
            .unwrap_or_else(|| panic!("{request_id} was admitted"))
            .worker
    }

    /// A pinned request runs on its rank's engine; a pin past the worker's
    /// ranks is the caller's error, as the servicers answer it.
    #[tokio::test]
    async fn a_pinned_request_runs_on_its_rank() {
        let (service, _stop) = scheduler_with_ranks(4, 65_101);
        let mut stream = service
            .generate(Request::new(generate_request("dp-pin-2", Some(2), 2)))
            .await
            .expect("accepted")
            .into_inner();
        while stream.next().await.is_some() {}
        assert_eq!(admitted_on("dp-pin-2"), "grpc:65101:r2");
        let refused = match service
            .generate(Request::new(generate_request("dp-pin-4", Some(4), 2)))
            .await
        {
            Err(status) => status,
            Ok(_) => panic!("a pin past the ranks was accepted"),
        };
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(
            refused.message().contains("out of range [0, 4)"),
            "{refused}"
        );
    }

    /// Unpinned requests go to the rank holding the fewest of the worker's
    /// requests, the scan start rotating, so three in a row on an idle worker
    /// cover its three ranks; the counts come down when the streams end.
    #[tokio::test]
    async fn an_unpinned_request_goes_to_the_rank_with_the_fewest_requests() {
        let (service, _stop) = scheduler_with_ranks(3, 65_102);
        let mut streams = Vec::new();
        for id in ["dp-free-a", "dp-free-b", "dp-free-c"] {
            streams.push(
                service
                    .generate(Request::new(generate_request(id, None, 2)))
                    .await
                    .expect("accepted")
                    .into_inner(),
            );
        }
        let counts: Vec<usize> = service
            .ranks
            .iter()
            .map(|rank| rank.in_flight.load(Ordering::Relaxed))
            .collect();
        assert_eq!(counts, vec![1, 1, 1]);
        for stream in &mut streams {
            while stream.next().await.is_some() {}
        }
        let mut placed: Vec<String> = ["dp-free-a", "dp-free-b", "dp-free-c"]
            .iter()
            .map(|id| admitted_on(id))
            .collect();
        placed.sort();
        assert_eq!(
            placed,
            vec!["grpc:65102:r0", "grpc:65102:r1", "grpc:65102:r2"]
        );
        drop(streams);
        let counts: Vec<usize> = service
            .ranks
            .iter()
            .map(|rank| rank.in_flight.load(Ordering::Relaxed))
            .collect();
        assert_eq!(counts, vec![0, 0, 0]);
    }

    /// `GetLoads` lists one entry per rank, the request's `dp_rank` keeping
    /// one; `GetServerInfo` advertises the width.
    #[tokio::test]
    async fn loads_and_server_info_report_the_ranks() {
        let (service, _stop) = scheduler_with_ranks(4, 65_103);
        let loads = service
            .get_loads(Request::new(ts::GetLoadsRequest::default()))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!(loads.dp_rank_count, 4);
        let ranks: Vec<i32> = loads.loads.iter().map(|load| load.dp_rank).collect();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
        let one = service
            .get_loads(Request::new(ts::GetLoadsRequest {
                dp_rank: Some(3),
                include: Vec::new(),
            }))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!(
            (one.dp_rank_count, one.loads.len(), one.loads[0].dp_rank),
            (1, 1, 3)
        );
        let info = service
            .get_server_info(Request::new(ts::GetServerInfoRequest::default()))
            .await
            .expect("info")
            .into_inner();
        let width = info
            .server_args
            .expect("server args")
            .fields
            .remove("dp_size")
            .expect("dp_size");
        assert_eq!(
            width.kind,
            Some(prost_types::value::Kind::StringValue("4".to_string()))
        );
    }

    /// One subscription carries every rank's stream with one record state,
    /// as the relay's: the first record (before any batch) is rank 0's, a
    /// batch carries its own rank's record, and the heartbeat that follows
    /// names the last batch's rank.
    #[tokio::test]
    async fn one_subscription_carries_every_ranks_batches_with_their_ranks_records() {
        let (service, _stop) = scheduler_with_ranks(3, 65_104);
        let mut events = service
            .subscribe_kv_events(Request::new(common::SubscribeKvEventsRequest {
                start_sequence_number: 0,
                load_records_before_first_batch: true,
            }))
            .await
            .expect("subscribed")
            .into_inner();
        let first = tokio::time::timeout(Duration::from_secs(2), events.next())
            .await
            .expect("a record before the first batch")
            .expect("an item")
            .expect("a batch");
        assert!(first.load.as_ref().is_some_and(|load| load.load_only));
        assert_eq!((first.sequence_number, first.dp_rank), (0, Some(0)));

        // A request on rank 2 long enough to fill blocks (the engine
        // publishes a batch per completed block): its batches name rank 2
        // and carry a record.
        let mut request = generate_request("dp-sub-2", Some(2), 4);
        request.tokenized = Some(ts::TokenizedInput {
            input_ids: (100..180).collect(),
            ..Default::default()
        });
        let mut stream = service
            .generate(Request::new(request))
            .await
            .expect("accepted")
            .into_inner();
        while stream.next().await.is_some() {}
        let batch = loop {
            let batch = tokio::time::timeout(Duration::from_secs(5), events.next())
                .await
                .expect("rank 2's batch in time")
                .expect("an item")
                .expect("a batch");
            if !batch.events.is_empty() {
                break batch;
            }
        };
        assert_eq!(batch.dp_rank, Some(2));
        assert!(batch.load.as_ref().is_some_and(|load| !load.load_only));
        let beat = loop {
            let batch = tokio::time::timeout(Duration::from_secs(5), events.next())
                .await
                .expect("a heartbeat in time")
                .expect("an item")
                .expect("a batch");
            if batch.events.is_empty() {
                break batch;
            }
        };
        assert_eq!(
            beat.dp_rank,
            Some(2),
            "the heartbeat names the last batch's rank"
        );
    }

    /// As the Rust relay: a subscriber that asks for records before the
    /// stream's first batch gets a `load_only` one numbered 0 within a tick;
    /// one that does not waits for the first batch, whose sequence the
    /// heartbeats repeat from there.
    #[tokio::test]
    async fn load_records_before_the_first_batch_go_to_the_subscriber_that_asked() {
        let engine = Engine::spawn(engine::EngineParams::default());
        let records = |asked: bool, kv: KvEventStream| {
            with_load_records(
                vec![RankLoadSource {
                    engine: engine.clone(),
                    prefill_pending: Arc::new(AtomicU64::new(0)),
                }],
                LoadsLike::Mock,
                false,
                asked,
                kv,
            )
        };
        let quiet = || Box::pin(stream::pending()) as KvEventStream;
        let load_only =
            |batch: &common::KvEventBatch| batch.load.as_ref().is_some_and(|load| load.load_only);

        let mut asked = records(true, quiet());
        let first = tokio::time::timeout(Duration::from_millis(500), asked.next())
            .await
            .expect("a record before the stream's first batch")
            .expect("an item")
            .expect("a batch");
        assert!(load_only(&first), "{first:?}");
        assert_eq!((first.sequence_number, first.events.len()), (0, 0));

        let mut waiting = records(false, quiet());
        assert!(
            tokio::time::timeout(Duration::from_millis(400), waiting.next())
                .await
                .is_err(),
            "no batch may precede the stream's first for a subscriber that did not ask"
        );

        let seventh = common::KvEventBatch {
            sequence_number: 7,
            ..Default::default()
        };
        let mut after = records(
            false,
            Box::pin(stream::iter([Ok::<_, Status>(seventh)]).chain(stream::pending())),
        );
        let live = after
            .next()
            .await
            .expect("the first batch")
            .expect("a batch");
        assert_eq!(live.sequence_number, 7);
        assert!(!load_only(&live));
        let beat = tokio::time::timeout(
            HEARTBEAT_INTERVAL + Duration::from_millis(500),
            after.next(),
        )
        .await
        .expect("a heartbeat after the interval")
        .expect("an item")
        .expect("a batch");
        assert!(load_only(&beat));
        assert_eq!(beat.sequence_number, 7, "repeats the last sequence sent");
    }

    #[tokio::test]
    async fn a_cut_stream_ends_with_the_status_after_the_given_tokens() {
        let (tx, rx) = mpsc::unbounded_channel();
        for id in 1..=4 {
            tx.send(token(id)).unwrap();
        }
        tx.send(done(4)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let cut = Some((
            2,
            CutEnd::Status(Status::unavailable("injected")),
            Arc::clone(&cuts),
        ));
        let mut stream = generate_stream(rx, true, "r".to_string(), None, cut, None);
        assert_eq!(chunk_ids(stream.next().await), vec![1]);
        assert_eq!(chunk_ids(stream.next().await), vec![2]);
        let cut = stream.next().await.expect("the cut");
        assert!(matches!(cut, Err(ref status) if status.code() == tonic::Code::Unavailable));
        assert!(stream.next().await.is_none(), "nothing after the cut");
        assert_eq!(cuts.load(Ordering::Relaxed), 1, "the cut is counted once");
    }

    #[tokio::test]
    async fn an_engine_failure_cut_ends_with_an_error_complete_after_the_given_tokens() {
        let (tx, rx) = mpsc::unbounded_channel();
        for id in 1..=4 {
            tx.send(token(id)).unwrap();
        }
        tx.send(done(4)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let cut = Some((2, CutEnd::EngineError, Arc::clone(&cuts)));
        let mut stream = generate_stream(rx, true, "r".to_string(), None, cut, None);
        assert_eq!(chunk_ids(stream.next().await), vec![1]);
        assert_eq!(chunk_ids(stream.next().await), vec![2]);
        let item = stream
            .next()
            .await
            .expect("the cut")
            .expect("a response, not a status");
        match item.response {
            Some(GenResp::Complete(complete)) => {
                assert_eq!(complete.finish_reason, "error");
                assert_eq!(
                    complete.output_ids,
                    vec![1, 2],
                    "the output the engine had sent"
                );
                assert_eq!(complete.completion_tokens, 2);
            }
            other => panic!("not an error completion: {other:?}"),
        }
        assert!(stream.next().await.is_none(), "nothing after the cut");
        assert_eq!(cuts.load(Ordering::Relaxed), 1, "the cut is counted once");
    }

    #[tokio::test]
    async fn an_output_no_longer_than_the_cut_completes_as_usual() {
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(token(1)).unwrap();
        tx.send(done(1)).unwrap();
        drop(tx);
        let cuts = Arc::new(AtomicU64::new(0));
        let cut = Some((
            1,
            CutEnd::Status(Status::internal("injected")),
            Arc::clone(&cuts),
        ));
        let mut stream = generate_stream(rx, true, "r".to_string(), None, cut, None);
        assert_eq!(chunk_ids(stream.next().await), vec![1]);
        let complete = stream.next().await.expect("complete").expect("ok");
        assert!(matches!(complete.response, Some(GenResp::Complete(_))));
        assert!(stream.next().await.is_none());
        assert_eq!(
            cuts.load(Ordering::Relaxed),
            0,
            "an output no longer than the cut is not counted as cut"
        );
    }

    #[test]
    fn injected_statuses_map_to_the_closest_grpc_code() {
        assert_eq!(injected_status(503).code(), tonic::Code::Unavailable);
        assert_eq!(injected_status(502).code(), tonic::Code::Unavailable);
        assert_eq!(injected_status(500).code(), tonic::Code::Internal);
        assert_eq!(injected_status(429).code(), tonic::Code::ResourceExhausted);
        assert_eq!(injected_status(504).code(), tonic::Code::DeadlineExceeded);
        assert_eq!(injected_status(400).code(), tonic::Code::InvalidArgument);
        assert_eq!(injected_status(499).code(), tonic::Code::Cancelled);
        // The three non-inverse cases (408 and 502 read back as 504 and 503,
        // 499 as 502) and a 4xx without its own code (read back as 400).
        assert_eq!(injected_status(408).code(), tonic::Code::DeadlineExceeded);
        assert_eq!(injected_status(502).code(), tonic::Code::Unavailable);
        assert_eq!(injected_status(418).code(), tonic::Code::InvalidArgument);
        // 422 is the engine's report that it could not continue the request
        // under its constraint, which the gateway reads back as its 422.
        let constraint = injected_status(422);
        assert_eq!(constraint.code(), tonic::Code::FailedPrecondition);
        assert!(smg_grpc_client::is_constraint_failure(&constraint));
    }
}
