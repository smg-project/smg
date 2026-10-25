//! Mock gRPC worker implementing TokenSpeed and vLLM control services. The gateway
//! tokenizes and sends token ids; this service streams back canned token ids.
//! Beside it the server hosts the standard `grpc.health.v1` service, which
//! answers as the scheduler's own `HealthCheck` does.

mod admission;
mod native;
mod vllm;

use std::{
    future::{pending, Future},
    pin::{pin, Pin},
    sync::{
        atomic::{AtomicU64, Ordering},
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

/// Serve the mock gRPC services on `port` until the process exits.
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
    let engine = cfg
        .realistic
        .then(|| Engine::spawn_named(cfg.engine.clone(), name.clone(), true));
    // A default base of 0 denotes one already-bound library listener, not
    // a fleet offset derived from the OS-assigned ephemeral port.
    let publisher_index = if cfg.grpc_base_port == 0 {
        Some(0)
    } else {
        addr.and_then(|a| a.port().checked_sub(cfg.grpc_base_port))
    };
    let publisher_cfg = publisher_index.and_then(|index| cfg.kv_zmq_for(index, 0));
    let publisher = match (engine.as_ref(), publisher_cfg) {
        (Some(engine), Some(kv)) => {
            match crate::kv_zmq::BoundPublisher::bind(engine.clone(), kv).await {
                Ok(publisher) => Some(publisher),
                Err(error) => {
                    tracing::error!("{error}");
                    None
                }
            }
        }
        _ => None,
    };
    let kv_source = publisher.as_ref().map(|publisher| publisher.source());
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
        vllm: cfg.grpc_vllm,
        native_vllm: cfg.grpc_native_vllm,
        overrides: Arc::clone(&overrides),
        stopped: stopped.clone(),
    };
    let service = MockScheduler {
        cfg,
        name,
        engine,
        capture,
        stopped,
        prefill_pending: Arc::new(AtomicU64::new(0)),
        admissions: admission::Admissions::default(),
        kv_source,
        overrides,
    };
    let vllm_service = service.cfg.grpc_vllm.then(|| {
        vllm::VllmServer::new(vllm::MockVllm::new(service.clone()))
            .max_decoding_message_size(max_message_bytes)
            .max_encoding_message_size(max_message_bytes)
    });
    let native_service = service.cfg.grpc_native_vllm.then(|| {
        native::NativeServer::new(native::MockNative(service.clone()))
            .max_decoding_message_size(max_message_bytes)
            .max_encoding_message_size(max_message_bytes)
    });
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
            .add_optional_service(vllm_service)
            .add_optional_service(native_service)
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
            let mut publisher = pin!(publisher.run());
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
    /// Present iff the worker runs the realistic engine simulator.
    engine: Option<Engine>,
    /// Present iff `--capture` is set: every `Generate` request is recorded.
    capture: Option<Arc<Capture>>,
    /// Flips when the worker stops: ends the streams that have no end of
    /// their own.
    stopped: watch::Receiver<bool>,
    /// Prompt tokens of the requests handed to the engine with no token yet:
    /// the prefill work in hand, as a servicer reports it on its load record.
    prefill_pending: Arc<AtomicU64>,
    /// Requests waiting before engine submission, shared by all gRPC adapters.
    admissions: admission::Admissions,
    /// Bound publisher metadata with a lease that expires when its task stops.
    kv_source: Option<crate::kv_zmq::Source>,
    /// What the admin API made this worker report about itself (serving,
    /// server args, model capabilities).
    overrides: Arc<crate::overrides::WorkerOverrides>,
}

impl MockScheduler {
    /// Abort before or after submission with the same ordering for every adapter.
    fn abort_request(&self, id: String) {
        if let Some(engine) = &self.engine {
            self.admissions.abort(&id, || engine.abort(id.clone()));
        }
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

        // Realistic mode: submit to the engine simulator and stream its output.
        if let Some(engine) = &self.engine {
            let mut admission = self.admissions.register(req.request_id.clone());
            // The fault hook's answer for this request, when one is armed: a
            // stall, a status before admission, or a cut after some output.
            let injected = engine.inject();
            if let Some(fault) = injected {
                if !fault.stall.is_zero() {
                    admission.wait(tokio::time::sleep(fault.stall)).await?;
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
            admission.wait(engine.admit()).await?;
            let weight_version = self.cfg.weight_version.clone();
            let (tx, rx) = mpsc::unbounded_channel();
            let pending = (
                Arc::clone(&self.prefill_pending),
                prompt_token_ids.len() as u64,
            );
            admission.submit(|| {
                pending.0.fetch_add(pending.1, Ordering::Relaxed);
                engine.submit(NewRequest {
                    request_id: request_id.clone(),
                    prompt_token_ids,
                    max_new,
                    events: tx,
                });
            })?;
            let stream = generate_stream(
                rx,
                stream_chunks,
                request_id,
                Some(pending),
                cut,
                weight_version,
            );
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
        self.abort_request(request.into_inner().request_id);
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
                let request = request.into_inner();
                let events = with_load_records(
                    engine.clone(),
                    self.cfg.loads_like,
                    self.cfg.step_reports,
                    Arc::clone(&self.prefill_pending),
                    request.load_records_before_first_batch,
                    engine.subscribe_kv(request.start_sequence_number),
                );
                // A subscription ends with the worker's stop, as a serving
                // engine's ends with its process; nothing else ends it, and
                // the server's shutdown waits for it.
                let mut stopped = self.stopped.clone();
                let stop = async move {
                    let _ = stopped.wait_for(|stop| *stop).await;
                };
                Ok(Response::new(Box::pin(events.take_until(stop))))
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

/// `Health.Watch` poll interval: the override is read, not signalled.
const HEALTH_WATCH_INTERVAL: Duration = Duration::from_millis(250);

type HealthWatchStream = Pin<Box<dyn Stream<Item = Result<HealthCheckResponse, Status>> + Send>>;

/// The standard `grpc.health.v1` service of a gRPC worker: the whole server
/// (the empty name) and enabled engine services are SERVING exactly when the
/// scheduler's own `HealthCheck` answers healthy, i.e. unless the admin API
/// set the worker not serving (`/admin/serving/{port}`); any other name is
/// unknown. It is the service a harness or a probe asks before it asks the
/// engine's own protocol, as every engine servicer hosts it.
#[derive(Clone)]
struct MockHealth {
    /// The optional vLLM service is known only when explicitly enabled.
    vllm: bool,
    native_vllm: bool,
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
            && !(self.vllm && service == <vllm::VllmServer as NamedService>::NAME)
            && !(self.native_vllm && service == <native::NativeServer as NamedService>::NAME)
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

struct LoadRecords {
    stream: KvEventStream,
    engine: engine::Engine,
    like: LoadsLike,
    /// What the record says of the wire (`--step-reports`).
    step_reports: bool,
    /// The worker's prompt tokens handed to the engine with no token yet.
    prefill_pending: Arc<AtomicU64>,
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
    fn record(&mut self, load_only: bool) -> common::EngineLoad {
        self.sample += 1;
        let snapshot = self.engine.load().as_reported_by(self.like);
        let mut record = load_record(&snapshot, self.like, self.sample);
        record.engine_silence_ms = silence_ms(&self.engine);
        record.engine_reports_steps = Some(self.step_reports);
        record.prefill_pending_tokens = Some(pending_u32(&self.prefill_pending));
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
        let mut current = load_record(&self.engine.load().as_reported_by(self.like), self.like, 0);
        current.engine_silence_ms = silence_ms(&self.engine);
        current.engine_reports_steps = Some(self.step_reports);
        current.prefill_pending_tokens = Some(pending_u32(&self.prefill_pending));
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
            sequence_number,
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
/// them (`crates/engine_servicer/src/kv_events.rs`): from the subscription
/// on for a subscriber that takes records before the first batch
/// (`records_before_first_batch`), from the stream's first batch otherwise.
fn with_load_records(
    engine: engine::Engine,
    like: LoadsLike,
    step_reports: bool,
    prefill_pending: Arc<AtomicU64>,
    records_before_first_batch: bool,
    stream: KvEventStream,
) -> KvEventStream {
    let records = LoadRecords {
        stream,
        engine,
        like,
        step_reports,
        prefill_pending,
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
            check(<vllm::VllmServer as NamedService>::NAME)
                .await
                .expect_err("disabled")
                .code(),
            Code::NotFound
        );
        assert_eq!(
            check(<native::NativeServer as NamedService>::NAME)
                .await
                .expect_err("native control disabled")
                .code(),
            Code::NotFound
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

    /// As the Rust relay: a subscriber that asks for records before the
    /// stream's first batch gets a `load_only` one numbered 0 within a tick;
    /// one that does not waits for the first batch, whose sequence the
    /// heartbeats repeat from there.
    #[tokio::test]
    async fn load_records_before_the_first_batch_go_to_the_subscriber_that_asked() {
        let engine = Engine::spawn(engine::EngineParams::default());
        let records = |asked: bool, kv: KvEventStream| {
            with_load_records(
                engine.clone(),
                LoadsLike::Mock,
                false,
                Arc::new(AtomicU64::new(0)),
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
