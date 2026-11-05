//! The cost of the generate path, in process, over the crate's mock engine:
//! a servicer answers streamed requests whose engine side is a mock that
//! answers in steps, as an engine does: one output message per step carrying
//! the next token of every running request, a `length` finish on each
//! request's last. What criterion times is a closed loop of
//! `GENERATE_PATH_STREAMS` concurrent requests end to end (the gRPC calls,
//! the engine adds, the steps back through the servicer, the terminals; `1`
//! and `48` by default, a comma-separated list), with the metrics endpoint
//! off or on (`GENERATE_PATH_METRICS=on`), so the two can be compared and the
//! per-request and per-token cost of the observations read, alone and under
//! the per-output work of a step of many.
//!
//! For a counter-based reading (`perf stat` on the binary),
//! `GENERATE_PATH_REQUESTS=<n>` runs `n` requests, in rounds of the first
//! `GENERATE_PATH_STREAMS` entry at once (one by default; `48` for the
//! counters of the 48-stream shape), and exits without criterion;
//! `GENERATE_PATH_TOKENS` sets the tokens per request (64 by default), so two
//! runs at different token counts separate the per-token cost from the
//! per-request one. With the endpoint on, a scrape every
//! `GENERATE_PATH_SCRAPE_EVERY` requests reports its render time.
//!
//! `GENERATE_PATH_WIRE=vllm|sglang|tokenspeed|trtllm` (vllm by default) picks
//! the wire: the same closed loop over that wire's servicer, proto and mock
//! engine, so a change to what the four share is read on each of them.
//!
//! Run: `cargo bench -p engine-servicer --bench generate_path`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_precision_loss
)]

use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use engine_servicer::{
    ServicerConfig, SglangModelInfo, SglangServicerConfig, SglangServicerServer,
    TokenSpeedModelInfo, TokenSpeedServicerConfig, TokenSpeedServicerServer, TrtllmModelInfo,
    TrtllmServicerConfig, TrtllmServicerServer, VllmModelInfo, VllmServicerConfig,
    VllmServicerServer,
};
use engine_zmq_client::{
    codec::{decode_msgpack, encode_msgpack},
    mock_engine::{
        connect_to_frontend, default_ready_response, EngineInbound, MockEngine, MockEngineInput,
    },
    protocol::{
        handshake::EngineCoreReadyResponse,
        sglang::{
            output::BatchTokenIDSlimOutput,
            request::{SglangRequestType, TokenizedGenerateReqInput as SglangAdd},
        },
        tokenspeed::{
            output::BatchTokenIDOutSlim,
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput as TokenSpeedAdd},
        },
        trtllm::request::{
            TrtllmGenerateReqInput, TrtllmRequestType, GENERATE_REQ_TAG, GENERATE_WIRE_FIELDS,
            SAMPLING_PARAMS_TAG, SAMPLING_WIRE_FIELDS,
        },
        vllm::output::{
            EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
        },
    },
    EngineId,
};
use futures::{future::join_all, StreamExt};
use smg_grpc_client::{
    sglang_proto as sg, sglang_proto::sglang_scheduler_client::SglangSchedulerClient,
    tokenspeed_proto as ts,
    tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient, trtllm_proto as trt,
    trtllm_proto::trtllm_service_client::TrtllmServiceClient, vllm_proto as vllm,
    vllm_proto::vllm_engine_client::VllmEngineClient,
};
use tokio::sync::mpsc;
use tonic::transport::Channel;

/// The wire under the bench: its servicer, its proto and its mock engine.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wire {
    Vllm,
    Sglang,
    TokenSpeed,
    Trtllm,
}

impl Wire {
    /// `GENERATE_PATH_WIRE`, vllm by default; an unknown name fails the run
    /// by name, as every misconfigured piece of this bed does.
    fn from_env() -> Self {
        match std::env::var("GENERATE_PATH_WIRE").as_deref() {
            Err(_) | Ok("vllm") => Self::Vllm,
            Ok("sglang") => Self::Sglang,
            Ok("tokenspeed") => Self::TokenSpeed,
            Ok("trtllm") => Self::Trtllm,
            Ok(other) => panic!(
                "generate_path: GENERATE_PATH_WIRE={other}: vllm, sglang, tokenspeed or trtllm"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
            Self::TokenSpeed => "tokenspeed",
            Self::Trtllm => "trtllm",
        }
    }

    /// The handshake the mock engine answers with: the shim of the
    /// TensorRT-LLM wire declares the field order of the two structs it
    /// decodes, the others take the default.
    fn ready_response(self) -> EngineCoreReadyResponse {
        match self {
            Self::Trtllm => {
                let names =
                    |fields: &[&str]| fields.iter().map(|f| f.to_string()).collect::<Vec<_>>();
                EngineCoreReadyResponse {
                    positional_layouts: BTreeMap::from([
                        (GENERATE_REQ_TAG.to_string(), names(&GENERATE_WIRE_FIELDS)),
                        (
                            SAMPLING_PARAMS_TAG.to_string(),
                            names(&SAMPLING_WIRE_FIELDS),
                        ),
                    ]),
                    ..default_ready_response()
                }
            }
            _ => default_ready_response(),
        }
    }
}

const DEFAULT_TOKENS: u32 = 64;

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn tokens_per_request() -> u32 {
    env_u64("GENERATE_PATH_TOKENS").map_or(DEFAULT_TOKENS, |tokens| tokens as u32)
}

fn metrics_on() -> bool {
    std::env::var("GENERATE_PATH_METRICS").is_ok_and(|value| value == "on")
}

/// The concurrent streams of a closed loop (`GENERATE_PATH_STREAMS`, a
/// comma-separated list; `1,48` by default).
fn streams_per_loop() -> Vec<u64> {
    std::env::var("GENERATE_PATH_STREAMS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|streams| streams.trim().parse().ok())
                .filter(|streams| *streams > 0)
                .collect::<Vec<u64>>()
        })
        .filter(|streams| !streams.is_empty())
        .unwrap_or_else(|| vec![1, 48])
}

/// The mock engine's side of the wire, answering in steps as an engine does:
/// every add joins the running set, and each step sends one message in the
/// wire's output shape with the next token of every running request
/// (`tokens` tokens each, a `length` finish on the last). A reader task takes
/// the adds off the socket, so the stepper never waits for one while requests
/// run.
async fn answer_adds(wire: Wire, engine: MockEngine, tokens: u32) {
    let (mut input, mut output) = engine.split();
    let (adds, mut added) = mpsc::unbounded_channel::<String>();
    #[expect(
        clippy::disallowed_methods,
        reason = "the reader ends with the frontend's socket, as the mock engine does"
    )]
    let _reader = tokio::spawn(async move {
        loop {
            let add = match wire {
                Wire::Vllm => match input.recv().await {
                    Ok(EngineInbound::Add(request)) => Some(request.request_id.clone()),
                    Ok(_) => None,
                    Err(_) => return,
                },
                Wire::Sglang | Wire::TokenSpeed | Wire::Trtllm => {
                    let Some(add) = next_add_rid(wire, &mut input).await else {
                        return;
                    };
                    add
                }
            };
            if let Some(rid) = add {
                if adds.send(rid).is_err() {
                    return;
                }
            }
        }
    });
    let mut running: Vec<(String, u32)> = Vec::new();
    loop {
        if running.is_empty() {
            match added.recv().await {
                Some(rid) => running.push((rid, 0)),
                None => return,
            }
        }
        while let Ok(rid) = added.try_recv() {
            running.push((rid, 0));
        }
        for (_, sent) in &mut running {
            *sent += 1;
        }
        if output
            .send_frames(step_frames(wire, &running, tokens))
            .await
            .is_err()
        {
            return;
        }
        running.retain(|(_, sent)| *sent < tokens);
    }
}

/// The request id of the next add on a msgpack wire (`None` on anything
/// else: an abort, or a frame of another kind).
async fn next_add_rid(wire: Wire, input: &mut MockEngineInput) -> Option<Option<String>> {
    let frames = input.recv_frames().await.ok()?;
    let type_frame = frames.first()?.as_ref();
    let payload = frames.get(1)?.as_ref();
    Some(match wire {
        Wire::Sglang
            if SglangRequestType::from_frame(type_frame) == Some(SglangRequestType::Add) =>
        {
            Some(
                decode_msgpack::<SglangAdd>(payload)
                    .expect("the add decodes")
                    .rid,
            )
        }
        Wire::TokenSpeed
            if TokenSpeedRequestType::from_frame(type_frame)
                == Some(TokenSpeedRequestType::Add) =>
        {
            Some(
                decode_msgpack::<TokenSpeedAdd>(payload)
                    .expect("the add decodes")
                    .rid,
            )
        }
        Wire::Trtllm
            if TrtllmRequestType::from_frame(type_frame) == Some(TrtllmRequestType::Add) =>
        {
            Some(
                decode_msgpack::<TrtllmGenerateReqInput>(payload)
                    .expect("the add decodes")
                    .rid,
            )
        }
        _ => None,
    })
}

/// One step's message in the wire's output shape: the token every running
/// request got this step (its `sent`th), a `length` finish where it was the
/// last of the `tokens`.
fn step_frames(wire: Wire, running: &[(String, u32)], tokens: u32) -> Vec<Bytes> {
    let count = running.len();
    let rids: Vec<String> = running.iter().map(|(rid, _)| rid.clone()).collect();
    let output_ids: Vec<Vec<u32>> = running.iter().map(|(_, sent)| vec![sent + 9]).collect();
    let finished_reasons: Vec<String> = running
        .iter()
        .map(|(_, sent)| {
            if *sent == tokens {
                "length".to_string()
            } else {
                String::new()
            }
        })
        .collect();
    let completion_tokens: Vec<u32> = running.iter().map(|(_, sent)| *sent).collect();
    let encoded = match wire {
        Wire::Vllm => {
            let finished: BTreeSet<String> = running
                .iter()
                .filter(|(_, sent)| *sent == tokens)
                .map(|(rid, _)| rid.clone())
                .collect();
            encode_msgpack(&EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
                engine_index: 0,
                outputs: running
                    .iter()
                    .map(|(rid, sent)| EngineCoreOutput {
                        request_id: rid.clone(),
                        new_token_ids: vec![sent + 9],
                        finish_reason: (*sent == tokens).then_some(EngineCoreFinishReason::Length),
                        ..Default::default()
                    })
                    .collect(),
                finished_requests: (!finished.is_empty()).then_some(finished),
                ..Default::default()
            }))
        }
        Wire::Sglang | Wire::Trtllm => encode_msgpack(&BatchTokenIDSlimOutput {
            rids,
            output_ids,
            finished_reasons,
            finished_messages: vec![None; count],
            finished_matched: (0..count).map(|_| None).collect(),
            prompt_tokens: vec![3; count],
            completion_tokens,
            cached_tokens: vec![0; count],
            output_token_logprobs_val: vec![Vec::new(); count],
            output_token_logprobs_idx: vec![Vec::new(); count],
            engine_index: 0,
            num_running: u64::try_from(count).unwrap_or(u64::MAX),
            num_waiting: 0,
            kv_used_tokens: 10,
            kv_total_tokens: 100,
            finished_status: vec![None; count],
            ..Default::default()
        }),
        Wire::TokenSpeed => encode_msgpack(&BatchTokenIDOutSlim {
            rids,
            output_ids,
            finished_reasons,
            prompt_tokens: vec![3; count],
            completion_tokens,
            cached_tokens: vec![0; count],
            output_token_logprobs_val: vec![Vec::new(); count],
            output_token_logprobs_idx: vec![Vec::new(); count],
            engine_index: 0,
            num_running: u64::try_from(count).unwrap_or(u64::MAX),
            num_waiting: 0,
            kv_active_pages: 1,
            kv_total_pages: 100,
            ..Default::default()
        }),
    };
    vec![Bytes::from(encoded.expect("a step encodes"))]
}

/// The wire's running servicer.
enum Server {
    Vllm(VllmServicerServer),
    Sglang(SglangServicerServer),
    TokenSpeed(TokenSpeedServicerServer),
    Trtllm(TrtllmServicerServer),
}

impl Server {
    fn start(wire: Wire, servicer: ServicerConfig) -> Self {
        let model_path = "org/model".to_string();
        let served = "served".to_string();
        match wire {
            Wire::Vllm => Self::Vllm(
                VllmServicerServer::start(VllmServicerConfig {
                    servicer,
                    model: VllmModelInfo {
                        model_path,
                        served_model_name: served,
                        tokenizer_path: "org/model".to_string(),
                        is_generation: true,
                        max_context_length: 4096,
                        vocab_size: 1024,
                        data_parallel_size: 1,
                        ..Default::default()
                    },
                    media_processor: None,
                })
                .expect("the servicer starts"),
            ),
            Wire::Sglang => Self::Sglang(
                SglangServicerServer::start(SglangServicerConfig {
                    servicer,
                    model: SglangModelInfo {
                        model_path,
                        served_model_name: served,
                        tokenizer_path: "org/model".to_string(),
                        is_generation: true,
                        max_context_length: 4096,
                        vocab_size: 1024,
                        eos_token_ids: vec![999],
                        server_args_json: r#"{"model_path":"org/model","tp_size":1,"dp_size":1}"#
                            .to_string(),
                        scheduler_info_json: r#"{"is_generation":true}"#.to_string(),
                        max_running_requests: 64,
                        data_parallel_size: 1,
                        ..Default::default()
                    },
                    encode_dispatcher: None,
                    media_processor: None,
                })
                .expect("the servicer starts"),
            ),
            Wire::TokenSpeed => Self::TokenSpeed(
                TokenSpeedServicerServer::start(TokenSpeedServicerConfig {
                    servicer,
                    model: TokenSpeedModelInfo {
                        model_path,
                        served_model_name: served,
                        tokenizer_path: "org/model".to_string(),
                        max_context_length: 4096,
                        vocab_size: 1024,
                        eos_token_ids: vec![999],
                        server_args_json: r#"{"model":"org/model","max_num_seqs":64}"#.to_string(),
                        scheduler_info_json: "{}".to_string(),
                        max_running_requests: 64,
                        data_parallel_size: 1,
                        ..Default::default()
                    },
                })
                .expect("the servicer starts"),
            ),
            Wire::Trtllm => Self::Trtllm(
                TrtllmServicerServer::start(TrtllmServicerConfig {
                    servicer,
                    model: TrtllmModelInfo {
                        model_path,
                        served_model_name: served,
                        max_input_len: 4096,
                        max_seq_len: 8192,
                        vocab_size: 1024,
                        eos_token_ids: vec![999],
                        data_parallel_size: 1,
                        max_running_requests: 64,
                        ..Default::default()
                    },
                })
                .expect("the servicer starts"),
            ),
        }
    }

    fn address(&self) -> String {
        match self {
            Self::Vllm(server) => server.address(),
            Self::Sglang(server) => server.address(),
            Self::TokenSpeed(server) => server.address(),
            Self::Trtllm(server) => server.address(),
        }
    }

    fn metrics_address(&self) -> Option<String> {
        match self {
            Self::Vllm(server) => server.metrics_address(),
            Self::Sglang(server) => server.metrics_address(),
            Self::TokenSpeed(server) => server.metrics_address(),
            Self::Trtllm(server) => server.metrics_address(),
        }
    }

    fn engine_ready(&self) -> bool {
        match self {
            Self::Vllm(server) => server.engine_ready(),
            Self::Sglang(server) => server.engine_ready(),
            Self::TokenSpeed(server) => server.engine_ready(),
            Self::Trtllm(server) => server.engine_ready(),
        }
    }

    fn stop(&self, timeout: Duration) {
        match self {
            Self::Vllm(server) => server.stop(timeout),
            Self::Sglang(server) => server.stop(timeout),
            Self::TokenSpeed(server) => server.stop(timeout),
            Self::Trtllm(server) => server.stop(timeout),
        }
        .expect("a clean stop");
    }
}

/// The wire's gRPC client, driving one streamed request at a time.
#[derive(Clone)]
enum Client {
    Vllm(VllmEngineClient<Channel>),
    Sglang(SglangSchedulerClient<Channel>),
    TokenSpeed(TokenSpeedSchedulerClient<Channel>),
    Trtllm(TrtllmServiceClient<Channel>),
}

impl Client {
    fn new(wire: Wire, channel: Channel) -> Self {
        match wire {
            Wire::Vllm => Self::Vllm(VllmEngineClient::new(channel)),
            Wire::Sglang => Self::Sglang(SglangSchedulerClient::new(channel)),
            Wire::TokenSpeed => Self::TokenSpeed(TokenSpeedSchedulerClient::new(channel)),
            Wire::Trtllm => Self::Trtllm(TrtllmServiceClient::new(channel)),
        }
    }

    /// One streamed request of `max_tokens` tokens under `request_id`, read
    /// to its end; returns the tokens received in chunks.
    async fn request(&mut self, request_id: String, max_tokens: u32) -> u32 {
        let input_ids = vec![1, 2, 3];
        let mut tokens = 0;
        match self {
            Self::Vllm(client) => {
                let mut stream = client
                    .generate(vllm::GenerateRequest {
                        request_id,
                        input: Some(vllm::generate_request::Input::Tokenized(
                            vllm::TokenizedInput {
                                original_text: String::new(),
                                input_ids,
                            },
                        )),
                        sampling_params: Some(vllm::SamplingParams {
                            max_tokens: Some(max_tokens),
                            ..Default::default()
                        }),
                        stream: true,
                        ..Default::default()
                    })
                    .await
                    .expect("the call is accepted")
                    .into_inner();
                while let Some(item) = stream.next().await {
                    if let Some(vllm::generate_response::Response::Chunk(chunk)) =
                        item.expect("a response").response
                    {
                        tokens += chunk.token_ids.len() as u32;
                    }
                }
            }
            Self::Sglang(client) => {
                let mut stream = client
                    .generate(sg::GenerateRequest {
                        request_id,
                        tokenized: Some(sg::TokenizedInput {
                            input_ids,
                            original_text: String::new(),
                        }),
                        sampling_params: Some(sg::SamplingParams {
                            max_new_tokens: Some(max_tokens),
                            ..Default::default()
                        }),
                        logprob_start_len: -1,
                        stream: true,
                        ..Default::default()
                    })
                    .await
                    .expect("the call is accepted")
                    .into_inner();
                while let Some(item) = stream.next().await {
                    if let Some(sg::generate_response::Response::Chunk(chunk)) =
                        item.expect("a response").response
                    {
                        tokens += chunk.token_ids.len() as u32;
                    }
                }
            }
            Self::TokenSpeed(client) => {
                let mut stream = client
                    .generate(ts::GenerateRequest {
                        request_id,
                        tokenized: Some(ts::TokenizedInput {
                            input_ids,
                            original_text: String::new(),
                        }),
                        sampling_params: Some(ts::SamplingParams {
                            max_new_tokens: Some(max_tokens),
                            ..Default::default()
                        }),
                        stream: true,
                        ..Default::default()
                    })
                    .await
                    .expect("the call is accepted")
                    .into_inner();
                while let Some(item) = stream.next().await {
                    if let Some(ts::generate_response::Response::Chunk(chunk)) =
                        item.expect("a response").response
                    {
                        tokens += chunk.token_ids.len() as u32;
                    }
                }
            }
            Self::Trtllm(client) => {
                let mut stream = client
                    .generate(trt::GenerateRequest {
                        request_id,
                        tokenized: Some(trt::TokenizedInput {
                            original_text: String::new(),
                            input_token_ids: input_ids,
                            query_token_ids: vec![],
                        }),
                        sampling_config: Some(trt::SamplingConfig {
                            beam_width: 1,
                            num_return_sequences: 1,
                            ..Default::default()
                        }),
                        output_config: Some(trt::OutputConfig {
                            exclude_input_from_output: true,
                            ..Default::default()
                        }),
                        max_tokens,
                        streaming: true,
                        ..Default::default()
                    })
                    .await
                    .expect("the call is accepted")
                    .into_inner();
                while let Some(item) = stream.next().await {
                    if let Some(trt::generate_response::Response::Chunk(chunk)) =
                        item.expect("a response").response
                    {
                        tokens += chunk.token_ids.len() as u32;
                    }
                }
            }
        }
        tokens
    }
}

struct Bed {
    wire: Wire,
    server: Server,
    client: Client,
    tokens: u32,
    sent: AtomicU64,
    _dir: tempfile::TempDir,
}

impl Bed {
    async fn start(wire: Wire, tokens: u32, metrics: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let handshake = format!("ipc://{}", dir.path().join("handshake").display());
        let servicer = ServicerConfig {
            engine_startup_timeout: Duration::from_secs(30),
            engine_startup_ceiling: None,
            metrics_bind_address: metrics.then(|| "127.0.0.1:0".to_string()),
            ..ServicerConfig::new(
                "127.0.0.1:0",
                format!("ipc://{}", dir.path().join("engine").display()),
                &handshake,
            )
        };
        let server = Server::start(wire, servicer);
        let engine = connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            wire.ready_response(),
        )
        .await
        .expect("the mock engine handshakes");
        while !server.engine_ready() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        #[expect(
            clippy::disallowed_methods,
            reason = "the mock engine answers until the bench's runtime drops"
        )]
        let _mock = tokio::spawn(answer_adds(wire, engine, tokens));
        let channel = Channel::from_shared(format!("http://{}", server.address()))
            .unwrap()
            .connect()
            .await
            .expect("the gRPC channel connects");
        Self {
            wire,
            server,
            client: Client::new(wire, channel),
            tokens,
            sent: AtomicU64::new(0),
            _dir: dir,
        }
    }

    /// `streams` streamed requests at once, each read to its end; returns
    /// the tokens received over all of them, every token every stream was
    /// due: a step whose outputs did not all reach their streams fails here.
    async fn requests(&self, streams: u64) -> u32 {
        let first = self.sent.fetch_add(streams, Ordering::Relaxed) + 1;
        let received: u32 = join_all((first..first + streams).map(|id| self.request(id)))
            .await
            .into_iter()
            .sum();
        assert_eq!(
            u64::from(received),
            streams * u64::from(self.tokens),
            "every stream ends with its tokens"
        );
        received
    }

    /// One streamed request under `id`, read to its end; returns the tokens
    /// received.
    async fn request(&self, id: u64) -> u32 {
        self.client
            .clone()
            .request(format!("r{id}"), self.tokens)
            .await
    }

    fn scrape(&self) -> Option<(Duration, usize)> {
        let address: SocketAddr = self.server.metrics_address()?.parse().ok()?;
        let started = Instant::now();
        let mut stream = TcpStream::connect(address).ok()?;
        write!(
            stream,
            "GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .ok()?;
        let mut body = String::new();
        stream.read_to_string(&mut body).ok()?;
        Some((started.elapsed(), body.len()))
    }
}

/// The process's resident set in kB (`VmRSS` of `/proc/self/status`), where
/// the platform has it.
fn resident_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// The counter-based run: `GENERATE_PATH_REQUESTS` requests in a closed
/// loop of rounds of the first `GENERATE_PATH_STREAMS` entry at once, the
/// wall time, the scrape times and the resident set after the first thousand
/// requests and at the end printed.
fn counted_run(runtime: &tokio::runtime::Runtime, requests: u64) {
    let tokens = tokens_per_request();
    let metrics = metrics_on();
    let streams = streams_per_loop()[0];
    let scrape_every = env_u64("GENERATE_PATH_SCRAPE_EVERY").unwrap_or(0);
    let wire = Wire::from_env();
    let bed = runtime.block_on(Bed::start(wire, tokens, metrics));
    let started = Instant::now();
    let mut received = 0u64;
    let mut scrapes = Vec::new();
    let mut rss_after_1k = None;
    let mut sent = 0u64;
    let mut scrapes_due = 0u64;
    while sent < requests {
        let round = streams.min(requests - sent);
        received += u64::from(runtime.block_on(bed.requests(round)));
        sent += round;
        if scrape_every > 0 && sent / scrape_every > scrapes_due {
            scrapes_due = sent / scrape_every;
            if let Some(scrape) = bed.scrape() {
                scrapes.push(scrape);
            }
        }
        if rss_after_1k.is_none() && sent >= 1000 {
            rss_after_1k = resident_kb();
        }
    }
    let elapsed = started.elapsed();
    if let (Some(after_1k), Some(at_end)) = (rss_after_1k, resident_kb()) {
        eprintln!(
            "generate_path: resident set {after_1k} kB after 1,000 requests, {at_end} kB after \
             {requests} ({:+} kB)",
            i64::try_from(at_end).unwrap_or(i64::MAX) - i64::try_from(after_1k).unwrap_or(0)
        );
    }
    eprintln!(
        "generate_path: {requests} requests x {tokens} tokens, {streams} at once, on the {} wire, \
         metrics {}: {received} tokens in {:.3} s ({:.1} us per request)",
        bed.wire.name(),
        if metrics { "on" } else { "off" },
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1e6 / requests as f64
    );
    if !scrapes.is_empty() {
        let mut times: Vec<Duration> = scrapes.iter().map(|(time, _)| *time).collect();
        times.sort();
        eprintln!(
            "generate_path: {} scrapes, median {:.3} ms, max {:.3} ms, {} bytes",
            times.len(),
            times[times.len() / 2].as_secs_f64() * 1e3,
            times[times.len() - 1].as_secs_f64() * 1e3,
            scrapes[scrapes.len() - 1].1
        );
    }
    bed.server.stop(Duration::from_secs(5));
}

/// A panic anywhere in the bed ends the run with a non-zero exit. The mock
/// engine answers from a detached task, where Tokio catches a panic and the
/// client would keep waiting on a stream nobody answers; Cargo ignores
/// `panic = "abort"` for a bench profile, so the hook does it.
#[expect(
    clippy::disallowed_methods,
    reason = "a bench whose bed broke ends with a non-zero exit instead of hanging"
)]
fn exit_on_panic() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        std::process::exit(101)
    }));
}

fn bench_generate_path(c: &mut Criterion) {
    exit_on_panic();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    if let Some(requests) = env_u64("GENERATE_PATH_REQUESTS") {
        counted_run(&runtime, requests);
        return;
    }
    let tokens = tokens_per_request();
    let metrics = metrics_on();
    let wire = Wire::from_env();
    let bed = runtime.block_on(Bed::start(wire, tokens, metrics));
    let mut group = c.benchmark_group("generate_path");
    // The vLLM row keeps its name (the ledger's rows read it); the other
    // wires' rows carry theirs, and the rows of several streams at once
    // their count.
    let suffix = match wire {
        Wire::Vllm => String::new(),
        other => format!("_{}", other.name()),
    };
    let metrics_label = if metrics { "on" } else { "off" };
    for streams in streams_per_loop() {
        let id = if streams == 1 {
            format!("request_{tokens}_tokens_metrics_{metrics_label}{suffix}")
        } else {
            format!("streams_{streams}_x_{tokens}_tokens_metrics_{metrics_label}{suffix}")
        };
        group.throughput(Throughput::Elements(u64::from(tokens) * streams));
        group.bench_function(id, |b| b.iter(|| runtime.block_on(bed.requests(streams))));
    }
    group.finish();
    if let Some((time, bytes)) = bed.scrape() {
        eprintln!(
            "generate_path: one scrape after the run: {:.3} ms, {bytes} bytes",
            time.as_secs_f64() * 1e3
        );
    }
    let server = Arc::new(bed.server);
    server.stop(Duration::from_secs(5));
}

criterion_group!(benches, bench_generate_path);
criterion_main!(benches);
