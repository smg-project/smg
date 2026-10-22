//! The cost of the generate path, in process, over the crate's mock engine:
//! a servicer answers streamed requests whose engine side is a mock that
//! replies to every add with one chunk per token and a `length` finish. What
//! criterion times is one closed-loop request end to end (the gRPC call, the
//! engine add, the chunks back through the servicer, the terminal), with the
//! metrics endpoint off or on (`GENERATE_PATH_METRICS=on`), so the two can be
//! compared and the per-request and per-token cost of the observations read.
//!
//! For a counter-based reading (`perf stat` on the binary),
//! `GENERATE_PATH_REQUESTS=<n>` runs `n` requests and exits without
//! criterion; `GENERATE_PATH_TOKENS` sets the tokens per request (64 by
//! default), so two runs at different token counts separate the per-token
//! cost from the per-request one. With the endpoint on, a scrape every
//! `GENERATE_PATH_SCRAPE_EVERY` requests reports its render time.
//!
//! Run: `cargo bench -p engine-servicer --bench generate_path`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stderr,
    clippy::cast_precision_loss
)]

use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use engine_servicer::{ServicerConfig, VllmModelInfo, VllmServicerConfig, VllmServicerServer};
use engine_zmq_client::{
    mock_engine::{connect_to_frontend, default_ready_response, EngineInbound, MockEngine},
    protocol::vllm::output::{
        EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    },
    EngineId,
};
use futures::StreamExt;
use smg_grpc_client::{vllm_proto as vllm, vllm_proto::vllm_engine_client::VllmEngineClient};
use tonic::transport::Channel;

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

/// The mock engine's side of the wire: every add is answered with
/// `tokens` single-token chunks and a `length` finish.
async fn answer_adds(engine: MockEngine, tokens: u32) {
    let (mut input, mut output) = engine.split();
    while let Ok(inbound) = input.recv().await {
        let EngineInbound::Add(request) = inbound else {
            continue;
        };
        let id = request.request_id.clone();
        for token in 0..tokens {
            let finish = (token + 1 == tokens).then_some(EngineCoreFinishReason::Length);
            let batch = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
                engine_index: 0,
                outputs: vec![EngineCoreOutput {
                    request_id: id.clone(),
                    new_token_ids: vec![token + 10],
                    finish_reason: finish,
                    ..Default::default()
                }],
                finished_requests: finish.map(|_| BTreeSet::from([id.clone()])),
                ..Default::default()
            });
            output
                .send_outputs(&batch)
                .await
                .expect("the mock engine's output is sent");
        }
    }
}

struct Bed {
    server: VllmServicerServer,
    client: VllmEngineClient<Channel>,
    tokens: u32,
    sent: u64,
    _dir: tempfile::TempDir,
}

impl Bed {
    async fn start(tokens: u32, metrics: bool) -> Self {
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
        let model = VllmModelInfo {
            model_path: "org/model".to_string(),
            served_model_name: "served".to_string(),
            tokenizer_path: "org/model".to_string(),
            is_generation: true,
            max_context_length: 4096,
            vocab_size: 1024,
            data_parallel_size: 1,
            ..Default::default()
        };
        let server = VllmServicerServer::start(VllmServicerConfig {
            servicer,
            model,
            media_processor: None,
        })
        .expect("the servicer starts");
        let engine = connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response(),
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
        let _mock = tokio::spawn(answer_adds(engine, tokens));
        let channel = Channel::from_shared(format!("http://{}", server.address()))
            .unwrap()
            .connect()
            .await
            .expect("the gRPC channel connects");
        Self {
            server,
            client: VllmEngineClient::new(channel),
            tokens,
            sent: 0,
            _dir: dir,
        }
    }

    /// One streamed request, read to its end; returns the tokens received.
    async fn request(&mut self) -> u32 {
        self.sent += 1;
        let request = vllm::GenerateRequest {
            request_id: format!("r{}", self.sent),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(self.tokens),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = self
            .client
            .generate(request)
            .await
            .expect("the call is accepted")
            .into_inner();
        let mut tokens = 0;
        while let Some(item) = stream.next().await {
            if let Some(vllm::generate_response::Response::Chunk(chunk)) =
                item.expect("a response").response
            {
                tokens += chunk.token_ids.len() as u32;
            }
        }
        tokens
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
/// loop, the wall time, the scrape times and the resident set after the
/// first thousand requests and at the end printed.
fn counted_run(runtime: &tokio::runtime::Runtime, requests: u64) {
    let tokens = tokens_per_request();
    let metrics = metrics_on();
    let scrape_every = env_u64("GENERATE_PATH_SCRAPE_EVERY").unwrap_or(0);
    let mut bed = runtime.block_on(Bed::start(tokens, metrics));
    let started = Instant::now();
    let mut received = 0u64;
    let mut scrapes = Vec::new();
    let mut rss_after_1k = None;
    for sent in 1..=requests {
        received += u64::from(runtime.block_on(bed.request()));
        if scrape_every > 0 && sent % scrape_every == 0 {
            if let Some(scrape) = bed.scrape() {
                scrapes.push(scrape);
            }
        }
        if sent == 1000 {
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
        "generate_path: {requests} requests x {tokens} tokens, metrics {}: {received} tokens in \
         {:.3} s ({:.1} us per request)",
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
    bed.server.stop(Duration::from_secs(5)).unwrap();
}

fn bench_generate_path(c: &mut Criterion) {
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
    let mut bed = runtime.block_on(Bed::start(tokens, metrics));
    let mut group = c.benchmark_group("generate_path");
    group.throughput(Throughput::Elements(u64::from(tokens)));
    group.bench_function(
        format!(
            "request_{tokens}_tokens_metrics_{}",
            if metrics { "on" } else { "off" }
        ),
        |b| b.iter(|| runtime.block_on(bed.request())),
    );
    group.finish();
    if let Some((time, bytes)) = bed.scrape() {
        eprintln!(
            "generate_path: one scrape after the run: {:.3} ms, {bytes} bytes",
            time.as_secs_f64() * 1e3
        );
    }
    let server = Arc::new(bed.server);
    server.stop(Duration::from_secs(5)).unwrap();
}

criterion_group!(benches, bench_generate_path);
criterion_main!(benches);
