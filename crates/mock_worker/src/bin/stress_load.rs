//! `stress_load`: an open-loop load client for the gateway's gRPC request
//! path on a CPU-only bed (the release gateway in front of `mock-worker
//! --engine realistic` gRPC engines; `scripts/bench/grpc_stress/`).
//!
//! It sends requests at a fixed rate whatever the answer time, so what it
//! measures is the gateway's cost under a given offered load: the time to the
//! response head (TTFB), to the first content delta (TTFT), to the end of the
//! body (e2e), the SSE frames per stream and the gaps between them, and the
//! status codes and error kinds. Requests started during `--warmup` are driven
//! but not sampled. One JSON summary on stdout at the end (`--out` writes it
//! too); `STRESS_INTERIM {...}` counters on stderr every `--interim` seconds.
//!
//! Modes (`--mode`): `chat` (chat completions; `--stream` for SSE), `tools`
//! (a tool-heavy conversation: `--turns` tool-call rounds over `--tools`
//! function schemas), `responses` (the Responses API with `store: true`,
//! `--chain` requests per session linked by `previous_response_id`) and
//! `image` (`--image <file>` inline as a data URL, `--images-per-request`
//! copies, before the text). `--tenants N` spreads the requests over N
//! tenant ids sent in `--tenant-header`.
//!
//! Prompts are deterministic from `--seed`: `--distinct` distinct prompts of
//! about `--prompt-tokens` words (one token each for a large-vocabulary BPE),
//! each a shared head from a pool of `--prefix-pool` fixed texts
//! (`--prefix-ratio` of the prompt) followed by a unique tail, so cache-aware
//! routing sees shared prefixes and unique remainders as in production.
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, LazyLock, Mutex,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::task::JoinSet;

/// A few hundred common words: prompts are made of these so a BPE tokenizer
/// with a large vocabulary gives about one token per word.
const VOCAB_TEXT: &str = "the of and to in is that for it as with on be by this are from or an at which have not were but \
     one all their there when who will more if out so what up its about into than them can only other \
     new some could time these two may then do first any my now such like our over man me even most \
     made after also did many before must through back years where much your way well down should \
     because each just those people how too little state good very make world still own see men work \
     long get here between both life being under never day same another know while last might us \
     great old year off come since against go came right used take three states himself few house use \
     during without again place around however home small found thought went say part once general \
     high upon school every does got united left number course war until always away something fact \
     though water less public put think almost hand enough far took head yet government system better \
     set told nothing night end why called eyes find going look asked later knew point next program \
     city business give group toward young days let room president side social given present several \
     order national second possible rather per face among form important often things looked early \
     white case become large need big four within felt along children saw best church ever least \
     power development light thing seemed family interest want members mind country area others \
     turned although open service certain kind problem began different door thus help means sense \
     whole matter perhaps itself times human law line above name example action company local show \
     whether five history gave today either act feet across taken past quite anything seen having \
     death week body word field experience really class already money free information table value \
     cost market growth design test data model result process report rate level change read write \
     file code error check build run";

static VOCAB: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| VOCAB_TEXT.split_whitespace().collect());

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// `POST /v1/chat/completions` with one user message.
    Chat,
    /// Chat completions with `--tools` function schemas and `--turns`
    /// rounds of assistant tool calls and tool results before the question.
    Tools,
    /// `POST /v1/responses` with `store: true`; sessions chain `--chain`
    /// requests through `previous_response_id`.
    Responses,
    /// Chat completions with `--images-per-request` inline images
    /// (`--image`, as data URLs) before the text.
    Image,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "stress_load",
    about = "Open-loop load client for the gateway stress bed (chat, tools, responses, image)"
)]
struct Args {
    /// Gateway base URL.
    #[arg(long, default_value = "http://127.0.0.1:29600")]
    gateway: String,
    /// Served model id.
    #[arg(long, default_value = "mock-model")]
    model: String,
    #[arg(long, value_enum, default_value_t = Mode::Chat)]
    mode: Mode,
    /// Stream the answer (SSE) instead of one JSON body.
    #[arg(long, default_value_t = false)]
    stream: bool,
    /// Requests per second, open loop (fractional allowed).
    #[arg(long, default_value_t = 100.0)]
    rate: f64,
    /// Sampled seconds after the warm-up.
    #[arg(long, default_value_t = 60)]
    duration: u64,
    /// Seconds driven but not sampled, before the window.
    #[arg(long, default_value_t = 5)]
    warmup: u64,
    /// Seconds to wait for the requests in flight after the window.
    #[arg(long, default_value_t = 60)]
    drain: u64,
    /// Prompt length in words (about one token each).
    #[arg(long, default_value_t = 64)]
    prompt_tokens: usize,
    /// Number of fixed shared heads the prompts cycle through.
    #[arg(long, default_value_t = 16)]
    prefix_pool: usize,
    /// Share of the prompt that is the shared head.
    #[arg(long, default_value_t = 0.5)]
    prefix_ratio: f64,
    /// Distinct prompts built up front and cycled per request.
    #[arg(long, default_value_t = 4096)]
    distinct: usize,
    /// `max_tokens` (`max_output_tokens` for responses).
    #[arg(long, default_value_t = 8)]
    max_tokens: u32,
    /// Spread the requests over this many tenant ids (0: no tenant header).
    #[arg(long, default_value_t = 0)]
    tenants: usize,
    #[arg(long, default_value = "x-smg-tenant-id")]
    tenant_header: String,
    /// Image file(s) for `--mode image` (repeat the flag; cycled per prompt).
    #[arg(long)]
    image: Vec<PathBuf>,
    #[arg(long, default_value_t = 1)]
    images_per_request: usize,
    /// Function schemas in a `--mode tools` request.
    #[arg(long, default_value_t = 8)]
    tools: usize,
    /// Tool-call rounds (assistant call + tool result + user follow-up) in a
    /// `--mode tools` conversation.
    #[arg(long, default_value_t = 10)]
    turns: usize,
    /// Requests per session in `--mode responses` (linked by
    /// `previous_response_id`; a new session starts after the last).
    #[arg(long, default_value_t = 10)]
    chain: usize,
    /// Concurrent sessions kept in `--mode responses` (0: four per req/s,
    /// at least 64).
    #[arg(long, default_value_t = 0)]
    sessions: usize,
    /// Tokio worker threads of the client.
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// Requests in flight above which new ones are counted as `skipped`
    /// instead of sent (protects the client, not the gateway).
    #[arg(long, default_value_t = 50_000)]
    max_inflight: usize,
    /// Seconds between interim counter lines on stderr (0: none).
    #[arg(long, default_value_t = 10)]
    interim: u64,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 600)]
    timeout: u64,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Write the summary JSON here as well as to stdout.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Print the first response (status, headers, up to 2 KB of body) on
    /// stderr: a check that the bed answers what the mode expects.
    #[arg(long, default_value_t = false)]
    debug_first: bool,
}

// ---------------------------------------------------------------- prompts

/// splitmix64: a small deterministic generator, no dependency.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(mix(seed))
    }
    fn next(&mut self) -> u64 {
        self.0 = mix(self.0);
        self.0
    }
    fn word(&mut self) -> &'static str {
        VOCAB[(self.next() % VOCAB.len() as u64) as usize]
    }
}

/// `n` words from the vocabulary, deterministic in `seed`.
fn words(seed: u64, n: usize) -> String {
    let mut rng = Rng::new(seed);
    let mut text = String::with_capacity(n * 6);
    for i in 0..n {
        if i > 0 {
            text.push(' ');
        }
        text.push_str(rng.word());
    }
    text
}

/// Prompt `i`: one of `prefix_pool` fixed heads, a marker, a unique tail.
fn prompt(args: &Args, i: usize) -> String {
    let head_words =
        ((args.prompt_tokens as f64) * args.prefix_ratio.clamp(0.0, 1.0)).round() as usize;
    let tail_words = args.prompt_tokens.saturating_sub(head_words);
    let pool = args.prefix_pool.max(1);
    let head = words(args.seed ^ 0x5EED_0000 ^ (i % pool) as u64, head_words);
    let tail = words(args.seed ^ 0x7A11_0000 ^ ((i as u64) << 20), tail_words);
    format!("{head} [session {i}] {tail}")
}

fn user(text: impl Into<String>) -> Value {
    json!({"role": "user", "content": text.into()})
}

fn tool_schemas(n: usize) -> Vec<Value> {
    (0..n)
        .map(|k| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("lookup_{k}"),
                    "description": format!("Tool {k}: looks a record up by key and returns the fields asked for"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "the record key"},
                            "fields": {"type": "array", "items": {"type": "string"}, "description": "fields to return"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                            "filter": {
                                "type": "object",
                                "properties": {
                                    "since": {"type": "string", "format": "date-time"},
                                    "status": {"type": "string", "enum": ["open", "closed", "any"]}
                                }
                            }
                        },
                        "required": ["key"]
                    }
                }
            })
        })
        .collect()
}

/// A tool-heavy conversation: system, the question, `turns` rounds of an
/// assistant tool call, its result and a follow-up.
fn tools_messages(args: &Args, i: usize) -> Vec<Value> {
    let tools = args.tools.max(1);
    let mut messages = vec![
        json!({"role": "system", "content": words(args.seed ^ 0x5151_0000, 40)}),
        user(prompt(args, i)),
    ];
    for t in 0..args.turns {
        let call_id = format!("call_{i}_{t}");
        let arguments = json!({"key": format!("k{i}-{t}"), "fields": ["a", "b", "c"], "limit": 10});
        messages.push(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": format!("lookup_{}", t % tools), "arguments": arguments.to_string()}
            }]
        }));
        let result = json!({"key": format!("k{i}-{t}"), "a": words(args.seed ^ 0x7001 ^ (t as u64), 12), "b": t, "c": [1, 2, 3]});
        messages
            .push(json!({"role": "tool", "tool_call_id": call_id, "content": result.to_string()}));
        messages.push(user(words(
            args.seed ^ 0xF011 ^ ((i as u64) << 8) ^ (t as u64),
            12,
        )));
    }
    messages
}

/// Standard base64 with padding (the data URL alphabet).
fn base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn data_url(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        _ => "image/png",
    };
    Ok(format!("data:{mime};base64,{}", base64(&bytes)))
}

/// The request bodies the run cycles through (built once; `responses`
/// bodies carry a session link and are built per request instead).
fn build_bodies(args: &Args) -> Result<Vec<Bytes>> {
    let distinct = args.distinct.max(1);
    let images: Vec<String> = args
        .image
        .iter()
        .map(|p| data_url(p))
        .collect::<Result<_>>()?;
    if args.mode == Mode::Image && images.is_empty() {
        anyhow::bail!("--mode image needs at least one --image <file>");
    }
    let mut bodies = Vec::with_capacity(distinct);
    for i in 0..distinct {
        let body = match args.mode {
            Mode::Chat => json!({
                "model": args.model,
                "messages": [user(prompt(args, i))],
                "max_tokens": args.max_tokens,
                "stream": args.stream,
                "stream_options": if args.stream { json!({"include_usage": true}) } else { Value::Null },
            }),
            Mode::Tools => json!({
                "model": args.model,
                "messages": tools_messages(args, i),
                "tools": tool_schemas(args.tools),
                "tool_choice": "auto",
                "max_tokens": args.max_tokens,
                "stream": args.stream,
            }),
            Mode::Image => {
                let mut parts: Vec<Value> = (0..args.images_per_request.max(1))
                    .map(|k| json!({"type": "image_url", "image_url": {"url": images[(i + k) % images.len()]}}))
                    .collect();
                parts.push(json!({"type": "text", "text": prompt(args, i)}));
                json!({
                    "model": args.model,
                    "messages": [{"role": "user", "content": parts}],
                    "max_tokens": args.max_tokens,
                    "stream": args.stream,
                })
            }
            Mode::Responses => Value::Null,
        };
        let mut text = serde_json::to_vec(&body)?;
        if args.mode == Mode::Responses {
            text.clear();
        }
        bodies.push(Bytes::from(text));
    }
    Ok(bodies)
}

fn responses_body(args: &Args, i: usize, previous: Option<&str>) -> Bytes {
    let mut body = json!({
        "model": args.model,
        "input": prompt(args, i % args.distinct.max(1)),
        "max_output_tokens": args.max_tokens,
        "store": true,
        "stream": args.stream,
    });
    if let Some(prev) = previous {
        body["previous_response_id"] = Value::String(prev.to_string());
    }
    Bytes::from(body.to_string())
}

// ------------------------------------------------------------ measurement

/// A log-scale histogram of microseconds: 5 % buckets, lock-free recording.
struct LogHist {
    buckets: Vec<AtomicU64>,
}

const LOG_SCALE: f64 = 20.0;

impl LogHist {
    fn new() -> Self {
        LogHist {
            buckets: (0..400).map(|_| AtomicU64::new(0)).collect(),
        }
    }
    fn index(us: u64) -> usize {
        (((us + 1) as f64).ln() * LOG_SCALE) as usize
    }
    fn record(&self, us: u64) {
        let idx = Self::index(us).min(self.buckets.len() - 1);
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }
    fn count(&self) -> u64 {
        self.buckets.iter().map(|b| b.load(Ordering::Relaxed)).sum()
    }
    /// The value at quantile `q` (bucket upper edge), in microseconds.
    fn quantile(&self, q: f64) -> f64 {
        let total = self.count();
        if total == 0 {
            return 0.0;
        }
        let target = ((total as f64) * q).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (i, b) in self.buckets.iter().enumerate() {
            seen += b.load(Ordering::Relaxed);
            if seen >= target {
                return (((i + 1) as f64) / LOG_SCALE).exp() - 1.0;
            }
        }
        0.0
    }
}

#[derive(Clone, Copy, Default)]
struct Sample {
    ttfb_us: u32,
    ttft_us: u32,
    total_us: u32,
    frames: u32,
}

#[derive(Default)]
struct Stats {
    sent: AtomicU64,
    skipped: AtomicU64,
    done: AtomicU64,
    ok: AtomicU64,
    failed: AtomicU64,
    inflight: AtomicUsize,
    inflight_max: AtomicUsize,
    frames: AtomicU64,
    chunks: AtomicU64,
    bytes_rx: AtomicU64,
    session_overflow: AtomicU64,
    samples: Mutex<Vec<Sample>>,
    statuses: Mutex<BTreeMap<u16, u64>>,
    errors: Mutex<BTreeMap<String, u64>>,
    gaps: Option<LogHist>,
}

impl Stats {
    fn note_inflight(&self) {
        let now = self.inflight.fetch_add(1, Ordering::Relaxed) + 1;
        self.inflight_max.fetch_max(now, Ordering::Relaxed);
    }
    fn error(&self, kind: String) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        *self
            .errors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(kind)
            .or_insert(0) += 1;
    }
}

/// `previous_response_id` chains: a session has at most one request in
/// flight and links `chain` of them before it starts over.
struct Sessions {
    chain: usize,
    slots: Mutex<Vec<SessionSlot>>,
}

#[derive(Default, Clone)]
struct SessionSlot {
    busy: bool,
    depth: usize,
    previous: Option<String>,
}

impl Sessions {
    fn new(count: usize, chain: usize) -> Self {
        Sessions {
            chain: chain.max(1),
            slots: Mutex::new(vec![SessionSlot::default(); count.max(1)]),
        }
    }
    /// A free session (its index and the id to link to), or none.
    fn take(&self) -> Option<(usize, Option<String>)> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let idx = slots.iter().position(|s| !s.busy)?;
        slots[idx].busy = true;
        Some((idx, slots[idx].previous.clone()))
    }
    /// The answer's id continues the chain; after `chain` links, or on a
    /// failure, the session starts over.
    fn release(&self, idx: usize, response_id: Option<String>) {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let slot = &mut slots[idx];
        slot.busy = false;
        match response_id {
            Some(id) if slot.depth + 1 < self.chain => {
                slot.depth += 1;
                slot.previous = Some(id);
            }
            _ => {
                slot.depth = 0;
                slot.previous = None;
            }
        }
    }
}

struct Shared {
    args: Args,
    url: String,
    client: reqwest::Client,
    bodies: Vec<Bytes>,
    stats: Stats,
    sessions: Sessions,
    first_printed: AtomicU64,
}

/// A client builder with the process-level TLS crypto provider installed
/// first (the crate's rule for every reqwest client).
#[expect(
    clippy::disallowed_methods,
    reason = "the tool's one direct client construction; the process TLS crypto provider is installed first"
)]
fn http_client(args: &Args) -> Result<reqwest::Client> {
    mock_worker::tls::install_crypto_provider();
    Ok(reqwest::Client::builder()
        .pool_max_idle_per_host(args.max_inflight)
        .pool_idle_timeout(Duration::from_secs(300))
        .tcp_nodelay(true)
        .http1_only()
        .timeout(Duration::from_secs(args.timeout))
        .build()?)
}

fn find_frame_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// Whether an SSE frame carries a visible token: a `content` (or
/// `reasoning_content`, or a Responses `delta`) string that is not empty.
fn frame_has_token(frame: &[u8]) -> bool {
    for key in [
        &b"\"content\":\""[..],
        &b"\"reasoning_content\":\""[..],
        &b"\"delta\":\""[..],
    ] {
        if let Some(pos) = frame.windows(key.len()).position(|w| w == key) {
            if frame.get(pos + key.len()).is_some_and(|&c| c != b'"') {
                return true;
            }
        }
    }
    false
}

/// The payload of an SSE frame's `data:` line, its leading space dropped;
/// `None` for a frame without one (a comment, or a bare `event:` line).
fn frame_data(frame: &[u8]) -> Option<&[u8]> {
    frame.split(|&c| c == b'\n').find_map(|line| {
        let data = line.strip_prefix(b"data:")?;
        Some(data.strip_prefix(b" ").unwrap_or(data))
    })
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Whether an SSE frame reports a failure after the head already said 200:
/// the router's `data: {"error":{...}}` frame, or a Responses
/// `response.failed` event.
fn frame_is_error(frame: &[u8]) -> bool {
    frame_data(frame).is_some_and(|data| {
        data.starts_with(b"{\"error\"") || contains(data, b"\"type\":\"response.failed\"")
    })
}

/// Whether an SSE frame ends its stream in order: `data: [DONE]`, or a
/// Responses `response.completed` / `response.incomplete` event.
fn frame_is_last(frame: &[u8]) -> bool {
    frame_data(frame).is_some_and(|data| {
        data == b"[DONE]"
            || contains(data, b"\"type\":\"response.completed\"")
            || contains(data, b"\"type\":\"response.incomplete\"")
    })
}

/// The `id` of a Responses answer: the JSON body's, or the first streamed
/// `response.*` event's.
fn response_id(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(bytes);
    let value: Option<Value> = serde_json::from_str(&text).ok();
    if let Some(id) = value
        .as_ref()
        .and_then(|v| v.get("id"))
        .and_then(Value::as_str)
    {
        return Some(id.to_string());
    }
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data.trim()) else {
            continue;
        };
        if let Some(id) = v.pointer("/response/id").and_then(Value::as_str) {
            return Some(id.to_string());
        }
    }
    None
}

async fn one(shared: Arc<Shared>, i: u64, sampled: bool) {
    let args = &shared.args;
    let stats = &shared.stats;
    let session = if args.mode == Mode::Responses {
        match shared.sessions.take() {
            Some(s) => Some(s),
            None => {
                stats.session_overflow.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    } else {
        None
    };
    let body = if args.mode == Mode::Responses {
        responses_body(
            args,
            i as usize,
            session.as_ref().and_then(|(_, p)| p.as_deref()),
        )
    } else {
        shared.bodies[(i as usize) % shared.bodies.len()].clone()
    };
    let mut request = shared
        .client
        .post(&shared.url)
        .header("content-type", "application/json")
        .body(body);
    if args.tenants > 0 {
        request = request.header(
            &args.tenant_header,
            format!("tenant-{}", i % args.tenants as u64),
        );
    }
    let started = Instant::now();
    let keep_body = args.mode == Mode::Responses || args.debug_first;
    let mut sample = Sample::default();
    let mut body_head: Vec<u8> = Vec::new();
    let mut outcome: Result<Option<String>, String> = Ok(None);
    match request.send().await {
        Err(e) => outcome = Err(classify(&e)),
        Ok(response) => {
            sample.ttfb_us = started.elapsed().as_micros() as u32;
            let status = response.status().as_u16();
            *stats
                .statuses
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(status)
                .or_insert(0) += 1;
            let debug =
                args.debug_first && shared.first_printed.fetch_add(1, Ordering::Relaxed) == 0;
            if debug {
                eprintln!("first response: HTTP {status} {:?}", response.headers());
            }
            let mut stream = response.bytes_stream();
            let mut pending: Vec<u8> = Vec::new();
            let mut first_token: Option<Instant> = None;
            let mut last_frame: Option<Instant> = None;
            let mut frames: u32 = 0;
            let mut sse_error = false;
            let mut sse_last = false;
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        outcome = Err(format!("body: {}", classify(&e)));
                        break;
                    }
                };
                let now = Instant::now();
                stats.chunks.fetch_add(1, Ordering::Relaxed);
                stats
                    .bytes_rx
                    .fetch_add(chunk.len() as u64, Ordering::Relaxed);
                if keep_body && body_head.len() < 1 << 20 {
                    body_head.extend_from_slice(&chunk);
                }
                if !args.stream {
                    continue;
                }
                pending.extend_from_slice(&chunk);
                let mut start = 0;
                while let Some(pos) = find_frame_end(&pending[start..]) {
                    let frame = &pending[start..start + pos];
                    frames += 1;
                    sse_error |= frame_is_error(frame);
                    sse_last |= frame_is_last(frame);
                    if first_token.is_none() && frame_has_token(frame) {
                        first_token = Some(now);
                    }
                    if let (Some(prev), Some(gaps)) = (last_frame, stats.gaps.as_ref()) {
                        if sampled {
                            gaps.record((now - prev).as_micros() as u64);
                        }
                    }
                    last_frame = Some(now);
                    start += pos + 2;
                }
                pending.drain(..start);
            }
            if debug {
                eprintln!(
                    "first body ({} bytes): {}",
                    body_head.len(),
                    String::from_utf8_lossy(&body_head[..body_head.len().min(2048)])
                );
            }
            sample.frames = frames;
            sample.total_us = started.elapsed().as_micros() as u32;
            sample.ttft_us =
                first_token.map_or(sample.total_us, |t| (t - started).as_micros() as u32);
            if outcome.is_ok() {
                if !(200..300).contains(&status) {
                    outcome = Err(format!("http-{status}"));
                } else if args.stream && sse_error {
                    // A failure after the head (an engine error, a stall bound, shedding) is an
                    // error frame on a stream that already said 200: not an ok request.
                    outcome = Err("sse-error".into());
                } else if args.stream && !sse_last {
                    // The stream closed before its terminator.
                    outcome = Err("sse-truncated".into());
                } else if args.mode == Mode::Responses {
                    outcome = Ok(response_id(&body_head));
                }
            }
        }
    }
    if let Some((idx, _)) = session {
        shared
            .sessions
            .release(idx, outcome.as_ref().ok().cloned().flatten());
    }
    stats.done.fetch_add(1, Ordering::Relaxed);
    stats.inflight.fetch_sub(1, Ordering::Relaxed);
    match outcome {
        Ok(_) => {
            stats.ok.fetch_add(1, Ordering::Relaxed);
            if sampled {
                stats
                    .frames
                    .fetch_add(u64::from(sample.frames), Ordering::Relaxed);
                stats
                    .samples
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(sample);
            }
        }
        Err(kind) => stats.error(kind),
    }
}

fn classify(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "timeout".into()
    } else if e.is_connect() {
        "connect".into()
    } else if e.is_body() || e.is_decode() {
        "body".into()
    } else {
        "request".into()
    }
}

fn percentile(sorted: &[u32], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * q).round() as usize;
    f64::from(sorted[idx.min(sorted.len() - 1)])
}

#[derive(Serialize)]
struct Quantiles {
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

fn quantiles_ms(mut values: Vec<u32>) -> Quantiles {
    values.sort_unstable();
    Quantiles {
        p50: percentile(&values, 0.50) / 1000.0,
        p90: percentile(&values, 0.90) / 1000.0,
        p99: percentile(&values, 0.99) / 1000.0,
        max: percentile(&values, 1.0) / 1000.0,
    }
}

#[derive(Serialize)]
struct Summary {
    mode: String,
    stream: bool,
    rate_target: f64,
    duration_s: u64,
    warmup_s: u64,
    wall_s: f64,
    sent: u64,
    skipped: u64,
    done: u64,
    ok: u64,
    failed: u64,
    unfinished: u64,
    sampled: usize,
    achieved_rps: f64,
    inflight_max: usize,
    statuses: BTreeMap<u16, u64>,
    errors: BTreeMap<String, u64>,
    ttfb_ms: Quantiles,
    ttft_ms: Quantiles,
    e2e_ms: Quantiles,
    frames_total: u64,
    frames_per_request: f64,
    chunks_total: u64,
    bytes_rx: u64,
    gap_us: Quantiles,
    gaps_recorded: u64,
    session_overflow: u64,
}

fn summarize(shared: &Shared, wall: Duration, unfinished: u64) -> Summary {
    let stats = &shared.stats;
    let samples = stats
        .samples
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let sampled = samples.len();
    let frames_total = stats.frames.load(Ordering::Relaxed);
    let gaps = stats.gaps.as_ref();
    Summary {
        mode: format!("{:?}", shared.args.mode).to_lowercase(),
        stream: shared.args.stream,
        rate_target: shared.args.rate,
        duration_s: shared.args.duration,
        warmup_s: shared.args.warmup,
        wall_s: wall.as_secs_f64(),
        sent: stats.sent.load(Ordering::Relaxed),
        skipped: stats.skipped.load(Ordering::Relaxed),
        done: stats.done.load(Ordering::Relaxed),
        ok: stats.ok.load(Ordering::Relaxed),
        failed: stats.failed.load(Ordering::Relaxed),
        unfinished,
        sampled,
        achieved_rps: sampled as f64 / shared.args.duration.max(1) as f64,
        inflight_max: stats.inflight_max.load(Ordering::Relaxed),
        statuses: stats
            .statuses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone(),
        errors: stats
            .errors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone(),
        ttfb_ms: quantiles_ms(samples.iter().map(|s| s.ttfb_us).collect()),
        ttft_ms: quantiles_ms(samples.iter().map(|s| s.ttft_us).collect()),
        e2e_ms: quantiles_ms(samples.iter().map(|s| s.total_us).collect()),
        frames_total,
        frames_per_request: if sampled == 0 {
            0.0
        } else {
            frames_total as f64 / sampled as f64
        },
        chunks_total: stats.chunks.load(Ordering::Relaxed),
        bytes_rx: stats.bytes_rx.load(Ordering::Relaxed),
        gap_us: Quantiles {
            p50: gaps.map_or(0.0, |g| g.quantile(0.50)),
            p90: gaps.map_or(0.0, |g| g.quantile(0.90)),
            p99: gaps.map_or(0.0, |g| g.quantile(0.99)),
            max: gaps.map_or(0.0, |g| g.quantile(1.0)),
        },
        gaps_recorded: gaps.map_or(0, LogHist::count),
        session_overflow: stats.session_overflow.load(Ordering::Relaxed),
    }
}

fn interim_line(stats: &Stats, t: f64) -> String {
    json!({
        "t": (t * 10.0).round() / 10.0,
        "sent": stats.sent.load(Ordering::Relaxed),
        "done": stats.done.load(Ordering::Relaxed),
        "ok": stats.ok.load(Ordering::Relaxed),
        "failed": stats.failed.load(Ordering::Relaxed),
        "skipped": stats.skipped.load(Ordering::Relaxed),
        "inflight": stats.inflight.load(Ordering::Relaxed),
    })
    .to_string()
}

async fn run(args: Args) -> Result<Summary> {
    let path = match args.mode {
        Mode::Responses => "/v1/responses",
        _ => "/v1/chat/completions",
    };
    let url = format!("{}{}", args.gateway.trim_end_matches('/'), path);
    let bodies = build_bodies(&args)?;
    let sessions = if args.sessions > 0 {
        args.sessions
    } else {
        ((args.rate * 4.0).ceil() as usize).max(64)
    };
    let shared = Arc::new(Shared {
        client: http_client(&args)?,
        url,
        bodies,
        stats: Stats {
            gaps: args.stream.then(LogHist::new),
            ..Stats::default()
        },
        sessions: Sessions::new(sessions, args.chain),
        first_printed: AtomicU64::new(0),
        args: args.clone(),
    });

    let start = Instant::now();
    let total = Duration::from_secs(args.warmup + args.duration);
    let warm = Duration::from_secs(args.warmup);
    let mut tasks: JoinSet<()> = JoinSet::new();
    let mut ticker = tokio::time::interval(Duration::from_millis(1));
    let mut launched: u64 = 0;
    // The interim lines are written by a thread of their own: a write to a
    // slow file must not hold the pacing loop (a stalled loop releases its
    // backlog as one burst, which is not the load asked for).
    let interim = (args.interim > 0).then(|| {
        let shared = Arc::clone(&shared);
        let every = Duration::from_secs(args.interim);
        std::thread::spawn(move || {
            while start.elapsed() < total {
                std::thread::sleep(every);
                eprintln!(
                    "STRESS_INTERIM {}",
                    interim_line(&shared.stats, start.elapsed().as_secs_f64())
                );
            }
        })
    });
    loop {
        ticker.tick().await;
        let elapsed = start.elapsed();
        if elapsed >= total {
            break;
        }
        while tasks.try_join_next().is_some() {}
        let due = (elapsed.as_secs_f64() * args.rate).floor() as u64;
        while launched < due {
            let sampled = elapsed >= warm;
            launched += 1;
            if shared.stats.inflight.load(Ordering::Relaxed) >= args.max_inflight {
                shared.stats.skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            shared.stats.sent.fetch_add(1, Ordering::Relaxed);
            shared.stats.note_inflight();
            tasks.spawn(one(Arc::clone(&shared), launched, sampled));
        }
    }
    let drain_until = Instant::now() + Duration::from_secs(args.drain);
    while let Ok(Some(_)) = tokio::time::timeout_at(drain_until.into(), tasks.join_next()).await {}
    let unfinished = tasks.len() as u64;
    tasks.abort_all();
    if let Some(thread) = interim {
        let _ = thread.join();
    }
    Ok(summarize(&shared, start.elapsed(), unfinished))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads.max(1))
        .enable_all()
        .build()?;
    let out = args.out.clone();
    let summary = runtime.block_on(run(args))?;
    let text = serde_json::to_string_pretty(&summary)?;
    if let Some(path) = out {
        std::fs::write(&path, &text).with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{text}");
    // A run in which no request succeeded is a failed cell, not a measurement: the exit status
    // carries that to cell.sh and run_all.sh.
    if summary.sent > 0 && summary.ok == 0 {
        anyhow::bail!(
            "no request succeeded ({} failed): see the errors in the summary",
            summary.failed
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(mode: Mode) -> Args {
        Args::parse_from([
            "stress_load",
            "--mode",
            &format!("{mode:?}").to_lowercase(),
            "--distinct",
            "4",
        ])
    }

    #[test]
    fn prompts_are_deterministic_and_share_heads_within_the_pool() {
        let mut a = args(Mode::Chat);
        a.prefix_pool = 2;
        a.prompt_tokens = 40;
        let p0 = prompt(&a, 0);
        let p2 = prompt(&a, 2);
        assert_eq!(p0, prompt(&a, 0), "the same index gives the same prompt");
        assert_ne!(p0, p2, "a different index is a different prompt");
        let head = |p: &str| p.split(" [session").next().unwrap().to_string();
        assert_eq!(head(&p0), head(&p2), "indexes 0 and 2 share the pool head");
        assert_ne!(head(&p0), head(&prompt(&a, 1)));
        assert_eq!(
            p0.split_whitespace().count(),
            42,
            "40 words plus the marker"
        );
    }

    #[test]
    fn bodies_have_the_shape_of_their_mode() {
        let chat = build_bodies(&args(Mode::Chat)).unwrap();
        assert_eq!(chat.len(), 4);
        let v: Value = serde_json::from_slice(&chat[0]).unwrap();
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "user");
        assert!(v["stream_options"].is_null());

        let mut t = args(Mode::Tools);
        t.stream = true;
        t.turns = 3;
        t.tools = 2;
        let tools = build_bodies(&t).unwrap();
        let v: Value = serde_json::from_slice(&tools[1]).unwrap();
        assert_eq!(v["tools"].as_array().unwrap().len(), 2);
        // system, question, then 3 x (assistant call, tool result, follow-up)
        assert_eq!(v["messages"].as_array().unwrap().len(), 2 + 3 * 3);
        assert_eq!(
            v["messages"][2]["tool_calls"][0]["function"]["name"],
            "lookup_0"
        );
        assert_eq!(v["messages"][3]["role"], "tool");
        assert_eq!(v["stream"], true);

        let r = args(Mode::Responses);
        let body: Value = serde_json::from_slice(&responses_body(&r, 3, Some("resp_1"))).unwrap();
        assert_eq!(body["previous_response_id"], "resp_1");
        assert_eq!(body["store"], true);
        assert!(responses_body(&r, 3, None)
            .windows(20)
            .all(|w| w != b"previous_response_id"));
    }

    #[test]
    fn image_bodies_inline_the_file_as_a_data_url() {
        let dir = std::env::temp_dir().join(format!("stress-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("pixel.png");
        std::fs::write(&file, b"\x89PNG\r\n\x1a\n").unwrap();
        let mut a = args(Mode::Image);
        a.image = vec![file.clone()];
        a.images_per_request = 2;
        let v: Value = serde_json::from_slice(&build_bodies(&a).unwrap()[0]).unwrap();
        let parts = v["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(
            parts[0]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
        assert_eq!(parts[2]["type"], "text");
        assert!(
            build_bodies(&args(Mode::Image)).is_err(),
            "no image file is refused"
        );
        let _ = std::fs::remove_file(file);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn base64_matches_the_standard_alphabet_and_padding() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8="); // codespell:ignore fo
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn sse_frames_and_first_token_detection() {
        let buf = b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";
        let mut frames = Vec::new();
        let mut start = 0;
        while let Some(pos) = find_frame_end(&buf[start..]) {
            frames.push(&buf[start..start + pos]);
            start += pos + 2;
        }
        assert_eq!(frames.len(), 3);
        assert!(
            !frame_has_token(frames[0]),
            "an empty content string is no token"
        );
        assert!(frame_has_token(frames[1]));
        assert!(!frame_has_token(frames[2]));
        assert!(frame_has_token(
            b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"x\"}"
        ));
    }

    #[test]
    fn error_frames_and_last_frames_are_told_apart() {
        assert!(frame_is_error(
            b"data: {\"error\":{\"message\":\"boom\",\"type\":\"internal_error\"}}"
        ));
        assert!(frame_is_error(
            b"event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{}}"
        ));
        assert!(!frame_is_error(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"error\"}}]}"
        ));
        assert!(frame_is_last(b"data: [DONE]"));
        assert!(frame_is_last(
            b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_c\"}}"
        ));
        assert!(!frame_is_last(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"[DONE]\"}}]}"
        ));
        assert!(!frame_is_last(b": keep-alive"));
        assert!(!frame_is_error(b": keep-alive"));
    }

    #[test]
    fn response_ids_come_from_json_bodies_and_streams() {
        assert_eq!(
            response_id(br#"{"id":"resp_a","object":"response"}"#).as_deref(),
            Some("resp_a")
        );
        let sse = b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_b\"}}\n\n";
        assert_eq!(response_id(sse).as_deref(), Some("resp_b"));
        assert_eq!(response_id(b"nonsense"), None);
    }

    #[test]
    fn sessions_chain_and_start_over() {
        let s = Sessions::new(1, 3);
        let (idx, prev) = s.take().unwrap();
        assert_eq!(prev, None);
        assert!(s.take().is_none(), "a busy session is not handed out twice");
        s.release(idx, Some("r1".into()));
        assert_eq!(s.take().unwrap().1.as_deref(), Some("r1"));
        s.release(idx, Some("r2".into()));
        assert_eq!(s.take().unwrap().1.as_deref(), Some("r2"));
        s.release(idx, Some("r3".into()));
        assert_eq!(
            s.take().unwrap().1,
            None,
            "after the chain length the session starts over"
        );
        s.release(idx, None);
        assert_eq!(
            s.take().unwrap().1,
            None,
            "a failure starts the session over"
        );
    }

    #[test]
    fn histogram_quantiles_and_percentiles() {
        let h = LogHist::new();
        for us in [100u64, 200, 300, 400, 1000] {
            h.record(us);
        }
        assert_eq!(h.count(), 5);
        let p50 = h.quantile(0.5);
        assert!((280.0..=330.0).contains(&p50), "p50 bucket near 300: {p50}");
        assert!(h.quantile(1.0) >= 1000.0);
        let sorted = [1u32, 2, 3, 4, 100];
        assert_eq!(percentile(&sorted, 0.5), 3.0);
        assert_eq!(percentile(&sorted, 1.0), 100.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
}
