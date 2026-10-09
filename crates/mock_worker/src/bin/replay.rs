//! Replay a Mooncake-format trace through the gateway and score the routing.
//!
//! Each trace row is `{timestamp (ms), input_length, output_length, hash_ids}`,
//! where every `hash_id` stands for one 512-token block of prompt. The replayer
//! synthesizes a deterministic text block per `hash_id` (so rows sharing ids
//! share prompt prefixes exactly as the trace intends), sends each request as a
//! streaming chat completion at `timestamp / speedup` (open loop), and records
//! time to first token, inter-token latencies, the serving worker (the
//! gateway's `system_fingerprint`, set from the worker's `weight_version`
//! label) and the engine-reported `cached_tokens`. When the mock fleet's admin
//! API is given, every request is joined with the fleet's record of it, which
//! carries the arrival-time oracle: the most cached tokens any worker held.
//! A thinking model's streamed `reasoning_content` counts as output for TTFT
//! and ITL (reported separately as `reasoning_tokens`), and
//! `--chat-template-kwargs` reaches the chat template, e.g. to turn thinking off.
//! Requests share HTTP/1.1 keep-alive connections by default (`--connections
//! pooled`); `--connections fresh` gives every request a connection of its
//! own, for tail-sensitive runs, so a request is never queued behind a stream
//! in flight on a reused connection, which would show as one stream's worth
//! of TTFT on an otherwise idle gateway.

// A command-line tool: the summary goes to stdout, progress to stderr.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::Write,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    sync::{watch, Semaphore},
    task::JoinSet,
};

#[derive(Parser, Debug, Clone)]
#[command(about = "Replay a Mooncake trace through the gateway and score routing quality")]
struct Args {
    /// Mooncake JSONL trace.
    #[arg(long)]
    trace: PathBuf,
    /// Gateway base URL.
    #[arg(long, default_value = "http://127.0.0.1:31000")]
    gateway: String,
    /// Model id to request.
    #[arg(long, default_value = "mock-model")]
    model: String,
    /// Arrival speedup: trace time is divided by this (finite, greater than 0).
    #[arg(long, default_value_t = 2.0, value_parser = parse_speedup)]
    speedup: f64,
    /// Rows to skip from the start of the trace.
    #[arg(long, default_value_t = 0)]
    skip: usize,
    /// Rows to replay after `skip` (0 = all).
    #[arg(long, default_value_t = 4000)]
    limit: usize,
    /// Words synthesized per 512-token trace block (about one token each).
    #[arg(long, default_value_t = 480)]
    words_per_block: usize,
    /// Cap on `max_tokens` per request (the trace's output_length otherwise).
    #[arg(long, default_value_t = 512)]
    max_output: u32,
    /// Safety cap on concurrently open requests.
    #[arg(long, default_value_t = 4096)]
    max_inflight: usize,
    /// Mock fleet admin API base URL (enables the oracle join).
    #[arg(long)]
    admin: Option<String>,
    /// First rows whose results are excluded from the statistics (warm-up).
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    /// TTFT SLO in ms for goodput.
    #[arg(long, default_value_t = 500.0)]
    slo_ttft_ms: f64,
    /// Per-request mean inter-token latency (TPOT) SLO in ms for goodput.
    #[arg(long, default_value_t = 50.0)]
    slo_itl_ms: f64,
    /// Output directory for `summary.json` and `requests.csv`.
    #[arg(long, default_value = "replay-out")]
    out: PathBuf,
    /// Label stored in the summary.
    #[arg(long, default_value = "run")]
    label: String,
    /// Seed mixed into the synthesized text.
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// The gateway's log at debug level: its routing decisions are joined to
    /// the requests by request id, so the gateway's cache credit can be
    /// compared with what the engine served.
    #[arg(long)]
    gateway_log: Option<PathBuf>,
    /// The fleet's block size, to turn a credit in blocks into tokens.
    #[arg(long, default_value_t = 16)]
    block_size: u32,
    /// Rows of the per-request decision table written to `t4.md`.
    #[arg(long, default_value_t = 40)]
    t4_rows: usize,
    /// JSON object sent as `chat_template_kwargs` in every request body, for
    /// example `{"enable_thinking":false}` on a model that reasons by default;
    /// nothing is sent when absent.
    #[arg(long, value_parser = parse_json_object)]
    chat_template_kwargs: Option<Value>,
    /// How requests use connections: `pooled` (the default) reuses HTTP/1.1
    /// keep-alive connections, fewer handshakes, but a request can be handed
    /// a connection whose previous response is still streaming and then
    /// waits for that stream to end before the gateway reads it (rare: a
    /// few per hundred thousand streamed requests); `fresh` opens a
    /// connection per request and has it closed after the response, so a
    /// request never shares a connection with a stream in flight, the
    /// choice for tail-sensitive runs (max TTFT, stall hunting).
    #[arg(long, value_enum, default_value_t = Connections::Pooled)]
    connections: Connections,
}

/// The addresses a URL's host name resolves to now, for the client to pin:
/// a connection then never waits on a name lookup, which is one per
/// connection otherwise, one per request with `Connections::Fresh`, and a
/// lost lookup costs a resolver timeout of seconds. A host that is already
/// an address, or one that does not resolve here, is left to the client.
async fn pinned_addresses(url: &str) -> Option<(String, Vec<std::net::SocketAddr>)> {
    let url = reqwest::Url::parse(url).ok()?;
    let host = url.host_str()?.trim_matches(['[', ']']).to_string();
    if host.parse::<std::net::IpAddr>().is_ok() {
        return None;
    }
    let port = url.port_or_known_default()?;
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
        .await
        .ok()?
        .collect();
    (!addrs.is_empty()).then_some((host, addrs))
}

/// How the replayer's requests use connections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Connections {
    /// A connection per request, closed after the response.
    Fresh,
    /// HTTP/1.1 keep-alive connections reused from a pool.
    Pooled,
}

/// The HTTP client for `connections`. With `Fresh` no idle connection is
/// kept and every request asks the server to close after the response
/// (`Connection: close`), so the server's side carries the TIME_WAIT and the
/// client's ephemeral ports stay free at rate; with `Pooled` up to
/// `max_inflight` keep-alive connections are kept and reused. The `pins`
/// are host names resolved once, so that no connection waits on a lookup.
fn http_client(
    connections: Connections,
    max_inflight: usize,
    pins: &[(String, Vec<std::net::SocketAddr>)],
) -> reqwest::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(600));
    for (host, addrs) in pins {
        builder = builder.resolve_to_addrs(host, addrs);
    }
    match connections {
        Connections::Fresh => {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::CONNECTION,
                reqwest::header::HeaderValue::from_static("close"),
            );
            builder
                .pool_max_idle_per_host(0)
                .default_headers(headers)
                .build()
        }
        Connections::Pooled => builder.pool_max_idle_per_host(max_inflight).build(),
    }
}

#[derive(Deserialize, Debug, Clone)]
struct TraceRow {
    timestamp: u64,
    input_length: u32,
    output_length: u32,
    hash_ids: Vec<u64>,
}

#[derive(Serialize, Debug, Clone, Default)]
struct ReqResult {
    row: usize,
    trace_ts_ms: u64,
    trace_input_length: u32,
    sent_at_ms: f64,
    status: String,
    request_id: String,
    worker: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    oracle_tokens: Option<u32>,
    queued_ms: Option<f64>,
    ttft_ms: Option<f64>,
    latency_ms: f64,
    itl_mean_ms: Option<f64>,
    itl_p99_ms: Option<f64>,
    tokens_seen: u32,
    /// Reasoning deltas (`reasoning_content`) the stream showed: they count
    /// for TTFT and ITL like text, separately from `tokens_seen`.
    reasoning_tokens: u32,
    /// Output tokens the engine counted but the stream never showed as text
    /// or reasoning (an incomplete UTF-8 piece the detokenizer holds back).
    invisible_tokens: u32,
    /// The gateway's `x-request-id` (the response id without its uuid tail).
    gateway_request_id: String,
    /// The gateway's routing branch for this request (from its debug log).
    branch: String,
    /// The gateway's cache credit in tokens, when its log states one.
    credit_tokens: Option<u32>,
    /// Whether the gateway's credit agrees with what the engine served.
    agree: Option<bool>,
}

const WORDS: &[&str] = &[
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "thing",
    "woman",
    "life",
    "child",
    "world",
    "school",
    "state",
    "family",
    "student",
    "group",
    "country",
    "problem",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "system",
    "program",
    "question",
    "work",
    "government",
    "number",
    "night",
    "point",
    "home",
    "water",
    "room",
    "mother",
    "area",
    "money",
    "story",
    "fact",
    "month",
    "lot",
    "right",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "friend",
    "father",
    "power",
    "hour",
    "game",
    "line",
    "end",
    "member",
    "law",
    "car",
    "city",
    "community",
    "name",
    "president",
    "team",
    "minute",
    "idea",
    "kid",
    "body",
    "information",
    "back",
    "parent",
    "face",
    "others",
    "level",
    "office",
    "door",
    "health",
    "person",
    "art",
    "war",
    "history",
    "party",
    "result",
    "change",
    "morning",
    "reason",
    "research",
    "girl",
    "guy",
    "moment",
    "air",
    "teacher",
    "force",
    "education",
    "foot",
    "boy",
    "age",
    "policy",
    "process",
    "music",
    "market",
    "sense",
    "nation",
    "plan",
    "college",
    "interest",
    "death",
    "experience",
    "effect",
    "use",
    "class",
    "control",
    "care",
    "field",
    "development",
    "role",
    "effort",
    "rate",
    "heart",
    "drug",
    "show",
    "leader",
    "light",
    "voice",
    "wife",
    "police",
    "mind",
    "price",
    "report",
    "decision",
    "son",
    "view",
    "relationship",
    "town",
    "road",
    "arm",
    "difference",
    "value",
    "building",
    "action",
    "model",
    "season",
    "society",
    "tax",
    "director",
    "position",
    "player",
    "record",
    "paper",
    "space",
    "ground",
    "form",
    "event",
    "official",
    "matter",
    "center",
    "couple",
    "site",
    "project",
    "activity",
    "star",
    "table",
    "need",
    "court",
    "oil",
    "situation",
    "cost",
    "industry",
    "figure",
    "street",
    "image",
    "phone",
    "data",
    "picture",
    "practice",
    "piece",
    "land",
    "product",
    "doctor",
    "wall",
    "patient",
    "worker",
    "news",
    "test",
    "movie",
    "north",
    "love",
    "support",
    "technology",
    "step",
    "baby",
    "computer",
    "type",
    "attention",
    "film",
    "tree",
    "source",
    "nothing",
    "network",
    "trade",
    "economy",
    "author",
    "window",
    "energy",
    "letter",
    "church",
    "cell",
    "ship",
    "island",
    "plant",
    "garden",
    "river",
    "bridge",
    "engine",
    "metal",
    "glass",
    "stone",
    "storm",
    "forest",
    "valley",
    "ocean",
    "desert",
    "signal",
    "memory",
    "logic",
];

/// Deterministic text for one trace block: `words_per_block` words drawn from
/// the vocabulary by a splitmix64 stream seeded with the block id.
fn block_text(hash_id: u64, seed: u64, words_per_block: usize) -> String {
    let mut x = hash_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed ^ 0xD1B5_4A32_D192_ED03);
    let mut out = String::with_capacity(words_per_block * 7);
    for i in 0..words_per_block {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if i > 0 {
            out.push(' ');
        }
        out.push_str(WORDS[(z % WORDS.len() as u64) as usize]);
    }
    out
}

fn prompt_for(row: &TraceRow, seed: u64, words_per_block: usize) -> String {
    let mut prompt = String::new();
    for (i, id) in row.hash_ids.iter().enumerate() {
        if i > 0 {
            prompt.push('\n');
        }
        prompt.push_str(&block_text(*id, seed, words_per_block));
    }
    prompt
}

/// `--speedup`: a divisor of trace time, so it must be a finite number
/// greater than zero; anything else would make a due time infinite, negative
/// or NaN.
fn parse_speedup(raw: &str) -> Result<f64, String> {
    match raw.trim().parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
        _ => Err(format!(
            "speedup must be a finite number greater than 0, got {raw}"
        )),
    }
}

/// `--chat-template-kwargs` must be a JSON object.
fn parse_json_object(raw: &str) -> Result<Value, String> {
    match serde_json::from_str::<Value>(raw) {
        Ok(v) if v.is_object() => Ok(v),
        Ok(_) => Err(format!("expected a JSON object, got {raw}")),
        Err(e) => Err(format!("not JSON: {e}")),
    }
}

/// The streaming chat completion for one trace row.
fn request_body(
    model: &str,
    prompt: &str,
    max_tokens: u32,
    chat_template_kwargs: Option<&Value>,
) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(kwargs) = chat_template_kwargs {
        body["chat_template_kwargs"] = kwargs.clone();
    }
    body
}

/// When a row stamped `trace_ts_ms` is due, relative to the window's first
/// row at `t0_ms`, at `speedup`. A row stamped before the first one (an
/// unsorted or hand-edited trace, or a window that starts on a reordered
/// row) is due at once rather than never: the difference saturates at zero
/// instead of wrapping. A due time too far ahead to represent is clamped.
fn due_after(trace_ts_ms: u64, t0_ms: u64, speedup: f64) -> Duration {
    let secs = trace_ts_ms.saturating_sub(t0_ms) as f64 / 1000.0 / speedup;
    Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

struct Job {
    client: reqwest::Client,
    url: String,
    model: String,
    row_index: usize,
    row: TraceRow,
    body_prompt: String,
    max_output: u32,
    sent_at_ms: f64,
    chat_template_kwargs: Option<Value>,
}

async fn run_one(job: Job) -> ReqResult {
    let Job {
        client,
        url,
        model,
        row_index,
        row,
        body_prompt,
        max_output,
        sent_at_ms,
        chat_template_kwargs,
    } = job;
    let max_tokens = row.output_length.clamp(1, max_output);
    let body = request_body(
        &model,
        &body_prompt,
        max_tokens,
        chat_template_kwargs.as_ref(),
    );
    let mut result = ReqResult {
        row: row_index,
        trace_ts_ms: row.timestamp,
        trace_input_length: row.input_length,
        sent_at_ms,
        status: "ok".to_string(),
        ..Default::default()
    };
    let started = Instant::now();
    let response = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            result.status = format!("send-error: {e}");
            result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
            return result;
        }
    };
    if !response.status().is_success() {
        result.status = format!("http-{}", response.status().as_u16());
        result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        return result;
    }
    result.gateway_request_id = response
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut observer = StreamObserver::default();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                result.status = format!("stream-error: {e}");
                break;
            }
        };
        buf.extend_from_slice(&chunk);
        // SSE events end with a blank line.
        while let Some(pos) = find_double_newline(&buf) {
            let event: Vec<u8> = buf.drain(..pos + 2).collect();
            let text = String::from_utf8_lossy(&event);
            for line in text.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                observer.observe(&v, Instant::now());
            }
        }
    }
    result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    observer.finish(started, &mut result);
    result
}

/// What one streamed response reveals, chunk by chunk.
#[derive(Default)]
struct StreamObserver {
    request_id: String,
    worker: String,
    /// The first chunk that carried a token (text or reasoning) or the
    /// finish: a one-token answer whose token has no visible text still
    /// arrives here.
    first_signal: Option<Instant>,
    last_token: Option<Instant>,
    itls: Vec<f64>,
    tokens_seen: u32,
    reasoning_tokens: u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    saw_usage: bool,
}

impl StreamObserver {
    fn observe(&mut self, v: &Value, now: Instant) {
        if self.request_id.is_empty() {
            if let Some(id) = v.get("id").and_then(Value::as_str) {
                self.request_id = id.to_string();
            }
        }
        if self.worker.is_empty() {
            if let Some(fp) = v.get("system_fingerprint").and_then(Value::as_str) {
                self.worker = fp.to_string();
            }
        }
        let choice = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first());
        let delta = choice.and_then(|c| c.get("delta"));
        let non_empty = |key: &str| {
            delta
                .and_then(|d| d.get(key))
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        };
        let has_content = non_empty("content");
        // A thinking model streams its reasoning first: `reasoning_content`
        // on chat deltas (`reasoning` on some engines). It is output too.
        let has_reasoning = non_empty("reasoning_content") || non_empty("reasoning");
        let finished = choice
            .and_then(|c| c.get("finish_reason"))
            .is_some_and(|f| !f.is_null());
        if has_content || has_reasoning {
            if has_content {
                self.tokens_seen += 1;
            } else {
                self.reasoning_tokens += 1;
            }
            if let Some(prev) = self.last_token {
                self.itls.push((now - prev).as_secs_f64() * 1000.0);
            }
            self.first_signal.get_or_insert(now);
            self.last_token = Some(now);
        } else if finished {
            self.first_signal.get_or_insert(now);
        }
        if let Some(usage) = v.get("usage").filter(|u| !u.is_null()) {
            self.saw_usage = true;
            self.prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            self.completion_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            self.cached_tokens = usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .or_else(|| usage.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
        }
    }

    /// Fold the observations into the result. A stream that showed neither
    /// text nor reasoning but finished with a counted output token is a
    /// served request whose token was invisible; one with no token at all is
    /// `no-tokens`.
    fn finish(self, started: Instant, result: &mut ReqResult) {
        result.request_id = self.request_id;
        result.worker = self.worker;
        result.prompt_tokens = self.prompt_tokens;
        result.completion_tokens = self.completion_tokens;
        result.cached_tokens = self.cached_tokens;
        result.tokens_seen = self.tokens_seen;
        result.reasoning_tokens = self.reasoning_tokens;
        let shown = self.tokens_seen + self.reasoning_tokens;
        result.ttft_ms = self
            .first_signal
            .map(|t| (t - started).as_secs_f64() * 1000.0);
        if !self.itls.is_empty() {
            let mut sorted = self.itls.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            result.itl_mean_ms = Some(mean(&self.itls));
            result.itl_p99_ms = Some(percentile(&sorted, 0.99));
        }
        if result.completion_tokens < shown {
            // No usage arrived: count what was seen.
            result.completion_tokens = shown;
        }
        if shown == 0 && self.completion_tokens > 0 && result.ttft_ms.is_some() {
            result.invisible_tokens = self.completion_tokens;
        }
        if result.status == "ok" && (result.ttft_ms.is_none() || result.completion_tokens == 0) {
            result.status = "no-tokens".to_string();
        }
    }
}

/// One routing decision from the gateway's debug log.
#[derive(Clone, Debug, Default, PartialEq)]
struct Decision {
    worker: String,
    branch: String,
    /// `matched_ratio=` on the tree path.
    matched_ratio: Option<f64>,
    /// `overlap_tokens=` / `overlap_blocks=`, when the gateway logs them.
    overlap_tokens: Option<u32>,
    overlap_blocks: Option<u32>,
}

impl Decision {
    /// The credit in tokens, floored to the block, when the log states one.
    fn credit_tokens(&self, prompt_tokens: u32, block_size: u32) -> Option<u32> {
        let bs = block_size.max(1);
        if let Some(t) = self.overlap_tokens {
            return Some(t / bs * bs);
        }
        if let Some(b) = self.overlap_blocks {
            return Some(b * bs);
        }
        self.matched_ratio
            .map(|r| ((r * f64::from(prompt_tokens)) as u32) / bs * bs)
    }

    fn claims_overlap(&self) -> bool {
        matches!(self.branch.as_str(), "event_hit" | "event_spill")
            || self.branch.starts_with("hit")
            || self.branch.contains("cache_hit")
    }
}

/// The value of `name="..."` or `name=<token>` in a log line.
fn log_field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let start = line.find(&format!("{name}="))? + name.len() + 1;
    let rest = &line[start..];
    if let Some(quoted) = rest.strip_prefix('"') {
        quoted.split('"').next()
    } else {
        rest.split(|c: char| c.is_whitespace() || c == ',' || c == '}')
            .next()
    }
}

/// A log line without its ANSI colour sequences (`ESC [ ... m`), which the
/// gateway writes even into a file.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for d in chars.by_ref() {
                if ('@'..='~').contains(&d) {
                    break;
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// The gateway's routing decisions by request id: the last decision line of
/// each request (`Event-driven routing` and `Cache-aware selection` lines).
fn parse_decisions(text: &str) -> HashMap<String, Decision> {
    let mut out = HashMap::new();
    for raw in text.lines() {
        if !(raw.contains("Event-driven routing") || raw.contains("Cache-aware selection")) {
            continue;
        }
        let clean = strip_ansi(raw);
        let line = clean.as_str();
        let Some(id) = log_field(line, "request_id") else {
            continue;
        };
        let branch = log_field(line, "branch").unwrap_or_else(|| {
            if line.contains("no overlap") {
                "expected_wait_fallback"
            } else {
                "?"
            }
        });
        out.insert(
            id.to_string(),
            Decision {
                worker: log_field(line, "worker").unwrap_or_default().to_string(),
                branch: branch.to_string(),
                matched_ratio: log_field(line, "matched_ratio").and_then(|v| v.parse().ok()),
                overlap_tokens: log_field(line, "overlap_tokens").and_then(|v| v.parse().ok()),
                overlap_blocks: log_field(line, "overlap_blocks").and_then(|v| v.parse().ok()),
            },
        );
    }
    out
}

/// The gateway's request id is the response id without its uuid tail
/// (`-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`, 37 bytes).
fn request_id_prefix(id: &str) -> String {
    let bytes = id.as_bytes();
    if bytes.len() > 37 {
        let tail = &bytes[bytes.len() - 37..];
        let uuid_shaped = [0usize, 9, 14, 19, 24].iter().all(|&i| tail[i] == b'-')
            && tail
                .iter()
                .enumerate()
                .all(|(i, b)| matches!(i, 0 | 9 | 14 | 19 | 24) || b.is_ascii_hexdigit());
        if uuid_shaped {
            return id[..bytes.len() - 37].to_string();
        }
    }
    id.to_string()
}

fn decision_summary(ok: &[&ReqResult], decisions: usize) -> Value {
    let joined: Vec<&&ReqResult> = ok.iter().filter(|r| !r.branch.is_empty()).collect();
    let agree = joined.iter().filter(|r| r.agree == Some(true)).count();
    let credit_known = joined.iter().filter(|r| r.credit_tokens.is_some()).count();
    let mut by_branch: BTreeMap<String, (usize, usize, u64)> = BTreeMap::new();
    for r in &joined {
        let e = by_branch.entry(r.branch.clone()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += usize::from(r.agree == Some(true));
        e.2 += u64::from(r.cached_tokens);
    }
    json!({
        "decisions_in_log": decisions,
        "joined": joined.len(),
        "credit_known": credit_known,
        "agree": agree,
        "disagree": joined.len() - agree,
        "by_branch": by_branch.iter().map(|(b, (n, a, cached))| json!({
            "branch": b, "requests": n, "agree": a,
            "engine_cached_tokens_mean": if *n == 0 { 0.0 } else { *cached as f64 / *n as f64 },
        })).collect::<Vec<_>>(),
    })
}

/// The per-request decision table (`t4.md`): `implied overlap` is the
/// gateway's stated credit when its log carries one, `-` otherwise (then
/// `agree` compares the branch's claim of an overlap with the engine).
fn decision_table(results: &[ReqResult], rows: usize) -> String {
    let mut out = String::from(
        "| phase | idx | worker | branch | prompt_tokens | engine cached_tokens | implied overlap | agree |
|---|---|---|---|---|---|---|---|
",
    );
    let joined: Vec<&ReqResult> = results
        .iter()
        .filter(|r| r.status == "ok" && !r.branch.is_empty())
        .collect();
    for r in joined.iter().take(rows) {
        out.push_str(&format!(
            "| replay | {} | {} | {} | {} | {} | {} | {} |
",
            r.row,
            r.worker.rsplit(':').next().unwrap_or(&r.worker),
            r.branch,
            r.prompt_tokens,
            r.cached_tokens,
            r.credit_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.agree.map(|v| v.to_string()).unwrap_or_default()
        ));
    }
    let agree = joined.iter().filter(|r| r.agree == Some(true)).count();
    out.push_str(&format!(
        "
agreement: {agree}/{} (gateway credit vs engine truth; {} rows shown)
",
        joined.len(),
        joined.len().min(rows)
    ));
    out
}

fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

#[derive(Deserialize, Debug)]
struct AdminRecords {
    records: Vec<AdminRecord>,
    next: u64,
}

#[derive(Deserialize, Debug, Clone)]
struct AdminRecord {
    request_id: String,
    worker: String,
    cached_tokens: u32,
    oracle_tokens: u32,
    queued_ms: f64,
}

async fn fetch_admin_records(client: &reqwest::Client, admin: &str) -> Result<Vec<AdminRecord>> {
    let mut since = 0u64;
    let mut all = Vec::new();
    loop {
        let page: AdminRecords = client
            .get(format!("{admin}/admin/requests?since={since}&limit=50000"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if page.records.is_empty() {
            break;
        }
        since = page.next;
        all.extend(page.records);
    }
    Ok(all)
}

/// One worker's line of the mock admin API's `GET /admin/fleet` snapshot.
#[derive(Deserialize, Debug, Clone)]
struct FleetWorker {
    worker: String,
    #[serde(default)]
    num_running_reqs: u64,
    #[serde(default)]
    num_waiting_reqs: u64,
    #[serde(default)]
    num_waiting_uncached_tokens: u64,
    #[serde(default)]
    token_usage: f64,
    #[serde(default)]
    num_cached_blocks: u64,
    #[serde(default)]
    num_preemptions: u64,
    #[serde(default)]
    num_kv_batches: u64,
}

#[derive(Deserialize, Debug)]
struct FleetSnapshot {
    workers: Vec<FleetWorker>,
}

const FLEET_CSV_HEADER: &str =
    "t_s,worker,running,waiting,waiting_uncached_tokens,token_usage,cached_blocks,preemptions,kv_batches";

/// Samples the fleet's admin snapshot once a second into `path` (`fleet.csv`:
/// one row per worker per tick, seconds since `start`) until `stop` turns
/// true. The header is written before the first poll, so the file exists
/// with a header even when the admin API never answers; a failed poll is
/// skipped, not fatal. Returns the number of rows written.
async fn sample_fleet(
    client: reqwest::Client,
    admin: String,
    path: PathBuf,
    start: Instant,
    mut stop: watch::Receiver<bool>,
) -> Result<usize> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut out = String::from(FLEET_CSV_HEADER);
    out.push('\n');
    fs::write(&path, &out)?;
    let mut file = fs::OpenOptions::new().append(true).open(&path)?;
    let url = format!("{}/admin/fleet", admin.trim_end_matches('/'));
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut rows = 0usize;
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() {
                    break;
                }
                continue;
            }
        }
        let t_s = start.elapsed().as_secs_f64();
        let snapshot = match client.get(&url).send().await {
            Ok(resp) => resp.json::<FleetSnapshot>().await,
            Err(e) => Err(e),
        };
        let Ok(snapshot) = snapshot else {
            continue;
        };
        let mut chunk = String::new();
        for w in &snapshot.workers {
            chunk.push_str(&format!(
                "{t_s:.1},{},{},{},{},{:.4},{},{},{}\n",
                w.worker,
                w.num_running_reqs,
                w.num_waiting_reqs,
                w.num_waiting_uncached_tokens,
                w.token_usage,
                w.num_cached_blocks,
                w.num_preemptions,
                w.num_kv_batches
            ));
            rows += 1;
        }
        file.write_all(chunk.as_bytes())?;
    }
    file.flush()?;
    Ok(rows)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let text = fs::read_to_string(&args.trace)
        .with_context(|| format!("reading {}", args.trace.display()))?;
    let rows: Vec<TraceRow> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<TraceRow>(l).map_err(|e| anyhow!("{e}: {l}")))
        .collect::<Result<_>>()?;
    let end = if args.limit == 0 {
        rows.len()
    } else {
        (args.skip + args.limit).min(rows.len())
    };
    let rows: Vec<TraceRow> = rows[args.skip.min(rows.len())..end].to_vec();
    if rows.is_empty() {
        return Err(anyhow!("no rows selected"));
    }
    let t0 = rows[0].timestamp;
    let mut pins = Vec::new();
    for url in std::iter::once(&args.gateway).chain(args.admin.iter()) {
        if let Some(pin) = pinned_addresses(url).await {
            pins.push(pin);
        }
    }
    let client = http_client(args.connections, args.max_inflight, &pins)?;
    let url = format!("{}/v1/chat/completions", args.gateway.trim_end_matches('/'));
    let inflight = Arc::new(Semaphore::new(args.max_inflight));
    let mut set: JoinSet<ReqResult> = JoinSet::new();
    let start = Instant::now();
    // Per-second fleet series (admin snapshots) for the whole run, when the
    // admin API is given; stopped once every request has finished.
    let (stop_fleet, fleet_stop_rx) = watch::channel(false);
    let mut fleet_set: JoinSet<Result<usize>> = JoinSet::new();
    if let Some(admin) = &args.admin {
        fleet_set.spawn(sample_fleet(
            client.clone(),
            admin.clone(),
            args.out.join("fleet.csv"),
            start,
            fleet_stop_rx,
        ));
    }
    eprintln!(
        "replaying {} rows at {}x ({:.0} s of trace), {} words/block, max_output {}",
        rows.len(),
        args.speedup,
        rows.last().map_or(0, |r| r.timestamp.saturating_sub(t0)) as f64 / 1000.0,
        args.words_per_block,
        args.max_output
    );
    for (i, row) in rows.iter().enumerate() {
        let due = due_after(row.timestamp, t0, args.speedup);
        let elapsed = start.elapsed();
        if due > elapsed {
            tokio::time::sleep(due - elapsed).await;
        }
        let permit = inflight.clone().acquire_owned().await?;
        let prompt = prompt_for(row, args.seed, args.words_per_block);
        let job = Job {
            client: client.clone(),
            url: url.clone(),
            model: args.model.clone(),
            row_index: i,
            row: row.clone(),
            body_prompt: prompt,
            max_output: args.max_output,
            sent_at_ms: start.elapsed().as_secs_f64() * 1000.0,
            chat_template_kwargs: args.chat_template_kwargs.clone(),
        };
        set.spawn(async move {
            let _permit = permit;
            run_one(job).await
        });
        if (i + 1) % 500 == 0 {
            eprintln!(
                "  sent {} / {} ({:.0} s)",
                i + 1,
                rows.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    let mut results: Vec<ReqResult> = Vec::with_capacity(rows.len());
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("task failed: {e}"),
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    results.sort_by_key(|r| r.row);
    if !fleet_set.is_empty() {
        let _ = stop_fleet.send(true);
        match fleet_set.join_next().await {
            Some(Ok(Ok(rows))) => eprintln!("fleet series: {rows} rows in fleet.csv"),
            Some(Ok(Err(e))) => eprintln!("fleet series not written: {e}"),
            Some(Err(e)) => eprintln!("fleet series task failed: {e}"),
            None => {}
        }
    }

    // Oracle join (optional).
    let mut admin_rows: HashMap<String, AdminRecord> = HashMap::new();
    if let Some(admin) = &args.admin {
        match fetch_admin_records(&client, admin.trim_end_matches('/')).await {
            Ok(recs) => {
                for r in recs {
                    admin_rows.insert(r.request_id.clone(), r);
                }
            }
            Err(e) => eprintln!("admin records unavailable: {e}"),
        }
        for r in &mut results {
            if let Some(a) = admin_rows.get(&r.request_id) {
                r.oracle_tokens = Some(a.oracle_tokens.max(a.cached_tokens));
                r.queued_ms = Some(a.queued_ms);
                if r.worker.is_empty() {
                    r.worker = a.worker.clone();
                }
                if r.cached_tokens == 0 && a.cached_tokens > 0 {
                    r.cached_tokens = a.cached_tokens;
                }
            }
        }
    }

    // Gateway routing decisions: join by request id.
    let mut decisions: HashMap<String, Decision> = HashMap::new();
    if let Some(log) = &args.gateway_log {
        match fs::read_to_string(log) {
            Ok(text) => decisions = parse_decisions(&text),
            Err(e) => eprintln!("gateway log unreadable: {e}"),
        }
        for r in &mut results {
            let key = if r.gateway_request_id.is_empty() {
                request_id_prefix(&r.request_id)
            } else {
                r.gateway_request_id.clone()
            };
            if let Some(d) = decisions.get(&key) {
                r.branch.clone_from(&d.branch);
                r.credit_tokens = d.credit_tokens(r.prompt_tokens, args.block_size);
                r.agree = Some(match r.credit_tokens {
                    Some(credit) => credit == r.cached_tokens,
                    None => d.claims_overlap() == (r.cached_tokens > 0),
                });
            }
        }
    }
    let mut truth_per_worker: Value = Value::Null;
    if let Some(admin) = &args.admin {
        match client
            .get(format!("{}/admin/truth", admin.trim_end_matches('/')))
            .send()
            .await
        {
            Ok(resp) => truth_per_worker = resp.json().await.unwrap_or(Value::Null),
            Err(e) => eprintln!("engine truth unavailable: {e}"),
        }
    }

    // Statistics over the non-warm-up rows.
    let scored: Vec<&ReqResult> = results.iter().filter(|r| r.row >= args.warmup).collect();
    let ok: Vec<&ReqResult> = scored
        .iter()
        .copied()
        .filter(|r| r.status == "ok")
        .collect();
    let mut ttft: Vec<f64> = ok.iter().filter_map(|r| r.ttft_ms).collect();
    ttft.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut itl_means: Vec<f64> = ok.iter().filter_map(|r| r.itl_mean_ms).collect();
    itl_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut itl_p99s: Vec<f64> = ok.iter().filter_map(|r| r.itl_p99_ms).collect();
    itl_p99s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut e2e: Vec<f64> = ok.iter().map(|r| r.latency_ms).collect();
    e2e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let prompt_total: u64 = ok.iter().map(|r| u64::from(r.prompt_tokens)).sum();
    let cached_total: u64 = ok.iter().map(|r| u64::from(r.cached_tokens)).sum();
    let oracle_total: u64 = ok
        .iter()
        .filter_map(|r| r.oracle_tokens)
        .map(u64::from)
        .sum();
    let oracle_known = ok.iter().filter(|r| r.oracle_tokens.is_some()).count();
    let within_slo = ok
        .iter()
        .filter(|r| {
            r.ttft_ms.is_some_and(|t| t <= args.slo_ttft_ms)
                && r.itl_mean_ms.is_none_or(|m| m <= args.slo_itl_ms)
        })
        .count();
    let within_slo_strict = ok
        .iter()
        .filter(|r| {
            r.ttft_ms.is_some_and(|t| t <= args.slo_ttft_ms)
                && r.itl_p99_ms.is_none_or(|p| p <= args.slo_itl_ms)
        })
        .count();
    let mut per_worker: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for r in &ok {
        let e = per_worker.entry(r.worker.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += u64::from(r.prompt_tokens.saturating_sub(r.cached_tokens));
    }
    let counts: Vec<f64> = per_worker.values().map(|v| v.0 as f64).collect();
    let balance_max_over_mean = if counts.is_empty() {
        f64::NAN
    } else {
        counts.iter().copied().fold(0.0, f64::max) / mean(&counts)
    };
    let summary = json!({
        "label": args.label,
        "rows": rows.len(),
        "scored": scored.len(),
        "ok": ok.len(),
        "errors": scored.len() - ok.len(),
        "invisible_token_requests": ok.iter().filter(|r| r.invisible_tokens > 0).count(),
        "reasoning_tokens": ok.iter().map(|r| u64::from(r.reasoning_tokens)).sum::<u64>(),
        "speedup": args.speedup,
        "wall_s": wall_s,
        "req_per_s": ok.len() as f64 / wall_s,
        "ttft_ms": {"mean": mean(&ttft), "p50": percentile(&ttft, 0.5), "p90": percentile(&ttft, 0.9), "p99": percentile(&ttft, 0.99)},
        "itl_mean_ms": {"mean": mean(&itl_means), "p90": percentile(&itl_means, 0.9), "p99": percentile(&itl_means, 0.99)},
        "itl_p99_ms_per_request": {"p50": percentile(&itl_p99s, 0.5), "p90": percentile(&itl_p99s, 0.9)},
        "e2e_ms": {"p50": percentile(&e2e, 0.5), "p99": percentile(&e2e, 0.99)},
        "goodput_req_per_s": within_slo as f64 / wall_s,
        "within_slo_fraction": if ok.is_empty() { f64::NAN } else { within_slo as f64 / ok.len() as f64 },
        "within_slo_strict_fraction": if ok.is_empty() { f64::NAN } else { within_slo_strict as f64 / ok.len() as f64 },
        "prefix_reuse": if prompt_total == 0 { f64::NAN } else { cached_total as f64 / prompt_total as f64 },
        "oracle_prefix_reuse": if prompt_total == 0 || oracle_known == 0 { f64::NAN } else { oracle_total as f64 / prompt_total as f64 },
        "hit_over_oracle": if oracle_total == 0 { f64::NAN } else { cached_total as f64 / oracle_total as f64 },
        "oracle_known": oracle_known,
        "per_worker_requests": per_worker.iter().map(|(w, v)| json!({"worker": w, "requests": v.0, "uncached_prompt_tokens": v.1})).collect::<Vec<_>>(),
        "t4": decision_summary(&ok, decisions.len()),
        "engine_truth_per_worker": truth_per_worker.get("workers").cloned().unwrap_or(Value::Array(Vec::new())),
        "balance_max_over_mean": balance_max_over_mean,
        "slo": {"ttft_ms": args.slo_ttft_ms, "itl_ms": args.slo_itl_ms, "itl_metric": "per-request mean (strict variant: per-request p99)"},
    });
    fs::create_dir_all(&args.out)?;
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    let mut csv = String::from("row,trace_ts_ms,trace_input_length,sent_at_ms,status,request_id,worker,prompt_tokens,completion_tokens,cached_tokens,oracle_tokens,queued_ms,ttft_ms,latency_ms,itl_mean_ms,itl_p99_ms,tokens_seen,reasoning_tokens,invisible_tokens,gateway_request_id,branch,credit_tokens,agree\n");
    for r in &results {
        csv.push_str(&format!(
            "{},{},{},{:.1},{},{},{},{},{},{},{},{},{},{:.1},{},{},{},{},{},{},{},{},{}\n",
            r.row,
            r.trace_ts_ms,
            r.trace_input_length,
            r.sent_at_ms,
            r.status.replace(',', ";"),
            r.request_id,
            r.worker,
            r.prompt_tokens,
            r.completion_tokens,
            r.cached_tokens,
            r.oracle_tokens.map(|v| v.to_string()).unwrap_or_default(),
            r.queued_ms.map(|v| format!("{v:.1}")).unwrap_or_default(),
            r.ttft_ms.map(|v| format!("{v:.1}")).unwrap_or_default(),
            r.latency_ms,
            r.itl_mean_ms.map(|v| format!("{v:.2}")).unwrap_or_default(),
            r.itl_p99_ms.map(|v| format!("{v:.2}")).unwrap_or_default(),
            r.tokens_seen,
            r.reasoning_tokens,
            r.invisible_tokens,
            r.gateway_request_id,
            r.branch,
            r.credit_tokens.map(|v| v.to_string()).unwrap_or_default(),
            r.agree.map(|v| v.to_string()).unwrap_or_default()
        ));
    }
    fs::write(args.out.join("requests.csv"), csv)?;
    fs::write(
        args.out.join("t4.md"),
        decision_table(&results, args.t4_rows),
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-endpoint HTTP/1.1 server answering every request with the given
    /// body (what `GET /admin/fleet` returns), for the fleet sampler test.
    async fn serve_json(body: &'static str) -> (String, JoinSet<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut server: JoinSet<()> = JoinSet::new();
        server.spawn(async move {
            let mut conns: JoinSet<()> = JoinSet::new();
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                conns.spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let mut n = 0;
                    while n < buf.len() {
                        let Ok(k) = sock.read(&mut buf[n..]).await else {
                            return;
                        };
                        if k == 0 {
                            return;
                        }
                        n += k;
                        if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        (format!("http://{addr}"), server)
    }

    #[tokio::test]
    async fn fleet_series_has_a_header_and_one_row_per_worker_per_second() {
        let (admin, _server) = serve_json(
            r#"{"workers":[{"worker":"grpc:1","num_running_reqs":2,"num_waiting_reqs":1,"num_waiting_uncached_tokens":300,"token_usage":0.25,"num_cached_blocks":40,"num_preemptions":0,"num_kv_batches":9},{"worker":"grpc:2","num_running_reqs":0,"num_waiting_reqs":0,"token_usage":0.0,"num_cached_blocks":0,"num_preemptions":0,"num_kv_batches":0}]}"#,
        )
        .await;
        let dir = std::env::temp_dir().join(format!("replay-fleet-{}", std::process::id()));
        let path = dir.join("fleet.csv");
        let (stop, rx) = watch::channel(false);
        // The test server is local: no proxy from the environment.
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client");
        let mut sampler: JoinSet<Result<usize>> = JoinSet::new();
        sampler.spawn(sample_fleet(
            client,
            admin,
            path.clone(),
            Instant::now(),
            rx,
        ));
        tokio::time::sleep(Duration::from_millis(2600)).await;
        let _ = stop.send(true);
        let rows = sampler
            .join_next()
            .await
            .expect("a task")
            .expect("join")
            .expect("sampler");
        let text = fs::read_to_string(&path).expect("fleet.csv");
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some(FLEET_CSV_HEADER));
        let data: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
        assert_eq!(data.len(), rows);
        // ticks at 0, 1 and 2 s inside 2.6 s, two workers each
        assert!((4..=8).contains(&rows) && rows % 2 == 0, "rows {rows}");
        assert_eq!(data[0][1], "grpc:1");
        assert_eq!(data[1][1], "grpc:2");
        assert_eq!(&data[0][2..], &["2", "1", "300", "0.2500", "40", "0", "9"]);
        let t: Vec<f64> = data
            .iter()
            .step_by(2)
            .map(|r| r[0].parse().expect("t_s"))
            .collect();
        for w in t.windows(2) {
            assert!(
                (w[1] - w[0] - 1.0).abs() < 0.3,
                "one tick per second: {t:?}"
            );
        }
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn block_text_is_deterministic_per_hash_id() {
        assert_eq!(block_text(42, 7, 32), block_text(42, 7, 32));
        assert_ne!(block_text(42, 7, 32), block_text(43, 7, 32));
        assert_ne!(block_text(42, 7, 32), block_text(42, 8, 32));
        assert_eq!(block_text(1, 7, 32).split(' ').count(), 32);
    }

    #[test]
    fn prompts_share_prefixes_when_rows_share_hash_ids() {
        let a = TraceRow {
            timestamp: 0,
            input_length: 1024,
            output_length: 1,
            hash_ids: vec![0, 1],
        };
        let b = TraceRow {
            timestamp: 0,
            input_length: 1024,
            output_length: 1,
            hash_ids: vec![0, 2],
        };
        let pa = prompt_for(&a, 7, 16);
        let pb = prompt_for(&b, 7, 16);
        let shared = block_text(0, 7, 16);
        assert!(pa.starts_with(&shared) && pb.starts_with(&shared));
        assert_ne!(pa, pb);
    }

    #[test]
    fn percentile_and_mean() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&v, 0.5), 3.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 1.0), 5.0);
        assert_eq!(mean(&v), 3.0);
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn a_finish_only_stream_counts_as_served() {
        // Captured from the gateway: a one-token answer whose token is an
        // incomplete UTF-8 piece, so no chunk carries text; the finish chunk
        // and the usage still arrive.
        let finish: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"grpc:19600","choices":[{"index":0,"delta":{"reasoning_content":null},"logprobs":null,"finish_reason":"length"}]}"#,
        )
        .unwrap();
        let usage: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"grpc:19600","choices":[],"usage":{"prompt_tokens":34,"completion_tokens":1,"total_tokens":35,"prompt_tokens_details":{"cached_tokens":16}}}"#,
        )
        .unwrap();
        let started = Instant::now();
        let mut observer = StreamObserver::default();
        observer.observe(&finish, started + Duration::from_millis(40));
        observer.observe(&usage, started + Duration::from_millis(41));
        let mut result = ReqResult {
            status: "ok".to_string(),
            ..Default::default()
        };
        observer.finish(started, &mut result);
        assert_eq!(result.status, "ok");
        assert_eq!(result.request_id, "chatcmpl-x");
        assert_eq!(result.worker, "grpc:19600");
        assert_eq!(result.completion_tokens, 1);
        assert_eq!(result.cached_tokens, 16);
        assert_eq!(result.tokens_seen, 0);
        assert_eq!(result.invisible_tokens, 1);
        assert!((result.ttft_ms.unwrap() - 40.0).abs() < 1.0);
    }

    #[test]
    fn a_stream_with_no_output_at_all_is_no_tokens() {
        let usage: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-y","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":0,"total_tokens":3}}"#,
        )
        .unwrap();
        let started = Instant::now();
        let mut observer = StreamObserver::default();
        observer.observe(&usage, started + Duration::from_millis(5));
        let mut result = ReqResult {
            status: "ok".to_string(),
            ..Default::default()
        };
        observer.finish(started, &mut result);
        assert_eq!(result.status, "no-tokens");
        assert!(result.ttft_ms.is_none());
    }

    #[test]
    fn visible_tokens_give_ttft_and_itl() {
        let tok = |s: &str| -> Value {
            serde_json::from_str(&format!(
                r#"{{"id":"chatcmpl-z","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}"#
            ))
            .unwrap()
        };
        let started = Instant::now();
        let mut observer = StreamObserver::default();
        observer.observe(&tok("a"), started + Duration::from_millis(100));
        observer.observe(&tok("b"), started + Duration::from_millis(120));
        observer.observe(&tok("c"), started + Duration::from_millis(150));
        let mut result = ReqResult {
            status: "ok".to_string(),
            ..Default::default()
        };
        observer.finish(started, &mut result);
        assert_eq!(result.tokens_seen, 3);
        assert!((result.ttft_ms.unwrap() - 100.0).abs() < 1.0);
        assert!((result.itl_mean_ms.unwrap() - 25.0).abs() < 1.0);
        assert_eq!(result.invisible_tokens, 0);
        // No usage arrived: the stream is still a served one, counted as seen.
        assert_eq!(result.status, "ok");
        assert_eq!(result.completion_tokens, 3);
    }

    /// A chat completion chunk whose delta carries the given field.
    fn delta_chunk(field: &str, text: &str) -> Value {
        serde_json::from_str(&format!(
            r#"{{"id":"chatcmpl-r","choices":[{{"index":0,"delta":{{"{field}":"{text}"}},"finish_reason":null}}]}}"#
        ))
        .unwrap()
    }

    #[test]
    fn reasoning_deltas_count_for_ttft_and_itl() {
        // A thinking model: two reasoning deltas, then one of text, then the
        // finish and the usage, as the gateway streams them.
        let finish: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-r","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        )
        .unwrap();
        let usage: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-r","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}"#,
        )
        .unwrap();
        let started = Instant::now();
        let mut observer = StreamObserver::default();
        observer.observe(
            &delta_chunk("reasoning_content", "think"),
            started + Duration::from_millis(40),
        );
        observer.observe(
            &delta_chunk("reasoning_content", "more"),
            started + Duration::from_millis(60),
        );
        observer.observe(
            &delta_chunk("content", "answer"),
            started + Duration::from_millis(80),
        );
        observer.observe(&finish, started + Duration::from_millis(81));
        observer.observe(&usage, started + Duration::from_millis(82));
        let mut result = ReqResult {
            status: "ok".to_string(),
            ..Default::default()
        };
        observer.finish(started, &mut result);
        assert_eq!(result.status, "ok");
        assert!((result.ttft_ms.unwrap() - 40.0).abs() < 1.0);
        assert!((result.itl_mean_ms.unwrap() - 20.0).abs() < 1.0);
        assert_eq!(result.reasoning_tokens, 2);
        assert_eq!(result.tokens_seen, 1);
        assert_eq!(result.completion_tokens, 3);
        assert_eq!(result.invisible_tokens, 0);
    }

    #[test]
    fn a_reasoning_only_stream_is_served_not_invisible() {
        // The output budget ran out inside the reasoning: no text at all.
        let finish: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-r","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
        )
        .unwrap();
        let usage: Value = serde_json::from_str(
            r#"{"id":"chatcmpl-r","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}"#,
        )
        .unwrap();
        let started = Instant::now();
        let mut observer = StreamObserver::default();
        observer.observe(
            &delta_chunk("reasoning", "a"),
            started + Duration::from_millis(30),
        );
        observer.observe(
            &delta_chunk("reasoning", "b"),
            started + Duration::from_millis(50),
        );
        observer.observe(&finish, started + Duration::from_millis(500));
        observer.observe(&usage, started + Duration::from_millis(501));
        let mut result = ReqResult {
            status: "ok".to_string(),
            ..Default::default()
        };
        observer.finish(started, &mut result);
        assert_eq!(result.status, "ok");
        // The first reasoning delta is the first token, not the finish.
        assert!((result.ttft_ms.unwrap() - 30.0).abs() < 1.0);
        assert!((result.itl_mean_ms.unwrap() - 20.0).abs() < 1.0);
        assert_eq!(result.reasoning_tokens, 2);
        assert_eq!(result.tokens_seen, 0);
        assert_eq!(result.invisible_tokens, 0);
    }

    #[test]
    fn chat_template_kwargs_are_forwarded_only_when_given() {
        let plain = request_body("m", "hi", 5, None);
        assert!(plain.get("chat_template_kwargs").is_none());
        assert_eq!(plain["max_tokens"], 5);
        let kwargs = parse_json_object(r#"{"enable_thinking": false}"#).unwrap();
        let body = request_body("m", "hi", 5, Some(&kwargs));
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["messages"][0]["content"], "hi");
        assert!(parse_json_object("[1]").is_err());
        assert!(parse_json_object("nope").is_err());
    }

    /// A one-route HTTP/1.1 server that records, per request, the client's
    /// port and whether the request asked for the connection to close; it
    /// keeps a connection open until the client closes it or asked it to.
    #[expect(
        clippy::disallowed_methods,
        reason = "test server tasks end with the test's runtime"
    )]
    async fn recording_server() -> (String, Arc<std::sync::Mutex<Vec<(u16, bool)>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let url = format!("http://{}/", listener.local_addr().expect("addr"));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, peer)) = listener.accept().await else {
                    return;
                };
                let log = log.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut byte = [0u8; 1];
                    loop {
                        // One request head at a time (the bodies are empty).
                        buf.clear();
                        while !buf.ends_with(b"\r\n\r\n") {
                            match socket.read(&mut byte).await {
                                Ok(0) | Err(_) => return,
                                Ok(_) => buf.push(byte[0]),
                            }
                        }
                        let head = String::from_utf8_lossy(&buf).to_ascii_lowercase();
                        let close = head.contains("connection: close");
                        log.lock().unwrap().push((peer.port(), close));
                        let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
                        if socket.write_all(response.as_bytes()).await.is_err() || close {
                            return;
                        }
                    }
                });
            }
        });
        (url, seen)
    }

    #[tokio::test]
    async fn fresh_connections_are_never_shared_between_requests() {
        let (url, seen) = recording_server().await;
        let client = http_client(Connections::Fresh, 16, &[]).expect("client");
        for _ in 0..3 {
            let response = client.get(&url).send().await.expect("send");
            assert_eq!(response.text().await.expect("body"), "ok");
        }
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 3);
        assert!(
            seen.iter().all(|(_, close)| *close),
            "every request asks the server to close: {seen:?}"
        );
        let ports: std::collections::BTreeSet<u16> = seen.iter().map(|(port, _)| *port).collect();
        assert_eq!(ports.len(), 3, "a connection per request: {seen:?}");
    }

    #[tokio::test]
    async fn host_names_are_pinned_once_and_addresses_left_alone() {
        assert!(pinned_addresses("http://127.0.0.1:1/").await.is_none());
        assert!(pinned_addresses("http://[::1]:1/").await.is_none());
        assert!(pinned_addresses("not a url").await.is_none());
        let (host, addrs) = pinned_addresses("http://localhost:1/")
            .await
            .expect("localhost resolves");
        assert_eq!(host, "localhost");
        assert!(!addrs.is_empty());
        assert!(addrs.iter().all(|a| a.ip().is_loopback()), "{addrs:?}");
    }

    #[tokio::test]
    async fn pooled_connections_are_reused_between_requests() {
        let (url, seen) = recording_server().await;
        let client = http_client(Connections::Pooled, 16, &[]).expect("client");
        for _ in 0..3 {
            let response = client.get(&url).send().await.expect("send");
            assert_eq!(response.text().await.expect("body"), "ok");
        }
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 3);
        assert!(
            seen.iter().all(|(_, close)| !*close),
            "keep-alive requests: {seen:?}"
        );
        let ports: std::collections::BTreeSet<u16> = seen.iter().map(|(port, _)| *port).collect();
        assert_eq!(ports.len(), 1, "one connection reused: {seen:?}");
    }

    #[test]
    fn gateway_decisions_are_parsed_and_joined_by_request_id() {
        let log = concat!(
            "2026-10-05 10:23:43 DEBUG http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module=\"smg\" request_id=\"chatcmpl-nukU\"}: smg::policies::cache_aware: cache_aware.rs:1681: Cache-aware selection index=\"tree\" branch=\"expected_wait_fallback\" worker=\"grpc://127.0.0.1:19611\" model_id=\"mock-model\" matched_ratio=0.0 threshold=0.30000001192092896\n",
            "2026-10-05 10:23:46 DEBUG http_request{method=POST uri=/v1/chat/completions version=HTTP/1.1 module=\"smg\" request_id=\"chatcmpl-UNiu\"}: smg::policies::cache_aware: cache_aware.rs:1493: Event-driven routing: overlap match worker=\"grpc://127.0.0.1:19611\" branch=\"event_hit\" model_id=\"mock-model\"\n",
            "2026-10-05 10:23:48 DEBUG http_request{request_id=\"chatcmpl-fut\"}: smg::policies::cache_aware: Event-driven routing: overlap match worker=\"grpc://127.0.0.1:19610\" branch=\"event_hit\" overlap_blocks=12 model_id=\"mock-model\"\n",
            "2026-10-05 10:23:49 DEBUG http_request{request_id=\"chatcmpl-none\"}: smg::policies::cache_aware: Event-driven routing: no overlap, expected-wait fallback worker=\"grpc://127.0.0.1:19610\" model_id=\"mock-model\"\n",
            "2026-10-05 10:23:50 DEBUG some other line request_id=\"x\" branch=\"nope\"\n",
            // As the gateway really writes it: ANSI colour sequences around every field.
            "\u{1b}[2m2026-10-05 10:59:48\u{1b}[0m \u{1b}[34mDEBUG\u{1b}[0m \u{1b}[1mhttp_request\u{1b}[0m\u{1b}[1m{\u{1b}[0m\u{1b}[3mrequest_id\u{1b}[0m\u{1b}[2m=\u{1b}[0m\"chatcmpl-7lE1\"\u{1b}[1m}\u{1b}[0m\u{1b}[2m:\u{1b}[0m Event-driven routing: overlap match \u{1b}[3mworker\u{1b}[0m\u{1b}[2m=\u{1b}[0m\"grpc://127.0.0.1:19702\" \u{1b}[3mbranch\u{1b}[0m\u{1b}[2m=\u{1b}[0m\"event_spill\"\n",
        );
        let d = parse_decisions(log);
        assert_eq!(d.len(), 5);
        assert_eq!(d["chatcmpl-7lE1"].branch, "event_spill");
        assert_eq!(d["chatcmpl-7lE1"].worker, "grpc://127.0.0.1:19702");
        assert_eq!(d["chatcmpl-nukU"].branch, "expected_wait_fallback");
        assert_eq!(d["chatcmpl-nukU"].matched_ratio, Some(0.0));
        assert_eq!(d["chatcmpl-nukU"].credit_tokens(1000, 16), Some(0));
        assert_eq!(d["chatcmpl-UNiu"].branch, "event_hit");
        assert!(d["chatcmpl-UNiu"].claims_overlap());
        assert_eq!(d["chatcmpl-UNiu"].credit_tokens(1000, 16), None);
        assert_eq!(d["chatcmpl-fut"].credit_tokens(1000, 16), Some(192));
        assert_eq!(d["chatcmpl-none"].branch, "expected_wait_fallback");
        assert!(!d["chatcmpl-none"].claims_overlap());
        assert_eq!(
            request_id_prefix(
                "chatcmpl-nukUelurU5QeHF4GhOUknktv-01a10b97-428b-7f72-9728-a08fad5787ae"
            ),
            "chatcmpl-nukUelurU5QeHF4GhOUknktv"
        );
    }

    #[test]
    fn decision_table_has_the_expected_columns() {
        let mut r = ReqResult {
            row: 3,
            status: "ok".to_string(),
            worker: "grpc:19500".to_string(),
            branch: "event_hit".to_string(),
            prompt_tokens: 640,
            cached_tokens: 512,
            agree: Some(true),
            ..Default::default()
        };
        let table = decision_table(std::slice::from_ref(&r), 10);
        assert!(table.starts_with("| phase | idx | worker | branch | prompt_tokens | engine cached_tokens | implied overlap | agree |"));
        assert!(table.contains("| replay | 3 | 19500 | event_hit | 640 | 512 | - | true |"));
        assert!(table.contains("agreement: 1/1"));
        r.credit_tokens = Some(512);
        let table = decision_table(std::slice::from_ref(&r), 10);
        assert!(table.contains("| 640 | 512 | 512 | true |"));
    }

    #[test]
    fn a_row_stamped_before_the_first_is_due_at_once() {
        // 3 s of trace at 2x is 1.5 s of wall time.
        assert_eq!(due_after(3_010, 10, 2.0), Duration::from_millis(1_500));
        assert_eq!(due_after(10, 10, 2.0), Duration::ZERO);
        // An earlier stamp (unsorted trace, or a window starting on a
        // reordered row) saturates to "now" instead of wrapping around.
        assert_eq!(due_after(5, 10, 2.0), Duration::ZERO);
        assert_eq!(due_after(0, u64::MAX, 0.5), Duration::ZERO);
        // A due time beyond what a Duration holds is clamped, not a panic.
        assert_eq!(due_after(u64::MAX, 0, 1e-300), Duration::MAX);
    }

    #[test]
    fn speedup_must_be_finite_and_positive() {
        assert_eq!(parse_speedup("2"), Ok(2.0));
        assert_eq!(parse_speedup(" 0.5 "), Ok(0.5));
        for bad in ["0", "-1", "inf", "-inf", "NaN", "fast", ""] {
            assert!(parse_speedup(bad).is_err(), "{bad:?} must be rejected");
        }
        // The flag rejects them at parsing, before any request is sent.
        for bad in ["0", "-2", "inf", "nan"] {
            assert!(
                Args::try_parse_from(["replay", "--trace", "t.jsonl", "--speedup", bad]).is_err(),
                "--speedup {bad} must be rejected"
            );
        }
        let ok = Args::try_parse_from(["replay", "--trace", "t.jsonl", "--speedup", "4"])
            .expect("a valid speedup parses");
        assert_eq!(ok.speedup, 4.0);
    }

    #[test]
    fn sse_event_boundary() {
        assert_eq!(find_double_newline(b"data: x\n\ndata: y"), Some(7));
        assert_eq!(find_double_newline(b"data: x\n"), None);
    }
}
