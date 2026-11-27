//! `stress_load`: a load client, open loop or closed loop, for the gateway's
//! gRPC request path on a CPU-only bed (the release gateway in front of
//! `mock-worker --engine realistic` gRPC engines; `scripts/bench/grpc_stress/`).
//!
//! It sends requests at a fixed rate whatever the answer time, so what it
//! measures is the gateway's cost under a given offered load: the time to the
//! response head (TTFB), to the first content delta (TTFT), to the end of the
//! body (e2e), the SSE frames per stream and the gaps between them, and the
//! status codes and error kinds. Requests started during `--warmup` are driven
//! but not sampled. One JSON summary on stdout at the end (`--out` writes it
//! too); `STRESS_INTERIM {...}` counters on stderr every `--interim` seconds.
//!
//! `--closed-loop N` drives a closed loop instead: N callers, each sending
//! its next request the moment its last one ended, so the answers set the
//! rate and no request is skipped. `achieved_rps` counts, as in the open
//! loop, the requests that started inside the window and ended well.
//!
//! `--conversation-turns T` makes each caller of a closed loop of `--mode
//! chat` hold a conversation: its first turn is the next of the `--distinct`
//! prompts, and each turn behind it sends the conversation so far (every turn
//! answered by an assistant message the client makes, of `--max-tokens`
//! words: the worker's own answer is not carried) with one more user message
//! of `--turn-tokens` words, `--think-ms` after the answer before. After T
//! turns, or a turn that failed, the caller starts a new conversation. The
//! summary's `conversations` says how many sampled turns went to the worker
//! of the turn before, by the answers' `x-smg-routed-worker-id` header (a
//! turn that failed behind its headers counts where they name one), and at
//! which turn a conversation moved.
//!
//! Modes (`--mode`): `chat` (chat completions; `--stream` for SSE), `tools`
//! (a tool-heavy conversation: `--turns` tool-call rounds over `--tools`
//! function schemas), `responses` (the Responses API with `store: true`,
//! `--chain` requests per session linked by `previous_response_id`) and
//! `image` (`--image <file>` inline as a data URL, `--images-per-request`
//! copies, before the text), `messages` (the Messages API, one user message),
//! `embeddings` (the prompt as the input), `generate` (the engine-native
//! `/generate` route, the prompt as its `text`) and `completion`
//! (`/v1/completions`, the prompt as its `prompt`). `--json-schema` asks a
//! chat request for a JSON answer under a fixed schema. `--tenants N` spreads
//! the requests over N tenant ids sent in `--tenant-header`. Those ids come
//! round again and the warm-up sends them before the window opens, so
//! `--new-tenants S` sends the share S of the requests under an id that no
//! request carried before.
//!
//! Prompts are deterministic from `--seed`: `--distinct` distinct prompts of
//! about `--prompt-tokens` words (one token each for a large-vocabulary BPE),
//! each a shared head from a pool of `--prefix-pool` fixed texts
//! (`--prefix-ratio` of the prompt) followed by a unique tail, so cache-aware
//! routing sees shared prefixes and unique remainders as in production.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line tool reports on its standard streams"
)]

use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
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
use tokio::task::{coop, JoinSet};

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
    /// `POST /v1/messages` (the Messages API) with one user message.
    Messages,
    /// `POST /v1/embeddings` with the prompt as the input.
    Embeddings,
    /// `POST /generate` (the engine-native route) with the prompt as its
    /// `text`; a streamed answer repeats the whole text so far in every frame.
    Generate,
    /// `POST /v1/completions` with the prompt as its `prompt`; a streamed
    /// answer is one `text` delta per frame.
    Completion,
}

/// The flags a conversation's flags are refused beside. Each names them
/// itself: a flag's `requires` is waived when one of these is given.
const NO_CONVERSATION: [&str; 4] = ["tenants", "new_tenants", "rate", "max_inflight"];

#[derive(Parser, Debug, Clone)]
#[command(
    name = "stress_load",
    about = "Load client for the gateway's stress runs, open loop or closed loop (chat, tools, responses, image, messages, embeddings, generate, completion)"
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
    /// A closed loop of this many callers instead of a rate: each caller
    /// sends its next request the moment its last one ended (answered or
    /// failed), so none is skipped.
    #[arg(long, conflicts_with_all = ["rate", "max_inflight"])]
    closed_loop: Option<NonZeroUsize>,
    /// Conversations in a closed loop of `--mode chat`: a caller sends this
    /// many turns, each carrying the turns before it, every one answered by
    /// an assistant message the client makes (`--max-tokens` words, not the
    /// worker's answer), then starts a new conversation. Not with
    /// `--tenants` or `--new-tenants`: a turn's tenant would not be its
    /// conversation's.
    #[arg(long, requires = "closed_loop", conflicts_with_all = NO_CONVERSATION)]
    conversation_turns: Option<NonZeroUsize>,
    /// Words of each user message behind a conversation's first, which is
    /// one of the `--distinct` prompts.
    #[arg(
        long,
        default_value_t = 64,
        requires = "conversation_turns",
        conflicts_with_all = NO_CONVERSATION
    )]
    turn_tokens: usize,
    /// Milliseconds a conversation waits between an answer and its next turn.
    #[arg(
        long,
        default_value_t = 0,
        requires = "conversation_turns",
        conflicts_with_all = NO_CONVERSATION
    )]
    think_ms: u64,
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
    /// Maximum text content parts per chat user message (0: one string).
    /// Splits on word boundaries without dropping text; other modes are unchanged.
    #[arg(long, default_value_t = 0)]
    text_parts: usize,
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
    /// Chat requests ask for a JSON answer under a fixed object schema
    /// (`response_format: json_schema`).
    #[arg(long, default_value_t = false)]
    json_schema: bool,
    /// Chat requests ask for logprobs: the sampled token's and this many
    /// alternatives per token (`logprobs: true, top_logprobs: N`).
    #[arg(long)]
    logprobs: Option<u32>,
    /// Spread the requests over this many tenant ids (0: no tenant header).
    #[arg(long, default_value_t = 0)]
    tenants: usize,
    #[arg(long, default_value = "x-smg-tenant-id")]
    tenant_header: String,
    /// Share of the requests (0 to 1) sent under a tenant id of their own,
    /// `tenant-new-<seed>-<request number>`, in place of one of the
    /// `--tenants` ids: an id no request carried before, in the warm-up or
    /// in the window. A later run against the same gateway sends other ids
    /// under another `--seed`. Refused with the linked requests of `--mode
    /// responses` (`--chain` above 1): a link holds under one tenant.
    #[arg(long, default_value_t = 0.0, value_parser = share)]
    new_tenants: f64,
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
    /// Concurrent sessions kept in `--mode responses` (0: four per req/s, or
    /// one per caller of a closed loop; at least 64).
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

/// A share from 0 to 1 (`--new-tenants`).
fn share(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    (0.0..=1.0)
        .contains(&v)
        .then_some(v)
        .ok_or_else(|| format!("{s} is not a share from 0 to 1"))
}

/// The tenant id of request `i`, if it carries one: an id of its own for the
/// `new_tenants` share of the requests, else the next of the `tenants` ids,
/// which the other requests take in turn. Of the first `n` requests
/// `floor(n * share)` carry a new id.
fn tenant(args: &Args, i: u64) -> Option<String> {
    let new = |n: u64| (n as f64 * args.new_tenants).floor() as u64;
    if new(i) > new(i.saturating_sub(1)) {
        return Some(format!("tenant-new-{}-{i}", args.seed));
    }
    (args.tenants > 0).then(|| format!("tenant-{}", (i - new(i)) % args.tenants as u64))
}

/// Builds a chat user message as a string (zero parts) or text content blocks.
/// Splits after ASCII spaces, retaining every byte and emitting at most the
/// requested number of parts; empty text produces an empty array. This controls
/// request shape, not token equivalence across model chat templates.
fn chat_user(text: String, text_parts: usize) -> Value {
    if text_parts == 0 {
        return user(text);
    }
    let words: Vec<_> = text.split_inclusive(' ').collect();
    let per_part = words.len().div_ceil(text_parts).max(1);
    let parts: Vec<_> = words
        .chunks(per_part)
        .map(|words| json!({"type": "text", "text": words.concat()}))
        .collect();
    json!({"role": "user", "content": parts})
}

fn user(text: impl Into<String>) -> Value {
    json!({"role": "user", "content": text.into()})
}

/// The `response_format` of a `--json-schema` chat request: a small object
/// with a string, a bounded integer and an enum, every field required.
fn answer_schema() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                    "status": {"type": "string", "enum": ["ok", "unsure"]}
                },
                "required": ["answer", "confidence", "status"],
                "additionalProperties": false
            }
        }
    })
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

/// A chat completions request over `messages`, with the options the flags ask.
fn chat_body(args: &Args, messages: Vec<Value>) -> Value {
    let mut body = json!({
        "model": args.model,
        "messages": [],
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "stream_options": if args.stream { json!({"include_usage": true}) } else { Value::Null },
    });
    // Moved in, not copied: a conversation's messages grow by its turns.
    body["messages"] = Value::Array(messages);
    if args.json_schema {
        body["response_format"] = answer_schema();
    }
    if let Some(top_logprobs) = args.logprobs {
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(top_logprobs);
    }
    body
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
    if args.mode == Mode::Responses && args.chain > 1 && args.new_tenants > 0.0 {
        anyhow::bail!("--new-tenants parts the linked requests of --mode responses: use --chain 1");
    }
    if args.mode == Mode::Image && images.is_empty() {
        anyhow::bail!("--mode image needs at least one --image <file>");
    }
    if args.mode == Mode::Embeddings && args.stream {
        anyhow::bail!("--mode embeddings answers one JSON object: drop --stream");
    }
    if args.json_schema && args.mode != Mode::Chat {
        anyhow::bail!("--json-schema applies to --mode chat only");
    }
    if args.logprobs.is_some() && args.mode != Mode::Chat {
        anyhow::bail!("--logprobs applies to --mode chat only");
    }
    if args.conversation_turns.is_some() && args.mode != Mode::Chat {
        anyhow::bail!("--conversation-turns applies to --mode chat only");
    }
    let mut bodies = Vec::with_capacity(distinct);
    for i in 0..distinct {
        let body = match args.mode {
            Mode::Chat => chat_body(args, vec![chat_user(prompt(args, i), args.text_parts)]),
            Mode::Messages => json!({
                "model": args.model,
                "max_tokens": args.max_tokens,
                "messages": [user(prompt(args, i))],
                "stream": args.stream,
            }),
            Mode::Embeddings => json!({
                "model": args.model,
                "input": prompt(args, i),
            }),
            Mode::Generate => json!({
                "model": args.model,
                "text": prompt(args, i),
                "sampling_params": {"max_new_tokens": args.max_tokens},
                "stream": args.stream,
            }),
            Mode::Completion => json!({
                "model": args.model,
                "prompt": prompt(args, i),
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
    /// Conversations opened: the next one's place in the `--distinct` prompts.
    opened: AtomicU64,
    samples: Mutex<Vec<Sample>>,
    statuses: Mutex<BTreeMap<u16, u64>>,
    errors: Mutex<BTreeMap<String, u64>>,
    conversations: Mutex<Conversations>,
    gaps: Option<LogHist>,
}

/// What `--conversation-turns` adds to the summary, over the sampled turns.
#[derive(Serialize, Default, Clone)]
struct Conversations {
    /// The turns of a conversation, as asked.
    turns: usize,
    /// Conversations whose last turn ended well.
    finished: u64,
    /// Turns whose answer names a worker, as the answer of the turn before
    /// in their conversation did.
    followed: u64,
    /// The followed turns that went to the worker of the turn before.
    returned: u64,
    /// `returned` over `followed`.
    return_share: f64,
    /// The turn (2 is a conversation's second) at which conversations left
    /// the worker of the turn before, and how many did.
    moves_by_turn: BTreeMap<usize, u64>,
}

/// A conversation's turn as `one` sends it: the body, and what `one` reads
/// from the answer.
struct Turn {
    body: Bytes,
    /// The answer ended well.
    ok: bool,
    /// The worker the answer names in `x-smg-routed-worker-id`, also where
    /// the answer failed behind its headers: the turn went there.
    worker: Option<String>,
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

/// Whether an SSE frame carries a visible token: a `content` or
/// `reasoning_content` string (chat), a Responses `delta`, or a Messages
/// `text` or `thinking` delta, that is not empty.
fn frame_has_token(frame: &[u8]) -> bool {
    for key in [
        &b"\"content\":\""[..],
        &b"\"reasoning_content\":\""[..],
        &b"\"delta\":\""[..],
        &b"\"text\":\""[..],
        &b"\"thinking\":\""[..],
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
/// the router's `data: {"error":{...}}` frame, a Responses `response.failed`
/// event, or the Messages API's `error` event (`{"type":"error",...}`).
fn frame_is_error(frame: &[u8]) -> bool {
    frame_data(frame).is_some_and(|data| {
        data.starts_with(b"{\"error\"")
            || data.starts_with(b"{\"type\":\"error\"")
            || contains(data, b"\"type\":\"response.failed\"")
    })
}

/// Whether an SSE frame ends its stream in order: `data: [DONE]`, a
/// Responses `response.completed` / `response.incomplete` event, or the
/// Messages API's `message_stop`.
fn frame_is_last(frame: &[u8]) -> bool {
    frame_data(frame).is_some_and(|data| {
        data == b"[DONE]"
            || contains(data, b"\"type\":\"response.completed\"")
            || contains(data, b"\"type\":\"response.incomplete\"")
            || contains(data, b"\"type\":\"message_stop\"")
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

async fn one(shared: Arc<Shared>, i: u64, sampled: bool, turn: Option<&mut Turn>) {
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
    let body = if let Some(turn) = turn.as_deref() {
        turn.body.clone()
    } else if args.mode == Mode::Responses {
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
    if let Some(tenant) = tenant(args, i) {
        request = request.header(&args.tenant_header, tenant);
    }
    let started = Instant::now();
    let keep_body = args.mode == Mode::Responses || args.debug_first;
    let mut sample = Sample::default();
    let mut body_head: Vec<u8> = Vec::new();
    let mut outcome: Result<Option<String>, String> = Ok(None);
    let mut worker: Option<String> = None;
    match request.send().await {
        Err(e) => outcome = Err(classify(&e)),
        Ok(response) => {
            sample.ttfb_us = started.elapsed().as_micros() as u32;
            let status = response.status().as_u16();
            if turn.is_some() {
                let named = response.headers().get("x-smg-routed-worker-id");
                worker = named.and_then(|v| v.to_str().ok()).map(str::to_owned);
            }
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
    if let Some(turn) = turn {
        turn.ok = outcome.is_ok();
        turn.worker = worker;
    }
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
    /// The callers of a closed loop; an open-loop summary has no such key.
    #[serde(skip_serializing_if = "Option::is_none")]
    closed_loop: Option<NonZeroUsize>,
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
    /// A summary without `--conversation-turns` has no such key.
    #[serde(skip_serializing_if = "Option::is_none")]
    conversations: Option<Conversations>,
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
        rate_target: shared.args.closed_loop.map_or(shared.args.rate, |_| 0.0),
        closed_loop: shared.args.closed_loop,
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
        conversations: shared.args.conversation_turns.map(|turns| {
            let counts = stats
                .conversations
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            Conversations {
                turns: turns.get(),
                return_share: counts.returned as f64 / counts.followed.max(1) as f64,
                ..counts.clone()
            }
        }),
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
        Mode::Messages => "/v1/messages",
        Mode::Embeddings => "/v1/embeddings",
        Mode::Generate => "/generate",
        Mode::Completion => "/v1/completions",
        _ => "/v1/chat/completions",
    };
    let url = format!("{}{}", args.gateway.trim_end_matches('/'), path);
    let bodies = build_bodies(&args)?;
    let sessions = if args.sessions > 0 {
        args.sessions
    } else if let Some(callers) = args.closed_loop {
        callers.get().max(64)
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
    if let Some(callers) = args.closed_loop {
        for _ in 0..callers.get() {
            let shared = Arc::clone(&shared);
            tasks.spawn(async move {
                // This caller's conversation: its messages so far, and the
                // worker of its last answer.
                let mut messages: Vec<Value> = Vec::new();
                let mut worker: Option<String> = None;
                loop {
                    // A send that fails without a wait must not hold its
                    // thread to the end of the window.
                    coop::consume_budget().await;
                    let elapsed = start.elapsed();
                    if elapsed >= total {
                        break;
                    }
                    let i = shared.stats.sent.fetch_add(1, Ordering::Relaxed) + 1;
                    shared.stats.note_inflight();
                    let sampled = elapsed >= warm;
                    let args = &shared.args;
                    let Some(turns) = args.conversation_turns else {
                        one(Arc::clone(&shared), i, sampled, None).await;
                        continue;
                    };
                    let text = if messages.is_empty() {
                        // An opening is the next of the `--distinct` prompts,
                        // as a request outside a conversation is.
                        let opened = shared.stats.opened.fetch_add(1, Ordering::Relaxed);
                        prompt(args, opened as usize % args.distinct.max(1))
                    } else {
                        let seed = args.seed ^ (i << 20);
                        let said = words(seed ^ 0xA55E_0000, args.max_tokens as usize);
                        messages.push(json!({"role": "assistant", "content": said}));
                        words(seed ^ 0x7012_0000, args.turn_tokens)
                    };
                    messages.push(chat_user(text, args.text_parts));
                    let nth = messages.len() / 2 + 1;
                    let mut body = chat_body(args, std::mem::take(&mut messages));
                    let mut turn = Turn {
                        body: Bytes::from(body.to_string()),
                        ok: false,
                        worker: None,
                    };
                    if let Value::Array(sent) = body["messages"].take() {
                        messages = sent;
                    }
                    one(Arc::clone(&shared), i, sampled, Some(&mut turn)).await;
                    if sampled {
                        let mut counts = shared
                            .stats
                            .conversations
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        if let (Some(before), Some(now)) = (&worker, &turn.worker) {
                            counts.followed += 1;
                            if before == now {
                                counts.returned += 1;
                            } else {
                                *counts.moves_by_turn.entry(nth).or_insert(0) += 1;
                            }
                        }
                        counts.finished += u64::from(turn.ok && nth == turns.get());
                    }
                    if turn.ok && nth < turns.get() {
                        worker = turn.worker;
                        // A caller whose next turn would start behind the
                        // window leaves: asleep at the end of the drain it
                        // would count as a request that did not finish.
                        let think = Duration::from_millis(args.think_ms);
                        if start.elapsed() + think >= total {
                            break;
                        }
                        if !think.is_zero() {
                            tokio::time::sleep(think).await;
                        }
                    } else {
                        messages.clear();
                        worker = None;
                    }
                }
            });
        }
        tokio::time::sleep_until((start + total).into()).await;
    } else {
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
                tasks.spawn(one(Arc::clone(&shared), launched, sampled, None));
            }
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

    /// Text blocks preserve whitespace and Unicode, including boundary inputs.
    #[test]
    fn chat_text_parts_keep_every_byte_and_default_string_shape() {
        for text in ["", "hello", "hello 世界  again ", "one two three four five"] {
            assert_eq!(chat_user(text.to_owned(), 0), user(text));
            for count in [1, 2, 16, usize::MAX] {
                let body = chat_user(text.to_owned(), count);
                let parts = body["content"].as_array().unwrap();
                assert!(parts.len() <= count);
                let joined: String = parts
                    .iter()
                    .map(|part| {
                        assert_eq!(part["type"], "text");
                        part["text"].as_str().unwrap()
                    })
                    .collect();
                assert_eq!(joined, text);
            }
        }
    }

    /// Multiple parts are produced in order, with spaces retained at boundaries.
    #[test]
    fn chat_text_parts_split_at_word_boundaries() {
        for (count, expected) in [
            (1, vec!["one two three four five"]),
            (2, vec!["one two three ", "four five"]),
            (usize::MAX, vec!["one ", "two ", "three ", "four ", "five"]),
        ] {
            let actual = chat_user("one two three four five".to_owned(), count);
            let expected: Vec<_> = expected
                .into_iter()
                .map(|text| json!({"type": "text", "text": text}))
                .collect();
            assert_eq!(actual, json!({"role": "user", "content": expected}));
        }
        assert_eq!(chat_user(String::new(), 1)["content"], json!([]));
    }

    /// The CLI option reaches serialized chat bodies without changing other fields.
    #[test]
    fn chat_text_parts_are_wired_into_request_bodies() {
        let mut options = Args::parse_from([
            "stress_load",
            "--mode",
            "chat",
            "--text-parts",
            "4",
            "--distinct",
            "3",
            "--stream",
            "--json-schema",
            "--logprobs",
            "2",
        ]);
        let multipart = build_bodies(&options).unwrap();
        options.text_parts = 0;
        let strings = build_bodies(&options).unwrap();
        assert_eq!(args(Mode::Chat).text_parts, 0);
        assert_eq!(multipart.len(), strings.len());
        for (parts, string) in multipart.iter().zip(&strings) {
            let mut parts: Value = serde_json::from_slice(parts).unwrap();
            let string: Value = serde_json::from_slice(string).unwrap();
            let content = parts["messages"][0]["content"].as_array().unwrap();
            assert_eq!(content.len(), 4);
            let joined: String = content
                .iter()
                .map(|part| {
                    assert_eq!(part["type"], "text");
                    part["text"].as_str().unwrap()
                })
                .collect();
            assert_eq!(
                Some(joined.as_str()),
                string["messages"][0]["content"].as_str()
            );
            parts["messages"][0]["content"] = Value::String(joined);
            assert_eq!(parts, string);
        }
    }

    /// Text partitioning is limited to chat, leaving every other mode unchanged.
    #[test]
    fn chat_text_parts_leave_other_modes_unchanged() {
        let directory = tempfile::tempdir().unwrap();
        let image = directory.path().join("image.png");
        std::fs::write(&image, b"image fixture").unwrap();
        for &mode in Mode::value_variants() {
            if mode == Mode::Chat {
                continue;
            }
            let mut options = args(mode);
            if mode == Mode::Image {
                options.image = vec![image.clone()];
            }
            let baseline = build_bodies(&options).unwrap();
            let response = responses_body(&options, 3, Some("resp_1"));
            options.text_parts = 4;
            assert_eq!(build_bodies(&options).unwrap(), baseline, "{mode:?}");
            if mode == Mode::Responses {
                assert_eq!(responses_body(&options, 3, Some("resp_1")), response);
            }
        }
    }

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
    fn new_tenants_are_an_exact_share_and_no_id_comes_twice() {
        let mut a = args(Mode::Chat);
        a.tenants = 4;
        let ids = |a: &Args| (1..=4000).filter_map(|i| tenant(a, i)).collect::<Vec<_>>();
        let distinct = |ids: &[String]| ids.iter().collect::<std::collections::BTreeSet<_>>().len();
        assert_eq!(tenant(&a, 6).as_deref(), Some("tenant-2"));
        assert_eq!(distinct(&ids(&a)), 4, "without the flag: the set in turn");

        a.new_tenants = 0.25;
        let mixed = ids(&a);
        let new: Vec<String> = mixed
            .iter()
            .filter(|id| id.starts_with("tenant-new-"))
            .cloned()
            .collect();
        assert_eq!(new.len(), 1000, "a quarter of 4,000 requests");
        assert_eq!(distinct(&new), 1000, "no new id comes twice");
        assert_eq!(distinct(&mixed), 1004, "the others take the four in turn");

        a.seed += 1;
        let other = ids(&a);
        assert!(new.iter().all(|id| !other.contains(id)), "another seed");

        a.tenants = 0;
        a.new_tenants = 1.0;
        assert_eq!(distinct(&ids(&a)), 4000, "every request, with no set");

        for bad in ["25", "-0.1", "nan", "inf"] {
            let read = Args::try_parse_from(["stress_load", "--new-tenants", bad]);
            assert!(read.is_err(), "{bad} is no share");
        }
        let mut linked = args(Mode::Responses);
        linked.new_tenants = 0.5;
        assert!(build_bodies(&linked).is_err(), "one tenant a link");
        linked.chain = 1;
        assert!(build_bodies(&linked).is_ok(), "no link, no refusal");
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
    fn messages_embeddings_and_json_schema_bodies() {
        let m = build_bodies(&args(Mode::Messages)).unwrap();
        let v: Value = serde_json::from_slice(&m[0]).unwrap();
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["max_tokens"], 8);
        assert!(v.get("tools").is_none());

        let e = build_bodies(&args(Mode::Embeddings)).unwrap();
        let v: Value = serde_json::from_slice(&e[1]).unwrap();
        assert!(v["input"].as_str().unwrap().contains("[session 1]"));
        assert!(v.get("messages").is_none());

        let mut g = args(Mode::Generate);
        g.stream = true;
        let g = build_bodies(&g).unwrap();
        let v: Value = serde_json::from_slice(&g[2]).unwrap();
        assert!(v["text"].as_str().unwrap().contains("[session 2]"));
        assert_eq!(v["sampling_params"]["max_new_tokens"], 8);
        assert_eq!(v["stream"], true);
        assert!(v.get("messages").is_none());

        let mut c = args(Mode::Completion);
        c.stream = true;
        let c = build_bodies(&c).unwrap();
        let v: Value = serde_json::from_slice(&c[3]).unwrap();
        assert!(v["prompt"].as_str().unwrap().contains("[session 3]"));
        assert_eq!(v["max_tokens"], 8);
        assert_eq!(v["stream"], true);
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert!(v.get("messages").is_none());
        let v: Value =
            serde_json::from_slice(&build_bodies(&args(Mode::Completion)).unwrap()[0]).unwrap();
        assert!(v["stream_options"].is_null());

        let mut c = args(Mode::Chat);
        c.json_schema = true;
        let v: Value = serde_json::from_slice(&build_bodies(&c).unwrap()[0]).unwrap();
        assert_eq!(v["response_format"]["type"], "json_schema");
        assert_eq!(
            v["response_format"]["json_schema"]["schema"]["required"][0],
            "answer"
        );
        assert!(build_bodies(&args(Mode::Chat)).unwrap()[0]
            .windows(15)
            .all(|w| w != b"response_format"));

        assert!(frame_is_last(
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}"
        ));
        assert!(frame_has_token(
            b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}"
        ));
        assert!(!frame_has_token(
            b"event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"\"}}"
        ));
        assert!(frame_has_token(
            b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Hm\"}}"
        ));
        assert!(frame_is_error(
            b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}"
        ));
        assert!(!frame_is_error(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"type\\\":\\\"error\\\"}\"}}]}"
        ));

        let mut e = args(Mode::Embeddings);
        e.stream = true;
        assert!(build_bodies(&e).is_err(), "an embeddings stream is refused");
        let mut m = args(Mode::Messages);
        m.json_schema = true;
        assert!(
            build_bodies(&m).is_err(),
            "--json-schema outside chat is refused"
        );
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

    /// Three callers against a server that holds every answer for a permit:
    /// three requests are out and no fourth, and one answer, a refusal,
    /// brings one more.
    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "the server and the load end with the test's runtime"
    )]
    async fn a_closed_loop_caller_sends_its_next_request_at_its_answer() {
        for refused in [
            &["--closed-loop", "0"][..],
            &["--closed-loop", "3", "--rate", "5"],
            &["--closed-loop", "3", "--max-inflight", "8"],
        ] {
            let words = ["stress_load"].iter().chain(refused);
            assert!(Args::try_parse_from(words).is_err(), "{refused:?}");
        }

        let seen = Arc::new(AtomicU64::new(0));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let held = {
            let (seen, gate) = (Arc::clone(&seen), Arc::clone(&gate));
            move || async move {
                seen.fetch_add(1, Ordering::Relaxed);
                match gate.acquire().await {
                    Ok(permit) => {
                        permit.forget();
                        axum::http::StatusCode::SERVICE_UNAVAILABLE
                    }
                    Err(_) => axum::http::StatusCode::OK,
                }
            }
        };
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(held));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });
        let args = Args::try_parse_from([
            "stress_load",
            "--gateway",
            &gateway,
            "--closed-loop",
            "3",
            "--warmup",
            "0",
            "--duration",
            "3",
            "--interim",
            "0",
            "--distinct",
            "1",
        ]);
        let load = tokio::spawn(run(args.expect("a closed loop is a flag")));

        let requests = |n: u64| {
            let seen = Arc::clone(&seen);
            tokio::time::timeout(Duration::from_secs(5), async move {
                while seen.load(Ordering::Relaxed) < n {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
        };
        requests(3).await.expect("each caller sends a request");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            seen.load(Ordering::Relaxed),
            3,
            "a caller waits for its answer"
        );
        gate.add_permits(1);
        requests(4)
            .await
            .expect("the answered caller sends its next request");

        gate.close();
        let summary = load.await.unwrap().unwrap();
        assert_eq!(summary.sent, seen.load(Ordering::Relaxed));
        assert_eq!((summary.ok + 1, summary.failed), (summary.sent, 1));
        assert_eq!((summary.skipped, summary.unfinished), (0, 0));
        assert_eq!(summary.inflight_max, 3);
        let read = serde_json::to_value(&summary).unwrap();
        assert_eq!(
            (&read["closed_loop"], &read["rate_target"]),
            (&json!(3), &json!(0.0))
        );
    }

    /// Two callers hold conversations of four turns against a server that
    /// names worker `a` for a conversation's first two turns and `b` behind
    /// them, and `lost` for a turn that does not carry the turn before it;
    /// it fails the last turn of the conversation that opens with the first
    /// of the `--distinct` prompts, behind a header that names worker `c`.
    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "the server ends with the test's runtime"
    )]
    async fn a_conversation_carries_its_turns_and_counts_where_they_went() {
        use axum::http::StatusCode;

        for refused in [
            &["--conversation-turns=4"][..],
            &["--closed-loop=2", "--conversation-turns=0"],
            &["--closed-loop=2", "--conversation-turns=4", "--tenants=2"],
            &[
                "--closed-loop=2",
                "--conversation-turns=4",
                "--new-tenants=0.5",
            ],
            &["--closed-loop=2", "--think-ms=2"],
            &["--conversation-turns=4", "--rate=10"],
            &["--closed-loop=2", "--think-ms=2", "--tenants=2"],
        ] {
            let words = ["stress_load"].iter().chain(refused);
            assert!(Args::try_parse_from(words).is_err(), "{refused:?}");
        }

        // Each conversation's messages at its last turn, by its first message.
        let seen: Arc<Mutex<BTreeMap<String, Vec<Value>>>> = Arc::default();
        let answer = move |axum::Json(body): axum::Json<Value>| async move {
            let messages = body["messages"].as_array().cloned().unwrap_or_default();
            let first = messages[0]["content"].to_string();
            let opens_the_prompts = first.contains("[session 0]");
            let before = seen.lock().unwrap().insert(first, messages.clone());
            let carried = before.is_none_or(|before| {
                messages.len() == before.len() + 2
                    && messages.starts_with(&before)
                    && messages[before.len()]["role"] == "assistant"
            });
            let (status, worker) = match messages.len() {
                _ if !carried => (StatusCode::OK, "lost"),
                7 if opens_the_prompts => (StatusCode::INTERNAL_SERVER_ERROR, "c"),
                1 | 3 => (StatusCode::OK, "a"),
                _ => (StatusCode::OK, "b"),
            };
            (status, [("x-smg-routed-worker-id", worker)], "{}")
        };
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(answer));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gateway = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });
        let args = Args::try_parse_from([
            "stress_load",
            "--gateway",
            &gateway,
            "--closed-loop",
            "2",
            "--conversation-turns",
            "4",
            "--think-ms",
            "2",
            "--warmup",
            "0",
            "--duration",
            "1",
            "--interim",
            "0",
        ]);
        let args = args.expect("a conversation is a flag of a closed loop");
        let summary = run(args.clone()).await.unwrap();

        let read = serde_json::to_value(&summary).unwrap();
        let counts = summary.conversations.expect("the summary counts the turns");
        assert!(counts.finished > 0, "a conversation ends at its last turn");
        // The one failure is the turn the server fails.
        assert_eq!((summary.failed, summary.unfinished), (1, 0));
        // A whole conversation returns at its turns 2 and 4 and moves at 3;
        // the window's end may cut each caller's last one short. The failed
        // turn moved to the worker its header names.
        let moved = counts.moves_by_turn.get(&3).copied().unwrap_or(0);
        assert_eq!(counts.moves_by_turn.get(&4), Some(&1));
        assert_eq!(counts.moves_by_turn.len(), 2, "moves at turns 3 and 4");
        assert!((counts.finished + 1..=counts.finished + 3).contains(&moved));
        assert_eq!(counts.returned + moved + 1, counts.followed);
        assert_eq!(
            read["conversations"]["return_share"],
            json!(counts.returned as f64 / counts.followed as f64)
        );
        assert_eq!(read["conversations"]["turns"], json!(4));

        // A caller whose think time ends behind the window leaves at its
        // answer: it is not a request that did not finish.
        let thinker = Args {
            closed_loop: NonZeroUsize::new(1),
            think_ms: 60_000,
            drain: 0,
            ..args
        };
        let summary = run(thinker).await.unwrap();
        assert_eq!((summary.sent, summary.ok, summary.unfinished), (1, 1, 0));
    }
}
