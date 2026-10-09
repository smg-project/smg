//! A lightweight, CPU-only simulation of a continuous-batching LLM engine.
//!
//! The goal is fidelity, not fakery: the mock should exhibit the behaviors that
//! drive SMG's routing decisions so the whole gateway can be exercised without
//! GPUs. Concretely this models
//!
//! - **prefill latency that scales with input length** — time-to-first-token
//!   grows with the (uncached) prompt size, chunked across scheduler steps;
//! - **decode latency that grows with load** — a decode pass costs more as
//!   the decoding requests' KV utilisation rises (with the linear model, as
//!   the batch widens), so a busy replica is slower per token;
//! - **finite KV capacity with admission/queueing** — when KV is full requests
//!   wait, producing the `num_waiting_uncached_tokens` signal `least_load` uses;
//! - **prefix caching** — a request sharing a prefix with cached blocks pays
//!   less prefill, reports `cached_tokens`, and the engine emits the KV-cache
//!   events (`KvBlocksStored` / `KvBlocksRemoved`) that drive `cache_aware`.
//!
//! The engine is an actor: one `tokio` task per virtual worker owns all mutable
//! state and advances it in real wall-clock time, but the per-step work is plain
//! arithmetic. Idle engines block on their request channel, so a fleet of mostly
//! idle workers stays cheap. The scheduling math lives in `SchedulerState::step`,
//! a pure synchronous function returning the work done plus the time it took,
//! which makes it deterministically unit-testable with no real timers.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex, OnceLock, RwLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::common_proto as common;
use tokio::sync::{broadcast, mpsc, Notify};
use tonic::Status;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How a pass's duration follows from the work it contains.
#[derive(Clone, Debug, PartialEq)]
pub enum TimingModel {
    /// Polynomials over the pass: prefill `a + b·T + c·T²` ms with `T` the
    /// uncached tokens it processes, decode `max(1, a + b·u + c·u²)` ms with
    /// `u` the KV utilisation of the decoding requests (their context tokens
    /// over capacity). The defaults are AISimulate's uncalibrated baseline, so
    /// results compare with other simulators built on the same polynomials.
    Polynomial { prefill: [f64; 3], decode: [f64; 3] },
    /// Prefill at a fixed token rate; decode `base + per_req × batch` ms.
    Linear {
        prefill_tps: f64,
        decode_base_ms: f64,
        decode_per_req_ms: f64,
    },
    /// A hardware calibration: a single request's prefill from a measured
    /// point table (piecewise-linear over `(tokens, ms)`, extrapolated with
    /// the last slope); a pass that prefills several requests costs the
    /// pass-total form `intercept + slope × total uncached tokens in the
    /// pass`, never less than the table value of its largest request (the
    /// per-request plateau is the floor). Decode as the polynomial.
    Calibrated {
        table: Vec<(f64, f64)>,
        pass_intercept_ms: f64,
        pass_ms_per_token: f64,
        decode: [f64; 3],
    },
}

impl TimingModel {
    /// AISimulate's prefill polynomial (ms over uncached tokens in the pass).
    pub const POLY_PREFILL: [f64; 3] = [16.501_42, 1.518_344e-2, 4.209_989e-7];
    /// AISimulate's decode polynomial (ms over KV utilisation).
    pub const POLY_DECODE: [f64; 3] = [5.74, 54.01, -25.74];

    pub fn polynomial() -> Self {
        Self::Polynomial {
            prefill: Self::POLY_PREFILL,
            decode: Self::POLY_DECODE,
        }
    }

    pub fn linear() -> Self {
        Self::Linear {
            prefill_tps: 8000.0,
            decode_base_ms: 6.0,
            decode_per_req_ms: 0.35,
        }
    }

    /// A calibration file's model (`Calibration::load`): the point table and
    /// pass-total form when the file has them, else its polynomials.
    pub(crate) fn fitted(c: &Calibration) -> Self {
        match (&c.prefill_table, c.prefill_pass) {
            (Some(table), pass) if table.len() >= 2 => {
                let (pass_intercept_ms, pass_ms_per_token) = pass.unwrap_or_else(|| {
                    // No pass form: the table's last slope, from its first value.
                    let n = table.len();
                    let slope =
                        (table[n - 1].1 - table[n - 2].1) / (table[n - 1].0 - table[n - 2].0);
                    (table[0].1, slope.max(0.0))
                });
                Self::Calibrated {
                    table: table.clone(),
                    pass_intercept_ms,
                    pass_ms_per_token,
                    decode: c.decode,
                }
            }
            _ => Self::Polynomial {
                prefill: c.prefill,
                decode: c.decode,
            },
        }
    }

    /// Piecewise-linear value of a `(tokens, ms)` table at `tokens`: the
    /// first point's value below the table, the last segment's slope (never
    /// negative) above it.
    fn table_ms(table: &[(f64, f64)], tokens: f64) -> f64 {
        let Some(first) = table.first() else {
            return 0.0;
        };
        if tokens <= first.0 || table.len() == 1 {
            return first.1;
        }
        for w in table.windows(2) {
            let ((x0, y0), (x1, y1)) = (w[0], w[1]);
            if tokens <= x1 {
                return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0).max(f64::EPSILON);
            }
        }
        let n = table.len();
        let ((x0, y0), (x1, y1)) = (table[n - 2], table[n - 1]);
        let slope = ((y1 - y0) / (x1 - x0).max(f64::EPSILON)).max(0.0);
        y1 + slope * (tokens - x1)
    }

    /// Prefill time of a pass that computes `total` uncached tokens, the
    /// largest single request's share being `largest`: a lone request costs
    /// its table value; a batched pass costs the pass-total form, never less
    /// than the table value of its largest request.
    fn prefill_pass_ms(&self, total: u32, largest: u32) -> f64 {
        if total == 0 {
            return 0.0;
        }
        match self {
            Self::Calibrated {
                table,
                pass_intercept_ms,
                pass_ms_per_token,
                ..
            } => {
                let largest = largest.min(total).max(1);
                let single = Self::table_ms(table, f64::from(largest));
                if largest >= total {
                    // A lone request costs its table value, whatever the pass form says.
                    return single;
                }
                let batched = pass_intercept_ms + pass_ms_per_token * f64::from(total);
                single.max(batched)
            }
            other => other.prefill_ms(total),
        }
    }

    fn prefill_ms(&self, tokens: u32) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        match self {
            Self::Polynomial {
                prefill: [a, b, c], ..
            } => {
                let t = f64::from(tokens);
                a + b * t + c * t * t
            }
            Self::Linear { prefill_tps, .. } => f64::from(tokens) / prefill_tps.max(1.0) * 1000.0,
            Self::Calibrated { .. } => self.prefill_pass_ms(tokens, tokens),
        }
    }

    fn decode_ms(&self, batch: usize, active_tokens: u64, capacity_tokens: u64) -> f64 {
        if batch == 0 {
            return 0.0;
        }
        match self {
            Self::Polynomial {
                decode: [a, b, c], ..
            } => {
                let u = (active_tokens as f64 / capacity_tokens.max(1) as f64).min(1.0);
                (a + b * u + c * u * u).max(1.0)
            }
            Self::Linear {
                decode_base_ms,
                decode_per_req_ms,
                ..
            } => decode_base_ms + decode_per_req_ms * batch as f64,
            Self::Calibrated {
                decode: [a, b, c], ..
            } => {
                let u = (active_tokens as f64 / capacity_tokens.max(1) as f64).min(1.0);
                (a + b * u + c * u * u).max(1.0)
            }
        }
    }
}

/// A hardware calibration of the pass model, as the GPU harness writes it:
/// prefill `a + b·T + c·T²` ms over the uncached tokens of a pass, decode
/// `d + e·u + f·u²` ms over KV utilisation, the engine's KV capacity and a
/// fixed per-request overhead. The JSON accepts the harness's own layout
/// (`prefill_fit_ms: {a_ms, b_ms_per_token, c_ms_per_token2}`,
/// `decode_fit_vs_utilisation_ms: {d_ms, e_ms_per_u, f_ms_per_u2}`), or the
/// coefficients as objects (`{"a":..,"b":..,"c":..}` / `{"d":..,"e":..,"f":..}`)
/// or arrays under `prefill_ms`/`prefill` and `decode_ms`/`decode`; capacity
/// as `kv_capacity_tokens` or `kv_capacity_blocks` (with `block_size`); the
/// overhead only as an explicit `request_overhead_ms` (the harness's
/// `fixed_overhead_ms` is the measured one-token TTFT, which the prefill
/// intercept already covers, so it is not added again). Unknown keys are
/// ignored.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Calibration {
    pub prefill: [f64; 3],
    pub decode: [f64; 3],
    /// Measured single-request prefill `(tokens, ms)` points, sorted by tokens
    /// (`prefill_table_ms: [[tokens, ms], ...]`, or the harness's
    /// `prefill_points_ms: {"<tokens>": {"median_ms": ..}}`).
    pub prefill_table: Option<Vec<(f64, f64)>>,
    /// Pass-total form for batched prefills, `(intercept_ms, ms_per_token)`
    /// (`prefill_pass_ms: {"intercept_ms": .., "ms_per_token": ..}`).
    pub prefill_pass: Option<(f64, f64)>,
    pub kv_capacity_tokens: Option<u64>,
    pub block_size: Option<u32>,
    pub request_overhead_ms: f64,
}

impl Calibration {
    pub(crate) fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read calibration {path}: {e}"))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("calibration {path} is not JSON: {e}"))?;
        Self::from_value(&value).map_err(|e| format!("calibration {path}: {e}"))
    }

    pub(crate) fn from_value(v: &serde_json::Value) -> Result<Self, String> {
        let poly = |keys: [&str; 3], names: [&str; 3]| -> Result<[f64; 3], String> {
            let node = keys
                .iter()
                .find_map(|k| v.get(*k))
                .ok_or_else(|| format!("missing {}", keys[0]))?;
            if let Some(arr) = node.as_array() {
                let got: Vec<f64> = arr.iter().filter_map(serde_json::Value::as_f64).collect();
                return match got.as_slice() {
                    [a, b, c] => Ok([*a, *b, *c]),
                    _ => Err(format!("{} needs three numbers", keys[0])),
                };
            }
            let mut out = [0.0; 3];
            for (slot, name) in out.iter_mut().zip(names) {
                *slot = node
                    .get(name)
                    .and_then(serde_json::Value::as_f64)
                    .ok_or_else(|| format!("{} is missing {name}", keys[0]))?;
            }
            Ok(out)
        };
        let prefill_table = Self::table_of(v);
        let prefill_pass = v
            .get("prefill_pass_ms")
            .or_else(|| v.get("batched_prefill_ms"))
            .and_then(|node| {
                Some((
                    node.get("intercept_ms")?.as_f64()?,
                    node.get("ms_per_token")?.as_f64()?,
                ))
            });
        let prefill = poly(
            ["prefill_fit_ms", "prefill_ms", "prefill"],
            ["a_ms", "b_ms_per_token", "c_ms_per_token2"],
        )
        .or_else(|_| poly(["prefill_ms", "prefill", "prefill_fit_ms"], ["a", "b", "c"]))
        .or_else(|e| {
            // A table alone is a complete prefill model.
            if prefill_table.is_some() {
                Ok([0.0; 3])
            } else {
                Err(e)
            }
        })?;
        let decode_keys = ["decode_fit_vs_utilisation_ms", "decode_ms", "decode"];
        let decode = poly(decode_keys, ["d_ms", "e_ms_per_u", "f_ms_per_u2"])
            .or_else(|_| poly(decode_keys, ["d", "e", "f"]))
            .or_else(|_| poly(decode_keys, ["a", "b", "c"]))?;
        let block_size = v
            .get("block_size")
            .and_then(serde_json::Value::as_u64)
            .map(|b| b as u32);
        let kv_capacity_tokens = v
            .get("kv_capacity_tokens")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| {
                let blocks = v.get("kv_capacity_blocks")?.as_u64()?;
                Some(blocks * u64::from(block_size?))
            });
        let request_overhead_ms = v
            .get("request_overhead_ms")
            .or_else(|| v.get("per_request_overhead_ms"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        Ok(Self {
            prefill,
            decode,
            prefill_table,
            prefill_pass,
            kv_capacity_tokens,
            block_size,
            request_overhead_ms,
        })
    }

    /// The prefill point table, from `prefill_table_ms` (`[[tokens, ms], ..]`),
    /// `prefill_points` (`[{tokens, ms | prefill_ms | median_ms}, ..]`) or the
    /// harness's `prefill_points_ms` (`{"<tokens>": {"median_ms": ..}}`).
    fn table_of(v: &serde_json::Value) -> Option<Vec<(f64, f64)>> {
        let mut points: Vec<(f64, f64)> = Vec::new();
        if let Some(rows) = v
            .get("prefill_table_ms")
            .and_then(serde_json::Value::as_array)
        {
            for row in rows {
                if let (Some(t), Some(ms)) = (
                    row.get(0).and_then(serde_json::Value::as_f64),
                    row.get(1).and_then(serde_json::Value::as_f64),
                ) {
                    points.push((t, ms));
                }
            }
        } else if let Some(rows) = v
            .get("prefill_points")
            .and_then(serde_json::Value::as_array)
        {
            for row in rows {
                let ms = ["ms", "prefill_ms", "median_ms"]
                    .iter()
                    .find_map(|k| row.get(*k).and_then(serde_json::Value::as_f64));
                if let (Some(t), Some(ms)) =
                    (row.get("tokens").and_then(serde_json::Value::as_f64), ms)
                {
                    points.push((t, ms));
                }
            }
        } else if let Some(map) = v
            .get("prefill_points_ms")
            .and_then(serde_json::Value::as_object)
        {
            for (tokens, entry) in map {
                let ms = entry
                    .get("median_ms")
                    .and_then(serde_json::Value::as_f64)
                    .or_else(|| entry.as_f64());
                if let (Ok(t), Some(ms)) = (tokens.parse::<f64>(), ms) {
                    points.push((t, ms));
                }
            }
        }
        if points.len() < 2 {
            return None;
        }
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Some(points)
    }
}

/// Tunable parameters of the simulated engine. The scheduler is vLLM's pass
/// loop (a token budget per pass, running requests first, then FCFS waiting,
/// LIFO preemption when KV runs out) over a block-level KV pool with
/// reference counts and an LRU of idle cached blocks; override via
/// mock-worker flags.
#[derive(Clone, Debug)]
pub struct EngineParams {
    /// Pass duration model.
    pub timing: TimingModel,
    /// Token budget per pass (`max_num_batched_tokens`): each decode token
    /// costs one, prefill chunks take the rest.
    pub max_batched_tokens: u32,
    /// Max sequences in a pass (`max_num_seqs`).
    pub max_running: usize,
    /// KV cache capacity in tokens (`num_blocks × block_size`).
    pub kv_capacity_tokens: u64,
    /// The KV capacity the decode cost was calibrated against. When set, a
    /// decode step's utilisation is the decoding context over this reference,
    /// not over the configured pool, so shrinking the pool (`--kv-blocks`)
    /// changes how much fits, not how long a step takes: the engine's step
    /// time depends on the tokens it attends to, not on the pool size.
    pub decode_reference_tokens: Option<u64>,
    /// Cache block (page) size in tokens.
    pub block_size: u32,
    /// Whether prefix caching + KV-event emission are enabled.
    pub prefix_cache: bool,
    /// SGLang-style scheduling: a pass that contains prefill runs prefill only.
    pub prefill_first: bool,
    /// vLLM's `scheduler_reserve_full_isl`: a waiting request is admitted only
    /// when the free and evictable blocks could hold its whole prompt beyond
    /// its cached blocks, and admission stops at the first request that does
    /// not fit; only the pass's chunk is allocated at admission, the next
    /// chunks allocate as they run (the gate is read, not reserved, as in
    /// vLLM's `full_sequence_must_fit`). Off, the gate is skipped (the older,
    /// over-admitting behaviour).
    pub reserve_full_isl: bool,
    /// Output tokens to generate when a request does not specify `max_new_tokens`.
    pub max_new_default: u32,
    /// Fixed per-request overhead in ms (a calibration's intercept that the
    /// pass polynomials do not explain): every event of a request's stream is
    /// delivered that much later, so TTFT and e2e grow by it and ITL does not.
    pub request_overhead_ms: f64,
    /// Capacity of the live KV-event broadcast channel.
    pub kv_broadcast_capacity: usize,
    /// How many recent KV-event batches to retain for subscriber replay.
    pub kv_replay_capacity: usize,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            timing: TimingModel::polynomial(),
            max_batched_tokens: 8192,
            max_running: 256,
            kv_capacity_tokens: 524_288,
            decode_reference_tokens: None,
            block_size: 16,
            prefix_cache: true,
            prefill_first: false,
            reserve_full_isl: true,
            max_new_default: 128,
            request_overhead_ms: 0.0,
            kv_broadcast_capacity: 1024,
            kv_replay_capacity: 4096,
        }
    }
}

impl EngineParams {
    /// KV capacity in blocks.
    fn capacity_blocks(&self) -> u64 {
        (self.kv_capacity_tokens / u64::from(self.block_size.max(1))).max(1)
    }
}

// ---------------------------------------------------------------------------
// Public request / event types (transport-agnostic)
// ---------------------------------------------------------------------------

/// A request submitted to the engine. The transport (gRPC/HTTP) renders the
/// resulting [`GenEvent`] stream into the wire format the gateway expects.
pub struct NewRequest {
    pub request_id: String,
    /// The prompt's token ids (gRPC supplies real ids from the gateway's
    /// tokenizer; HTTP supplies synthetic ids derived from the prompt text).
    pub prompt_token_ids: Vec<u32>,
    /// Requested output length; 0 means "use the engine default".
    pub max_new: u32,
    /// Sink for this request's generation events.
    pub events: mpsc::UnboundedSender<GenEvent>,
}

/// One unit of generation output for a request.
#[derive(Clone, Debug)]
pub enum GenEvent {
    /// A single decoded token. `prompt_tokens` / `cached_tokens` are constant
    /// for the request and repeated for parity with real engines' chunk fields.
    Token {
        token_id: u32,
        prompt_tokens: u32,
        cached_tokens: u32,
    },
    /// Terminal event; the stream ends after this.
    Done {
        finish_reason: &'static str,
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
    },
}

/// A point-in-time view of engine load, served via `GetLoads` / `/v1/loads`.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadSnapshot {
    pub num_running_reqs: i32,
    pub num_waiting_reqs: i32,
    pub num_waiting_uncached_tokens: i32,
    pub num_used_tokens: i32,
    pub max_total_num_tokens: i32,
    pub max_running_requests: i32,
    pub token_usage: f64,
    pub gen_throughput: f64,
    pub cache_hit_rate: f64,
    /// Cached blocks (referenced or idle) in the KV pool.
    pub num_cached_blocks: i32,
    /// Requests preempted so far (LIFO, on KV exhaustion).
    pub num_preemptions: i64,
    /// KV-event batches produced so far (the current gRPC sequence number).
    pub num_kv_batches: i64,
}

// ---------------------------------------------------------------------------
// Engine handle
// ---------------------------------------------------------------------------

/// Shared state readable from the transport handlers while the actor runs.
struct EngineShared {
    snapshot: RwLock<LoadSnapshot>,
    /// Every published batch, as the publisher task releases it.
    kv_tx: broadcast::Sender<Published>,
    /// Hands batches from the actor to the publisher task with their release time.
    publish_tx: mpsc::UnboundedSender<(Published, Instant)>,
    /// Hands a request's events to the delivery task with their release time
    /// (the per-request overhead); unused when the overhead is zero.
    deliver_tx: mpsc::UnboundedSender<(Instant, mpsc::UnboundedSender<GenEvent>, GenEvent)>,
    kv_replay: Mutex<VecDeque<common::KvEventBatch>>,
    prefix_cache: bool,
    block_size: u32,
    /// Worker name for records and the admin API (`grpc:<port>` / `http:<port>`).
    name: String,
    /// Mirror of the actor's cache block keys, so sibling engines and the
    /// admin API can read it without entering the actor: the fleet oracle
    /// ("which worker holds the most of this prompt") is a read over these.
    cache_mirror: RwLock<HashSet<u64>>,
    /// Fault hooks, switched on through the admin API.
    faults: Faults,
    /// Wakes a paused actor.
    resume: Notify,
}

/// Fault hooks the admin API switches on: batches lost on the wire, delayed
/// publishing, delayed admission, a frozen engine, a publisher restart, and
/// requests answered with an error status, held back or cut mid-stream.
#[derive(Default)]
struct Faults {
    /// Event batches still to lose on the wire.
    drop_batches: AtomicU32,
    dropped_total: AtomicU64,
    /// Publishing delay after a pass ends, in ms.
    delay_ms: AtomicU64,
    /// How long each new request is held before the engine sees it, in ms:
    /// a backlog in transit between the gateway and the engine's queue. The
    /// load record does not count a held request.
    admit_delay_ms: AtomicU64,
    /// At most this many new requests per second enter the engine (0 = no
    /// cap); the rest wait their turn, unseen by the load record: a
    /// throttled input path.
    admit_per_sec: AtomicU64,
    /// The next admission slot under the cap.
    admit_next: tokio::sync::Mutex<Option<Instant>>,
    /// The engine is frozen: no passes until resumed.
    paused: AtomicBool,
    /// Publisher generation; a restart bumps it.
    generation: AtomicU64,
    restarts: AtomicU64,
    /// The request fault, while one is armed.
    fail: Mutex<Option<RequestFault>>,
    /// Requests answered with the fault's status before admission.
    failed_total: AtomicU64,
    /// Streams cut with the fault's status after their first tokens, counted
    /// by the worker paths where they end the stream (`Engine::cut_counter`).
    cut_total: Arc<AtomicU64>,
    /// Requests held back by the fault's stall.
    stalled_total: AtomicU64,
}

/// A request fault the admin API arms: what the worker answers instead of
/// serving, for how long, and whether before admission or after some output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RequestFault {
    /// The HTTP status to answer (the gRPC worker maps it to a status code);
    /// 0 serves the request after the stall.
    pub status: u16,
    /// How long the request is held before the answer.
    pub stall: Duration,
    /// Serve this many tokens, then cut the stream with `status`; `None`
    /// answers before admission, so the request never counts as served.
    pub after_tokens: Option<u32>,
    /// How long the fault stays armed.
    pub scope: FaultScope,
}

/// How long a request fault stays armed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaultScope {
    /// Until cleared.
    Open,
    /// The next N requests.
    Requests(u32),
    /// Every request until this instant.
    Until(Instant),
}

/// What the armed request fault does to one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Injected {
    pub status: u16,
    pub stall: Duration,
    pub after_tokens: Option<u32>,
}

/// The hooks' current state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FaultStatus {
    pub drop_pending: u32,
    pub dropped_total: u64,
    pub delay_ms: u64,
    pub admit_delay_ms: u64,
    pub admit_per_sec: u64,
    pub paused: bool,
    pub generation: u64,
    pub restarts: u64,
    /// The armed request fault, if any.
    pub fail: Option<RequestFault>,
    pub failed_total: u64,
    pub cut_total: u64,
    pub stalled_total: u64,
}

/// What the engine itself would serve from cache for a prompt right now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheTruth {
    pub cached_tokens: u32,
    pub cached_blocks: u32,
    pub block_size: u32,
}

/// A batch as the publisher releases it to every transport.
#[derive(Clone, Debug)]
pub(crate) struct Published {
    pub batch: common::KvEventBatch,
    /// Lost on the wire by a drop hook: not delivered live, kept for replay.
    pub dropped: bool,
    /// The publisher generation the batch belongs to.
    pub generation: u64,
}

/// Messages into the engine actor.
enum EngineMsg {
    Request(NewRequest),
    /// Drop every cached block and announce `AllBlocksCleared` (an engine
    /// restart, as far as the gateway's index is concerned).
    Reset,
    /// The publisher restarts: sequence numbers start over, the replay
    /// buffer is emptied, the cache is kept. Acknowledged once applied.
    RestartPublisher(tokio::sync::oneshot::Sender<()>),
}

/// One admitted request as seen by the engine: the ground truth for routing
/// accuracy. `oracle_tokens` is the best cached prefix any worker of this
/// process held when the request arrived (the router's ideal choice).
#[derive(Clone, Debug)]
pub(crate) struct RequestRecord {
    pub seq: u64,
    pub request_id: String,
    pub worker: String,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub oracle_tokens: u32,
    pub queued_ms: f64,
    pub running_at_admit: u32,
    pub waiting_at_admit: u32,
    pub admitted_unix_ms: u64,
}

/// Every engine in this process, for the fleet oracle and the admin API.
fn fleet() -> &'static Mutex<Vec<Engine>> {
    static FLEET: OnceLock<Mutex<Vec<Engine>>> = OnceLock::new();
    FLEET.get_or_init(|| Mutex::new(Vec::new()))
}

/// Recent admitted-request records across the fleet (a ring buffer).
fn records() -> &'static Mutex<(u64, VecDeque<RequestRecord>)> {
    static RECORDS: OnceLock<Mutex<(u64, VecDeque<RequestRecord>)>> = OnceLock::new();
    RECORDS.get_or_init(|| Mutex::new((0, VecDeque::new())))
}

const RECORD_CAPACITY: usize = 500_000;

/// All engines registered in this process.
pub(crate) fn fleet_engines() -> Vec<Engine> {
    fleet().lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Records with `seq > since`, oldest first, at most `limit`.
pub(crate) fn records_since(since: u64, limit: usize) -> Vec<RequestRecord> {
    let guard = records().lock().unwrap_or_else(|p| p.into_inner());
    guard
        .1
        .iter()
        .filter(|r| r.seq > since)
        .take(limit)
        .cloned()
        .collect()
}

fn push_record(mut record: RequestRecord) {
    let mut guard = records().lock().unwrap_or_else(|p| p.into_inner());
    guard.0 += 1;
    record.seq = guard.0;
    guard.1.push_back(record);
    while guard.1.len() > RECORD_CAPACITY {
        guard.1.pop_front();
    }
}

fn unix_ms(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Seconds since the Unix epoch, as the engines stamp their event batches.
pub(crate) fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A cloneable handle to one simulated engine.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::UnboundedSender<EngineMsg>,
    shared: Arc<EngineShared>,
}

/// Concrete KV-event stream type returned to the gRPC `subscribe_kv_events` RPC.
pub type KvEventStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;

impl Engine {
    /// Spawn the engine actor and return a handle to it.
    ///
    /// The actor task is detached intentionally: when the last [`Engine`] handle
    /// drops, its request channel closes and `run` returns, so there is nothing
    /// to wait on at shutdown.
    pub fn spawn(params: EngineParams) -> Engine {
        Self::spawn_named(params, String::new(), false)
    }

    /// Spawn a named engine; `register` adds it to the process fleet, which
    /// the arrival-time oracle and the admin API read.
    #[expect(
        clippy::disallowed_methods,
        reason = "engine actor self-terminates when its request channel closes"
    )]
    pub fn spawn_named(params: EngineParams, name: String, register: bool) -> Engine {
        let (tx, rx) = mpsc::unbounded_channel();
        let (kv_tx, _) = broadcast::channel(params.kv_broadcast_capacity.max(1));
        let (publish_tx, publish_rx) = mpsc::unbounded_channel();
        let (deliver_tx, deliver_rx) = mpsc::unbounded_channel();
        let shared = Arc::new(EngineShared {
            snapshot: RwLock::new(LoadSnapshot::idle(&params)),
            kv_tx: kv_tx.clone(),
            publish_tx,
            deliver_tx,
            kv_replay: Mutex::new(VecDeque::new()),
            prefix_cache: params.prefix_cache,
            block_size: params.block_size,
            name,
            cache_mirror: RwLock::new(HashSet::new()),
            faults: Faults::default(),
            resume: Notify::new(),
        });
        // The publisher task releases batches in order at their release time
        // (the delay hook) and ends with the actor, which owns its sender.
        tokio::spawn(publish(publish_rx, kv_tx));
        tokio::spawn(deliver(deliver_rx));
        tokio::spawn(run(params, rx, shared.clone()));
        let engine = Engine { tx, shared };
        if register {
            fleet()
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(engine.clone());
        }
        engine
    }

    /// Submit a request. Dropped silently if the engine has shut down.
    pub fn submit(&self, req: NewRequest) {
        let _ = self.tx.send(EngineMsg::Request(req));
    }

    /// Clear the prefix cache and announce it (`AllBlocksCleared`).
    pub fn reset(&self) {
        let _ = self.tx.send(EngineMsg::Reset);
    }

    /// Lose the next `batches` event batches on the wire (they stay in the
    /// replay buffer).
    pub(crate) fn fault_drop(&self, batches: u32) {
        self.shared
            .faults
            .drop_batches
            .store(batches, Ordering::Relaxed);
    }

    /// Publish every batch `ms` milliseconds after its pass ends (0 clears).
    pub(crate) fn fault_delay_ms(&self, ms: u64) {
        self.shared.faults.delay_ms.store(ms, Ordering::Relaxed);
    }

    /// Hold every new request `ms` milliseconds before the engine sees it
    /// (0 clears): the gateway has dispatched it, the load record does not
    /// count it yet.
    pub(crate) fn fault_admit_delay_ms(&self, ms: u64) {
        self.shared
            .faults
            .admit_delay_ms
            .store(ms, Ordering::Relaxed);
    }

    /// Let at most `per_sec` new requests per second into the engine (0
    /// clears); the rest wait their turn, unseen by the load record.
    pub(crate) fn fault_admit_per_sec(&self, per_sec: u64) {
        self.shared
            .faults
            .admit_per_sec
            .store(per_sec, Ordering::Relaxed);
    }

    /// The admission gate a new request passes before the engine sees it:
    /// the hold, then the rate cap. Returns at once while neither is set.
    pub(crate) async fn admit(&self) {
        let faults = &self.shared.faults;
        let hold = Duration::from_millis(faults.admit_delay_ms.load(Ordering::Relaxed));
        if !hold.is_zero() {
            tokio::time::sleep(hold).await;
        }
        let per_sec = faults.admit_per_sec.load(Ordering::Relaxed);
        if per_sec == 0 {
            return;
        }
        let slot = {
            let mut next = faults.admit_next.lock().await;
            let now = Instant::now();
            let at = next.map_or(now, |next| next.max(now));
            *next = Some(at + Duration::from_secs_f64(1.0 / per_sec as f64));
            at
        };
        tokio::time::sleep_until(slot.into()).await;
    }

    /// Restart the publisher: sequence numbers start over, the replay buffer
    /// is emptied, the cache is kept. Returns once the actor has applied it
    /// (after its current pass, at most), so a status read right after sees
    /// the new generation.
    pub(crate) async fn restart_publisher(&self) {
        let (ack, applied) = tokio::sync::oneshot::channel();
        if self.tx.send(EngineMsg::RestartPublisher(ack)).is_ok() {
            let _ = tokio::time::timeout(Duration::from_secs(2), applied).await;
        }
    }

    /// Freeze the engine: no passes, no tokens, no events; requests queue.
    pub(crate) fn pause(&self) {
        self.shared.faults.paused.store(true, Ordering::Relaxed);
    }

    /// Run again after a pause.
    pub(crate) fn resume(&self) {
        self.shared.faults.paused.store(false, Ordering::Relaxed);
        self.shared.resume.notify_one();
    }

    /// Arm a request fault (`None` clears it).
    pub(crate) fn fault_fail(&self, fault: Option<RequestFault>) {
        *self
            .shared
            .faults
            .fail
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = fault;
    }

    /// What the armed request fault does to the request arriving now; `None`
    /// when no fault applies. A `Requests(n)` scope is spent by one, an
    /// elapsed `Until` disarms the fault. A stall is counted here; a status
    /// answered before admission is counted by the worker path that answers
    /// it ([`Engine::record_failure`], after the stall), a cut where the stream
    /// is actually cut (an output no longer than `after_tokens` is never cut).
    pub(crate) fn inject(&self) -> Option<Injected> {
        let faults = &self.shared.faults;
        let mut armed = faults.fail.lock().unwrap_or_else(|p| p.into_inner());
        let fault = (*armed)?;
        let next_scope = match fault.scope {
            FaultScope::Until(at) if Instant::now() >= at => {
                *armed = None;
                return None;
            }
            FaultScope::Requests(n) => n
                .checked_sub(1)
                .filter(|left| *left > 0)
                .map(FaultScope::Requests),
            scope => Some(scope),
        };
        *armed = next_scope.map(|scope| RequestFault { scope, ..fault });
        drop(armed);
        if !fault.stall.is_zero() {
            faults.stalled_total.fetch_add(1, Ordering::Relaxed);
        }
        Some(Injected {
            status: fault.status,
            stall: fault.stall,
            after_tokens: fault.after_tokens,
        })
    }

    /// The count of streams the request fault actually cut; the worker paths
    /// bump it at the point where they end a stream with the fault's status.
    pub(crate) fn cut_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.shared.faults.cut_total)
    }

    /// One request answered with the fault's status before admission; the
    /// worker paths call it at the point where they return that status.
    pub(crate) fn record_failure(&self) {
        self.shared
            .faults
            .failed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn fault_status(&self) -> FaultStatus {
        let f = &self.shared.faults;
        // A timed fault whose deadline passed without another request reads as
        // disarmed, as the next request would find it.
        let fail = {
            let mut armed = f.fail.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(RequestFault {
                scope: FaultScope::Until(at),
                ..
            }) = *armed
            {
                if Instant::now() >= at {
                    *armed = None;
                }
            }
            *armed
        };
        FaultStatus {
            drop_pending: f.drop_batches.load(Ordering::Relaxed),
            dropped_total: f.dropped_total.load(Ordering::Relaxed),
            delay_ms: f.delay_ms.load(Ordering::Relaxed),
            admit_delay_ms: f.admit_delay_ms.load(Ordering::Relaxed),
            admit_per_sec: f.admit_per_sec.load(Ordering::Relaxed),
            paused: f.paused.load(Ordering::Relaxed),
            generation: f.generation.load(Ordering::Relaxed),
            restarts: f.restarts.load(Ordering::Relaxed),
            fail,
            failed_total: f.failed_total.load(Ordering::Relaxed),
            cut_total: f.cut_total.load(Ordering::Relaxed),
            stalled_total: f.stalled_total.load(Ordering::Relaxed),
        }
    }

    /// What this engine would serve from cache for `token_ids` right now:
    /// its own prefix match over the mirror, last block recomputed, as
    /// admission would compute it.
    pub(crate) fn cached_tokens_for(&self, token_ids: &[u32]) -> CacheTruth {
        let bs = self.shared.block_size.max(1);
        let keys = prompt_blocks(token_ids, bs as usize).0;
        let mut matched = self.match_prefix(&keys) as u32;
        if matched > 0 && matched * bs >= token_ids.len() as u32 {
            matched -= 1;
        }
        CacheTruth {
            cached_tokens: matched * bs,
            cached_blocks: matched,
            block_size: bs,
        }
    }

    /// Every batch the publisher releases from now on, with its drop mark and
    /// generation (for transports that keep their own replay buffer).
    pub(crate) fn subscribe_published(&self) -> Pin<Box<dyn Stream<Item = Published> + Send>> {
        let live_rx = self.shared.kv_tx.subscribe();
        Box::pin(stream::unfold(live_rx, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(item) => return Some((item, rx)),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        }))
    }

    /// This engine's name (`grpc:<port>` / `http:<port>`; empty when unnamed).
    pub(crate) fn name(&self) -> &str {
        &self.shared.name
    }

    /// Number of consecutive blocks from the start of `keys` this engine holds.
    fn match_prefix(&self, keys: &[u64]) -> usize {
        let mirror = self
            .shared
            .cache_mirror
            .read()
            .unwrap_or_else(|p| p.into_inner());
        keys.iter().take_while(|k| mirror.contains(k)).count()
    }

    /// A copy of the cached block keys.
    pub fn cache_keys(&self) -> Vec<u64> {
        self.shared
            .cache_mirror
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .copied()
            .collect()
    }

    /// Block keys of a prompt, exactly as the engine computes them.
    pub fn block_keys(prompt_token_ids: &[u32], block_size: usize) -> Vec<u64> {
        prompt_blocks(prompt_token_ids, block_size).0
    }

    /// Current load snapshot.
    pub fn load(&self) -> LoadSnapshot {
        self.shared
            .snapshot
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Whether this engine emits KV-cache events.
    pub(crate) fn kv_enabled(&self) -> bool {
        self.shared.prefix_cache
    }

    /// Build a KV-event stream: replay buffered batches newer than `start_seq`,
    /// then live events. Subscribing to the live channel *before* snapshotting
    /// the replay buffer guarantees no batch falls between the two (the gateway
    /// dedups any overlap by `sequence_number`).
    pub fn subscribe_kv(&self, start_seq: u64) -> KvEventStream {
        let live_rx = self.shared.kv_tx.subscribe();
        let replay: Vec<common::KvEventBatch> = {
            let buf = self
                .shared
                .kv_replay
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            buf.iter()
                .filter(|b| b.sequence_number > start_seq)
                .cloned()
                .collect()
        };
        let replay_stream = stream::iter(replay.into_iter().map(Ok));
        let live_stream = stream::unfold(live_rx, |mut rx| async move {
            loop {
                match rx.recv().await {
                    // A batch lost on the wire (drop hook) never reaches a live
                    // subscriber; it waits in the replay buffer.
                    Ok(item) if item.dropped => continue,
                    Ok(item) => return Some((Ok(item.batch), rx)),
                    // A lagged slow consumer leaves a gap; the gateway detects it
                    // and reconnects, replaying from the buffer. Skip and continue.
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        });
        Box::pin(replay_stream.chain(live_stream))
    }
}

// ---------------------------------------------------------------------------
// The actor loop
// ---------------------------------------------------------------------------

async fn run(
    params: EngineParams,
    mut rx: mpsc::UnboundedReceiver<EngineMsg>,
    shared: Arc<EngineShared>,
) {
    let mut state = SchedulerState::new();
    loop {
        // When there is nothing to do, publish an idle snapshot and block on the
        // request channel — an idle engine consumes no CPU.
        if state.is_idle() {
            *shared.snapshot.write().unwrap_or_else(|p| p.into_inner()) = state.snapshot(&params);
            match rx.recv().await {
                Some(msg) => handle_msg(&mut state, &shared, &params, msg),
                None => return, // all handles dropped
            }
        }
        // Drain any other already-queued submissions without blocking.
        while let Ok(msg) = rx.try_recv() {
            handle_msg(&mut state, &shared, &params, msg);
        }
        // A paused engine takes messages (requests queue, scored at arrival)
        // but runs no pass until resumed.
        while shared.faults.paused.load(Ordering::Relaxed) {
            *shared.snapshot.write().unwrap_or_else(|p| p.into_inner()) = state.snapshot(&params);
            tokio::select! {
                _ = shared.resume.notified() => {}
                msg = rx.recv() => match msg {
                    Some(msg) => handle_msg(&mut state, &shared, &params, msg),
                    None => return,
                },
            }
        }

        let step = state.step(&params);
        if step.duration > Duration::ZERO {
            tokio::time::sleep(step.duration).await;
        }
        // The step's outputs become observable only after its simulated time,
        // plus the fixed per-request overhead when one is calibrated (every
        // event of a stream shifts by the same amount, so order is kept).
        if params.request_overhead_ms > 0.0 {
            let at = Instant::now() + Duration::from_secs_f64(params.request_overhead_ms / 1000.0);
            for (tx, ev) in step.sends {
                let _ = shared.deliver_tx.send((at, tx, ev));
            }
        } else {
            for (tx, ev) in step.sends {
                let _ = tx.send(ev);
            }
        }
        if step.cleared || !step.inserted.is_empty() || !step.evicted.is_empty() {
            let mut mirror = shared
                .cache_mirror
                .write()
                .unwrap_or_else(|p| p.into_inner());
            if step.cleared {
                mirror.clear();
            }
            for k in &step.evicted {
                mirror.remove(k);
            }
            mirror.extend(step.inserted.iter().copied());
        }
        let admitted_at = SystemTime::now();
        for admitted in step.admitted {
            push_record(RequestRecord {
                seq: 0,
                request_id: admitted.request_id,
                worker: shared.name.clone(),
                prompt_tokens: admitted.prompt_tokens,
                cached_tokens: admitted.cached_tokens,
                oracle_tokens: admitted.oracle_tokens,
                queued_ms: admitted.enqueued_at.elapsed().as_secs_f64() * 1000.0,
                running_at_admit: admitted.running_at_admit,
                waiting_at_admit: admitted.waiting_at_admit,
                admitted_unix_ms: unix_ms(admitted_at),
            });
        }
        if let Some(batch) = step.batch {
            {
                let mut buf = shared.kv_replay.lock().unwrap_or_else(|p| p.into_inner());
                buf.push_back(batch.clone());
                while buf.len() > params.kv_replay_capacity {
                    buf.pop_front();
                }
            }
            let faults = &shared.faults;
            let dropped = faults
                .drop_batches
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                .is_ok();
            if dropped {
                faults.dropped_total.fetch_add(1, Ordering::Relaxed);
            }
            let release_at =
                Instant::now() + Duration::from_millis(faults.delay_ms.load(Ordering::Relaxed));
            let item = Published {
                batch,
                dropped,
                generation: faults.generation.load(Ordering::Relaxed),
            };
            let _ = shared.publish_tx.send((item, release_at));
        }
        *shared.snapshot.write().unwrap_or_else(|p| p.into_inner()) = step.snapshot;
    }
}

/// Release batches to every transport in order, each at its release time.
async fn publish(
    mut rx: mpsc::UnboundedReceiver<(Published, Instant)>,
    kv_tx: broadcast::Sender<Published>,
) {
    while let Some((item, release_at)) = rx.recv().await {
        tokio::time::sleep_until(release_at.into()).await;
        let _ = kv_tx.send(item);
    }
}

/// Deliver request events in order, each at its release time.
async fn deliver(
    mut rx: mpsc::UnboundedReceiver<(Instant, mpsc::UnboundedSender<GenEvent>, GenEvent)>,
) {
    while let Some((at, tx, ev)) = rx.recv().await {
        tokio::time::sleep_until(at.into()).await;
        let _ = tx.send(ev);
    }
}

/// Apply one actor message: enqueue a request (scoring the fleet oracle at
/// arrival, before this engine's own cache changes), reset the cache, or
/// restart the publisher.
fn handle_msg(
    state: &mut SchedulerState,
    shared: &Arc<EngineShared>,
    params: &EngineParams,
    msg: EngineMsg,
) {
    match msg {
        EngineMsg::Request(req) => {
            let oracle_tokens = if params.prefix_cache {
                let keys = prompt_blocks(&req.prompt_token_ids, params.block_size as usize).0;
                let fleet_best = fleet_engines()
                    .iter()
                    .map(|e| e.match_prefix(&keys))
                    .max()
                    .unwrap_or(0);
                let own_best = {
                    let own = shared
                        .cache_mirror
                        .read()
                        .unwrap_or_else(|p| p.into_inner());
                    keys.iter().take_while(|k| own.contains(k)).count()
                };
                (fleet_best.max(own_best) as u32 * params.block_size)
                    .min(req.prompt_token_ids.len() as u32)
            } else {
                0
            };
            state.enqueue_with_oracle(req, params, oracle_tokens);
        }
        EngineMsg::Reset => state.reset(),
        EngineMsg::RestartPublisher(ack) => {
            state.restart_publisher();
            shared
                .kv_replay
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clear();
            shared.faults.generation.fetch_add(1, Ordering::Relaxed);
            shared.faults.restarts.fetch_add(1, Ordering::Relaxed);
            let _ = ack.send(());
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler state + pass (pure, deterministic, unit-testable)
// ---------------------------------------------------------------------------

/// A request in the running batch (being prefilled or decoded).
struct RunningReq {
    request_id: String,
    events: mpsc::UnboundedSender<GenEvent>,
    /// Tokens to (re)compute: the prompt, extended by the output generated
    /// before a preemption.
    seq_prompt: Vec<u32>,
    /// Content keys of `seq_prompt`'s full blocks.
    seq_keys: Vec<u64>,
    /// Prompt / cached tokens as reported on the stream (constant per request).
    prompt_tokens: u32,
    cached_tokens: u32,
    /// `seq_prompt` tokens computed so far (cached + prefilled); prefill ends
    /// when it reaches the prompt length.
    computed: u32,
    max_new: u32,
    /// Output tokens produced in total, including those folded into
    /// `seq_prompt` by a preemption.
    generated: u32,
    /// Output tokens already folded into `seq_prompt`.
    resumed_output: u32,
    output_ids: Vec<u32>,
    /// Full blocks this request references, in sequence order.
    held: Vec<u64>,
    /// Whether an anonymous (not yet full) block is allocated for the tail.
    partial: bool,
    /// Tokens of the not-yet-full tail block (the next stored event's payload).
    pending: Vec<u32>,
    /// FNV over every token of the sequence so far (the next block's key).
    rolling_hash: u64,
    /// Per-request seed so decode blocks never collide across requests.
    token_seed: u32,
}

impl RunningReq {
    fn prefilling(&self) -> bool {
        (self.computed as usize) < self.seq_prompt.len()
    }

    /// Tokens of the sequence resident in KV right now.
    fn seq_len(&self) -> u32 {
        if self.prefilling() {
            self.computed
        } else {
            self.seq_prompt.len() as u32 + self.generated - self.resumed_output
        }
    }

    fn finished(&self) -> bool {
        !self.prefilling() && self.generated >= self.max_new
    }
}

/// State a preempted request carries back to the queue.
struct Resume {
    generated: u32,
    output_ids: Vec<u32>,
    cached_tokens: u32,
    token_seed: u32,
}

/// A request admitted to the queue but not yet running.
struct WaitingReq {
    req: NewRequest,
    /// Content keys of the prompt's full blocks and the hash of every token.
    keys: Vec<u64>,
    rolling_hash: u64,
    /// Prompt tokens as reported on the stream (the original prompt).
    prompt_tokens: u32,
    /// Uncached prompt tokens at enqueue time (the queued token-work it adds).
    uncached_tokens: u32,
    /// Best cached prefix any fleet worker held at arrival (ground truth).
    oracle_tokens: u32,
    enqueued_at: Instant,
    /// Set when this is a preempted request coming back for recompute.
    resume: Option<Resume>,
}

/// What a pass reports about a request it admitted for the first time.
struct Admitted {
    request_id: String,
    prompt_tokens: u32,
    cached_tokens: u32,
    oracle_tokens: u32,
    running_at_admit: u32,
    waiting_at_admit: u32,
    enqueued_at: Instant,
}

/// Block-level KV pool: cached blocks keyed by content hash with reference
/// counts, idle (unreferenced) cached blocks in an LRU whose head is evicted
/// first, and anonymous blocks for tails that are not full yet. `allocated`
/// counts every physical block: referenced, idle-cached and anonymous.
#[derive(Default)]
struct BlockPool {
    refs: HashMap<u64, u32>,
    tick_of: HashMap<u64, u64>,
    free_lru: BTreeSet<(u64, u64)>,
    tick: u64,
    allocated: u64,
}

impl BlockPool {
    fn cached(&self) -> usize {
        self.refs.len()
    }

    /// Number of consecutive cached blocks from the start of `keys`.
    fn match_prefix(&self, keys: &[u64]) -> usize {
        keys.iter()
            .take_while(|k| self.refs.contains_key(k))
            .count()
    }

    /// Reserve `n` anonymous blocks, evicting idle cached blocks (LRU first)
    /// as needed. Reserves nothing and returns false when even that is not
    /// enough.
    fn reserve(&mut self, n: u64, capacity: u64, evicted: &mut Vec<u64>) -> bool {
        if !self.fits(n, capacity) {
            return false;
        }
        while capacity.saturating_sub(self.allocated) < n {
            match self.evict_lru() {
                Some(h) => evicted.push(h),
                None => return false,
            }
        }
        self.allocated += n;
        true
    }

    /// Whether `n` more blocks could be reserved now: never-used capacity
    /// plus the idle cached blocks an eviction may take (what vLLM's
    /// free-block count holds).
    fn fits(&self, n: u64, capacity: u64) -> bool {
        capacity.saturating_sub(self.allocated) + self.free_lru.len() as u64 >= n
    }

    /// Reserve without a capacity check (a prompt larger than all of KV on
    /// an otherwise empty engine must still run).
    fn force_reserve(&mut self, n: u64) {
        self.allocated += n;
    }

    fn release_anonymous(&mut self, n: u64) {
        self.allocated = self.allocated.saturating_sub(n);
    }

    /// Take a reference to cached block `h` (leaving the idle LRU if it was there).
    fn hit(&mut self, h: u64) {
        if let Some(r) = self.refs.get_mut(&h) {
            if *r == 0 {
                if let Some(t) = self.tick_of.remove(&h) {
                    self.free_lru.remove(&(t, h));
                }
            }
            *r += 1;
        }
    }

    /// Turn one of the caller's anonymous blocks into cached block `h`.
    /// Returns true when the hash is new (a stored event is due); when it
    /// already exists the anonymous block is given back and a reference taken.
    fn register(&mut self, h: u64) -> bool {
        match self.refs.entry(h) {
            std::collections::hash_map::Entry::Occupied(_) => {
                self.hit(h);
                self.release_anonymous(1);
                false
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(1);
                true
            }
        }
    }

    /// Drop a reference; an unreferenced block becomes idle (evictable).
    fn unref(&mut self, h: u64) {
        if let Some(r) = self.refs.get_mut(&h) {
            *r = r.saturating_sub(1);
            if *r == 0 {
                self.tick += 1;
                self.tick_of.insert(h, self.tick);
                self.free_lru.insert((self.tick, h));
            }
        }
    }

    /// Drop the references a finished (or preempted) request holds, tail
    /// block first, so the head of its prefix carries the newest LRU stamp
    /// and outlives its tail. vLLM's `free()` appends a request's blocks to
    /// the free queue in reverse order for the same reason: a later repeat of
    /// the prefix still finds its first blocks, and the chain behind a lost
    /// head would be unhittable anyway.
    fn unref_request(&mut self, held: &[u64]) {
        for h in held.iter().rev() {
            self.unref(*h);
        }
    }

    /// Evict the least recently idle cached block; returns its hash.
    fn evict_lru(&mut self) -> Option<u64> {
        let &(t, h) = self.free_lru.iter().next()?;
        self.free_lru.remove(&(t, h));
        self.tick_of.remove(&h);
        self.refs.remove(&h);
        self.allocated = self.allocated.saturating_sub(1);
        Some(h)
    }
}

/// The actor-owned scheduler state. `running` stays in admission order, which
/// is what LIFO preemption relies on.
struct SchedulerState {
    running: Vec<RunningReq>,
    waiting: VecDeque<WaitingReq>,
    pool: BlockPool,
    kv_seq: u64,
    kv_event_id: u64,
    gen_tp_ewma: f64,
    cache_hit_ewma: f64,
    preemptions: u64,
    /// A reset was requested; the next pass clears the cache and says so.
    reset_pending: bool,
}

/// The result of one pass.
struct Step {
    duration: Duration,
    sends: Vec<(mpsc::UnboundedSender<GenEvent>, GenEvent)>,
    batch: Option<common::KvEventBatch>,
    snapshot: LoadSnapshot,
    /// Cache deltas this pass, for the actor's mirror.
    inserted: Vec<u64>,
    evicted: Vec<u64>,
    cleared: bool,
    admitted: Vec<Admitted>,
}

/// A block a request completed this pass: its position in the request's
/// block list, its content key and its tokens.
struct Completed {
    index: usize,
    key: u64,
    tokens: Vec<u32>,
}

/// Making room for a running request's next tokens preempted the request
/// itself (it is back in the queue; the pass moves on without it).
struct SelfPreempted;

fn blocks_for(tokens: u32, block_size: u32) -> u64 {
    u64::from(tokens.div_ceil(block_size.max(1)))
}

impl SchedulerState {
    fn new() -> Self {
        Self {
            running: Vec::new(),
            waiting: VecDeque::new(),
            pool: BlockPool::default(),
            kv_seq: 0,
            kv_event_id: 0,
            gen_tp_ewma: 0.0,
            cache_hit_ewma: 0.0,
            preemptions: 0,
            reset_pending: false,
        }
    }

    /// Ask the next pass to clear the cache and announce `AllBlocksCleared`.
    fn reset(&mut self) {
        self.reset_pending = true;
    }

    /// The publisher restarted: batch sequence numbers start over.
    fn restart_publisher(&mut self) {
        self.kv_seq = 0;
    }

    fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty() && !self.reset_pending
    }

    /// Queue a request with no oracle information (tests).
    #[cfg(test)]
    fn enqueue(&mut self, req: NewRequest, p: &EngineParams) {
        self.enqueue_with_oracle(req, p, 0);
    }

    /// Queue a request, recording the queued token-work it contributes, together
    /// with the fleet oracle's cached-token count at arrival (what the
    /// best-informed router could have obtained).
    fn enqueue_with_oracle(&mut self, req: NewRequest, p: &EngineParams, oracle_tokens: u32) {
        let prompt_tokens = req.prompt_token_ids.len() as u32;
        let (keys, rolling_hash, _) = prompt_blocks(&req.prompt_token_ids, p.block_size as usize);
        let cached_blocks = if p.prefix_cache {
            self.pool.match_prefix(&keys)
        } else {
            0
        };
        let cached = (cached_blocks as u32 * p.block_size).min(prompt_tokens);
        self.waiting.push_back(WaitingReq {
            req,
            keys,
            rolling_hash,
            prompt_tokens,
            uncached_tokens: prompt_tokens - cached,
            oracle_tokens,
            enqueued_at: Instant::now(),
            resume: None,
        });
    }

    /// Tokens of running sequences resident in KV (what `token_usage` reports;
    /// idle cached blocks are evictable and not counted, as in SGLang).
    fn active_tokens(&self) -> u64 {
        self.running.iter().map(|r| u64::from(r.seq_len())).sum()
    }

    fn snapshot(&self, p: &EngineParams) -> LoadSnapshot {
        let used = self.active_tokens();
        let waiting_uncached: i64 = self
            .waiting
            .iter()
            .map(|w| i64::from(w.uncached_tokens))
            .sum();
        LoadSnapshot {
            num_running_reqs: self.running.len() as i32,
            num_waiting_reqs: self.waiting.len() as i32,
            num_waiting_uncached_tokens: waiting_uncached.min(i64::from(i32::MAX)) as i32,
            num_used_tokens: used.min(i32::MAX as u64) as i32,
            max_total_num_tokens: p.kv_capacity_tokens.min(i32::MAX as u64) as i32,
            max_running_requests: p.max_running.min(i32::MAX as usize) as i32,
            token_usage: (used as f64 / p.kv_capacity_tokens.max(1) as f64).clamp(0.0, 1.0),
            gen_throughput: self.gen_tp_ewma,
            cache_hit_rate: self.cache_hit_ewma,
            num_cached_blocks: self.pool.cached().min(i32::MAX as usize) as i32,
            num_preemptions: self.preemptions.min(i64::MAX as u64) as i64,
            num_kv_batches: self.kv_seq.min(i64::MAX as u64) as i64,
        }
    }

    /// Run one pass. Pure: mutates state and returns the work produced plus
    /// how long it took, but performs no I/O. Order within the pass follows
    /// vLLM: running requests first (a prefill chunk or one decode token
    /// each), then FCFS admission from the queue while the token budget and
    /// KV room last. Per request, KV events are the blocks evicted for its
    /// allocation (`Removed`) followed by the blocks it completed (`Stored`,
    /// contiguous, chained to a parent). The caller makes every event of the
    /// pass visible at its end.
    fn step(&mut self, p: &EngineParams) -> Step {
        let mut kv: Vec<common::KvCacheEvent> = Vec::new();
        let mut inserted: Vec<u64> = Vec::new();
        let mut evicted: Vec<u64> = Vec::new();
        let mut sends: Vec<(mpsc::UnboundedSender<GenEvent>, GenEvent)> = Vec::new();
        let mut admitted: Vec<Admitted> = Vec::new();
        let bs = p.block_size.max(1);

        // ---- 0. Reset (an engine restart, from the index's point of view) ----
        let cleared = std::mem::take(&mut self.reset_pending);
        if cleared {
            while !self.running.is_empty() {
                self.preempt_last(p);
            }
            self.pool = BlockPool::default();
            if p.prefix_cache {
                self.kv_event_id += 1;
                kv.push(common::KvCacheEvent {
                    event_id: self.kv_event_id,
                    data: Some(common::kv_cache_event::Data::Cleared(
                        common::KvCacheCleared::default(),
                    )),
                });
            }
        }

        let mut budget = p.max_batched_tokens;
        let mut prefill_tokens = 0u32;
        let mut largest_chunk = 0u32;
        let mut num_decode = 0usize;
        let mut decode_tokens = 0u32;
        let mut decode_ctx = 0u64;

        // ---- 1. Running requests: a prefill chunk or one decode token each ----
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].events.is_closed() || self.running[i].finished() {
                i += 1;
                continue;
            }
            if budget == 0 {
                break;
            }
            if self.running[i].prefilling() {
                let remaining = self.running[i].seq_prompt.len() as u32 - self.running[i].computed;
                let chunk = remaining.min(budget);
                let completes = chunk == remaining;
                // The chunk's blocks, plus the first output token's slot when
                // this chunk finishes the prompt.
                let add = chunk + u32::from(completes);
                let mut freed = Vec::new();
                if self.ensure(i, add, p, &mut freed).is_err() {
                    continue;
                }
                evicted.extend(freed.iter().copied());
                self.push_removed(&mut kv, freed, p);
                let stored = self.apply_prefill(i, chunk, p);
                prefill_tokens += chunk;
                largest_chunk = largest_chunk.max(chunk);
                budget -= chunk;
                let mut completed = stored;
                if completes {
                    // The pass that finishes a prefill samples its first token;
                    // that token adds no decode time to the pass.
                    completed.extend(self.emit_token(i, &mut sends, p));
                }
                inserted.extend(completed.iter().map(|c| c.key));
                self.push_stored(&mut kv, i, completed, p);
            } else if !p.prefill_first {
                let mut freed = Vec::new();
                if self.ensure(i, 1, p, &mut freed).is_err() {
                    continue;
                }
                evicted.extend(freed.iter().copied());
                self.push_removed(&mut kv, freed, p);
                decode_ctx += u64::from(self.running[i].seq_len());
                let completed = self.emit_token(i, &mut sends, p);
                num_decode += 1;
                decode_tokens += 1;
                budget -= 1;
                inserted.extend(completed.iter().map(|c| c.key));
                self.push_stored(&mut kv, i, completed, p);
            }
            i += 1;
        }

        // ---- 2. FCFS admission while the budget and KV room last ----
        while budget > 0 && self.running.len() < p.max_running {
            let Some(front) = self.waiting.front() else {
                break;
            };
            let prompt_len = front.req.prompt_token_ids.len() as u32;
            let mut cached_blocks = if p.prefix_cache {
                self.pool.match_prefix(&front.keys)
            } else {
                0
            };
            // A fully cached prompt still recomputes its last block.
            if cached_blocks > 0 && cached_blocks as u32 * bs >= prompt_len {
                cached_blocks -= 1;
            }
            let cached = cached_blocks as u32 * bs;
            let remaining = prompt_len - cached;
            let chunk = remaining.min(budget);
            let completes = chunk == remaining;
            let add = chunk + u32::from(completes);
            // This pass's chunk is what gets allocated. Under full-ISL
            // admission the whole prompt (plus the first output token's slot)
            // must also fit in the free and evictable blocks right now: vLLM's
            // `full_sequence_must_fit` is a gate read at admission, not a
            // reservation, so an admitted prompt holds no more than it has
            // computed and the next chunks allocate (evict, preempt) as they run.
            let need = blocks_for(cached + add, bs) - cached_blocks as u64;
            let need_full = blocks_for(cached + remaining + 1, bs) - cached_blocks as u64;
            // Reference the cached prefix before making room, so the eviction
            // cannot take the very blocks this request is about to reuse.
            for k in &front.keys[..cached_blocks] {
                self.pool.hit(*k);
            }
            let mut freed = Vec::new();
            let room = (!p.reserve_full_isl || self.pool.fits(need_full, p.capacity_blocks()))
                && self.pool.reserve(need, p.capacity_blocks(), &mut freed);
            if !room {
                if self.running.is_empty() {
                    self.pool.force_reserve(need);
                } else {
                    let Some(front) = self.waiting.front() else {
                        break;
                    };
                    for k in &front.keys[..cached_blocks] {
                        self.pool.unref(*k);
                    }
                    break;
                }
            }
            let running_at_admit = self.running.len() as u32;
            let waiting_at_admit = self.waiting.len() as u32;
            let w = self.waiting.pop_front().expect("front exists");
            let NewRequest {
                request_id,
                prompt_token_ids,
                max_new,
                events,
            } = w.req;
            let (reported_cached, generated, resumed_output, output_ids, token_seed) =
                match w.resume {
                    Some(r) => (
                        r.cached_tokens,
                        r.generated,
                        r.generated,
                        r.output_ids,
                        r.token_seed,
                    ),
                    None => {
                        admitted.push(Admitted {
                            request_id: request_id.clone(),
                            prompt_tokens: w.prompt_tokens,
                            cached_tokens: cached,
                            oracle_tokens: w.oracle_tokens.max(cached),
                            running_at_admit,
                            waiting_at_admit,
                            enqueued_at: w.enqueued_at,
                        });
                        let sample = if w.prompt_tokens > 0 {
                            f64::from(cached) / f64::from(w.prompt_tokens)
                        } else {
                            0.0
                        };
                        self.cache_hit_ewma = ewma(self.cache_hit_ewma, sample, 0.2);
                        (cached, 0, 0, Vec::new(), fnv_hash_str(&request_id) as u32)
                    }
                };
            let resolved_max_new = if max_new == 0 {
                p.max_new_default
            } else {
                max_new
            };
            self.running.push(RunningReq {
                request_id,
                events,
                seq_keys: w.keys,
                seq_prompt: prompt_token_ids,
                prompt_tokens: w.prompt_tokens,
                cached_tokens: reported_cached,
                computed: cached,
                max_new: resolved_max_new,
                generated,
                resumed_output,
                output_ids,
                held: Vec::new(),
                partial: false,
                pending: Vec::new(),
                rolling_hash: w.rolling_hash,
                token_seed,
            });
            let idx = self.running.len() - 1;
            self.running[idx].held = self.running[idx].seq_keys[..cached_blocks].to_vec();
            evicted.extend(freed.iter().copied());
            self.push_removed(&mut kv, freed, p);
            let mut completed = self.apply_prefill(idx, chunk, p);
            prefill_tokens += chunk;
            largest_chunk = largest_chunk.max(chunk);
            budget -= chunk;
            if completes {
                completed.extend(self.emit_token(idx, &mut sends, p));
            }
            inserted.extend(completed.iter().map(|c| c.key));
            self.push_stored(&mut kv, idx, completed, p);
        }

        // ---- 3. Prefill-first engines decode only in passes without prefill ----
        if p.prefill_first && prefill_tokens == 0 {
            let mut i = 0;
            while i < self.running.len() {
                if self.running[i].events.is_closed()
                    || self.running[i].finished()
                    || self.running[i].prefilling()
                    || budget == 0
                {
                    i += 1;
                    continue;
                }
                let mut freed = Vec::new();
                if self.ensure(i, 1, p, &mut freed).is_err() {
                    continue;
                }
                evicted.extend(freed.iter().copied());
                self.push_removed(&mut kv, freed, p);
                decode_ctx += u64::from(self.running[i].seq_len());
                let completed = self.emit_token(i, &mut sends, p);
                num_decode += 1;
                decode_tokens += 1;
                budget -= 1;
                inserted.extend(completed.iter().map(|c| c.key));
                self.push_stored(&mut kv, i, completed, p);
                i += 1;
            }
        }

        // ---- 4. Completion: release references, emit the terminal event ----
        let mut still = Vec::with_capacity(self.running.len());
        for r in std::mem::take(&mut self.running) {
            if r.finished() || r.events.is_closed() {
                if r.finished() {
                    sends.push((
                        r.events.clone(),
                        GenEvent::Done {
                            finish_reason: "length",
                            prompt_tokens: r.prompt_tokens,
                            completion_tokens: r.generated,
                            cached_tokens: r.cached_tokens,
                        },
                    ));
                }
                self.release(&r, p);
            } else {
                still.push(r);
            }
        }
        self.running = still;

        // ---- 5. Timing + bookkeeping ----
        let prefill_ms = p.timing.prefill_pass_ms(prefill_tokens, largest_chunk);
        let decode_ms = p.timing.decode_ms(
            num_decode,
            decode_ctx,
            p.decode_reference_tokens.unwrap_or(p.kv_capacity_tokens),
        );
        let mut secs = (prefill_ms + decode_ms) / 1000.0;
        if secs <= 0.0 && !self.is_idle() {
            secs = 0.001; // never busy-spin while work remains
        }
        let throughput_sample = if secs > 0.0 && decode_tokens > 0 {
            f64::from(decode_tokens) / secs
        } else {
            0.0
        };
        self.gen_tp_ewma = ewma(self.gen_tp_ewma, throughput_sample, 0.3);

        let batch = if kv.is_empty() {
            None
        } else {
            self.kv_seq += 1;
            Some(common::KvEventBatch {
                sequence_number: self.kv_seq,
                // Creation time, as the engines stamp their batches: a batch the
                // delay hook holds back keeps it, so the delay shows up as lag.
                timestamp: unix_seconds(),
                events: kv,
                dp_rank: Some(0),
                snapshot: None,
                load: None,
            })
        };

        Step {
            duration: Duration::from_secs_f64(secs.max(0.0)),
            sends,
            batch,
            snapshot: self.snapshot(p),
            inserted,
            evicted,
            cleared,
            admitted,
        }
    }

    /// Make room for `add` more tokens of `running[idx]`, evicting idle cached
    /// blocks first and then preempting the most recently admitted request
    /// (LIFO) until the allocation fits; `Err` when the victim was the request
    /// itself. A lone request that cannot fit even then is over-allocated
    /// rather than deadlocked.
    fn ensure(
        &mut self,
        idx: usize,
        add: u32,
        p: &EngineParams,
        freed: &mut Vec<u64>,
    ) -> Result<(), SelfPreempted> {
        let bs = p.block_size.max(1);
        let before = self.running[idx].seq_len();
        let need = blocks_for(before + add, bs) - blocks_for(before, bs);
        if need == 0 {
            return Ok(());
        }
        loop {
            if self.pool.reserve(need, p.capacity_blocks(), freed) {
                return Ok(());
            }
            if self.running.len() == 1 {
                self.pool.force_reserve(need);
                return Ok(());
            }
            // `running` is in admission order, so the LIFO victim is the last.
            let victim = self.running.len() - 1;
            self.preempt_last(p);
            if victim == idx {
                return Err(SelfPreempted);
            }
        }
    }

    /// Preempt the most recently admitted running request: free its KV and
    /// put it back at the head of the queue to recompute (its output so far
    /// becomes part of the prompt).
    fn preempt_last(&mut self, p: &EngineParams) {
        let Some(r) = self.running.pop() else {
            return;
        };
        self.release(&r, p);
        self.preemptions += 1;
        let mut seq = r.seq_prompt;
        seq.extend_from_slice(&r.output_ids[r.resumed_output as usize..]);
        let (keys, rolling_hash, _) = prompt_blocks(&seq, p.block_size.max(1) as usize);
        let len = seq.len() as u32;
        let cached = if p.prefix_cache {
            (self.pool.match_prefix(&keys) as u32 * p.block_size).min(len)
        } else {
            0
        };
        self.waiting.push_front(WaitingReq {
            req: NewRequest {
                request_id: r.request_id,
                prompt_token_ids: seq,
                max_new: r.max_new,
                events: r.events,
            },
            keys,
            rolling_hash,
            prompt_tokens: r.prompt_tokens,
            uncached_tokens: len - cached,
            oracle_tokens: 0,
            enqueued_at: Instant::now(),
            resume: Some(Resume {
                generated: r.generated,
                output_ids: r.output_ids,
                cached_tokens: r.cached_tokens,
                token_seed: r.token_seed,
            }),
        });
    }

    /// Drop every KV block the request holds. With prefix caching the full
    /// blocks stay cached and become evictable (no events); without it they
    /// were private and simply free.
    fn release(&mut self, r: &RunningReq, p: &EngineParams) {
        if p.prefix_cache {
            self.pool.unref_request(&r.held);
        } else {
            self.pool.release_anonymous(r.held.len() as u64);
        }
        if r.partial {
            self.pool.release_anonymous(1);
        }
    }

    /// Compute `chunk` more prompt tokens of `running[idx]`, registering the
    /// blocks the chunk completes. Returns the newly cached blocks.
    fn apply_prefill(&mut self, idx: usize, chunk: u32, p: &EngineParams) -> Vec<Completed> {
        let bs = p.block_size.max(1) as usize;
        let mut stored = Vec::new();
        let r = &mut self.running[idx];
        let start = r.computed as usize;
        let end = start + chunk as usize;
        r.pending.reserve(bs);
        for pos in start..end {
            r.pending.push(r.seq_prompt[pos]);
            if r.pending.len() == bs {
                let index = r.held.len();
                let key = r.seq_keys[index];
                let tokens = std::mem::take(&mut r.pending);
                r.held.push(key);
                if p.prefix_cache && self.pool.register(key) {
                    stored.push(Completed { index, key, tokens });
                }
            }
        }
        let r = &mut self.running[idx];
        r.computed = end as u32;
        r.partial = !r.pending.is_empty();
        stored
    }

    /// Generate one token for `running[idx]`, registering a block when the
    /// tail fills. Returns the newly cached blocks.
    fn emit_token(
        &mut self,
        idx: usize,
        sends: &mut Vec<(mpsc::UnboundedSender<GenEvent>, GenEvent)>,
        p: &EngineParams,
    ) -> Vec<Completed> {
        let bs = p.block_size.max(1) as usize;
        let mut stored = Vec::new();
        let r = &mut self.running[idx];
        let token_id = next_token(r);
        r.generated += 1;
        r.output_ids.push(token_id);
        r.rolling_hash = fnv_step(r.rolling_hash, token_id);
        r.pending.reserve(bs);
        r.pending.push(token_id);
        if r.pending.len() == bs {
            let index = r.held.len();
            let key = r.rolling_hash;
            let tokens = std::mem::take(&mut r.pending);
            r.held.push(key);
            if p.prefix_cache && self.pool.register(key) {
                stored.push(Completed { index, key, tokens });
            }
        }
        let r = &mut self.running[idx];
        r.partial = !r.pending.is_empty();
        sends.push((
            r.events.clone(),
            GenEvent::Token {
                token_id,
                prompt_tokens: r.prompt_tokens,
                cached_tokens: r.cached_tokens,
            },
        ));
        stored
    }

    fn push_removed(
        &mut self,
        kv: &mut Vec<common::KvCacheEvent>,
        freed: Vec<u64>,
        p: &EngineParams,
    ) {
        if freed.is_empty() || !p.prefix_cache {
            return;
        }
        self.kv_event_id += 1;
        kv.push(common::KvCacheEvent {
            event_id: self.kv_event_id,
            data: Some(common::kv_cache_event::Data::Removed(
                common::KvBlocksRemoved {
                    block_hashes: freed.into_iter().map(|k| k as i64).collect(),
                    cache_level: None,
                    ..Default::default()
                },
            )),
        });
    }

    /// `Stored` events for the blocks `running[idx]` completed this pass: one
    /// per contiguous run, chained to the block before the run's first.
    fn push_stored(
        &mut self,
        kv: &mut Vec<common::KvCacheEvent>,
        idx: usize,
        completed: Vec<Completed>,
        p: &EngineParams,
    ) {
        if completed.is_empty() || !p.prefix_cache {
            return;
        }
        let held = &self.running[idx].held;
        let mut runs: Vec<(Option<u64>, Vec<common::KvBlock>)> = Vec::new();
        let mut last_index: Option<usize> = None;
        for c in completed {
            let contiguous = last_index.is_some_and(|prev| prev + 1 == c.index);
            if !contiguous || runs.is_empty() {
                let parent = c.index.checked_sub(1).map(|pos| held[pos]);
                runs.push((parent, Vec::new()));
            }
            last_index = Some(c.index);
            runs.last_mut()
                .expect("run exists")
                .1
                .push(common::KvBlock {
                    block_hash: c.key as i64,
                    token_ids: c.tokens,
                    block_size: p.block_size as i32,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                });
        }
        for (parent, blocks) in runs {
            self.kv_event_id += 1;
            kv.push(common::KvCacheEvent {
                event_id: self.kv_event_id,
                data: Some(common::kv_cache_event::Data::Stored(
                    common::KvBlocksStored {
                        blocks,
                        parent_block_hash: parent.map(|k| k as i64),
                        ..Default::default()
                    },
                )),
            });
        }
    }
}

/// Which backend's load report the transports imitate. The vLLM servicer
/// fills only `num_running_reqs`, `num_waiting_reqs`, `token_usage` and the
/// maxima; the gateway's expected-wait then uses its default throughput and
/// `waiting_reqs × mean prefill` instead of the queued token-work and live
/// throughput the mock knows. Matching that makes routing on the mock agree
/// with routing on a vLLM fleet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadsLike {
    /// Everything the simulator knows (SGLang-style report).
    #[default]
    Mock,
    /// Only what the vLLM servicer reports.
    Vllm,
}

impl std::str::FromStr for LoadsLike {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mock" => Ok(Self::Mock),
            "vllm" => Ok(Self::Vllm),
            other => Err(format!("--loads-like must be mock|vllm, got {other}")),
        }
    }
}

impl LoadSnapshot {
    /// The snapshot as a backend of kind `like` would report it.
    pub(crate) fn as_reported_by(&self, like: LoadsLike) -> Self {
        match like {
            LoadsLike::Mock => self.clone(),
            LoadsLike::Vllm => Self {
                num_waiting_uncached_tokens: 0,
                num_used_tokens: 0,
                gen_throughput: 0.0,
                cache_hit_rate: 0.0,
                ..self.clone()
            },
        }
    }

    fn idle(p: &EngineParams) -> Self {
        Self {
            num_running_reqs: 0,
            num_waiting_reqs: 0,
            num_waiting_uncached_tokens: 0,
            num_used_tokens: 0,
            max_total_num_tokens: p.kv_capacity_tokens.min(i32::MAX as u64) as i32,
            max_running_requests: p.max_running.min(i32::MAX as usize) as i32,
            token_usage: 0.0,
            gen_throughput: 0.0,
            cache_hit_rate: 0.0,
            num_cached_blocks: 0,
            num_preemptions: 0,
            num_kv_batches: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_step(mut h: u64, token: u32) -> u64 {
    for b in token.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn fnv_hash_str(s: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Chunk `ids` into `block_size`-token blocks, returning each full block's
/// cumulative-prefix content key, the rolling hash after all tokens, and the
/// trailing partial block. The partial (< block_size) tail is never a block —
/// matching real engines, which only cache full pages.
fn prompt_blocks(ids: &[u32], block_size: usize) -> (Vec<u64>, u64, Vec<u32>) {
    let mut h = FNV_OFFSET;
    let mut keys = Vec::new();
    let mut pending = Vec::new();
    for &t in ids {
        h = fnv_step(h, t);
        pending.push(t);
        if block_size > 0 && pending.len() == block_size {
            keys.push(h);
            pending.clear();
        }
    }
    (keys, h, pending)
}

/// A synthetic, request-specific output token id. Decode blocks must never
/// collide across requests (only shared *prompt* prefixes should match), so the
/// id is derived from the request seed and position.
fn next_token(r: &RunningReq) -> u32 {
    let mixed = r
        .token_seed
        .wrapping_add(r.generated)
        .wrapping_mul(2_654_435_761);
    100 + (mixed % 30_000)
}

fn ewma(prev: f64, sample: f64, alpha: f64) -> f64 {
    alpha * sample + (1.0 - alpha) * prev
}

// ---------------------------------------------------------------------------
// Tests — drive the pure `step()` directly, no real timers.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Make a request plus a receiver to observe its events.
    fn req(
        id: &str,
        prompt: Vec<u32>,
        max_new: u32,
    ) -> (NewRequest, mpsc::UnboundedReceiver<GenEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            NewRequest {
                request_id: id.to_string(),
                prompt_token_ids: prompt,
                max_new,
                events: tx,
            },
            rx,
        )
    }

    /// Run passes until the given request emits its first Token, returning the
    /// accumulated simulated time (TTFT) and pass count.
    fn run_to_first_token(
        st: &mut SchedulerState,
        p: &EngineParams,
        rx: &mut mpsc::UnboundedReceiver<GenEvent>,
    ) -> (Duration, u32) {
        let mut total = Duration::ZERO;
        for n in 1..100_000 {
            let step = st.step(p);
            total += step.duration;
            for (tx, ev) in step.sends {
                let _ = tx.send(ev);
            }
            if let Ok(GenEvent::Token { .. }) = rx.try_recv() {
                return (total, n);
            }
        }
        panic!("no token produced");
    }

    fn stored_events(batch: &common::KvEventBatch) -> Vec<&common::KvBlocksStored> {
        batch
            .events
            .iter()
            .filter_map(|e| match &e.data {
                Some(common::kv_cache_event::Data::Stored(s)) => Some(s),
                _ => None,
            })
            .collect()
    }

    fn has_removed(batch: &common::KvEventBatch) -> bool {
        batch
            .events
            .iter()
            .any(|e| matches!(e.data, Some(common::kv_cache_event::Data::Removed(_))))
    }

    #[test]
    fn reset_clears_cache_and_publishes_cleared() {
        let p = EngineParams::default();
        let mut st = SchedulerState::new();
        let (r, rx) = req("a", vec![3; 64], 2);
        st.enqueue(r, &p);
        let step = st.step(&p);
        assert_eq!(
            step.inserted.len(),
            4,
            "64 tokens store four 16-token blocks"
        );
        assert!(!step.cleared);
        assert_ne!(st.pool.cached(), 0);

        st.reset();
        assert!(!st.is_idle(), "a pending reset keeps the actor stepping");
        let step = st.step(&p);
        assert!(step.cleared, "the reset pass reports the clear");
        assert!(
            step.batch.as_ref().is_some_and(|b| b
                .events
                .iter()
                .any(|e| matches!(e.data, Some(common::kv_cache_event::Data::Cleared(_))))),
            "the reset publishes AllBlocksCleared to subscribers"
        );
        assert!(
            step.batch.as_ref().is_some_and(|b| matches!(
                b.events[0].data,
                Some(common::kv_cache_event::Data::Cleared(_))
            )),
            "the clear precedes the recompute's stored events"
        );
        drop(rx);
    }

    #[test]
    fn admitted_records_carry_cached_and_oracle_tokens() {
        let p = EngineParams::default();
        let mut st = SchedulerState::new();
        let (r1, rx1) = req("first", vec![9; 64], 1);
        st.enqueue_with_oracle(r1, &p, 48);
        let step = st.step(&p);
        assert_eq!(step.admitted.len(), 1);
        assert_eq!(step.admitted[0].prompt_tokens, 64);
        assert_eq!(
            step.admitted[0].cached_tokens, 0,
            "a cold cache serves nothing"
        );
        assert_eq!(
            step.admitted[0].oracle_tokens, 48,
            "the oracle is what the fleet could have served"
        );

        // The same prompt again: every block is cached, and the last one is
        // recomputed anyway (vLLM's rule), so 48 of 64 tokens are served.
        let (r2, rx2) = req("second", vec![9; 64], 1);
        st.enqueue_with_oracle(r2, &p, 0);
        let step = st.step(&p);
        let admitted = step
            .admitted
            .iter()
            .find(|a| a.request_id == "second")
            .expect("second request admitted");
        assert_eq!(admitted.cached_tokens, 48);
        assert_eq!(
            admitted.oracle_tokens, 48,
            "the oracle is never below what the worker actually served"
        );
        drop((rx1, rx2));
    }

    #[test]
    fn block_keys_are_deterministic_and_prefix_stable() {
        let a = Engine::block_keys(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        let b = Engine::block_keys(&[1, 2, 3, 4, 9, 9, 9, 9], 4);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], b[0], "a shared first block hashes the same");
        assert_ne!(a[1], b[1], "a different second block hashes differently");
        assert_eq!(a, Engine::block_keys(&[1, 2, 3, 4, 5, 6, 7, 8], 4));
    }

    #[test]
    fn ttft_scales_with_uncached_prompt_length() {
        let p = EngineParams {
            prefix_cache: false,
            ..Default::default()
        };
        let mut s1 = SchedulerState::new();
        let (r1, mut rx1) = req("a", vec![7; 64], 4);
        s1.enqueue(r1, &p);
        let (short, _) = run_to_first_token(&mut s1, &p, &mut rx1);

        let mut s2 = SchedulerState::new();
        let (r2, mut rx2) = req("b", vec![7; 8192], 4);
        s2.enqueue(r2, &p);
        let (long, _) = run_to_first_token(&mut s2, &p, &mut rx2);

        assert!(
            long > short * 5,
            "TTFT should grow with prompt size: short={short:?} long={long:?}"
        );
    }

    #[test]
    fn pass_time_follows_the_polynomials() {
        let p = EngineParams::default();
        let mut st = SchedulerState::new();
        let (r, rx) = req("a", vec![7; 1024], 4);
        st.enqueue(r, &p);
        // One pass prefills 1024 uncached tokens: 16.50 + 15.55 + 0.44 ms.
        let prefill = st.step(&p).duration.as_secs_f64() * 1000.0;
        assert!((prefill - 32.49).abs() < 0.1, "prefill pass {prefill} ms");
        // A lone decoder at ~0 utilisation: 5.74 ms plus a sliver of 54u.
        let decode = st.step(&p).duration.as_secs_f64() * 1000.0;
        assert!((decode - 5.85).abs() < 0.1, "decode pass {decode} ms");
        drop(rx);
    }

    #[test]
    fn itl_grows_with_batch_size() {
        let p = EngineParams {
            timing: TimingModel::linear(),
            prefix_cache: false,
            ..Default::default()
        };

        let decode_step_duration = |n: usize| -> Duration {
            let mut st = SchedulerState::new();
            // Keep receivers alive: a dropped receiver looks like a disconnected
            // client and the engine would abort the request.
            let mut rxs = Vec::new();
            for i in 0..n {
                let (r, rx) = req(&format!("r{i}"), vec![1, 2, 3, 4], 8);
                st.enqueue(r, &p);
                rxs.push(rx);
            }
            st.step(&p); // admit + prefill + first tokens
            let duration = st.step(&p).duration; // a pure decode pass
            drop(rxs);
            duration
        };

        let one = decode_step_duration(1);
        let many = decode_step_duration(64);
        assert!(
            many > one,
            "decode pass should be slower with a bigger batch: one={one:?} many={many:?}"
        );
    }

    #[test]
    fn a_calibrated_decode_reference_keeps_step_time_when_the_pool_shrinks() {
        // The same 2048-token decoder: against a 4096-token pool the step
        // reads u = 0.5; with the pool cut to 2560 tokens but the decode
        // calibrated at 4096, the step must cost the same as before (the
        // pool changed room, not the engine's speed); without the reference
        // the smaller pool would read u = 0.8 and decode slower.
        let full = EngineParams {
            kv_capacity_tokens: 4096,
            block_size: 16,
            ..Default::default()
        };
        let small_pinned = EngineParams {
            kv_capacity_tokens: 2560,
            decode_reference_tokens: Some(4096),
            ..full.clone()
        };
        let small_unpinned = EngineParams {
            decode_reference_tokens: None,
            ..small_pinned.clone()
        };
        let step_time = |p: &EngineParams| {
            let mut st = SchedulerState::new();
            let (r, _rx) = req("big", vec![1; 2048], 8);
            st.enqueue(r, p);
            st.step(p);
            st.step(p).duration
        };
        assert_eq!(step_time(&full), step_time(&small_pinned));
        assert!(step_time(&small_unpinned) > step_time(&full));
    }

    #[test]
    fn decode_slows_with_kv_utilisation() {
        // Polynomial decode depends on the decoding requests' context over
        // capacity, not on the batch width.
        let p = EngineParams {
            kv_capacity_tokens: 4096,
            block_size: 16,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (r, rx) = req("big", vec![1; 2048], 8);
        st.enqueue(r, &p);
        st.step(&p);
        let busy = st.step(&p).duration;
        let mut st2 = SchedulerState::new();
        let (r2, rx2) = req("small", vec![1; 16], 8);
        st2.enqueue(r2, &p);
        st2.step(&p);
        let light = st2.step(&p).duration;
        assert!(
            busy > light,
            "u=0.5 should decode slower than u~0: {busy:?} vs {light:?}"
        );
        drop((rx, rx2));
    }

    #[test]
    fn shared_prefix_yields_cached_tokens() {
        let p = EngineParams {
            block_size: 4,
            ..Default::default()
        };
        let mut st = SchedulerState::new();

        // First request stores its prompt blocks.
        let prompt: Vec<u32> = (0..16).collect(); // 4 full blocks of 4
        let (r1, mut rx1) = req("first", prompt.clone(), 2);
        st.enqueue(r1, &p);
        for _ in 0..50 {
            let step = st.step(&p);
            for (tx, ev) in step.sends {
                let _ = tx.send(ev);
            }
        }

        // Second request shares the whole prompt: every block is cached, the
        // last one is recomputed anyway, so 12 of 16 tokens come from cache.
        let (r2, mut rx2) = req("second", prompt, 2);
        st.enqueue(r2, &p);
        st.step(&p); // admission computes the cache hit
        let _ = &mut rx1;

        let mut saw_cached = false;
        for _ in 0..50 {
            let step = st.step(&p);
            for (tx, ev) in step.sends {
                let _ = tx.send(ev);
            }
            while let Ok(ev) = rx2.try_recv() {
                if let GenEvent::Token { cached_tokens, .. } = ev {
                    assert_eq!(
                        cached_tokens, 12,
                        "all but the last block served from cache"
                    );
                    saw_cached = true;
                }
            }
        }
        assert!(saw_cached, "second request should report cached tokens");
    }

    #[test]
    fn saturation_produces_queued_token_work() {
        let p = EngineParams {
            max_running: 1,
            prefix_cache: false,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let mut rxs = Vec::new(); // keep receivers alive (see itl test)
        for i in 0..3 {
            let (r, rx) = req(&format!("r{i}"), vec![5; 100], 32);
            st.enqueue(r, &p);
            rxs.push(rx);
        }
        let step = st.step(&p); // admit only 1; 2 remain queued
        assert_eq!(step.snapshot.num_running_reqs, 1);
        assert_eq!(step.snapshot.num_waiting_reqs, 2);
        assert_eq!(step.snapshot.num_waiting_uncached_tokens, 200);
        drop(rxs);
    }

    #[test]
    fn token_budget_chunks_prefill_across_passes() {
        let p = EngineParams {
            max_batched_tokens: 1000,
            prefix_cache: false,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (r, mut rx) = req("long", vec![7; 2500], 1);
        st.enqueue(r, &p);
        let (_, passes) = run_to_first_token(&mut st, &p, &mut rx);
        assert_eq!(
            passes, 3,
            "2500 tokens at a 1000-token budget take three passes"
        );
    }

    #[test]
    fn calibration_file_is_read_in_either_spelling() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"prefill_ms": {"a": 20.0, "b": 0.01, "c": 1e-7}, "decode_ms": {"d": 7.0, "e": 40.0, "f": -10.0},
                "kv_capacity_blocks": 6144, "block_size": 64, "request_overhead_ms": 12.5, "engine": "vllm"}"#,
        )
        .unwrap();
        let c = Calibration::from_value(&v).unwrap();
        assert_eq!(c.prefill, [20.0, 0.01, 1e-7]);
        assert_eq!(c.decode, [7.0, 40.0, -10.0]);
        assert_eq!(c.kv_capacity_tokens, Some(6144 * 64));
        assert_eq!(c.block_size, Some(64));
        assert_eq!(c.request_overhead_ms, 12.5);
        assert_eq!(
            TimingModel::fitted(&c),
            TimingModel::Polynomial {
                prefill: [20.0, 0.01, 1e-7],
                decode: [7.0, 40.0, -10.0]
            }
        );

        let arrays: serde_json::Value = serde_json::from_str(
            r#"{"prefill": [16.5, 0.015, 4e-7], "decode": [5.7, 54.0, -25.7], "kv_capacity_tokens": 393216}"#,
        )
        .unwrap();
        let c = Calibration::from_value(&arrays).unwrap();
        assert_eq!(c.kv_capacity_tokens, Some(393_216));
        assert_eq!(c.block_size, None);
        assert_eq!(c.request_overhead_ms, 0.0);

        let bad: serde_json::Value = serde_json::from_str(r#"{"decode": [1, 2, 3]}"#).unwrap();
        assert!(Calibration::from_value(&bad)
            .unwrap_err()
            .contains("prefill"));
    }

    #[test]
    fn calibrated_prefill_uses_the_table_for_one_request_and_the_pass_form_for_a_batch() {
        let model = TimingModel::Calibrated {
            table: vec![
                (1.0, 24.0),
                (128.0, 37.0),
                (512.0, 82.0),
                (1024.0, 75.0),
                (4096.0, 98.0),
                (8192.0, 106.0),
                (16384.0, 211.0),
            ],
            pass_intercept_ms: 35.0,
            pass_ms_per_token: 0.0095,
            decode: TimingModel::POLY_DECODE,
        };
        let close = |a: f64, b: f64| (a - b).abs() < 0.5;
        // A lone request: its interpolated table value, whatever the pass form says.
        assert!(close(model.prefill_pass_ms(1024, 1024), 75.0));
        assert!(
            close(model.prefill_pass_ms(768, 768), 78.5),
            "{}",
            model.prefill_pass_ms(768, 768)
        );
        assert!(
            close(model.prefill_pass_ms(64, 64), 30.45),
            "{}",
            model.prefill_pass_ms(64, 64)
        );
        // Batched: four 1024-token prompts take the pass form once it exceeds the plateau.
        assert!(close(
            model.prefill_pass_ms(4096, 1024),
            75.0_f64.max(35.0 + 0.0095 * 4096.0)
        ));
        assert!(close(
            model.prefill_pass_ms(16384, 1024),
            35.0 + 0.0095 * 16384.0
        ));
        // One 16k request: its own table point wins over the pass form.
        assert!(close(model.prefill_pass_ms(16384, 16384), 211.0));
        // Beyond the table: the last slope.
        let slope = (211.0 - 106.0) / (16384.0 - 8192.0);
        assert!(close(
            model.prefill_pass_ms(20000, 20000),
            211.0 + slope * (20000.0 - 16384.0)
        ));
        assert!(close(model.prefill_ms(0), 0.0));
    }

    #[test]
    fn calibration_table_and_pass_form_are_read() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"prefill_points_ms": {"1": {"median_ms": 23.84, "min_ms": 23.27}, "1024": {"median_ms": 74.94}, "128": {"median_ms": 37.33}},
                "prefill_pass_ms": {"intercept_ms": 38.3, "ms_per_token": 0.0102},
                "decode_fit_vs_utilisation_ms": {"d_ms": 3.76, "e_ms_per_u": 15.24, "f_ms_per_u2": -2.87},
                "kv_capacity_tokens": 676128}"#,
        )
        .unwrap();
        let c = Calibration::from_value(&v).unwrap();
        assert_eq!(
            c.prefill_table.as_deref(),
            Some(&[(1.0, 23.84), (128.0, 37.33), (1024.0, 74.94)][..]),
            "sorted by tokens"
        );
        assert_eq!(c.prefill_pass, Some((38.3, 0.0102)));
        assert!(matches!(
            TimingModel::fitted(&c),
            TimingModel::Calibrated { pass_intercept_ms, .. } if (pass_intercept_ms - 38.3).abs() < 1e-9
        ));
        let canonical: serde_json::Value = serde_json::from_str(
            r#"{"prefill_table_ms": [[1, 24], [4096, 98]], "decode_ms": [3.76, 15.24, -2.87]}"#,
        )
        .unwrap();
        let c = Calibration::from_value(&canonical).unwrap();
        assert_eq!(
            c.prefill_table.as_deref(),
            Some(&[(1.0, 24.0), (4096.0, 98.0)][..])
        );
        assert_eq!(
            c.prefill, [0.0; 3],
            "a table alone is a complete prefill model"
        );
        assert!(matches!(
            TimingModel::fitted(&c),
            TimingModel::Calibrated { .. }
        ));
    }

    #[test]
    fn vllm_like_loads_drop_what_the_vllm_servicer_does_not_report() {
        let full = LoadSnapshot {
            num_running_reqs: 30,
            num_waiting_reqs: 2,
            num_waiting_uncached_tokens: 20_000,
            num_used_tokens: 400_000,
            max_total_num_tokens: 676_128,
            max_running_requests: 128,
            token_usage: 0.59,
            gen_throughput: 3_000.0,
            cache_hit_rate: 0.4,
            num_cached_blocks: 42_000,
            num_preemptions: 0,
            num_kv_batches: 10,
        };
        assert_eq!(full.as_reported_by(LoadsLike::Mock), full);
        let v = full.as_reported_by(LoadsLike::Vllm);
        assert_eq!((v.num_running_reqs, v.num_waiting_reqs), (30, 2));
        assert_eq!(
            (v.max_total_num_tokens, v.max_running_requests),
            (676_128, 128)
        );
        assert_eq!(v.token_usage, 0.59);
        assert_eq!(
            (v.num_waiting_uncached_tokens, v.num_used_tokens),
            (0, 0),
            "no queued token-work or used-token count"
        );
        assert_eq!((v.gen_throughput, v.cache_hit_rate), (0.0, 0.0));
        assert_eq!("vllm".parse::<LoadsLike>(), Ok(LoadsLike::Vllm));
        assert!("other".parse::<LoadsLike>().is_err());
    }

    #[test]
    fn calibrated_model_reproduces_a_measured_batched_sweep() {
        // A hardware calibration: the table of single-request medians and the
        // pass form fitted to its batched sweep.
        let model = TimingModel::Calibrated {
            table: vec![
                (1.0, 23.84),
                (128.0, 37.33),
                (256.0, 48.16),
                (512.0, 81.72),
                (1024.0, 74.94),
                (2048.0, 76.98),
                (3072.0, 76.44),
                (4096.0, 98.39),
                (6144.0, 103.1),
                (8192.0, 105.73),
                (12288.0, 157.74),
                (16384.0, 211.04),
            ],
            pass_intercept_ms: 36.12,
            pass_ms_per_token: 0.01028,
            decode: [3.7646, 15.2448, -2.8663],
        };
        let within = |value: f64, lo: f64, hi: f64| (lo..=hi).contains(&value);
        // Measured: 4x1024 in 79.6-80.8 ms, 8x1024 in 114.8-121.8 ms, 16x1024 in 200-210 ms.
        assert!(within(model.prefill_pass_ms(4096, 1024), 74.0, 86.0));
        assert!(within(model.prefill_pass_ms(8192, 1024), 110.0, 126.0));
        assert!(within(model.prefill_pass_ms(16384, 1024), 195.0, 215.0));
        // The single-request plateau: 512-3072 tokens cost 75-82 ms alone.
        assert!(within(model.prefill_pass_ms(2048, 2048), 74.0, 82.0));
        // 8x4096 is two passes of 16384 on a 16384-token budget: 409 ms by the
        // pass form against ~335 ms measured; the second pass is cheaper on the
        // hardware than the form says (recorded, not matched).
        assert!(within(
            2.0 * model.prefill_pass_ms(16384, 4096),
            390.0,
            430.0
        ));
    }

    #[test]
    fn calibration_reads_the_gpu_harness_layout() {
        // A calibration file as the GPU harness writes it, abridged.
        let harness: serde_json::Value = serde_json::from_str(
            r#"{"target": "127.0.0.1:20061", "kv_capacity_tokens": 676128,
                "prefill_points_ms": {"1": {"median_ms": 23.84}},
                "prefill_fit_ms": {"a_ms": 35.24, "b_ms_per_token": 0.005654, "c_ms_per_token2": 2.076e-07,
                                   "fixed_overhead_ms": 23.84, "residual_ms": [-22.48]},
                "decode_fit_vs_utilisation_ms": {"d_ms": 3.7645, "e_ms_per_u": 15.2448, "f_ms_per_u2": -2.8663,
                                                 "u": "KV tokens in use / capacity (computed)"},
                "decode_fit_vs_batch_ms": {"d_ms": 3.8, "e_ms_per_seq": 0.028, "f_ms_per_seq2": 1.3e-05}}"#,
        )
        .unwrap();
        let c = Calibration::from_value(&harness).unwrap();
        assert_eq!(c.prefill, [35.24, 0.005654, 2.076e-07]);
        assert_eq!(c.decode, [3.7645, 15.2448, -2.8663]);
        assert_eq!(c.kv_capacity_tokens, Some(676_128));
        assert_eq!(c.block_size, None);
        assert_eq!(
            c.request_overhead_ms, 0.0,
            "the one-token TTFT is in the prefill intercept already"
        );
    }

    #[tokio::test]
    async fn request_overhead_delays_the_stream_without_stretching_it() {
        let quick = Engine::spawn(EngineParams::default());
        let slow = Engine::spawn(EngineParams {
            request_overhead_ms: 300.0,
            ..Default::default()
        });
        async fn first_and_second(engine: &Engine) -> (Duration, Duration) {
            let (r, mut rx) = req("a", vec![1; 32], 3);
            let t0 = Instant::now();
            engine.submit(r);
            let mut times = Vec::new();
            while times.len() < 2 {
                let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("events")
                    .expect("open");
                if matches!(ev, GenEvent::Token { .. }) {
                    times.push(t0.elapsed());
                }
            }
            (times[0], times[1] - times[0])
        }
        let (ttft_quick, itl_quick) = first_and_second(&quick).await;
        let (ttft_slow, itl_slow) = first_and_second(&slow).await;
        assert!(
            ttft_slow >= ttft_quick + Duration::from_millis(250),
            "the overhead adds to TTFT: {ttft_quick:?} vs {ttft_slow:?}"
        );
        assert!(
            itl_slow < itl_quick + Duration::from_millis(50),
            "and not to the inter-token gap: {itl_quick:?} vs {itl_slow:?}"
        );
    }

    #[test]
    fn batches_are_stamped_with_their_creation_time() {
        let p = EngineParams::default();
        let mut st = SchedulerState::new();
        let (r, _rx) = req("x", vec![1; 32], 1);
        st.enqueue(r, &p);
        let batch = st.step(&p).batch.expect("a batch");
        let age = unix_seconds() - batch.timestamp;
        assert!(
            (0.0..5.0).contains(&age),
            "fresh wall-clock stamp: {age} s old"
        );
    }

    #[test]
    fn prompt_blocks_emit_one_chained_stored_event() {
        let p = EngineParams {
            block_size: 4,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (r, _rx) = req("x", (0..8).collect(), 1); // 2 full blocks
        st.enqueue(r, &p);
        let step = st.step(&p);
        let batch = step.batch.expect("stored events expected");
        assert_eq!(batch.sequence_number, 1);
        let stored = stored_events(&batch);
        assert_eq!(stored.len(), 1, "contiguous blocks share one event");
        assert_eq!(stored[0].blocks.len(), 2, "two prompt blocks");
        assert!(
            stored[0].parent_block_hash.is_none(),
            "first block has no parent"
        );
        assert_eq!(stored[0].blocks[0].token_ids, vec![0, 1, 2, 3]);
        assert_eq!(stored[0].blocks[1].token_ids, vec![4, 5, 6, 7]);

        // A longer prompt sharing the prefix stores only its tail, chained to
        // the last cached block.
        let (r2, _rx2) = req("y", (0..12).collect(), 1);
        st.enqueue(r2, &p);
        let step = st.step(&p);
        let batch = step.batch.expect("stored events expected");
        let stored = stored_events(&batch);
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].blocks.len(), 1, "only the third block is new");
        let second_key = Engine::block_keys(&(0..8).collect::<Vec<_>>(), 4)[1];
        assert_eq!(stored[0].parent_block_hash, Some(second_key as i64));
    }

    #[test]
    fn a_finished_request_frees_its_blocks_tail_first() {
        // Request A held blocks 1-4; request B shared the prefix 1-2 and added 5.
        let mut pool = BlockPool {
            allocated: 5,
            ..Default::default()
        };
        for h in 1..=4u64 {
            assert!(pool.register(h));
        }
        pool.hit(1);
        pool.hit(2);
        assert!(pool.register(5));
        pool.unref_request(&[1, 2, 3, 4]);
        pool.unref_request(&[1, 2, 5]);
        // Tails go first, the shared prefix head last: 4, 3 (A's tail), then
        // 5 (B's tail), then 2, then 1.
        let order: Vec<u64> = std::iter::from_fn(|| pool.evict_lru()).collect();
        assert_eq!(order, vec![4, 3, 5, 2, 1]);
    }

    #[test]
    fn kv_pressure_evicts_lru_and_emits_removed_before_stored() {
        // 16 blocks of 4 tokens; each prompt takes 4 blocks plus a tail slot.
        let p = EngineParams {
            block_size: 4,
            kv_capacity_tokens: 64,
            max_running: 64,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let mut rxs = Vec::new();
        for i in 0..8 {
            let base = (i as u32) * 1000;
            let (r, rx) = req(&format!("r{i}"), (base..base + 16).collect(), 1);
            st.enqueue(r, &p);
            rxs.push(rx);
        }
        let mut saw_removed = false;
        for _ in 0..50 {
            let step = st.step(&p);
            if let Some(batch) = step.batch {
                if has_removed(&batch) {
                    saw_removed = true;
                    let first_removed = batch
                        .events
                        .iter()
                        .position(|e| {
                            matches!(e.data, Some(common::kv_cache_event::Data::Removed(_)))
                        })
                        .expect("removed present");
                    let first_stored = batch
                        .events
                        .iter()
                        .position(|e| {
                            matches!(e.data, Some(common::kv_cache_event::Data::Stored(_)))
                        })
                        .expect("a stored event follows the eviction");
                    assert!(
                        first_removed < first_stored,
                        "evictions precede the blocks they made room for"
                    );
                }
            }
        }
        assert!(saw_removed, "KV pressure should emit a removed event");
        assert!(
            st.pool.allocated <= 16,
            "the pool never holds more than its capacity: {}",
            st.pool.allocated
        );
        drop(rxs);
    }

    #[test]
    fn full_isl_reservation_admits_only_prompts_that_fit_and_blocks_head_of_line() {
        // 16 blocks of 4 tokens. Two 40-token prompts need 10 blocks (+1 for the
        // first token) each: the second does not fit beside the first, and a
        // small prompt behind it waits too (head-of-line), as vLLM does.
        let p = EngineParams {
            block_size: 4,
            kv_capacity_tokens: 64,
            max_batched_tokens: 60,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (a, _ra) = req("a", (0..40).collect(), 1);
        let (b, _rb) = req("b", (100..140).collect(), 1);
        let (c, _rc) = req("c", (200..208).collect(), 1);
        st.enqueue(a, &p);
        st.enqueue(b, &p);
        st.enqueue(c, &p);
        let step = st.step(&p);
        assert_eq!(step.admitted.len(), 1, "only the first prompt fits");
        assert_eq!(
            step.snapshot.num_waiting_reqs, 2,
            "the small prompt waits behind the big one"
        );
        assert!(st.pool.allocated <= 16);

        // Without the reservation the old behaviour admits chunks of everything.
        let loose = EngineParams {
            reserve_full_isl: false,
            ..p.clone()
        };
        let mut st = SchedulerState::new();
        let (a, _ra) = req("a", (0..40).collect(), 1);
        let (b, _rb) = req("b", (100..140).collect(), 1);
        st.enqueue(a, &loose);
        st.enqueue(b, &loose);
        let step = st.step(&loose);
        assert_eq!(
            step.admitted.len(),
            2,
            "chunk-only reservation admits both long prompts"
        );
    }

    #[test]
    fn full_isl_admission_gates_on_the_whole_prompt_but_allocates_per_chunk() {
        // A 2500-token prompt against a 1000-token pass budget: the gate reads
        // the whole prompt, the first pass allocates its chunk only, the next
        // chunks allocate as they run, and nothing is held beyond what is
        // computed (plus the first token's slot once the prefill completes).
        let p = EngineParams {
            max_batched_tokens: 1000,
            block_size: 16,
            kv_capacity_tokens: 16 * 400,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (r, mut rx) = req("long", vec![7; 2500], 3);
        st.enqueue(r, &p);
        st.step(&p);
        assert_eq!(
            st.pool.allocated,
            blocks_for(1000, 16),
            "the first pass allocates its chunk, not the whole prompt"
        );
        let (_, passes) = run_to_first_token(&mut st, &p, &mut rx);
        assert_eq!(passes, 2, "two more passes finish the prefill");
        assert_eq!(
            st.pool.allocated,
            blocks_for(2501, 16),
            "after the prefill the prompt and the first token's slot are held"
        );
        for _ in 0..10 {
            st.step(&p);
        }
        assert!(st.is_idle());
        assert_eq!(
            st.pool.allocated as usize,
            st.pool.cached(),
            "once done, only cached blocks remain allocated (no leaked reservation)"
        );
    }

    #[test]
    fn lifo_preemption_recomputes_and_completes() {
        // 16 blocks of 4 tokens. Two requests of 24 prompt tokens generating
        // 40 tokens each need 32 blocks between them: the later one is
        // preempted when the earlier one needs room, recomputes, and both
        // still deliver every token exactly once.
        let p = EngineParams {
            block_size: 4,
            kv_capacity_tokens: 64,
            max_running: 64,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (r1, mut rx1) = req("first", (0..24).collect(), 40);
        let (r2, mut rx2) = req("second", (100..124).collect(), 40);
        st.enqueue(r1, &p);
        st.enqueue(r2, &p);
        for _ in 0..500 {
            let step = st.step(&p);
            for (tx, ev) in step.sends {
                let _ = tx.send(ev);
            }
            if st.is_idle() {
                break;
            }
        }
        assert!(st.is_idle(), "both requests finish");
        assert!(
            st.preemptions >= 1,
            "KV exhaustion preempted the later request"
        );
        for rx in [&mut rx1, &mut rx2] {
            let mut tokens = 0;
            let mut done = None;
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    GenEvent::Token { .. } => tokens += 1,
                    GenEvent::Done {
                        completion_tokens, ..
                    } => done = Some(completion_tokens),
                }
            }
            assert_eq!(tokens, 40, "every token delivered once");
            assert_eq!(done, Some(40));
        }
        assert_eq!(
            st.pool.allocated,
            st.pool.free_lru.len() as u64,
            "all blocks idle"
        );
    }

    /// A live engine for the hook tests, with one request helper.
    fn live() -> Engine {
        Engine::spawn(EngineParams::default())
    }

    fn submit(engine: &Engine, id: &str, prompt: Vec<u32>) -> mpsc::UnboundedReceiver<GenEvent> {
        let (r, rx) = req(id, prompt, 1);
        engine.submit(r);
        rx
    }

    async fn next_batch(
        stream: &mut KvEventStream,
        within: Duration,
    ) -> Option<common::KvEventBatch> {
        tokio::time::timeout(within, stream.next())
            .await
            .ok()
            .flatten()
            .and_then(Result::ok)
    }

    #[tokio::test]
    async fn drop_hook_loses_live_batches_but_keeps_them_for_replay() {
        let engine = live();
        let mut live_stream = engine.subscribe_kv(0);
        engine.fault_drop(1);
        let mut rx1 = submit(&engine, "a", vec![1; 64]);
        // Let the first request's pass (and its batch) complete before the
        // second arrives, so the two batches are distinct.
        while let Some(event) = tokio::time::timeout(Duration::from_secs(5), rx1.recv())
            .await
            .expect("events")
        {
            if matches!(event, GenEvent::Done { .. }) {
                break;
            }
        }
        let _rx2 = submit(&engine, "b", vec![2; 64]);
        let first_live = next_batch(&mut live_stream, Duration::from_secs(5))
            .await
            .expect("a live batch");
        assert_eq!(
            first_live.sequence_number, 2,
            "the first batch was lost on the wire, the second arrives"
        );
        let mut replay = engine.subscribe_kv(0);
        let replayed = next_batch(&mut replay, Duration::from_secs(1))
            .await
            .expect("replay");
        assert_eq!(
            replayed.sequence_number, 1,
            "the lost batch is still replayable"
        );
        let status = engine.fault_status();
        assert_eq!((status.drop_pending, status.dropped_total), (0, 1));
    }

    #[tokio::test]
    async fn fail_hook_answers_the_next_n_requests_then_clears() {
        let engine = live();
        assert_eq!(engine.inject(), None, "nothing armed");
        engine.fault_fail(Some(RequestFault {
            status: 503,
            stall: Duration::ZERO,
            after_tokens: None,
            scope: FaultScope::Requests(2),
        }));
        let injected = Injected {
            status: 503,
            stall: Duration::ZERO,
            after_tokens: None,
        };
        assert_eq!(engine.inject(), Some(injected));
        assert_eq!(
            engine.fault_status().fail.map(|f| f.scope),
            Some(FaultScope::Requests(1)),
            "one request left"
        );
        assert_eq!(engine.inject(), Some(injected));
        assert_eq!(engine.inject(), None, "spent");
        let status = engine.fault_status();
        assert_eq!(status.fail, None, "cleared once spent");
        assert_eq!(
            (status.failed_total, status.cut_total, status.stalled_total),
            (0, 0, 0),
            "a refusal is counted by the path that answers it"
        );
        engine.record_failure();
        engine.record_failure();
        assert_eq!(engine.fault_status().failed_total, 2);
    }

    #[tokio::test]
    async fn fail_hook_timed_scope_elapses_and_cuts_count_apart() {
        let engine = live();
        engine.fault_fail(Some(RequestFault {
            status: 500,
            stall: Duration::from_millis(5),
            after_tokens: Some(3),
            scope: FaultScope::Until(Instant::now() + Duration::from_millis(100)),
        }));
        let injected = engine.inject().expect("armed");
        assert_eq!(
            (injected.status, injected.stall, injected.after_tokens),
            (500, Duration::from_millis(5), Some(3))
        );
        assert!(engine.fault_status().fail.is_some(), "still armed");
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert_eq!(
            engine.fault_status().fail,
            None,
            "an elapsed fault reads as disarmed before the next request"
        );
        assert_eq!(engine.inject(), None, "elapsed");
        let status = engine.fault_status();
        assert_eq!(status.fail, None, "disarmed once elapsed");
        assert_eq!(
            (status.failed_total, status.cut_total, status.stalled_total),
            (0, 0, 1),
            "a cut is counted where the stream is cut, not when it is armed"
        );
        engine.cut_counter().fetch_add(1, Ordering::Relaxed);
        assert_eq!(engine.fault_status().cut_total, 1);
        // Open scope: every request until cleared.
        engine.fault_fail(Some(RequestFault {
            status: 429,
            stall: Duration::ZERO,
            after_tokens: None,
            scope: FaultScope::Open,
        }));
        assert!(engine.inject().is_some());
        assert!(engine.inject().is_some());
        engine.fault_fail(None);
        assert_eq!(engine.inject(), None, "cleared");
        assert_eq!(
            engine.fault_status().failed_total,
            0,
            "inject() counts no refusal of its own"
        );
    }

    #[tokio::test]
    async fn pause_freezes_the_engine_until_resume() {
        let engine = live();
        engine.pause();
        let mut rx = submit(&engine, "a", vec![3; 32]);
        assert!(
            tokio::time::timeout(Duration::from_millis(200), rx.recv())
                .await
                .is_err(),
            "no token while paused"
        );
        assert!(engine.fault_status().paused);
        assert_eq!(engine.load().num_waiting_reqs, 1, "the request queued");
        engine.resume();
        let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("a token after resume")
            .expect("stream open");
        assert!(matches!(event, GenEvent::Token { .. }));
        assert!(!engine.fault_status().paused);
    }

    #[tokio::test]
    async fn restart_publisher_starts_the_sequence_over_and_empties_replay() {
        let engine = live();
        let mut live_stream = engine.subscribe_kv(0);
        let _rx1 = submit(&engine, "a", vec![4; 64]);
        let before = next_batch(&mut live_stream, Duration::from_secs(5))
            .await
            .expect("a batch");
        assert_eq!(before.sequence_number, 1);
        engine.restart_publisher().await;
        let _rx2 = submit(&engine, "b", vec![5; 64]);
        let after = next_batch(&mut live_stream, Duration::from_secs(5))
            .await
            .expect("a batch after the restart");
        assert_eq!(after.sequence_number, 1, "sequence numbers start over");
        let status = engine.fault_status();
        assert_eq!((status.generation, status.restarts), (1, 1));
        let mut replay = engine.subscribe_kv(0);
        let replayed = next_batch(&mut replay, Duration::from_secs(1))
            .await
            .expect("replay");
        assert_eq!(
            replayed.sequence_number, 1,
            "only the new generation is replayable"
        );
    }

    #[tokio::test]
    async fn delay_hook_defers_publishing_but_not_tokens() {
        let engine = live();
        let mut live_stream = engine.subscribe_kv(0);
        engine.fault_delay_ms(300);
        let mut rx = submit(&engine, "a", vec![6; 64]);
        let token = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("a token")
            .expect("stream open");
        assert!(matches!(token, GenEvent::Token { .. }));
        let token_at = Instant::now();
        let batch = next_batch(&mut live_stream, Duration::from_secs(5))
            .await
            .expect("the batch");
        let lag = token_at.elapsed();
        assert!(
            lag >= Duration::from_millis(200),
            "events trail the pass by the delay: {lag:?}"
        );
        let age = unix_seconds() - batch.timestamp;
        assert!(
            (0.2..30.0).contains(&age),
            "the batch keeps its creation time, so the delay is visible as lag: {age} s"
        );
    }

    #[tokio::test]
    async fn admission_hooks_hold_and_pace_requests_before_the_engine_sees_them() {
        let engine = live();
        // Neither hook set: a request is admitted at once.
        let t0 = Instant::now();
        engine.admit().await;
        assert!(
            t0.elapsed() < Duration::from_millis(100),
            "{:?}",
            t0.elapsed()
        );
        // The hold: every admission waits the delay.
        engine.fault_admit_delay_ms(150);
        let t0 = Instant::now();
        engine.admit().await;
        assert!(
            t0.elapsed() >= Duration::from_millis(150),
            "held for the delay: {:?}",
            t0.elapsed()
        );
        engine.fault_admit_delay_ms(0);
        // The cap: admissions are spaced 1/per_sec apart, the first one free.
        engine.fault_admit_per_sec(10);
        let t0 = Instant::now();
        for _ in 0..3 {
            engine.admit().await;
        }
        assert!(
            t0.elapsed() >= Duration::from_millis(200),
            "three admissions at 10/s take two slots: {:?}",
            t0.elapsed()
        );
        let status = engine.fault_status();
        assert_eq!((status.admit_delay_ms, status.admit_per_sec), (0, 10));
        // Cleared: at once again.
        engine.fault_admit_per_sec(0);
        let t0 = Instant::now();
        engine.admit().await;
        assert!(
            t0.elapsed() < Duration::from_millis(100),
            "{:?}",
            t0.elapsed()
        );
    }

    #[tokio::test]
    async fn cached_tokens_for_reports_the_engine_truth() {
        let engine = live();
        let prompt: Vec<u32> = (0..64).collect();
        assert_eq!(engine.cached_tokens_for(&prompt).cached_tokens, 0);
        let mut rx = submit(&engine, "a", prompt.clone());
        while let Some(event) = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("events")
        {
            if matches!(event, GenEvent::Done { .. }) {
                break;
            }
        }
        // The mirror is updated at pass end, just before the token is sent.
        let truth = engine.cached_tokens_for(&prompt);
        assert_eq!(truth.block_size, 16);
        assert_eq!(
            (truth.cached_blocks, truth.cached_tokens),
            (3, 48),
            "all four blocks are cached; the last one is recomputed"
        );
        let longer: Vec<u32> = (0..80).collect();
        assert_eq!(engine.cached_tokens_for(&longer).cached_tokens, 64);
    }

    #[test]
    fn prefill_first_pass_holds_decoders() {
        let p = EngineParams {
            prefill_first: true,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        let (r1, mut rx1) = req("decoder", vec![1; 32], 8);
        st.enqueue(r1, &p);
        st.step(&p); // prefill + first token
        st.step(&p); // a decode pass
        while rx1.try_recv().is_ok() {}
        let (r2, _rx2) = req("arrival", vec![2; 32], 8);
        st.enqueue(r2, &p);
        let step = st.step(&p); // a pass with prefill: prefill only
        for (tx, ev) in step.sends {
            let _ = tx.send(ev);
        }
        assert!(
            rx1.try_recv().is_err(),
            "the decoder gets no token in a prefill pass"
        );
        let step = st.step(&p); // no prefill left: everyone decodes
        for (tx, ev) in step.sends {
            let _ = tx.send(ev);
        }
        assert!(rx1.try_recv().is_ok(), "the decoder resumes afterwards");
    }
}
