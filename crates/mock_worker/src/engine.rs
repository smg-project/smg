//! A lightweight, CPU-only simulation of a continuous-batching LLM engine.
//!
//! The goal is fidelity, not fakery: the mock should exhibit the behaviors that
//! drive SMG's routing decisions so the whole gateway can be exercised without
//! GPUs. Concretely this models
//!
//! - **prefill latency that scales with input length** — time-to-first-token
//!   grows with the (uncached) prompt size, chunked across scheduler steps;
//! - **decode latency that grows with batch size** — inter-token latency is
//!   `base + slope · batch`, so a busy replica is slower per token;
//! - **finite KV capacity with admission/queueing** — when KV is full requests
//!   wait, producing the `num_waiting_uncached_tokens` signal `least_load` uses;
//! - **prefix caching** — a request sharing a prefix with cached blocks pays
//!   less prefill, reports `cached_tokens`, and the engine emits the KV-cache
//!   events (`KvBlocksStored` / `KvBlocksRemoved`) that drive `cache_aware`.
//!
//! The engine is an actor: one `tokio` task per virtual worker owns all mutable
//! state and advances it in real wall-clock time, but the per-step work is plain
//! arithmetic. Idle engines block on their request channel, so a fleet of mostly
//! idle workers stays cheap. The scheduling math lives in [`SchedulerState::step`],
//! a pure synchronous function returning the work done plus the time it took —
//! which makes it deterministically unit-testable with no real timers.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    pin::Pin,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::common_proto as common;
use tokio::sync::{broadcast, mpsc};
use tonic::Status;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tunable parameters of the simulated engine. Defaults approximate a single
/// mid-size model replica on one accelerator; override via mock-worker flags.
#[derive(Clone, Debug)]
pub struct EngineParams {
    /// Prefill throughput (prompt tokens processed per second).
    pub prefill_tps: f64,
    /// Fixed decode-step latency (ms) independent of batch size.
    pub decode_base_ms: f64,
    /// Added decode-step latency (ms) per running request — the batch slope.
    pub decode_per_req_ms: f64,
    /// Max prompt tokens prefilled per scheduler step (chunked prefill); keeps a
    /// huge prompt from stalling the whole batch in a single step.
    pub prefill_chunk_tokens: u32,
    /// Max concurrent running requests (continuous-batching width).
    pub max_running: usize,
    /// KV cache capacity in tokens.
    pub kv_capacity_tokens: u64,
    /// Cache block (page) size in tokens.
    pub block_size: u32,
    /// Whether prefix caching + KV-event emission are enabled.
    pub prefix_cache: bool,
    /// Start evicting once KV usage exceeds this fraction of capacity.
    pub kv_high_watermark: f64,
    /// Evict down to this fraction of capacity once eviction starts.
    pub kv_low_watermark: f64,
    /// Output tokens to generate when a request does not specify `max_new_tokens`.
    pub max_new_default: u32,
    /// Capacity of the live KV-event broadcast channel.
    pub kv_broadcast_capacity: usize,
    /// How many recent KV-event batches to retain for subscriber replay.
    pub kv_replay_capacity: usize,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            prefill_tps: 8000.0,
            decode_base_ms: 6.0,
            decode_per_req_ms: 0.35,
            prefill_chunk_tokens: 2048,
            max_running: 256,
            kv_capacity_tokens: 524_288,
            block_size: 16,
            prefix_cache: true,
            kv_high_watermark: 0.92,
            kv_low_watermark: 0.85,
            max_new_default: 128,
            kv_broadcast_capacity: 1024,
            kv_replay_capacity: 4096,
        }
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
#[derive(Clone, Debug)]
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
}

// ---------------------------------------------------------------------------
// Engine handle
// ---------------------------------------------------------------------------

/// Shared state readable from the transport handlers while the actor runs.
struct EngineShared {
    snapshot: RwLock<LoadSnapshot>,
    kv_tx: broadcast::Sender<common::KvEventBatch>,
    kv_replay: Mutex<VecDeque<common::KvEventBatch>>,
    prefix_cache: bool,
}

/// A cloneable handle to one simulated engine.
#[derive(Clone)]
pub struct Engine {
    tx: mpsc::UnboundedSender<NewRequest>,
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
    #[expect(
        clippy::disallowed_methods,
        reason = "engine actor self-terminates when its request channel closes"
    )]
    pub fn spawn(params: EngineParams) -> Engine {
        let (tx, rx) = mpsc::unbounded_channel();
        let (kv_tx, _) = broadcast::channel(params.kv_broadcast_capacity.max(1));
        let shared = Arc::new(EngineShared {
            snapshot: RwLock::new(LoadSnapshot::idle(&params)),
            kv_tx,
            kv_replay: Mutex::new(VecDeque::new()),
            prefix_cache: params.prefix_cache,
        });
        tokio::spawn(run(params, rx, shared.clone()));
        Engine { tx, shared }
    }

    /// Submit a request. Dropped silently if the engine has shut down.
    pub fn submit(&self, req: NewRequest) {
        let _ = self.tx.send(req);
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
    pub fn kv_enabled(&self) -> bool {
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
                    Ok(batch) => return Some((Ok(batch), rx)),
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
    mut rx: mpsc::UnboundedReceiver<NewRequest>,
    shared: Arc<EngineShared>,
) {
    let mut state = SchedulerState::new();
    loop {
        // When there is nothing to do, publish an idle snapshot and block on the
        // request channel — an idle engine consumes no CPU.
        if state.is_idle() {
            *shared.snapshot.write().unwrap_or_else(|p| p.into_inner()) = state.snapshot(&params);
            match rx.recv().await {
                Some(req) => state.enqueue(req, &params),
                None => return, // all handles dropped
            }
        }
        // Drain any other already-queued submissions without blocking.
        while let Ok(req) = rx.try_recv() {
            state.enqueue(req, &params);
        }

        let step = state.step(&params);
        if step.duration > Duration::ZERO {
            tokio::time::sleep(step.duration).await;
        }
        // The step's outputs become observable only after its simulated time.
        for (tx, ev) in step.sends {
            let _ = tx.send(ev);
        }
        if let Some(batch) = step.batch {
            {
                let mut buf = shared.kv_replay.lock().unwrap_or_else(|p| p.into_inner());
                buf.push_back(batch.clone());
                while buf.len() > params.kv_replay_capacity {
                    buf.pop_front();
                }
            }
            let _ = shared.kv_tx.send(batch);
        }
        *shared.snapshot.write().unwrap_or_else(|p| p.into_inner()) = step.snapshot;
    }
}

// ---------------------------------------------------------------------------
// Scheduler state + step (pure, deterministic, unit-testable)
// ---------------------------------------------------------------------------

/// A request currently being prefilled or decoded.
struct RunningReq {
    events: mpsc::UnboundedSender<GenEvent>,
    prompt_tokens: u32,
    cached_tokens: u32,
    max_new: u32,
    generated: u32,
    /// Uncached prompt tokens still to be prefilled before the first token.
    prefill_remaining: u32,
    /// FNV hash of all tokens seen so far (prompt + generated) — the content key
    /// of the next block once `pending_block` fills.
    rolling_hash: u64,
    /// Block key of the most recently completed block (for parent chaining).
    prev_block_key: Option<u64>,
    /// Tokens accumulated toward the next (not-yet-full) block.
    pending_block: Vec<u32>,
    /// Generated token ids (for the terminal `output_ids`).
    output_ids: Vec<u32>,
    /// Per-request seed so decode blocks never collide across requests.
    token_seed: u32,
}

/// A request admitted to the queue but not yet running.
struct WaitingReq {
    req: NewRequest,
    prompt_tokens: u32,
    /// Uncached prompt tokens at enqueue time (the queued token-work it adds).
    uncached_tokens: u32,
}

/// Per-worker prefix cache: content-keyed block presence with LRU ordering.
#[derive(Default)]
struct Cache {
    present: HashSet<u64>,
    tick_of: HashMap<u64, u64>,
    lru: BTreeSet<(u64, u64)>,
    tick: u64,
}

impl Cache {
    fn len(&self) -> usize {
        self.present.len()
    }

    /// Number of consecutive cached blocks from the start of `keys`.
    fn match_prefix(&self, keys: &[u64]) -> usize {
        let mut n = 0;
        for &k in keys {
            if self.present.contains(&k) {
                n += 1;
            } else {
                break;
            }
        }
        n
    }

    /// Mark `keys` as just-used (warms shared prefixes so eviction prefers
    /// colder leaves, keeping the block set prefix-closed in the common case).
    fn touch(&mut self, keys: &[u64]) {
        self.tick += 1;
        let t = self.tick;
        for &k in keys {
            if self.present.contains(&k) {
                if let Some(old) = self.tick_of.insert(k, t) {
                    self.lru.remove(&(old, k));
                }
                self.lru.insert((t, k));
            }
        }
    }

    /// Insert a block; returns true if it was newly added.
    fn insert(&mut self, k: u64) -> bool {
        if self.present.contains(&k) {
            self.touch(&[k]);
            return false;
        }
        self.present.insert(k);
        self.tick += 1;
        let t = self.tick;
        self.tick_of.insert(k, t);
        self.lru.insert((t, k));
        true
    }

    /// Evict the least-recently-used block; returns its key.
    fn evict_one(&mut self) -> Option<u64> {
        let &(t, k) = self.lru.iter().next()?;
        self.lru.remove(&(t, k));
        self.tick_of.remove(&k);
        self.present.remove(&k);
        Some(k)
    }
}

/// The actor-owned scheduler state.
pub(crate) struct SchedulerState {
    running: Vec<RunningReq>,
    waiting: VecDeque<WaitingReq>,
    cache: Cache,
    kv_seq: u64,
    kv_event_id: u64,
    gen_tp_ewma: f64,
    cache_hit_ewma: f64,
}

/// The result of one scheduler step.
pub(crate) struct Step {
    duration: Duration,
    sends: Vec<(mpsc::UnboundedSender<GenEvent>, GenEvent)>,
    batch: Option<common::KvEventBatch>,
    snapshot: LoadSnapshot,
}

impl SchedulerState {
    pub(crate) fn new() -> Self {
        Self {
            running: Vec::new(),
            waiting: VecDeque::new(),
            cache: Cache::default(),
            kv_seq: 0,
            kv_event_id: 0,
            gen_tp_ewma: 0.0,
            cache_hit_ewma: 0.0,
        }
    }

    fn is_idle(&self) -> bool {
        self.running.is_empty() && self.waiting.is_empty()
    }

    /// Queue a request, recording the queued token-work it contributes.
    pub(crate) fn enqueue(&mut self, req: NewRequest, p: &EngineParams) {
        let prompt_tokens = req.prompt_token_ids.len() as u32;
        let (block_keys, _, _) = prompt_blocks(&req.prompt_token_ids, p.block_size as usize);
        let cached_blocks = if p.prefix_cache {
            self.cache.match_prefix(&block_keys)
        } else {
            0
        };
        let cached = (cached_blocks as u32 * p.block_size).min(prompt_tokens);
        let uncached = prompt_tokens - cached;
        self.waiting.push_back(WaitingReq {
            req,
            prompt_tokens,
            uncached_tokens: uncached,
        });
    }

    /// Tokens currently resident in KV.
    fn used_tokens(&self, p: &EngineParams) -> u64 {
        if p.prefix_cache {
            // KV holds the shared radix cache (blocks persist across requests
            // until evicted) plus each running request's not-yet-blocked tail.
            let blocks = self.cache.len() as u64 * p.block_size as u64;
            let partial: u64 = self
                .running
                .iter()
                .map(|r| r.pending_block.len() as u64)
                .sum();
            blocks + partial
        } else {
            // No sharing/persistence: each running request occupies its full
            // current context; that KV frees when it leaves the batch.
            self.running
                .iter()
                .map(|r| u64::from(r.prompt_tokens + r.generated))
                .sum()
        }
    }

    fn snapshot(&self, p: &EngineParams) -> LoadSnapshot {
        let used = self.used_tokens(p);
        let waiting_uncached: i64 = self.waiting.iter().map(|w| w.uncached_tokens as i64).sum();
        LoadSnapshot {
            num_running_reqs: self.running.len() as i32,
            num_waiting_reqs: self.waiting.len() as i32,
            num_waiting_uncached_tokens: waiting_uncached.min(i32::MAX as i64) as i32,
            num_used_tokens: used.min(i32::MAX as u64) as i32,
            max_total_num_tokens: p.kv_capacity_tokens.min(i32::MAX as u64) as i32,
            max_running_requests: p.max_running.min(i32::MAX as usize) as i32,
            token_usage: (used as f64 / p.kv_capacity_tokens.max(1) as f64).clamp(0.0, 1.0),
            gen_throughput: self.gen_tp_ewma,
            cache_hit_rate: self.cache_hit_ewma,
        }
    }

    /// Advance the engine by one scheduler iteration. Pure: mutates state and
    /// returns the work produced plus how long it took, but performs no I/O.
    pub(crate) fn step(&mut self, p: &EngineParams) -> Step {
        let mut kv: Vec<common::KvCacheEvent> = Vec::new();

        // ---- 1. Admission ----
        self.admit(p, &mut kv);

        // ---- 2. Chunked prefill ----
        let mut budget = p.prefill_chunk_tokens;
        let mut prefill_tokens = 0u32;
        for r in &mut self.running {
            if r.prefill_remaining > 0 && budget > 0 {
                let c = r.prefill_remaining.min(budget);
                r.prefill_remaining -= c;
                budget -= c;
                prefill_tokens += c;
            }
        }

        // ---- 3. Decode (one token per ready request) ----
        let block_size = p.block_size as usize;
        let mut sends: Vec<(mpsc::UnboundedSender<GenEvent>, GenEvent)> = Vec::new();
        let mut new_blocks: Vec<(u64, Vec<u32>, Option<u64>)> = Vec::new();
        let mut decode_tokens = 0u32;
        let num_decode = self
            .running
            .iter()
            .filter(|r| {
                r.prefill_remaining == 0 && r.generated < r.max_new && !r.events.is_closed()
            })
            .count();

        for r in &mut self.running {
            if r.prefill_remaining != 0 || r.generated >= r.max_new || r.events.is_closed() {
                continue;
            }
            let token_id = next_token(r);
            r.generated += 1;
            r.output_ids.push(token_id);
            r.rolling_hash = fnv_step(r.rolling_hash, token_id);
            r.pending_block.push(token_id);
            decode_tokens += 1;
            if r.pending_block.len() == block_size {
                let key = r.rolling_hash;
                new_blocks.push((key, std::mem::take(&mut r.pending_block), r.prev_block_key));
                r.prev_block_key = Some(key);
            }
            sends.push((
                r.events.clone(),
                GenEvent::Token {
                    token_id,
                    prompt_tokens: r.prompt_tokens,
                    cached_tokens: r.cached_tokens,
                },
            ));
        }

        // Commit newly completed decode blocks to the cache (+ stored events).
        for (key, tokens, parent) in new_blocks {
            if p.prefix_cache && self.cache.insert(key) {
                kv.push(self.stored_event(key, tokens, parent, p.block_size));
            }
        }

        // ---- 4. Completion ----
        let mut still = Vec::with_capacity(self.running.len());
        for r in std::mem::take(&mut self.running) {
            let done = r.prefill_remaining == 0 && r.generated >= r.max_new;
            if done || r.events.is_closed() {
                if done {
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
            } else {
                still.push(r);
            }
        }
        self.running = still;

        // ---- 5. Eviction under KV pressure ----
        self.evict(p, &mut kv);

        // ---- 6. Timing + bookkeeping ----
        let prefill_time = prefill_tokens as f64 / p.prefill_tps.max(1.0);
        let decode_time = if num_decode > 0 {
            (p.decode_base_ms + p.decode_per_req_ms * num_decode as f64) / 1000.0
        } else {
            0.0
        };
        let mut secs = prefill_time.max(decode_time);
        if secs <= 0.0 && !self.is_idle() {
            secs = 0.001; // never busy-spin while work remains
        }
        let throughput_sample = if secs > 0.0 && decode_tokens > 0 {
            decode_tokens as f64 / secs
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
                timestamp: 0.0,
                events: kv,
                dp_rank: Some(0),
            })
        };

        Step {
            duration: Duration::from_secs_f64(secs.max(0.0)),
            sends,
            batch,
            snapshot: self.snapshot(p),
        }
    }

    /// Admit waiting requests while batch width and KV capacity allow.
    fn admit(&mut self, p: &EngineParams, kv: &mut Vec<common::KvCacheEvent>) {
        let block_size = p.block_size as usize;
        while self.running.len() < p.max_running {
            let Some(front) = self.waiting.front() else {
                break;
            };
            let (block_keys, rolling, pending) =
                prompt_blocks(&front.req.prompt_token_ids, block_size);
            let cached_blocks = if p.prefix_cache {
                self.cache.match_prefix(&block_keys)
            } else {
                0
            };
            let cached_tokens = (cached_blocks as u32 * p.block_size).min(front.prompt_tokens);
            let uncached = front.prompt_tokens - cached_tokens;

            // Admission control: require KV room for the uncached prompt unless
            // the engine is empty (a prompt larger than all of KV must still run).
            let free = p.kv_capacity_tokens.saturating_sub(self.used_tokens(p));
            if u64::from(uncached) > free && !self.running.is_empty() {
                break;
            }

            let Some(w) = self.waiting.pop_front() else {
                break;
            };
            let NewRequest {
                request_id,
                prompt_token_ids,
                max_new,
                events,
            } = w.req;

            if cached_blocks > 0 {
                self.cache.touch(&block_keys[..cached_blocks]);
            }
            // Allocate + announce the uncached prompt blocks (resident during prefill).
            let mut prev = cached_blocks.checked_sub(1).map(|i| block_keys[i]);
            for j in cached_blocks..block_keys.len() {
                let key = block_keys[j];
                if p.prefix_cache && self.cache.insert(key) {
                    let toks = prompt_token_ids[j * block_size..(j + 1) * block_size].to_vec();
                    kv.push(self.stored_event(key, toks, prev, p.block_size));
                }
                prev = Some(key);
            }

            let resolved_max_new = if max_new == 0 {
                p.max_new_default
            } else {
                max_new
            };
            let sample = if w.prompt_tokens > 0 {
                f64::from(cached_tokens) / f64::from(w.prompt_tokens)
            } else {
                0.0
            };
            self.cache_hit_ewma = ewma(self.cache_hit_ewma, sample, 0.2);

            self.running.push(RunningReq {
                events,
                prompt_tokens: w.prompt_tokens,
                cached_tokens,
                max_new: resolved_max_new,
                generated: 0,
                prefill_remaining: uncached,
                rolling_hash: rolling,
                prev_block_key: prev,
                pending_block: pending,
                output_ids: Vec::new(),
                token_seed: fnv_hash_str(&request_id) as u32,
            });
        }
    }

    /// Evict LRU blocks once KV usage crosses the high watermark.
    fn evict(&mut self, p: &EngineParams, kv: &mut Vec<common::KvCacheEvent>) {
        if !p.prefix_cache {
            return;
        }
        let b = p.block_size as u64;
        let partial: u64 = self
            .running
            .iter()
            .map(|r| r.pending_block.len() as u64)
            .sum();
        let mut blocks = self.cache.len() as u64;
        let high = (p.kv_capacity_tokens as f64 * p.kv_high_watermark) as u64;
        if blocks * b + partial <= high {
            return;
        }
        let low = (p.kv_capacity_tokens as f64 * p.kv_low_watermark) as u64;
        let mut removed: Vec<i64> = Vec::new();
        while blocks * b + partial > low {
            match self.cache.evict_one() {
                Some(key) => {
                    removed.push(key as i64);
                    blocks -= 1;
                }
                None => break,
            }
        }
        if !removed.is_empty() {
            self.kv_event_id += 1;
            kv.push(common::KvCacheEvent {
                event_id: self.kv_event_id,
                data: Some(common::kv_cache_event::Data::Removed(
                    common::KvBlocksRemoved {
                        block_hashes: removed,
                        cache_level: None,
                    },
                )),
            });
        }
    }

    fn stored_event(
        &mut self,
        key: u64,
        token_ids: Vec<u32>,
        parent: Option<u64>,
        block_size: u32,
    ) -> common::KvCacheEvent {
        self.kv_event_id += 1;
        common::KvCacheEvent {
            event_id: self.kv_event_id,
            data: Some(common::kv_cache_event::Data::Stored(
                common::KvBlocksStored {
                    blocks: vec![common::KvBlock {
                        block_hash: key as i64,
                        token_ids,
                        block_size: block_size as i32,
                        lora_id: None,
                        cache_level: None,
                    }],
                    parent_block_hash: parent.map(|k| k as i64),
                },
            )),
        }
    }
}

impl LoadSnapshot {
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

    /// Run steps until the given request emits its first Token, returning the
    /// accumulated simulated time (TTFT) and step count.
    fn run_to_first_token(
        st: &mut SchedulerState,
        p: &EngineParams,
        rx: &mut mpsc::UnboundedReceiver<GenEvent>,
    ) -> (Duration, u32) {
        let mut total = Duration::ZERO;
        for _ in 0..100_000 {
            let step = st.step(p);
            total += step.duration;
            for (tx, ev) in step.sends {
                let _ = tx.send(ev);
            }
            if let Ok(GenEvent::Token { .. }) = rx.try_recv() {
                return (total, 1);
            }
        }
        panic!("no token produced");
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
    fn itl_grows_with_batch_size() {
        let p = EngineParams {
            prefix_cache: false,
            prefill_chunk_tokens: 1_000_000, // finish prefill in one step
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
            let duration = st.step(&p).duration; // a pure decode step
            drop(rxs);
            duration
        };

        let one = decode_step_duration(1);
        let many = decode_step_duration(64);
        assert!(
            many > one,
            "decode step should be slower with a bigger batch: one={one:?} many={many:?}"
        );
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

        // Second request shares the whole prompt prefix.
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
                    assert_eq!(cached_tokens, 16, "full prompt prefix should be cached");
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
    fn prompt_blocks_emit_chained_kv_events() {
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
        let stored: Vec<_> = batch
            .events
            .iter()
            .filter_map(|e| match &e.data {
                Some(common::kv_cache_event::Data::Stored(s)) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(stored.len(), 2, "two prompt blocks");
        assert!(
            stored[0].parent_block_hash.is_none(),
            "first block has no parent"
        );
        let first_hash = stored[0].blocks[0].block_hash;
        assert_eq!(
            stored[1].parent_block_hash,
            Some(first_hash),
            "second block chains to the first"
        );
    }

    #[test]
    fn kv_pressure_evicts_and_emits_removed() {
        // Tiny KV so a couple of prompts overflow it.
        let p = EngineParams {
            block_size: 4,
            kv_capacity_tokens: 64,
            kv_high_watermark: 0.5,
            kv_low_watermark: 0.25,
            max_running: 64,
            prefill_chunk_tokens: 1_000_000,
            ..Default::default()
        };
        let mut st = SchedulerState::new();
        for i in 0..8 {
            // Distinct prompts so each contributes its own blocks.
            let base = (i as u32) * 1000;
            let (r, _rx) = req(&format!("r{i}"), (base..base + 16).collect(), 1);
            st.enqueue(r, &p);
        }
        let mut saw_removed = false;
        for _ in 0..50 {
            let step = st.step(&p);
            if let Some(batch) = step.batch {
                if batch
                    .events
                    .iter()
                    .any(|e| matches!(e.data, Some(common::kv_cache_event::Data::Removed(_))))
                {
                    saw_removed = true;
                }
            }
        }
        assert!(saw_removed, "KV pressure should emit a removed event");
    }
}
