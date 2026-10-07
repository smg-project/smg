//! `SubscribeKvEvents`: an engine's ZMQ KV-cache event publisher relayed into
//! gRPC streams through one [`KvEventRelay`] per engine.
//!
//! The relay subscribes to the publisher once, for the servicer's lifetime,
//! when the servicer starts serving (so it sees the engine from its first
//! batch, and a servicer that outlives a gateway outage or restarts beside a
//! warm engine is not blind until eviction; `SMG_KV_EVENT_RELAY_START=lazy`
//! defers the subscription to the first `SubscribeKvEvents` for a
//! memory-constrained host), keeps what it relays in a bounded [`History`]
//! (the last
//! `SMG_KV_EVENT_HISTORY_BATCHES` batches, 10,000 by default like the
//! engines' own `buffer_steps`, within `SMG_KV_EVENT_HISTORY_BYTES`, 256 MiB
//! by default) and folds every batch into a record of the engine's live
//! blocks per rank ([`LiveState`]). A `SubscribeKvEvents` call is served
//! from them:
//!
//! - `start_sequence_number` is the last sequence the gateway applied. Inside
//!   the window, the batches after it come first, then live events; below
//!   the window (or beyond the newest sequence, which means another publisher
//!   incarnation) the call fails with `OUT_OF_RANGE`, and the gateway clears
//!   and resubscribes from zero. A relay that started after the publisher
//!   holds no history from before its start, so a cursor from before it is
//!   `OUT_OF_RANGE` too.
//! - zero means no cursor. When the window holds every batch the publisher
//!   ever numbered (it starts at 0, nothing lost or evicted) the subscriber
//!   gets all of it, which is the publisher's whole state. Once the window no
//!   longer starts there (it rolled past its caps, has a hole, or the relay
//!   started after the publisher) the subscriber gets a state snapshot
//!   instead: the live blocks as the relay recorded them, cut at the last
//!   relayed sequence under the same lock that admits batches, sent as
//!   chunks marked `KvSnapshotChunk` with consecutive sequence numbers
//!   ending at that cut (chunk 0 begins with `AllBlocksCleared`, the stores
//!   follow parents first), then live events from the next sequence: no
//!   gap, no duplicate. The gateway applies the chunks as a `snapshot`
//!   resync. A stale cursor below the window stays `OUT_OF_RANGE`: the
//!   gateway clears, resubscribes from zero and receives the snapshot. Before
//!   anything was relayed, zero is live only.
//!
//! The publisher's own sequence numbers are kept. A sequence the relay did
//! not receive is asked of the engine's replay socket (vLLM's and SGLang's
//! ROUTER, the mock engine's too); what the replay cannot give leaves a hole
//! in the window, counted, which a subscriber skips the way the live stream
//! did. A payload that does not decode relays as an empty batch under its
//! sequence, so nobody sees a gap for it. The batches before the first one
//! the relay sees are treated the same way: ZMQ delivers nothing from before
//! a subscription, so the relay asks the replay for everything from 0 when
//! it starts (retrying every second until the engine's replay answers, since
//! the engine may still be coming up), and again before relaying a first
//! live batch (of an incarnation) past sequence 1 (the publishers count from
//! 0, the mock engine from 1) unless that batch carries the engine's startup
//! clear, after which nothing earlier matters. The start replay is what
//! covers an engine that published a few batches at registration and nothing
//! since: without it the relay would never hear of them; and because those
//! batches may follow the first (empty) answer, a subscription from zero
//! that finds the relay still holding nothing asks the replay once more.
//! What the replay no longer reaches back to is counted as unknown before
//! the record's start (`unknown_before_start`): the window is then not
//! complete from the publisher's first batch, a subscriber from zero gets
//! the snapshot, and the snapshot's chunks carry `unknown_before` so the
//! gateway marks the worker degraded instead of trusting a silently partial
//! state.
//!
//! A publisher restart is read the same way on every wire, from three signs
//! (vLLM and SGLang count from 0 per process; SGLang's first batch after a
//! start carries `AllBlocksCleared`, vLLM's does not): the sequence goes
//! backwards on the same socket; the engine's startup clear arrives under a
//! sequence the relay already passed; the counter is back at 0 or 1 after a
//! cursor above them. Each starts a new incarnation: the history is cleared,
//! live subscribers end with `DATA_LOSS`, and the gateway clears and
//! resubscribes from zero, where the new incarnation's complete history is
//! waiting for it. A repeated sequence without a clear is a duplicate.
//!
//! Every batch a subscriber receives carries the servicer's load record
//! ([`LoadSource`], the figures `GetLoads` answers with, as `KvEventBatch.
//! load`), and a subscriber hears from the relay even when the publisher is
//! quiet: a batch with no events and the last sequence repeated, marked
//! `load_only`, goes out when the record changed (checked every
//! `load_tick`, 100 ms) and as a heartbeat after `heartbeat_interval` (1 s)
//! of silence, backing off to `heartbeat_backoff` (5 s) once the engine has
//! been idle for two heartbeats. The record's routing core rides every
//! batch; the engine's telemetry (cache hit rate, token counts, SGLang's
//! sections) rides the heartbeats and the stream's first record. The
//! gateway feeds the record where its `GetLoads` poll goes, admits nothing
//! from a `load_only` batch, and does not poll a worker while its records
//! arrive: `GetLoads` is the fallback.
//!
//! `SMG_KV_EVENT_HASH_CHECK=sglang|vllm-sha256-cbor` turns on the relay's
//! engine-hash verification ([`crate::engine_hash`]); mismatches are counted,
//! never dropped. The relay's counters ([`RelayCounts`]) are logged on every
//! gap, restart and refusal, and together with the normalizer's
//! ([`crate::kv_wire::Counts`]: forwarded, dropped by reason, the hash
//! check's tally) in a summary line every 500 relayed batches and when the
//! relay closes.
//!
//! Framing (`ZmqEventPublisher` in both engines): one PUB multipart message
//! per scheduler step, `[topic, sequence as u64 big-endian, msgpack batch]`.
//! The wire format and the normalization each event goes through live in
//! [`crate::kv_wire`].

use std::{
    collections::VecDeque,
    ops::RangeInclusive,
    sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use engine_zmq_client::codec::TrailingTolerant;
use futures::stream;
use smg_grpc_client::common_proto::{self as common};
use tokio::sync::{broadcast, Notify};
use tonic::Status;
use tracing::{debug, info, warn};
use zeromq::{
    prelude::{Socket, SocketRecv, SocketSend},
    DealerSocket, SocketOptions, SubSocket, ZmqError, ZmqMessage,
};

use crate::{
    kv_history::{History, Window},
    kv_state::{LiveState, Snapshot, SnapshotChunks},
    kv_wire::{low64_big_endian, Counts as WireCounts, Normalizer, WireBatch, WireEvent},
    BoxStream,
};

/// The Python vLLM servicer's refusal when vLLM runs without a ZMQ publisher.
pub(crate) const VLLM_DISABLED_MESSAGE: &str = "KV cache events not enabled. Start vLLM with \
     --kv-events-config '{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\"}'";

/// The Python SGLang servicer's refusal when SGLang runs without a ZMQ publisher.
pub(crate) const SGLANG_DISABLED_MESSAGE: &str = "KV cache events not enabled. Start SGLang \
     with --kv-events-config '{\"publisher\": \"zmq\"}'";

/// The Python TokenSpeed servicer's refusal without a publisher.
pub(crate) const TOKENSPEED_DISABLED_MESSAGE: &str = "KV cache events not enabled. Start \
     TokenSpeed with --kv-events-config '{\"enable_kv_cache_events\": true, \"publisher\": \
     \"zmq\"}'";

/// When the relay subscribes to the publisher: at the servicer's start
/// (default, any other value) or `lazy`, at the first `SubscribeKvEvents`.
pub(crate) const RELAY_START_ENV: &str = "SMG_KV_EVENT_RELAY_START";
/// How many relayed batches the history keeps (the engines' `buffer_steps`).
pub(crate) const HISTORY_BATCHES_ENV: &str = "SMG_KV_EVENT_HISTORY_BATCHES";
/// The history's byte budget over the encoded batches.
pub(crate) const HISTORY_BYTES_ENV: &str = "SMG_KV_EVENT_HISTORY_BYTES";
pub const DEFAULT_HISTORY_BATCHES: usize = 10_000;
pub const DEFAULT_HISTORY_BYTES: usize = 256 << 20;
const DEFAULT_REPLAY_TIMEOUT: Duration = Duration::from_secs(5);
/// How often the start replay is retried while the engine's replay socket
/// does not answer yet.
const PRIME_RETRY: Duration = Duration::from_secs(1);
/// Live batches a slow subscriber may fall behind before it is refilled from
/// the history.
const LIVE_CHANNEL: usize = 4_096;
/// A relay summary line (relay and normalizer counters) every this many
/// relayed batches, so a live run shows them before the relay closes.
const SUMMARY_EVERY_BATCHES: u64 = 500;

/// How often a subscriber's stream checks the load record for a change.
pub(crate) const DEFAULT_LOAD_TICK: Duration = Duration::from_millis(100);
/// Silence after which a `load_only` heartbeat goes out.
pub(crate) const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);
/// The heartbeat interval once the engine has been idle for two heartbeats.
pub(crate) const DEFAULT_HEARTBEAT_BACKOFF: Duration = Duration::from_secs(5);
/// The replay socket's end marker, eight 0xff bytes on both wires.
const END_SEQUENCE: [u8; 8] = [0xff; 8];

/// One publisher's relay settings.
#[derive(Clone, Debug)]
pub struct RelayConfig {
    /// The connectable SUB endpoint of rank 0's publisher.
    pub endpoint: String,
    /// Its replay ROUTER, when the engine runs one.
    pub replay_endpoint: Option<String>,
    pub topic: String,
    pub history_batches: usize,
    pub history_bytes: usize,
    /// How long a replay reply may take before the rest of a gap is lost.
    pub replay_timeout: Duration,
    /// How often a subscriber's stream checks the load record for a change
    /// while no batch flows.
    pub load_tick: Duration,
    /// Silence after which a `load_only` heartbeat goes out.
    pub heartbeat_interval: Duration,
    /// The heartbeat interval after two heartbeats with an unchanged record.
    pub heartbeat_backoff: Duration,
}

impl RelayConfig {
    /// Rank 0 of the publisher at `kv_events_endpoint` (bind wildcards
    /// resolved), with the history caps from the environment.
    pub(crate) fn for_publisher(
        kv_events_endpoint: &str,
        replay_endpoint: Option<&str>,
        topic: &str,
    ) -> Self {
        Self {
            endpoint: endpoint_for_rank(kv_events_endpoint, 0),
            replay_endpoint: replay_endpoint
                .filter(|endpoint| !endpoint.is_empty())
                .map(|endpoint| endpoint_for_rank(endpoint, 0)),
            topic: topic.to_string(),
            history_batches: env_usize(HISTORY_BATCHES_ENV, DEFAULT_HISTORY_BATCHES),
            history_bytes: env_usize(HISTORY_BYTES_ENV, DEFAULT_HISTORY_BYTES),
            replay_timeout: DEFAULT_REPLAY_TIMEOUT,
            load_tick: DEFAULT_LOAD_TICK,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
        }
    }
}

/// Where the relay reads the engine's load for the record it attaches to
/// every batch: the servicer's `GetLoads` bookkeeping.
pub(crate) trait LoadSource: Send + Sync {
    /// The load for `dp_rank` (the batch's rank; `None` for a publisher that
    /// names none) as `GetLoads` would report it now, or `None` while nothing
    /// is known (the engine is not up yet). `sample` and `load_only` are the
    /// relay's to set.
    fn load(&self, dp_rank: Option<i32>) -> Option<common::EngineLoad>;
}

/// Whether a record moved enough from `last` to be worth a `load_only`
/// batch: any queue, running or window change, KV usage by half a percent,
/// the rate by 5 % or 50 tokens per second.
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

/// Whether [`RELAY_START_ENV`] set to `value` keeps the start at boot.
fn starts_at_boot(value: Option<&str>) -> bool {
    !value.is_some_and(|value| value.trim().eq_ignore_ascii_case("lazy"))
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => match value.trim().parse::<usize>() {
            Ok(parsed) => parsed,
            Err(_) => {
                warn!(%value, "{name} is not a number; using {default}");
                default
            }
        },
        _ => default,
    }
}

/// Resolve a KV-events PUB endpoint to a connectable SUB address: bind
/// wildcards become loopback, and under data parallelism rank `dp_rank`
/// publishes on `base_port + dp_rank` (tcp only; ipc/inproc get no port
/// arithmetic).
pub(crate) fn endpoint_for_rank(endpoint: &str, dp_rank: u32) -> String {
    let resolved = endpoint
        .replace('*', "127.0.0.1")
        .replace("0.0.0.0", "127.0.0.1");
    if dp_rank == 0 || !resolved.starts_with("tcp://") {
        return resolved;
    }
    let Some((host, port)) = resolved.rsplit_once(':') else {
        return resolved;
    };
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return resolved;
    }
    match port.parse::<u64>() {
        Ok(port) => format!("{host}:{}", port.saturating_add(u64::from(dp_rank))),
        Err(_) => resolved,
    }
}

/// What one relay has done since it started.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelayCounts {
    /// Batches relayed under the publisher's sequence (live or replayed).
    pub relayed: u64,
    /// Payloads that did not decode and were relayed as empty batches.
    pub undecodable_batches: u64,
    /// Subscriptions whose first batches came from the history.
    pub served_from_history: u64,
    /// Subscriptions that began with a state snapshot.
    pub served_snapshots: u64,
    /// Subscriptions refused because their cursor was outside the window.
    pub out_of_range: u64,
    /// Sequence gaps seen on the publisher's socket.
    pub publisher_gaps: u64,
    /// Batches of those gaps the engine's replay gave back.
    pub gap_batches_recovered: u64,
    /// Batches of those gaps nobody had: holes in the window.
    pub gap_batches_lost: u64,
    /// Publisher sequences before the relay's record that neither the stream
    /// nor the engine's replay gave (summed over incarnations): how late the
    /// history and the snapshots start.
    pub unknown_before_start: u64,
    /// Batches taken from the publisher's replay at the relay's start, before
    /// any live batch reached it.
    pub primed_batches: u64,
    /// Publisher incarnations after the first.
    pub publisher_restarts: u64,
    /// Subscribers that fell behind the live channel and were refilled.
    pub subscribers_lagged: u64,
}

/// The state the publisher task and the subscribers share.
struct Shared {
    history: History,
    /// The engine's live blocks as the relayed stream describes them,
    /// current through `cursor`.
    state: LiveState,
    /// The last sequence relayed in this incarnation, or none yet.
    cursor: Option<u64>,
    /// The first sequence relayed in this incarnation: the relay holds
    /// nothing from before it, which is how a relay that started (or
    /// restarted) after the publisher tells a resume from before its time.
    started_at: Option<u64>,
    /// Sequences of this incarnation before the record that even the
    /// engine's replay no longer had: what a snapshot cannot cover.
    unknown_before: u64,
    /// Counts the publisher's restarts; sequences compare within one.
    generation: u64,
    counts: RelayCounts,
    /// The normalizer's counters as of the last relayed batch: what was
    /// forwarded, dropped by reason, and the engine-hash check's tally.
    wire: WireCounts,
    /// Why the publisher task gave up, when it did.
    failed: Option<String>,
}

/// How the publisher task classified a sequence against the cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    Accept,
    Duplicate,
    Gap { from: u64, to: u64 },
    Restart { reason: RestartReason, last: u64 },
}

/// What showed that the publisher started over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RestartReason {
    /// The sequence went backwards on the same socket.
    SequenceRegression,
    /// The engine's startup `AllBlocksCleared` arrived under a sequence the
    /// relay had already passed (SGLang's first batch after a start).
    StartupClear,
    /// The counter is back at 0 or 1 after a cursor above them (the mock
    /// engine's publisher restart, a vLLM process restart).
    CounterRestarted,
}

impl RestartReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SequenceRegression => "sequence_regression",
            Self::StartupClear => "startup_clear",
            Self::CounterRestarted => "counter_restarted",
        }
    }
}

impl Shared {
    /// Classify `seq` against the cursor; `startup_clear` says the batch
    /// begins with the engine's `AllBlocksCleared`.
    fn admit(&self, seq: u64, startup_clear: bool) -> Admission {
        let Some(last) = self.cursor else {
            return Admission::Accept;
        };
        if seq == last + 1 {
            return Admission::Accept;
        }
        if seq > last + 1 {
            return Admission::Gap {
                from: last + 1,
                to: seq - 1,
            };
        }
        let restart = |reason| Admission::Restart { reason, last };
        if seq <= 1 && last >= 2 {
            return restart(RestartReason::CounterRestarted);
        }
        if seq < last {
            return restart(RestartReason::SequenceRegression);
        }
        if startup_clear {
            return restart(RestartReason::StartupClear);
        }
        Admission::Duplicate
    }

    /// Start a new incarnation: the window is the old publisher's.
    fn restart(&mut self) -> u64 {
        self.generation += 1;
        self.history.clear();
        self.state.clear();
        self.cursor = None;
        self.started_at = None;
        self.unknown_before = 0;
        self.counts.publisher_restarts += 1;
        self.generation
    }
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What the live channel carries to subscribers.
#[derive(Clone)]
enum Live {
    Batch(Arc<common::KvEventBatch>),
    Restart { generation: u64 },
}

/// One engine's KV-event relay: the publisher subscription, its history and
/// the live channel its subscribers read.
pub struct KvEventRelay {
    config: RelayConfig,
    shared: Arc<Mutex<Shared>>,
    live: broadcast::Sender<Live>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// A subscriber came while the relay holds nothing: the publisher task
    /// asks the engine's replay for the publisher's start again.
    prime_request: Arc<Notify>,
    /// The servicer's load, attached to every batch sent; none until the
    /// servicer installs it.
    load_source: OnceLock<Arc<dyn LoadSource>>,
}

impl Drop for KvEventRelay {
    fn drop(&mut self) {
        if let Some(task) = self
            .task
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            task.abort();
        }
        let counts = self.counts();
        let shared = lock(&self.shared);
        info!(
            endpoint = %self.config.endpoint,
            ?counts,
            wire = ?shared.wire,
            window = shared.history.len(),
            holes = shared.history.holes(),
            bytes = shared.history.bytes(),
            live_blocks = shared.state.blocks(),
            live_entries = shared.state.entries(),
            state = ?shared.state.counts(),
            "KV event relay closed"
        );
    }
}

impl KvEventRelay {
    pub fn new(config: RelayConfig) -> Arc<Self> {
        let (live, _) = broadcast::channel(LIVE_CHANNEL);
        Arc::new(Self {
            shared: Arc::new(Mutex::new(Shared {
                history: History::new(config.history_batches, config.history_bytes),
                state: LiveState::new(),
                cursor: None,
                started_at: None,
                wire: WireCounts::default(),
                unknown_before: 0,
                generation: 0,
                counts: RelayCounts::default(),
                failed: None,
            })),
            config,
            live,
            task: Mutex::new(None),
            prime_request: Arc::new(Notify::new()),
            load_source: OnceLock::new(),
        })
    }

    /// Install where the load record on every sent batch comes from (once;
    /// a second call is ignored).
    pub(crate) fn set_load_source(&self, source: Arc<dyn LoadSource>) {
        let _ = self.load_source.set(source);
    }

    /// The relay for an engine's publisher, or `None` when events are off
    /// (an empty endpoint).
    pub(crate) fn for_publisher(
        kv_events_endpoint: &str,
        replay_endpoint: Option<&str>,
        topic: &str,
    ) -> Option<Arc<Self>> {
        (!kv_events_endpoint.is_empty()).then(|| {
            Self::new(RelayConfig::for_publisher(
                kv_events_endpoint,
                replay_endpoint,
                topic,
            ))
        })
    }

    pub fn counts(&self) -> RelayCounts {
        lock(&self.shared).counts.clone()
    }

    /// The normalizer's counters as of the last relayed batch (forwarded,
    /// dropped by reason, the engine-hash check's tally); what the summary
    /// and closing log lines print.
    #[cfg(test)]
    pub(crate) fn wire_counts(&self) -> WireCounts {
        lock(&self.shared).wire.clone()
    }

    /// Record `batches` as relayed without a publisher: the history, the
    /// live state and the cursor as [`Relaying::relay`] would leave them.
    #[cfg(test)]
    fn preload(&self, batches: impl IntoIterator<Item = common::KvEventBatch>) {
        let mut shared = lock(&self.shared);
        for batch in batches {
            let sequence = batch.sequence_number;
            let batch = Arc::new(batch);
            shared.history.push(sequence, Arc::clone(&batch));
            shared.state.apply(&batch);
            if shared.started_at.is_none() {
                shared.started_at = Some(sequence);
            }
            shared.cursor = Some(sequence);
            shared.counts.relayed += 1;
        }
    }

    /// [`Self::start`] when the servicer begins serving, unless
    /// [`RELAY_START_ENV`] is `lazy`; needs a Tokio runtime.
    pub(crate) fn start_at_boot(&self) {
        if starts_at_boot(std::env::var(RELAY_START_ENV).ok().as_deref()) {
            self.start();
        } else {
            info!(
                endpoint = %self.config.endpoint,
                "{RELAY_START_ENV}=lazy: the KV event relay subscribes at the first SubscribeKvEvents"
            );
        }
    }

    /// Subscribe to the publisher if not yet subscribed. Called at the
    /// servicer's start ([`Self::start_at_boot`]) and by
    /// [`Self::subscribe`]; needs a Tokio runtime.
    pub fn start(&self) {
        let mut task = self.task.lock().unwrap_or_else(PoisonError::into_inner);
        if task.as_ref().is_some_and(|task| !task.is_finished()) {
            return;
        }
        let config = self.config.clone();
        let shared = Arc::clone(&self.shared);
        let live = self.live.clone();
        let prime_request = Arc::clone(&self.prime_request);
        #[expect(
            clippy::disallowed_methods,
            reason = "the publisher subscription outlives any one RPC; aborted when the relay drops"
        )]
        let handle = tokio::spawn(run(config, shared, live, prime_request));
        *task = Some(handle);
    }

    /// Handle one `SubscribeKvEvents` call: the history after the cursor, or
    /// a state snapshot, then live events; see the module docs for the
    /// cursor rules.
    pub fn subscribe(
        &self,
        request: common::SubscribeKvEventsRequest,
    ) -> Result<BoxStream<common::KvEventBatch>, Status> {
        self.start();
        // Subscribed before the history is read, so nothing published between
        // the two is missed; the subscriber skips what it already replayed.
        let rx = self.live.subscribe();
        let cursor = request.start_sequence_number;
        let serve = {
            let mut shared = lock(&self.shared);
            if let Some(error) = &shared.failed {
                return Err(Status::internal(format!(
                    "SubscribeKvEvents: the relay for {} is down: {error}",
                    self.config.endpoint
                )));
            }
            if cursor == 0 {
                if shared.history.complete_from_start() {
                    shared.counts.served_from_history += 1;
                    Serve::History {
                        batches: shared.history.all(),
                        last_sent: None,
                    }
                } else if let Some(through) = shared.cursor {
                    // The window no longer reaches back to the publisher's
                    // first batch: the live set as of `through`, taken under
                    // the guard that admits batches, so every later batch
                    // reaches this subscriber through the live channel.
                    let started = Instant::now();
                    let snapshot = shared.state.snapshot();
                    let collected = started.elapsed();
                    shared.counts.served_snapshots += 1;
                    Serve::Snapshot {
                        snapshot,
                        through,
                        collected,
                        oldest: shared.history.oldest(),
                        holes: shared.history.holes(),
                        unknown_before: shared.unknown_before,
                    }
                } else {
                    // Nothing relayed yet: the publisher may have spoken
                    // before the subscription reached it and after the
                    // start replay answered; ask its replay once more.
                    if self.config.replay_endpoint.is_some() {
                        self.prime_request.notify_one();
                    }
                    Serve::Live
                }
            } else {
                match shared.history.after(cursor) {
                    Ok(batches) => {
                        shared.counts.served_from_history += 1;
                        Serve::History {
                            batches,
                            last_sent: Some(cursor),
                        }
                    }
                    Err(window) => {
                        shared.counts.out_of_range += 1;
                        let endpoint = &self.config.endpoint;
                        let message = match window {
                            Window::Empty => format!(
                                "SubscribeKvEvents: the relay for {endpoint} holds no history \
                                 yet (it started after the publisher); resubscribe from zero"
                            ),
                            Window::Behind { oldest } => match shared.started_at {
                                Some(started) if cursor + 1 < started => format!(
                                    "SubscribeKvEvents: the relay for {endpoint} started at \
                                     sequence {started} and holds nothing before it; resubscribe \
                                     from zero"
                                ),
                                _ => format!(
                                    "SubscribeKvEvents: the relay for {endpoint} keeps history \
                                     from sequence {oldest}, after cursor {cursor}; resubscribe \
                                     from zero for a state snapshot"
                                ),
                            },
                            Window::Ahead { newest } => format!(
                                "SubscribeKvEvents: cursor {cursor} is beyond the publisher's \
                                 last sequence {newest} at {endpoint}: the publisher restarted; \
                                 resubscribe from zero"
                            ),
                        };
                        info!(
                            counts = ?shared.counts,
                            window = shared.history.len(),
                            holes = shared.history.holes(),
                            "{message}"
                        );
                        return Err(Status::out_of_range(message));
                    }
                }
            }
        };
        let (replay, last_sent, snapshot) = match serve {
            Serve::History { batches, last_sent } => {
                if !batches.is_empty() {
                    debug!(
                        endpoint = %self.config.endpoint,
                        cursor,
                        batches = batches.len(),
                        "SubscribeKvEvents: serving from history"
                    );
                }
                (batches, last_sent, SnapshotPhase::None)
            }
            Serve::Snapshot {
                snapshot,
                through,
                collected,
                oldest,
                holes,
                unknown_before,
            } => {
                info!(
                    endpoint = %self.config.endpoint,
                    through,
                    blocks = snapshot.blocks,
                    entries = snapshot.entries(),
                    ranks = snapshot.ranks.len(),
                    collected_us = u64::try_from(collected.as_micros()).unwrap_or(u64::MAX),
                    oldest,
                    holes,
                    unknown_before,
                    "SubscribeKvEvents: the history no longer starts at the publisher's first \
                     batch; serving a state snapshot"
                );
                let timestamp = unix_seconds();
                // Ordering and sizing the chunks is a pass over every live
                // block: off the runtime's workers, and off the lock.
                let ordering = tokio::task::spawn_blocking(move || {
                    SnapshotChunks::new(snapshot, through, timestamp, unknown_before)
                });
                (Vec::new(), Some(through), SnapshotPhase::Ordering(ordering))
            }
            Serve::Live => (Vec::new(), None, SnapshotPhase::None),
        };
        let subscriber = Subscriber {
            snapshot,
            replay: replay.into(),
            rx,
            last_sent,
            shared: Arc::clone(&self.shared),
            endpoint: self.config.endpoint.clone(),
            done: false,
            source: self.load_source.get().cloned(),
            load_tick: self.config.load_tick,
            heartbeat_interval: self.config.heartbeat_interval,
            heartbeat_backoff: self.config.heartbeat_backoff,
            last_rank: None,
            last_record: None,
            last_sent_at: Instant::now(),
            idle_heartbeats: 0,
            sample: 0,
        };
        Ok(Box::pin(stream::unfold(
            subscriber,
            |mut subscriber| async move { subscriber.next().await.map(|item| (item, subscriber)) },
        )))
    }
}

/// What a subscription begins with, decided under the relay's lock.
enum Serve {
    History {
        batches: Vec<Arc<common::KvEventBatch>>,
        last_sent: Option<u64>,
    },
    Snapshot {
        snapshot: Snapshot,
        through: u64,
        collected: Duration,
        oldest: Option<u64>,
        holes: usize,
        unknown_before: u64,
    },
    Live,
}

/// Where a subscription's snapshot stands.
enum SnapshotPhase {
    None,
    /// Being ordered and sized on the blocking pool; awaited at the first poll.
    Ordering(tokio::task::JoinHandle<SnapshotChunks>),
    Serving(SnapshotChunks),
}

/// Seconds since the Unix epoch, the way the engines stamp their batches.
fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0)
}

type Item = Result<common::KvEventBatch, Status>;

/// One `SubscribeKvEvents` stream: the snapshot it owes, what it still owes
/// from the history, then the live channel, deduplicated by sequence.
struct Subscriber {
    snapshot: SnapshotPhase,
    replay: VecDeque<Arc<common::KvEventBatch>>,
    rx: broadcast::Receiver<Live>,
    last_sent: Option<u64>,
    shared: Arc<Mutex<Shared>>,
    endpoint: String,
    done: bool,
    /// The servicer's load for the record on every batch; `None` sends
    /// batches without one and no `load_only` batches.
    source: Option<Arc<dyn LoadSource>>,
    load_tick: Duration,
    heartbeat_interval: Duration,
    heartbeat_backoff: Duration,
    /// The rank of the last batch sent: what a `load_only` batch names.
    last_rank: Option<i32>,
    /// The last record sent, for the change check.
    last_record: Option<common::EngineLoad>,
    /// When the last batch of any kind went out.
    last_sent_at: Instant,
    /// Heartbeats in a row with an unchanged record.
    idle_heartbeats: u32,
    /// Records sent on this stream.
    sample: u64,
}

/// What the live wait ended with.
enum Waited {
    Live(Result<Live, broadcast::error::RecvError>),
    Tick,
}

impl Subscriber {
    /// Attach the load record to a batch about to go out and note it went.
    fn stamped(&mut self, mut batch: common::KvEventBatch) -> common::KvEventBatch {
        self.last_rank = batch.dp_rank;
        self.last_sent_at = Instant::now();
        self.idle_heartbeats = 0;
        batch.load = self.record(batch.dp_rank, false);
        batch
    }

    /// The next record for `rank` from the source, numbered.
    fn record(&mut self, rank: Option<i32>, load_only: bool) -> Option<common::EngineLoad> {
        let mut record = self.source.as_ref()?.load(rank)?;
        self.sample += 1;
        record.sample = self.sample;
        record.load_only = load_only;
        // The telemetry rides the heartbeats and the stream's first record;
        // an event batch, one per scheduler step, carries the core only.
        if !load_only && self.sample > 1 {
            smg_grpc_client::engine_load::core_only(&mut record);
        }
        self.last_record = Some(record.clone());
        Some(record)
    }

    /// A `load_only` batch when the record moved since the last one sent,
    /// or when the stream has been silent for the heartbeat interval (the
    /// backoff interval after two unchanged heartbeats); `None` otherwise.
    fn load_only_batch(&mut self) -> Option<common::KvEventBatch> {
        // Nothing sent yet, nothing to repeat: a heartbeat numbered 0 before
        // the publisher's first batch would make a gateway that predates
        // the field take that batch for a duplicate.
        let sequence_number = self.last_sent?;
        let current = self.source.as_ref()?.load(self.last_rank)?;
        let changed = self
            .last_record
            .as_ref()
            .is_none_or(|last| load_changed(last, &current));
        let due_after = if self.idle_heartbeats >= 2 {
            self.heartbeat_backoff
        } else {
            self.heartbeat_interval
        };
        if !changed && self.last_sent_at.elapsed() < due_after {
            return None;
        }
        let rank = self.last_rank;
        let record = self.record(rank, true)?;
        self.last_sent_at = Instant::now();
        self.idle_heartbeats = if changed {
            0
        } else {
            self.idle_heartbeats.saturating_add(1)
        };
        Some(common::KvEventBatch {
            sequence_number,
            timestamp: unix_seconds(),
            events: Vec::new(),
            dp_rank: rank,
            snapshot: None,
            load: Some(record),
        })
    }

    async fn next(&mut self) -> Option<Item> {
        if self.done {
            return None;
        }
        loop {
            match &mut self.snapshot {
                SnapshotPhase::Ordering(ordering) => {
                    let chunks = match ordering.await {
                        Ok(chunks) => chunks,
                        Err(error) => {
                            self.done = true;
                            return Some(Err(Status::internal(format!(
                                "SubscribeKvEvents: the snapshot of {} could not be ordered: \
                                 {error}",
                                self.endpoint
                            ))));
                        }
                    };
                    debug!(
                        endpoint = %self.endpoint,
                        chunks = chunks.chunk_count(),
                        blocks = chunks.blocks(),
                        through = chunks.through(),
                        "SubscribeKvEvents: snapshot ordered"
                    );
                    self.snapshot = SnapshotPhase::Serving(chunks);
                    continue;
                }
                SnapshotPhase::Serving(chunks) => {
                    if let Some(chunk) = chunks.next_chunk() {
                        return Some(Ok(self.stamped(chunk)));
                    }
                    self.snapshot = SnapshotPhase::None;
                }
                SnapshotPhase::None => {}
            }
            if let Some(batch) = self.replay.pop_front() {
                self.last_sent = Some(batch.sequence_number);
                return Some(Ok(self.stamped((*batch).clone())));
            }
            let waited = if self.source.is_some() {
                let tick = self.load_tick;
                tokio::select! {
                    received = self.rx.recv() => Waited::Live(received),
                    () = tokio::time::sleep(tick) => Waited::Tick,
                }
            } else {
                Waited::Live(self.rx.recv().await)
            };
            let received = match waited {
                Waited::Tick => {
                    if let Some(batch) = self.load_only_batch() {
                        return Some(Ok(batch));
                    }
                    continue;
                }
                Waited::Live(received) => received,
            };
            match received {
                Ok(Live::Batch(batch)) => {
                    if self
                        .last_sent
                        .is_some_and(|last| batch.sequence_number <= last)
                    {
                        continue;
                    }
                    self.last_sent = Some(batch.sequence_number);
                    return Some(Ok(self.stamped((*batch).clone())));
                }
                Ok(Live::Restart { generation }) => {
                    self.done = true;
                    return Some(Err(Status::data_loss(format!(
                        "SubscribeKvEvents: publisher at {} restarted (incarnation \
                         {generation}); resubscribe from zero",
                        self.endpoint
                    ))));
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    let refill = {
                        let mut shared = lock(&self.shared);
                        shared.counts.subscribers_lagged += 1;
                        match self.last_sent {
                            Some(last) => shared.history.after(last),
                            None => Err(Window::Empty),
                        }
                    };
                    match refill {
                        Ok(batches) => {
                            debug!(
                                endpoint = %self.endpoint,
                                skipped,
                                refilled = batches.len(),
                                "SubscribeKvEvents: subscriber fell behind; refilled from history"
                            );
                            self.replay = batches.into();
                        }
                        Err(_) => {
                            self.done = true;
                            return Some(Err(Status::data_loss(format!(
                                "SubscribeKvEvents: subscriber of {} fell {skipped} batches \
                                 behind and the history no longer reaches back; resubscribe \
                                 from zero",
                                self.endpoint
                            ))));
                        }
                    }
                }
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    }
}

/// A publisher payload as the relay read it.
enum Decoded {
    Batch(WireBatch),
    Undecodable,
}

fn decode(payload: &[u8], sequence: u64) -> Decoded {
    match rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(payload) {
        Ok(batch) => Decoded::Batch(batch.0),
        Err(error) => {
            warn!(%error, sequence, "Failed to decode KV event batch; relaying it empty");
            Decoded::Undecodable
        }
    }
}

/// The publisher task: one SUB socket for the relay's lifetime.
async fn run(
    config: RelayConfig,
    shared: Arc<Mutex<Shared>>,
    live: broadcast::Sender<Live>,
    prime_request: Arc<Notify>,
) {
    let mut socket = match connect(&config.endpoint, &config.topic).await {
        Ok(socket) => socket,
        Err(status) => {
            lock(&shared).failed = Some(status.message().to_string());
            return;
        }
    };
    let mut relay = Relaying {
        config: &config,
        shared: &shared,
        live: &live,
        normalizer: Normalizer::from_env(),
        event_id: 0,
    };
    if let Some(check) = relay.normalizer.hash_check() {
        info!(
            endpoint = %config.endpoint,
            check = check.as_str(),
            "KV event engine-hash check on; its tally is in the relay's summary and closing lines"
        );
    }
    // The publisher may have been counting before the subscription reached
    // it: take what its replay still holds before the first live batch, and
    // keep asking until the engine answers (it may still be starting). A
    // live batch arriving first settles the start on its own path.
    let mut primed = config.replay_endpoint.is_none();
    let mut next_prime = tokio::time::Instant::now();
    loop {
        let received = tokio::select! {
            received = socket.recv() => received,
            () = tokio::time::sleep_until(next_prime), if !primed => {
                primed = relay.prime_from_replay().await;
                next_prime = tokio::time::Instant::now() + PRIME_RETRY;
                continue;
            }
            () = prime_request.notified(), if config.replay_endpoint.is_some() => {
                if lock(&shared).cursor.is_none() {
                    primed = false;
                    next_prime = tokio::time::Instant::now();
                }
                continue;
            }
        };
        primed = true;
        let message = match received {
            Ok(message) => message,
            Err(error) => {
                warn!(endpoint = %config.endpoint, %error, "KV event receive failed; retrying");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let Some((sequence, payload)) = split_frames(&message) else {
            continue;
        };
        let decoded = decode(payload, sequence);
        let startup_clear = matches!(
            &decoded,
            Decoded::Batch(batch)
                if matches!(batch.events.first(), Some(WireEvent::AllBlocksCleared { .. }))
        );
        let (first, mut admission) = {
            let shared = lock(&shared);
            (
                shared.cursor.is_none(),
                shared.admit(sequence, startup_clear),
            )
        };
        if first && admission == Admission::Accept && joined_late(sequence, startup_clear) {
            relay.recover_start(sequence).await;
            // The replay may have given this sequence too.
            admission = lock(&shared).admit(sequence, startup_clear);
        }
        if let Admission::Gap { from, to } = admission {
            relay.recover_gap(from..=to).await;
            // The replay may have run past the live sequence.
            admission = lock(&shared).admit(sequence, startup_clear);
        }
        match admission {
            Admission::Accept => relay.relay(sequence, decoded),
            Admission::Gap { .. } | Admission::Duplicate => {}
            Admission::Restart { reason, last } => {
                let (generation, counts) = {
                    let mut shared = lock(&shared);
                    let generation = shared.restart();
                    (generation, shared.counts.clone())
                };
                warn!(
                    endpoint = %config.endpoint,
                    sequence,
                    last,
                    generation,
                    reason = reason.as_str(),
                    ?counts,
                    "KV event publisher restarted; live subscribers end with DATA_LOSS"
                );
                let _ = live.send(Live::Restart { generation });
                relay.normalizer = Normalizer::from_env();
                // The SUB re-joins a restarted publisher a moment late too.
                if joined_late(sequence, startup_clear) {
                    relay.recover_start(sequence).await;
                    if lock(&shared).admit(sequence, startup_clear) != Admission::Accept {
                        continue;
                    }
                }
                relay.relay(sequence, decoded);
            }
        }
    }
}

/// Whether the first batch of an incarnation shows the relay joined after the
/// publisher's start: the publishers count from 0 (vLLM, SGLang) or 1 (the
/// mock engine), and a batch carrying the engine's startup clear needs
/// nothing before it.
fn joined_late(sequence: u64, startup_clear: bool) -> bool {
    sequence > 1 && !startup_clear
}

/// What a replay request gave back for a range of sequences.
struct Filled {
    recovered: u64,
    lost: u64,
    /// Holes before the first recovered batch (the whole range when nothing
    /// came back).
    unknown_before: u64,
}

/// The publisher task's per-batch work.
struct Relaying<'a> {
    config: &'a RelayConfig,
    shared: &'a Mutex<Shared>,
    live: &'a broadcast::Sender<Live>,
    normalizer: Normalizer,
    event_id: u64,
}

impl Relaying<'_> {
    /// Normalize, remember and broadcast one batch under `sequence`.
    fn relay(&mut self, sequence: u64, decoded: Decoded) {
        let (batch, undecodable) = match decoded {
            Decoded::Batch(batch) => (
                self.normalizer
                    .normalize_batch(batch, sequence, &mut self.event_id),
                false,
            ),
            Decoded::Undecodable => (
                common::KvEventBatch {
                    sequence_number: sequence,
                    ..common::KvEventBatch::default()
                },
                true,
            ),
        };
        let batch = Arc::new(batch);
        {
            let mut shared = lock(self.shared);
            shared.history.push(sequence, Arc::clone(&batch));
            shared.state.apply(&batch);
            if shared.started_at.is_none() {
                shared.started_at = Some(sequence);
            }
            shared.cursor = Some(sequence);
            shared.counts.relayed += 1;
            if undecodable {
                shared.counts.undecodable_batches += 1;
            }
            shared.wire = self.normalizer.counts().clone();
            if shared.counts.relayed.is_multiple_of(SUMMARY_EVERY_BATCHES) {
                info!(
                    endpoint = %self.config.endpoint,
                    counts = ?shared.counts,
                    wire = ?shared.wire,
                    live_blocks = shared.state.blocks(),
                    live_entries = shared.state.entries(),
                    "KV event relay summary"
                );
            }
        }
        let _ = self.live.send(Live::Batch(batch));
    }

    /// The publisher skipped `gap` on the socket: fill it from the engine's
    /// replay; what it does not give becomes holes.
    async fn recover_gap(&mut self, gap: RangeInclusive<u64>) {
        let (from, to) = (*gap.start(), *gap.end());
        lock(self.shared).counts.publisher_gaps += 1;
        let filled = self.fill(gap).await;
        let counts = lock(self.shared).counts.clone();
        warn!(
            endpoint = %self.config.endpoint,
            from,
            to,
            recovered = filled.recovered,
            lost = filled.lost,
            ?counts,
            "KV event publisher skipped sequences"
        );
    }

    /// The publisher was already past `first_seen` when the relay's
    /// subscription reached it (ZMQ delivers nothing from before a join):
    /// ask its replay for everything before, and record what even the replay
    /// no longer had as unknown before the record's start.
    async fn recover_start(&mut self, first_seen: u64) {
        let filled = self.fill(0..=first_seen - 1).await;
        let counts = {
            let mut shared = lock(self.shared);
            shared.unknown_before = filled.unknown_before;
            shared.counts.unknown_before_start += filled.unknown_before;
            shared.counts.clone()
        };
        if filled.unknown_before == 0 {
            info!(
                endpoint = %self.config.endpoint,
                first_seen,
                recovered = filled.recovered,
                ?counts,
                "KV event relay joined a publisher already counting; its replay covered the \
                 batches before"
            );
        } else {
            warn!(
                endpoint = %self.config.endpoint,
                first_seen,
                recovered = filled.recovered,
                unknown_before = filled.unknown_before,
                ?counts,
                "KV event relay joined a publisher already counting and its replay does not \
                 reach back to the start; the history and the snapshots start late"
            );
        }
    }

    /// Before any live batch of an incarnation: everything the engine's
    /// replay still holds, so a publisher that counted before the
    /// subscription reached it is known even if it never publishes again.
    /// `true` once the replay answered (batches, or an empty buffer: the
    /// engine is up and has nothing yet), `false` when it could not be
    /// reached or did not finish, to be asked again.
    async fn prime_from_replay(&mut self) -> bool {
        let Some(endpoint) = &self.config.replay_endpoint else {
            return true;
        };
        let timeout = self.config.replay_timeout;
        let (replies, complete) = match tokio::time::timeout(
            timeout * 2,
            replay(endpoint, 0, timeout),
        )
        .await
        {
            Ok(Ok(answer)) => answer,
            Ok(Err(error)) => {
                debug!(endpoint = %self.config.endpoint, %error, "KV event replay not answering yet");
                return false;
            }
            Err(_) => return false,
        };
        if lock(self.shared).cursor.is_some() {
            // A live batch got in first and settled the start.
            return true;
        }
        if replies.is_empty() {
            if complete {
                info!(
                    endpoint = %self.config.endpoint,
                    "KV event relay asked the publisher's replay at start; it holds nothing yet"
                );
            }
            return complete;
        }
        let first = replies[0].0;
        // The publishers count from 0 (vLLM, SGLang) or 1 (the mock engine).
        let unknown_before = if first <= 1 { 0 } else { first };
        if unknown_before > 0 {
            self.lose(0, first - 1);
        }
        let mut expected = first;
        let mut taken = 0u64;
        let mut last = first;
        for (sequence, payload) in replies {
            if sequence < expected {
                continue;
            }
            if expected < sequence {
                self.lose(expected, sequence - 1);
            }
            self.relay(sequence, decode(&payload, sequence));
            taken += 1;
            last = sequence;
            expected = sequence + 1;
        }
        let counts = {
            let mut shared = lock(self.shared);
            shared.unknown_before = unknown_before;
            shared.counts.unknown_before_start += unknown_before;
            shared.counts.primed_batches += taken;
            shared.counts.clone()
        };
        if unknown_before == 0 {
            info!(
                endpoint = %self.config.endpoint,
                first,
                last,
                batches = taken,
                ?counts,
                "KV event relay primed from the publisher's replay at start"
            );
        } else {
            warn!(
                endpoint = %self.config.endpoint,
                first,
                last,
                batches = taken,
                unknown_before,
                ?counts,
                "KV event relay primed from the publisher's replay at start; the replay no \
                 longer reaches back to the publisher's first batch"
            );
        }
        true
    }

    /// Fill `gap` from the engine's replay socket: what it gives back is
    /// relayed under its sequence (past the gap's end too, when the replay
    /// ran ahead of the live socket), what it does not becomes holes.
    async fn fill(&mut self, gap: RangeInclusive<u64>) -> Filled {
        let (from, to) = (*gap.start(), *gap.end());
        let replies = match &self.config.replay_endpoint {
            Some(endpoint) => match replay(endpoint, from, self.config.replay_timeout).await {
                Ok((replies, _)) => replies,
                Err(error) => {
                    warn!(
                        endpoint = %self.config.endpoint,
                        %error,
                        "KV event replay failed; the gap stays"
                    );
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        let mut expected = from;
        let mut filled = Filled {
            recovered: 0,
            lost: 0,
            unknown_before: 0,
        };
        let mut first_known = None;
        for (sequence, payload) in replies {
            if sequence < expected {
                continue;
            }
            if expected < sequence {
                filled.lost += sequence - expected;
                self.lose(expected, sequence - 1);
            }
            if first_known.is_none() {
                first_known = Some(sequence);
            }
            self.relay(sequence, decode(&payload, sequence));
            filled.recovered += 1;
            expected = sequence + 1;
        }
        if expected <= to {
            filled.lost += to + 1 - expected;
            self.lose(expected, to);
        }
        filled.unknown_before = first_known.map_or(to + 1 - from, |first| first - from);
        lock(self.shared).counts.gap_batches_recovered += filled.recovered;
        filled
    }

    /// `from..=to` passed without batches: holes in the window.
    fn lose(&mut self, from: u64, to: u64) {
        let mut shared = lock(self.shared);
        shared.history.push_lost_range(from, to);
        shared.cursor = Some(to);
        shared.counts.gap_batches_lost += to + 1 - from;
    }
}

/// Ask a publisher's replay ROUTER for its buffered batches from `from`:
/// `[b"", from as 8 bytes big-endian]` on a DEALER; replies are
/// `[b"", topic, seq, payload]` (vLLM) or `[b"", seq, payload]` (SGLang),
/// ending with the all-ones sequence. Returned in arrival order with whether
/// the end marker arrived; a stop before it returns what arrived.
async fn replay(
    endpoint: &str,
    from: u64,
    timeout: Duration,
) -> Result<(Vec<(u64, Vec<u8>)>, bool), String> {
    let mut dealer = DealerSocket::new();
    dealer
        .connect(endpoint)
        .await
        .map_err(|error| format!("could not connect to replay {endpoint}: {error}"))?;
    let mut request = ZmqMessage::from(Vec::new());
    request.push_back(from.to_be_bytes().to_vec().into());
    dealer
        .send(request)
        .await
        .map_err(|error| format!("could not send the replay request to {endpoint}: {error}"))?;
    let mut replies = Vec::new();
    loop {
        let reply = match tokio::time::timeout(timeout, dealer.recv()).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(error)) => return Err(format!("replay from {endpoint} failed: {error}")),
            Err(_) => {
                warn!(
                    endpoint,
                    from,
                    received = replies.len(),
                    "KV event replay timed out"
                );
                return Ok((replies, false));
            }
        };
        let (sequence, payload) = match reply.len() {
            3 => (reply.get(1), reply.get(2)),
            4 => (reply.get(2), reply.get(3)),
            frames => {
                return Err(format!(
                    "malformed replay reply from {endpoint}: {frames} frames"
                ))
            }
        };
        let (Some(sequence), Some(payload)) = (sequence, payload) else {
            return Err(format!("malformed replay reply from {endpoint}"));
        };
        if sequence.as_ref() == &END_SEQUENCE[..] {
            return Ok((replies, true));
        }
        if sequence.len() != 8 {
            return Err(format!("malformed replay sequence from {endpoint}"));
        }
        replies.push((low64_big_endian(sequence.as_ref()), payload.to_vec()));
    }
}

/// A SUB socket subscribed to `topic` and connected to `endpoint`. The
/// subscription is recorded first and sent on connect (and on the crate's
/// reconnects), as libzmq does; a refused publisher is retried until the
/// socket is dropped, as libzmq's background connect would.
async fn connect(endpoint: &str, topic: &str) -> Result<SubSocket, Status> {
    let mut options = SocketOptions::default();
    options.no_connect_timeout();
    let mut socket = SubSocket::with_options(options);
    let failed = |step: &str, error: ZmqError| {
        Status::internal(format!("SubscribeKvEvents: {step} {endpoint}: {error}"))
    };
    socket
        .subscribe(topic)
        .await
        .map_err(|error| failed("could not subscribe to", error))?;
    socket
        .connect(endpoint)
        .await
        .map_err(|error| failed("could not connect to", error))?;
    info!(%endpoint, "SubscribeKvEvents: connected to ZMQ endpoint");
    Ok(socket)
}

/// A publisher message's `[topic, sequence, payload, ...]` as the sequence
/// number and payload, or `None` for fewer than three frames.
fn split_frames(message: &ZmqMessage) -> Option<(u64, &[u8])> {
    if message.len() < 3 {
        return None;
    }
    let sequence = message.get(1)?;
    let payload = message.get(2)?;
    Some((low64_big_endian(sequence), payload.as_ref()))
}

/// Golden publisher payloads encoded by vLLM 0.30.1rc1 (msgspec 0.22) with
/// `crates/engine_servicer/scripts/generate_kv_events_golden.py`, which also
/// prints the Python relay's conversion of them (the expected protos below).
#[cfg(test)]
pub(crate) mod golden {
    use zeromq::ZmqMessage;

    /// `KVEventBatch(ts=1700000000.5, data_parallel_rank=None)` with a
    /// `BlockStored` of two sha256-byte hashes (`00..00 80 00..00`,
    /// `00..00 ff..fe`), parent 7, tokens 1..=8, block size 4, medium GPU; a
    /// `BlockStored` of int hash 0x1234, no parent, tokens [9, 10], block
    /// size 2, lora_id 3, group_idx 0, kv_cache_spec_kind full_attention; an
    /// unaligned `BlockStored` (one hash, block size 4, three tokens); a
    /// `BlockRemoved` of [0x1234, ff..ff]; an `AllBlocksCleared`.
    pub(crate) const BATCH1: &str = "93cb41d954fc402000009588a474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657392c4200000000000000000000000000000000000000000000000008000000000000000c420000000000000000000000000000000000000000000000000fffffffffffffffeb1706172656e745f626c6f636b5f6861736807a9746f6b656e5f696473980102030405060708aa626c6f636b5f73697a6504a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c08aa474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657391cd1234b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647392090aaa626c6f636b5f73697a6502a76c6f72615f696403a66d656469756dc0a96c6f72615f6e616d65c0a967726f75705f69647800b26b765f63616368655f737065635f6b696e64ae66756c6c5f617474656e74696f6e88a474797065ab426c6f636b53746f726564ac626c6f636b5f6861736865739105b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f69647393010203aa626c6f636b5f73697a6504a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c083a474797065ac426c6f636b52656d6f766564ac626c6f636b5f68617368657392cd1234c420ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa66d656469756da347505581a474797065b0416c6c426c6f636b73436c6561726564c0";

    /// `KVEventBatch(ts=1700000001.0, data_parallel_rank=1)` with one
    /// `BlockStored`: int hash 42, parent 41, tokens [100, 101], block size 2.
    pub(crate) const BATCH2: &str = "93cb41d954fc404000009188a474797065ab426c6f636b53746f726564ac626c6f636b5f686173686573912ab1706172656e745f626c6f636b5f6861736829a9746f6b656e5f696473926465aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055a96c6f72615f6e616d65c001";

    pub(crate) fn bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
            .collect()
    }

    /// A publisher message: `[topic, sequence (u64 big-endian), payload]`.
    pub(crate) fn frame(topic: &[u8], sequence: u64, payload: &[u8]) -> ZmqMessage {
        let mut message = ZmqMessage::from(topic.to_vec());
        message.push_back(sequence.to_be_bytes().to_vec().into());
        message.push_back(payload.to_vec().into());
        message
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use futures::StreamExt;
    use smg_grpc_client::common_proto::{kv_cache_event, KvCacheLocality, KvCacheTier};
    use tokio::time::timeout;
    use zeromq::{PubSocket, RouterSocket, SocketEvent};

    use super::{golden, *};
    use crate::kv_wire::WireBatch;

    fn decode(hex: &str) -> WireBatch {
        rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&golden::bytes(hex))
            .expect("golden batch decodes")
            .0
    }

    fn convert_batch(
        batch: WireBatch,
        sequence_number: u64,
        event_id: &mut u64,
    ) -> common::KvEventBatch {
        Normalizer::new().normalize_batch(batch, sequence_number, event_id)
    }

    fn stored(event: &common::KvCacheEvent) -> &common::KvBlocksStored {
        match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => stored,
            other => panic!("expected a stored event, got {other:?}"),
        }
    }

    fn block(block_hash: i64, token_ids: Vec<u32>, lora_id: Option<i64>) -> common::KvBlock {
        common::KvBlock {
            block_hash,
            block_size: i32::try_from(token_ids.len()).unwrap(),
            token_ids,
            lora_id,
            cache_level: None,
            ..Default::default()
        }
    }

    #[test]
    fn the_relay_starts_at_boot_unless_told_to_wait() {
        assert!(starts_at_boot(None));
        assert!(starts_at_boot(Some("")));
        assert!(starts_at_boot(Some("eager")));
        assert!(!starts_at_boot(Some("lazy")));
        assert!(!starts_at_boot(Some(" Lazy ")));
    }

    #[test]
    fn endpoint_for_rank_mirrors_the_python_helper() {
        assert_eq!(endpoint_for_rank("tcp://*:5557", 0), "tcp://127.0.0.1:5557");
        assert_eq!(
            endpoint_for_rank("tcp://0.0.0.0:5557", 0),
            "tcp://127.0.0.1:5557"
        );
        assert_eq!(endpoint_for_rank("tcp://*:5557", 2), "tcp://127.0.0.1:5559");
        assert_eq!(
            endpoint_for_rank("tcp://10.0.0.1:5557", 1),
            "tcp://10.0.0.1:5558"
        );
        assert_eq!(endpoint_for_rank("tcp://host:port", 1), "tcp://host:port");
        assert_eq!(endpoint_for_rank("ipc:///tmp/kv", 1), "ipc:///tmp/kv");
    }

    /// The golden batches convert to exactly what the Python relay produced
    /// for them: sha256 hashes reduced to their low 64 bits, an unaligned
    /// store skipped but its event id consumed, `lora_id` and the parent
    /// carried, `dp_rank` set only when the publisher set it.
    #[test]
    fn golden_batches_convert_like_the_python_relay() {
        let mut event_id = 0;
        let batch = convert_batch(decode(golden::BATCH1), 9, &mut event_id);
        assert_eq!(event_id, 5);
        assert_eq!(batch.sequence_number, 9);
        assert!((batch.timestamp - 1_700_000_000.5).abs() < f64::EPSILON);
        assert_eq!(batch.dp_rank, None);
        assert_eq!(
            batch
                .events
                .iter()
                .map(|event| event.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 5]
        );
        let first = stored(&batch.events[0]);
        assert_eq!(first.parent_block_hash, Some(7));
        assert_eq!(
            first.blocks,
            vec![
                block(i64::MIN, vec![1, 2, 3, 4], None),
                block(-2, vec![5, 6, 7, 8], None),
            ]
        );
        let second = stored(&batch.events[1]);
        assert_eq!(second.parent_block_hash, None);
        assert_eq!(second.blocks, vec![block(0x1234, vec![9, 10], Some(3))]);
        assert_eq!(
            batch.events[2].data,
            Some(kv_cache_event::Data::Removed(common::KvBlocksRemoved {
                block_hashes: vec![0x1234, -1],
                cache_level: None,
                tier: Some(KvCacheTier::Device as i32),
                medium: Some("GPU".to_string()),
                locality: Some(KvCacheLocality::Local as i32),
                ..Default::default()
            }))
        );
        assert_eq!(
            batch.events[3].data,
            Some(kv_cache_event::Data::Cleared(
                common::KvCacheCleared::default()
            ))
        );

        let batch = convert_batch(decode(golden::BATCH2), 10, &mut event_id);
        assert_eq!(event_id, 6);
        assert_eq!(batch.sequence_number, 10);
        assert_eq!(batch.dp_rank, Some(1));
        assert_eq!(batch.events[0].event_id, 6);
        let only = stored(&batch.events[0]);
        assert_eq!(only.parent_block_hash, Some(41));
        assert_eq!(only.blocks, vec![block(42, vec![100, 101], None)]);
    }

    /// The batch array may omit the trailing rank and may grow new fields;
    /// an event of a type this relay does not convert is skipped on its own
    /// (consuming its event id), as the Python relay skips unknown types.
    #[test]
    fn batch_layout_tolerates_an_omitted_rank_and_trailing_fields() {
        let short = rmp_serde::to_vec(&(1.5f64, Vec::<u8>::new())).unwrap();
        let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&short)
            .unwrap()
            .0;
        assert!(batch.events.is_empty());
        assert_eq!(batch.dp_rank, None);

        let long = rmp_serde::to_vec(&(1.5f64, Vec::<u8>::new(), 2i32, "future")).unwrap();
        let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&long)
            .unwrap()
            .0;
        assert_eq!(batch.dp_rank, Some(2));

        let unknown = rmp_serde::to_vec(&serde_json::json!([
            1.5,
            [{"type": "Mystery", "x": 1}, {"type": "AllBlocksCleared"}]
        ]))
        .unwrap();
        let batch = rmp_serde::from_slice::<TrailingTolerant<WireBatch>>(&unknown)
            .unwrap()
            .0;
        let mut event_id = 4;
        let converted = convert_batch(batch, 7, &mut event_id);
        assert_eq!(event_id, 6, "the skipped event still consumed an id");
        assert_eq!(converted.events.len(), 1);
        assert_eq!(converted.events[0].event_id, 6);
        assert!(matches!(
            converted.events[0].data,
            Some(kv_cache_event::Data::Cleared(_))
        ));
    }

    #[test]
    fn frames_are_split_like_the_python_relay() {
        let payload = golden::bytes(golden::BATCH2);
        let message = golden::frame(b"kv", 5, &payload);
        let (sequence, body) = split_frames(&message).expect("three frames");
        assert_eq!(sequence, 5);
        assert_eq!(body, payload.as_slice());

        let mut short = ZmqMessage::from(b"kv".to_vec());
        short.push_back(5u64.to_be_bytes().to_vec().into());
        assert!(split_frames(&short).is_none());

        // A shorter sequence frame zero-extends, as `int.from_bytes` does.
        let mut narrow = ZmqMessage::from(b"kv".to_vec());
        narrow.push_back(vec![1, 2].into());
        narrow.push_back(payload.into());
        assert_eq!(
            split_frames(&narrow).map(|(sequence, _)| sequence),
            Some(0x0102)
        );
    }

    struct Lab {
        publisher: PubSocket,
        router: Option<RouterSocket>,
        relay: Arc<KvEventRelay>,
    }

    /// A local publisher (and replay ROUTER when asked) with a started relay.
    async fn start_lab(history_batches: usize, with_replay: bool) -> Lab {
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        let (router, replay_endpoint) = if with_replay {
            let mut router = RouterSocket::new();
            let endpoint = router
                .bind("tcp://127.0.0.1:0")
                .await
                .expect("router binds")
                .to_string();
            (Some(router), Some(endpoint))
        } else {
            (None, None)
        };
        let relay = KvEventRelay::new(RelayConfig {
            endpoint,
            replay_endpoint,
            topic: "kv".to_string(),
            history_batches,
            history_bytes: 64 << 20,
            replay_timeout: Duration::from_secs(2),
            load_tick: DEFAULT_LOAD_TICK,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
        });
        relay.start();
        let mut lab = Lab {
            publisher,
            router,
            relay,
        };
        if with_replay {
            // The relay asks the replay socket for the publisher's start
            // before anything else; this publisher has nothing yet.
            lab.answer_replay(0, &[]).await;
        }
        lab
    }

    /// A load source the tests steer: the record the relay attaches, and how
    /// often it was asked.
    struct StubLoads {
        record: Mutex<common::EngineLoad>,
        asked: AtomicU64,
    }

    impl StubLoads {
        fn new(running: u32, waiting: u32) -> Arc<Self> {
            Arc::new(Self {
                record: Mutex::new(common::EngineLoad {
                    running_requests: running,
                    waiting_requests: waiting,
                    waiting_uncached_tokens: Some(4_096),
                    token_usage: 0.25,
                    gen_throughput: 1_200.0,
                    max_running_requests: 64,
                    cache_hit_rate: Some(0.5),
                    num_used_tokens: Some(1_024),
                    max_total_num_tokens: Some(4_096),
                    ..Default::default()
                }),
                asked: AtomicU64::new(0),
            })
        }

        fn set(&self, running: u32, waiting: u32) {
            let mut record = self.record.lock().unwrap_or_else(PoisonError::into_inner);
            record.running_requests = running;
            record.waiting_requests = waiting;
        }
    }

    impl LoadSource for StubLoads {
        fn load(&self, _dp_rank: Option<i32>) -> Option<common::EngineLoad> {
            self.asked.fetch_add(1, Ordering::Relaxed);
            Some(
                self.record
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone(),
            )
        }
    }

    /// A relay with a load source and short record intervals (`tick`,
    /// heartbeat, backoff), primed with sequence 0.
    async fn start_lab_with_loads(
        loads: Arc<StubLoads>,
        tick: Duration,
        heartbeat: Duration,
        backoff: Duration,
    ) -> Lab {
        start_lab_with_loads_at(loads, tick, heartbeat, backoff, true).await
    }

    /// [`start_lab_with_loads`], with or without the publisher's first batch.
    async fn start_lab_with_loads_at(
        loads: Arc<StubLoads>,
        tick: Duration,
        heartbeat: Duration,
        backoff: Duration,
        prime: bool,
    ) -> Lab {
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        let relay = KvEventRelay::new(RelayConfig {
            endpoint,
            replay_endpoint: None,
            topic: "kv".to_string(),
            history_batches: 100,
            history_bytes: 64 << 20,
            replay_timeout: Duration::from_secs(2),
            load_tick: tick,
            heartbeat_interval: heartbeat,
            heartbeat_backoff: backoff,
        });
        relay.set_load_source(loads);
        relay.start();
        let mut lab = Lab {
            publisher,
            router: None,
            relay,
        };
        if prime {
            lab.prime().await;
        }
        lab
    }

    fn record_of(batch: &common::KvEventBatch) -> &common::EngineLoad {
        batch.load.as_ref().expect("a load record on the batch")
    }

    /// Every batch a subscriber receives, whether from the history, live or
    /// a snapshot chunk, carries the servicer's load record, numbered per
    /// stream; a relay without a source sends none.
    #[tokio::test]
    async fn every_batch_a_subscriber_receives_carries_the_load_record() {
        let loads = StubLoads::new(3, 1);
        let mut lab = start_lab_with_loads(
            loads,
            Duration::from_secs(10),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )
        .await;
        let batch2 = golden::bytes(golden::BATCH2);
        lab.publish(1, &batch2).await;
        lab.wait_relayed(2).await;
        // History first (0 and 1), then live (2).
        let mut stream = lab.subscribe(0).expect("history");
        let first = read(&mut stream).await;
        assert_eq!(first.sequence_number, 0);
        let record = record_of(&first);
        assert_eq!(
            (
                record.running_requests,
                record.waiting_requests,
                record.waiting_uncached_tokens,
                record.sample,
                record.load_only
            ),
            (3, 1, Some(4_096), 1, false)
        );
        // The first record carries the telemetry; the next event batch's
        // record is the core only.
        assert_eq!(record.cache_hit_rate, Some(0.5));
        assert_eq!(record.max_total_num_tokens, Some(4_096));
        let second = read(&mut stream).await;
        assert_eq!(record_of(&second).sample, 2);
        assert_eq!(record_of(&second).cache_hit_rate, None);
        assert_eq!(record_of(&second).max_total_num_tokens, None);
        lab.publish(2, &batch2).await;
        let live = read(&mut stream).await;
        assert_eq!((live.sequence_number, record_of(&live).sample), (2, 3));
        // A snapshot chunk (the window of this second lab has no start).
        let mut rolled = start_lab_with_loads(
            StubLoads::new(1, 0),
            Duration::from_secs(10),
            Duration::from_secs(10),
            Duration::from_secs(10),
        )
        .await;
        rolled.publish(1, &batch2).await;
        rolled.wait_relayed(2).await;
        let mut whole = rolled.subscribe(0).expect("history");
        assert_eq!(record_of(&read(&mut whole).await).running_requests, 1);
        // No source: no record.
        let mut plain = start_lab(10, false).await;
        plain.prime().await;
        let mut bare = plain.subscribe(0).expect("history");
        assert!(read(&mut bare).await.load.is_none());
    }

    /// While the publisher is quiet the stream still speaks: a `load_only`
    /// batch (no events, the last sequence repeated) when the record moved,
    /// a heartbeat after the interval, and after two unchanged heartbeats
    /// only every backoff interval.
    #[tokio::test]
    async fn a_quiet_publishers_stream_sends_load_only_batches_and_heartbeats() {
        let loads = StubLoads::new(2, 0);
        let mut lab = start_lab_with_loads(
            Arc::clone(&loads),
            Duration::from_millis(20),
            Duration::from_millis(200),
            Duration::from_millis(600),
        )
        .await;
        let mut stream = lab.subscribe(0).expect("history");
        let first = read(&mut stream).await;
        assert_eq!(first.sequence_number, 0);
        // A change with no KV event: a load-only batch within a tick or two.
        let started = Instant::now();
        loads.set(5, 2);
        let change = read(&mut stream).await;
        let record = record_of(&change);
        assert!(load_only_marker(&change), "{change:?}");
        assert_eq!((change.sequence_number, change.events.len()), (0, 0));
        assert_eq!((record.running_requests, record.waiting_requests), (5, 2));
        // A heartbeat carries the telemetry.
        assert_eq!(record.cache_hit_rate, Some(0.5));
        assert!(
            started.elapsed() < Duration::from_millis(150),
            "the change took {:?}",
            started.elapsed()
        );
        // Nothing changes: heartbeats at the interval, then at the backoff.
        let mut gaps = Vec::new();
        let mut last = Instant::now();
        for _ in 0..4 {
            let beat = read(&mut stream).await;
            assert!(load_only_marker(&beat));
            assert_eq!(record_of(&beat).running_requests, 5);
            gaps.push(last.elapsed());
            last = Instant::now();
        }
        assert!(
            gaps[0] >= Duration::from_millis(150) && gaps[0] < Duration::from_millis(500),
            "first heartbeat after {:?}",
            gaps[0]
        );
        assert!(
            gaps[1] < Duration::from_millis(500),
            "second heartbeat after {:?}",
            gaps[1]
        );
        assert!(
            gaps[2] >= Duration::from_millis(500) && gaps[3] >= Duration::from_millis(500),
            "backed off: {gaps:?}"
        );
        // A real batch resets the backoff and carries the record itself.
        lab.publish(1, &golden::bytes(golden::BATCH2)).await;
        let live = read(&mut stream).await;
        assert_eq!(live.sequence_number, 1);
        assert!(!load_only_marker(&live));
        let beat = read(&mut stream).await;
        assert!(load_only_marker(&beat));
        assert_eq!(beat.sequence_number, 1, "repeats the last sequence sent");
    }

    fn load_only_marker(batch: &common::KvEventBatch) -> bool {
        batch.load.as_ref().is_some_and(|load| load.load_only)
    }

    /// A relay whose replay socket is not bound yet: the relay keeps asking
    /// for the publisher's start until it is.
    async fn start_lab_without_replay_socket_yet() -> (Lab, String) {
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("publisher binds")
            .to_string();
        let replay_endpoint = format!(
            "tcp://127.0.0.1:{}",
            portpicker::pick_unused_port().expect("a free replay port")
        );
        let relay = KvEventRelay::new(RelayConfig {
            endpoint,
            replay_endpoint: Some(replay_endpoint.clone()),
            topic: "kv".to_string(),
            history_batches: 100,
            history_bytes: 64 << 20,
            replay_timeout: Duration::from_secs(2),
            load_tick: DEFAULT_LOAD_TICK,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
        });
        relay.start();
        (
            Lab {
                publisher,
                router: None,
                relay,
            },
            replay_endpoint,
        )
    }

    impl Lab {
        async fn publish(&mut self, sequence: u64, payload: &[u8]) {
            self.publisher
                .send(golden::frame(b"kv", sequence, payload))
                .await
                .expect("publish");
        }

        /// The subscription reaches the publisher a moment after the
        /// connect; publish sequence 0 until the relay has it.
        async fn prime(&mut self) {
            let batch1 = golden::bytes(golden::BATCH1);
            for _ in 0..250 {
                self.publish(0, &batch1).await;
                if self.relay.counts().relayed >= 1 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            panic!("the relay never saw sequence 0");
        }

        async fn wait_relayed(&self, count: u64) {
            for _ in 0..250 {
                if self.relay.counts().relayed >= count {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            panic!(
                "the relay relayed {} batches, not {count}",
                self.relay.counts().relayed
            );
        }

        fn subscribe(&self, cursor: u64) -> Result<BoxStream<common::KvEventBatch>, Status> {
            self.relay.subscribe(common::SubscribeKvEventsRequest {
                start_sequence_number: cursor,
            })
        }

        /// Answer the next replay request as vLLM's ROUTER does, with
        /// `replies` and the end marker; the request must ask from `start`.
        async fn answer_replay(&mut self, start: u64, replies: &[(u64, Vec<u8>)]) {
            let router = self.router.as_mut().expect("a replay socket");
            let request = timeout(Duration::from_secs(5), router.recv())
                .await
                .expect("a replay request in time")
                .expect("request");
            let frames = replay_request_frames(&request, start);
            self.send_replay(&frames, replies).await;
        }

        /// Publish `sequence` (the subscription reaches the publisher a
        /// moment after the connect) until the relay asks the replay socket,
        /// which it must do from `start`; the request's frames.
        async fn publish_until_replay_requested(
            &mut self,
            sequence: u64,
            payload: &[u8],
            start: u64,
        ) -> Vec<Vec<u8>> {
            let router = self.router.as_mut().expect("a replay socket");
            for _ in 0..250 {
                self.publisher
                    .send(golden::frame(b"kv", sequence, payload))
                    .await
                    .expect("publish");
                if let Ok(request) = timeout(Duration::from_millis(20), router.recv()).await {
                    return replay_request_frames(&request.expect("request"), start);
                }
            }
            panic!("the relay never asked the replay socket");
        }

        /// Reply to the request `frames` with `replies` and the end marker.
        async fn send_replay(&mut self, frames: &[Vec<u8>], replies: &[(u64, Vec<u8>)]) {
            let router = self.router.as_mut().expect("a replay socket");
            for (sequence, payload) in replies {
                let mut reply = ZmqMessage::from(frames[0].clone());
                reply.push_back(Vec::new().into());
                reply.push_back(b"kv".to_vec().into());
                reply.push_back(sequence.to_be_bytes().to_vec().into());
                reply.push_back(payload.clone().into());
                router.send(reply).await.expect("reply");
            }
            let mut end = ZmqMessage::from(frames[0].clone());
            end.push_back(Vec::new().into());
            end.push_back(Vec::new().into());
            end.push_back(END_SEQUENCE.to_vec().into());
            end.push_back(Vec::new().into());
            router.send(end).await.expect("end marker");
        }
    }

    /// A replay request's frames (`[identity, empty, start]`), checked to
    /// ask from `start`.
    fn replay_request_frames(request: &ZmqMessage, start: u64) -> Vec<Vec<u8>> {
        let frames: Vec<Vec<u8>> = request.iter().map(|frame| frame.to_vec()).collect();
        assert_eq!(frames.len(), 3, "[identity, empty, start]");
        assert_eq!(frames[1], b"");
        assert_eq!(frames[2], start.to_be_bytes());
        frames
    }

    fn refused(result: Result<BoxStream<common::KvEventBatch>, Status>) -> Status {
        match result {
            Err(status) => status,
            Ok(_) => panic!("the subscription was accepted"),
        }
    }

    async fn read(stream: &mut BoxStream<common::KvEventBatch>) -> common::KvEventBatch {
        timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("a batch in time")
            .expect("stream open")
            .expect("a batch")
    }

    async fn read_error(stream: &mut BoxStream<common::KvEventBatch>) -> Status {
        timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("an item in time")
            .expect("stream open")
            .expect_err("an error")
    }

    async fn read_end(stream: &mut BoxStream<common::KvEventBatch>) {
        assert!(timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("the end in time")
            .is_none());
    }

    #[tokio::test]
    async fn history_serves_a_cursor_inside_the_window_then_live() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        for sequence in 1..=5 {
            lab.publish(sequence, &batch2).await;
        }
        lab.wait_relayed(6).await;

        let mut stream = lab.subscribe(2).expect("inside the window");
        for expected in 3..=5 {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        lab.publish(6, &batch2).await;
        assert_eq!(read(&mut stream).await.sequence_number, 6);

        // The newest sequence as the cursor: nothing owed, live from here.
        let mut caught_up = lab.subscribe(6).expect("at the newest");
        lab.publish(7, &batch2).await;
        assert_eq!(read(&mut caught_up).await.sequence_number, 7);
        assert_eq!(read(&mut stream).await.sequence_number, 7);
        assert_eq!(lab.relay.counts().served_from_history, 2);
    }

    #[tokio::test]
    async fn cursors_outside_the_window_are_out_of_range() {
        let fresh = start_lab(10, false).await;
        let status = refused(fresh.subscribe(1));
        assert_eq!(status.code(), tonic::Code::OutOfRange);
        assert!(status.message().contains("holds no history yet"));

        let mut lab = start_lab(3, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        for sequence in 1..=9 {
            lab.publish(sequence, &batch2).await;
        }
        lab.wait_relayed(10).await;
        // The window is 7..=9: cursor 6 wants 7, served; cursor 5 wants 6, gone.
        let mut stream = lab.subscribe(6).expect("the oldest batch is wanted");
        for expected in 7..=9 {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        let behind = refused(lab.subscribe(5));
        assert_eq!(behind.code(), tonic::Code::OutOfRange);
        assert!(behind.message().contains("from sequence 7"));
        let ahead = refused(lab.subscribe(20));
        assert_eq!(ahead.code(), tonic::Code::OutOfRange);
        assert!(ahead.message().contains("publisher restarted"));
        assert_eq!(lab.relay.counts().out_of_range, 2);
    }

    #[tokio::test]
    async fn a_subscriber_without_a_cursor_gets_the_whole_history_or_a_snapshot() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        lab.publish(1, &batch2).await;
        lab.publish(2, &batch2).await;
        lab.wait_relayed(3).await;
        // Everything since the publisher's first batch is here: hand it over.
        let mut stream = lab.subscribe(0).expect("live");
        for expected in 0..=2 {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        lab.publish(3, &batch2).await;
        assert_eq!(read(&mut stream).await.sequence_number, 3);

        // A relay that joined after the publisher's first batches has an
        // incomplete window: what it knows of the state, as a snapshot cut
        // at its newest sequence, then live.
        let mut late = start_lab(100, false).await;
        for _ in 0..250 {
            late.publish(5, &batch2).await;
            if late.relay.counts().relayed >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(late.relay.counts().relayed, 1, "the relay saw sequence 5");
        let mut stream = late.subscribe(0).expect("a snapshot");
        let chunk = read(&mut stream).await;
        assert_eq!(chunk.sequence_number, 5, "cut at the newest sequence");
        assert_eq!(
            chunk.snapshot,
            Some(common::KvSnapshotChunk {
                index: 0,
                count: 1,
                blocks: 1,
                unknown_before: 5,
            }),
            "the five sequences before the relay joined are unknown"
        );
        assert_eq!(chunk.dp_rank, Some(1));
        assert!(is_cleared(&chunk.events[0]));
        assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
        late.publish(6, &batch2).await;
        assert_eq!(
            read(&mut stream).await.sequence_number,
            6,
            "live from the next sequence"
        );
        assert_eq!(late.relay.counts().served_snapshots, 1);
    }

    #[tokio::test]
    async fn a_publisher_gap_is_filled_from_the_engines_replay() {
        let mut lab = start_lab(100, true).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        lab.publish(1, &batch2).await;
        lab.wait_relayed(2).await;
        let mut stream = lab.subscribe(1).expect("caught up");

        // 2 and 3 never reach the SUB; 4 reveals the gap.
        lab.publish(4, &batch2).await;
        lab.answer_replay(
            2,
            &[
                (2, batch2.clone()),
                (3, batch2.clone()),
                (4, batch2.clone()),
            ],
        )
        .await;
        for expected in 2..=4 {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        lab.publish(5, &batch2).await;
        assert_eq!(read(&mut stream).await.sequence_number, 5);
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.publisher_gaps,
                counts.gap_batches_recovered,
                counts.gap_batches_lost
            ),
            (1, 3, 0)
        );
        assert_eq!(
            counts.relayed, 6,
            "the replay's copy of 4 was relayed, the live one skipped"
        );
        // The window is complete: a cursor inside it is served across the
        // recovered stretch.
        let mut resumed = lab.subscribe(1).expect("served");
        for expected in 2..=5 {
            assert_eq!(read(&mut resumed).await.sequence_number, expected);
        }
    }

    #[tokio::test]
    async fn a_gap_the_engine_cannot_fill_leaves_a_hole_not_a_dead_end() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        lab.publish(1, &batch2).await;
        lab.wait_relayed(2).await;
        let mut stream = lab.subscribe(1).expect("caught up");
        lab.publish(4, &batch2).await;
        assert_eq!(
            read(&mut stream).await.sequence_number,
            4,
            "the live stream jumps"
        );
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.publisher_gaps,
                counts.gap_batches_recovered,
                counts.gap_batches_lost
            ),
            (1, 0, 2)
        );
        // A resume from inside the hole or before it skips it, as the gateway
        // then settles the gap itself instead of looping on OUT_OF_RANGE.
        let mut from_before = lab.subscribe(1).expect("served");
        assert_eq!(read(&mut from_before).await.sequence_number, 4);
        let mut from_inside = lab.subscribe(2).expect("served");
        assert_eq!(read(&mut from_inside).await.sequence_number, 4);
        // The window has a hole, so it is not the publisher's whole state: a
        // subscriber without a cursor gets the snapshot (both copies of block
        // 42, from sequences 1 and 4) cut at 4, then live.
        let mut no_cursor = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut no_cursor).await;
        assert_eq!(chunk.sequence_number, 4);
        assert_eq!(chunk.snapshot.as_ref().map(|chunk| chunk.blocks), Some(2));
        assert_eq!(
            stored_hashes(&chunk),
            vec![(Some(41), vec![42]), (Some(41), vec![42])]
        );
        lab.publish(5, &batch2).await;
        assert_eq!(read(&mut no_cursor).await.sequence_number, 5);
    }

    #[tokio::test]
    async fn a_partial_replay_recovers_what_it_can_and_loses_the_rest() {
        let mut lab = start_lab(100, true).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        let mut stream = lab.subscribe(0).expect("live");
        assert_eq!(read(&mut stream).await.sequence_number, 0);
        lab.publish(5, &batch2).await;
        // The engine's buffer starts at 3: 1 and 2 are gone for good.
        lab.answer_replay(1, &[(3, batch2.clone()), (4, batch2.clone())])
            .await;
        for expected in [3, 4, 5] {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        let counts = lab.relay.counts();
        assert_eq!(
            (counts.gap_batches_recovered, counts.gap_batches_lost),
            (2, 2)
        );
    }

    #[tokio::test]
    async fn undecodable_payloads_relay_as_empty_batches() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let mut stream = lab.subscribe(0).expect("live");
        assert_eq!(read(&mut stream).await.events.len(), 4);
        lab.publish(1, b"not msgpack").await;
        let empty = read(&mut stream).await;
        assert_eq!((empty.sequence_number, empty.events.len()), (1, 0));
        let mut short = ZmqMessage::from(b"kv".to_vec());
        short.push_back(2u64.to_be_bytes().to_vec().into());
        lab.publisher.send(short).await.expect("publish");
        lab.publish(2, &golden::bytes(golden::BATCH2)).await;
        assert_eq!(
            read(&mut stream).await.sequence_number,
            2,
            "a short frame is nothing"
        );
        assert_eq!(lab.relay.counts().undecodable_batches, 1);
    }

    #[tokio::test]
    async fn a_sequence_regression_ends_live_streams_and_clears_the_history() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        for sequence in 1..=3 {
            lab.publish(sequence, &batch2).await;
        }
        lab.wait_relayed(4).await;
        let mut stream = lab.subscribe(0).expect("whole history");
        for expected in 0..=3 {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }

        // The publisher restarts and counts from 0 again.
        lab.publish(0, &batch2).await;
        let status = read_error(&mut stream).await;
        assert_eq!(status.code(), tonic::Code::DataLoss);
        assert!(status.message().contains("restarted"));
        read_end(&mut stream).await;
        assert_eq!(lab.relay.counts().publisher_restarts, 1);

        // The old incarnation's cursor is refused; the new one's complete
        // history is handed to a fresh subscriber.
        let stale = refused(lab.subscribe(3));
        assert_eq!(stale.code(), tonic::Code::OutOfRange);
        let mut fresh = lab
            .subscribe(0)
            .expect("the new incarnation from its start");
        assert_eq!(read(&mut fresh).await.sequence_number, 0);
        lab.publish(1, &batch2).await;
        assert_eq!(read(&mut fresh).await.sequence_number, 1);
    }

    fn cleared_payload() -> Vec<u8> {
        let batch = serde_json::json!([1700000002.0, [{"type": "AllBlocksCleared"}], 0]);
        rmp_serde::to_vec_named(&batch).expect("encodes")
    }

    #[test]
    fn restart_rules_read_the_same_on_every_wire() {
        let mut shared = Shared {
            history: History::new(10, usize::MAX),
            state: LiveState::new(),
            cursor: Some(500),
            started_at: Some(0),
            unknown_before: 0,
            generation: 0,
            counts: RelayCounts::default(),
            wire: WireCounts::default(),
            failed: None,
        };
        let restart = |reason| Admission::Restart { reason, last: 500 };
        assert_eq!(shared.admit(501, false), Admission::Accept);
        assert_eq!(
            shared.admit(501, true),
            Admission::Accept,
            "a flush continues the sequence"
        );
        assert_eq!(
            shared.admit(503, false),
            Admission::Gap { from: 501, to: 502 }
        );
        assert_eq!(shared.admit(500, false), Admission::Duplicate);
        assert_eq!(
            shared.admit(500, true),
            restart(RestartReason::StartupClear)
        );
        assert_eq!(
            shared.admit(499, false),
            restart(RestartReason::SequenceRegression)
        );
        assert_eq!(
            shared.admit(0, false),
            restart(RestartReason::CounterRestarted)
        );
        assert_eq!(
            shared.admit(1, true),
            restart(RestartReason::CounterRestarted)
        );
        // Under a cursor of 1, a repeated 1 is a duplicate unless it clears.
        shared.cursor = Some(1);
        assert_eq!(shared.admit(1, false), Admission::Duplicate);
        assert_eq!(
            shared.admit(1, true),
            Admission::Restart {
                reason: RestartReason::StartupClear,
                last: 1
            }
        );
        assert_eq!(
            shared.admit(0, false),
            Admission::Restart {
                reason: RestartReason::SequenceRegression,
                last: 1
            }
        );
        shared.cursor = None;
        assert_eq!(
            shared.admit(7, true),
            Admission::Accept,
            "no cursor yet: anything goes"
        );
    }

    /// SGLang's first batch after a start carries `AllBlocksCleared`; under a
    /// sequence the relay already passed it is a restart, not a duplicate.
    #[tokio::test]
    async fn the_engines_startup_clear_under_a_passed_cursor_is_a_restart() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        for sequence in 1..=3 {
            lab.publish(sequence, &batch2).await;
        }
        lab.wait_relayed(4).await;
        let mut stream = lab.subscribe(3).expect("caught up");
        // A repeated sequence without a clear is a duplicate.
        lab.publish(3, &batch2).await;
        lab.publish(4, &batch2).await;
        assert_eq!(read(&mut stream).await.sequence_number, 4);

        // The engine comes back and its startup clear lands on a sequence
        // the relay already passed.
        lab.publish(2, &cleared_payload()).await;
        let status = read_error(&mut stream).await;
        assert_eq!(status.code(), tonic::Code::DataLoss);
        assert_eq!(lab.relay.counts().publisher_restarts, 1);
        // The new incarnation began at 2 with the clear: nothing is live, and
        // a fresh subscriber is told so by a snapshot of one clear, cut at 2.
        let mut fresh = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut fresh).await;
        assert_eq!((chunk.sequence_number, chunk.events.len()), (2, 1));
        assert!(is_cleared(&chunk.events[0]));
        lab.publish(3, &batch2).await;
        assert_eq!(read(&mut fresh).await.sequence_number, 3);
    }

    /// A counter back at 0 or 1 after a cursor above them is a restart even
    /// when nothing else says so (the mock engine's restart-publisher hook,
    /// a vLLM process restart: no clear on that wire).
    #[tokio::test]
    async fn a_counter_back_at_its_start_is_a_restart() {
        let mut lab = start_lab(100, false).await;
        lab.prime().await;
        let batch2 = golden::bytes(golden::BATCH2);
        for sequence in 1..=5 {
            lab.publish(sequence, &batch2).await;
        }
        lab.wait_relayed(6).await;
        let mut stream = lab.subscribe(5).expect("caught up");
        lab.publish(1, &batch2).await;
        let status = read_error(&mut stream).await;
        assert_eq!(status.code(), tonic::Code::DataLoss);
        read_end(&mut stream).await;
        // The new incarnation started at 1, not 0: its window is not complete
        // from the publisher's first batch, so a fresh subscriber gets the
        // new incarnation's state as a snapshot (the restart emptied the old
        // one), then live.
        let mut fresh = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut fresh).await;
        assert_eq!(chunk.sequence_number, 1);
        assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
        lab.publish(2, &batch2).await;
        assert_eq!(read(&mut fresh).await.sequence_number, 2);
        let mut resumed = lab.subscribe(1).expect("inside the new window");
        assert_eq!(read(&mut resumed).await.sequence_number, 2);
    }

    /// A publisher already counting when the relay's subscription reaches
    /// it: the relay asks the engine's replay for everything from 0 before
    /// relaying what it saw, and the window is the publisher's whole life.
    #[tokio::test]
    async fn a_publisher_already_counting_when_the_relay_joins_is_replayed_from_its_start() {
        let mut lab = start_lab(100, true).await;
        let batch2 = golden::bytes(golden::BATCH2);
        // Sequences 0..=2 went out before the subscription landed; 3 is the
        // first the relay sees, and it must ask for 0 before relaying it.
        let request = lab.publish_until_replay_requested(3, &batch2, 0).await;
        lab.send_replay(
            &request,
            &[
                (0, batch2.clone()),
                (1, batch2.clone()),
                (2, batch2.clone()),
                (3, batch2.clone()),
            ],
        )
        .await;
        lab.wait_relayed(4).await;
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.gap_batches_recovered,
                counts.gap_batches_lost,
                counts.unknown_before_start,
                counts.publisher_gaps,
            ),
            (4, 0, 0, 0),
            "{counts:?}"
        );
        let mut stream = lab.subscribe(0).expect("the whole history");
        for expected in 0..=3 {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        lab.publish(4, &batch2).await;
        assert_eq!(read(&mut stream).await.sequence_number, 4);
        let counts = lab.relay.counts();
        assert_eq!(
            (counts.served_from_history, counts.served_snapshots),
            (1, 0)
        );
    }

    /// The replay that answers a late join may itself start past 0 (the
    /// engine's buffer rolled): what it gives is relayed, the sequences
    /// before it are holes and counted as unknown, a subscriber from zero
    /// gets a snapshot that says so, and cursors inside the holes are served
    /// past them.
    #[tokio::test]
    async fn a_replay_that_starts_past_zero_leaves_the_earlier_sequences_unknown() {
        let mut lab = start_lab(100, true).await;
        let batch2 = golden::bytes(golden::BATCH2);
        let request = lab.publish_until_replay_requested(7, &batch2, 0).await;
        lab.send_replay(
            &request,
            &[
                (5, batch2.clone()),
                (6, batch2.clone()),
                (7, batch2.clone()),
            ],
        )
        .await;
        lab.wait_relayed(3).await;
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.gap_batches_recovered,
                counts.gap_batches_lost,
                counts.unknown_before_start,
            ),
            (3, 5, 5),
            "{counts:?}"
        );
        let mut fresh = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut fresh).await;
        assert_eq!(chunk.sequence_number, 7);
        assert_eq!(
            chunk.snapshot,
            Some(common::KvSnapshotChunk {
                index: 0,
                count: 1,
                blocks: 3,
                unknown_before: 5,
            })
        );
        // A cursor inside the unknown stretch resumes at the first batch
        // after it; the gateway settles the jump as a gap of its own.
        let mut resumed = lab.subscribe(2).expect("served past the holes");
        for expected in [5, 6, 7] {
            assert_eq!(read(&mut resumed).await.sequence_number, expected);
        }
        lab.publish(8, &batch2).await;
        assert_eq!(read(&mut fresh).await.sequence_number, 8);
        assert_eq!(read(&mut resumed).await.sequence_number, 8);
    }

    /// Without a replay socket a late join cannot be filled: the sequences
    /// before the first seen are holes, counted as unknown, and the snapshot
    /// a fresh subscriber gets carries the count instead of passing as whole.
    #[tokio::test]
    async fn a_relay_without_a_replay_socket_marks_the_batches_before_its_start_unknown() {
        let mut late = start_lab(100, false).await;
        let batch2 = golden::bytes(golden::BATCH2);
        for _ in 0..250 {
            late.publish(5, &batch2).await;
            if late.relay.counts().relayed >= 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let counts = late.relay.counts();
        assert_eq!(
            (
                counts.relayed,
                counts.gap_batches_lost,
                counts.unknown_before_start,
                counts.publisher_gaps,
            ),
            (1, 5, 5, 0),
            "{counts:?}"
        );
        let mut fresh = late.subscribe(0).expect("a snapshot");
        let chunk = read(&mut fresh).await;
        assert_eq!((chunk.sequence_number, chunk.dp_rank), (5, Some(1)));
        assert_eq!(
            chunk.snapshot.as_ref().map(|chunk| chunk.unknown_before),
            Some(5)
        );
        assert_eq!(stored_hashes(&chunk), vec![(Some(41), vec![42])]);
        // Cursors in the unknown stretch are served from the first batch.
        let mut resumed = late.subscribe(3).expect("served past the holes");
        assert_eq!(read(&mut resumed).await.sequence_number, 5);
        late.publish(6, &batch2).await;
        assert_eq!(read(&mut resumed).await.sequence_number, 6);
        assert_eq!(read(&mut fresh).await.sequence_number, 6);
        assert_eq!(late.relay.counts().out_of_range, 0);
    }

    /// A publisher that counts from 1 (the mock engine) or whose first batch
    /// is its startup clear (SGLang) has nothing before it to ask for.
    #[test]
    fn a_first_batch_at_the_counters_start_or_with_a_clear_is_not_a_late_join() {
        assert!(!joined_late(0, false));
        assert!(!joined_late(1, false));
        assert!(joined_late(2, false));
        assert!(!joined_late(2, true));
        assert!(joined_late(500, false));
    }

    /// A publisher that published at registration and nothing since: the
    /// relay never gets a live batch to notice it by, so it takes the
    /// replay's buffer at start, before any live batch, and a subscriber
    /// from zero gets those blocks.
    #[tokio::test]
    async fn the_relay_takes_the_publishers_replay_at_start_before_any_live_batch() {
        let mut lab = start_lab_without_replay_socket_yet().await;
        let (lab, replay_endpoint) = (&mut lab.0, lab.1);
        let batch2 = golden::bytes(golden::BATCH2);
        // The engine comes up a moment after the servicer: the first attempt
        // found no replay socket, the retry finds it.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut router = RouterSocket::new();
        router.bind(&replay_endpoint).await.expect("replay binds");
        lab.router = Some(router);
        // The mock counts from 1: sequences 1..=3 went out at registration.
        lab.answer_replay(
            0,
            &[
                (1, batch2.clone()),
                (2, batch2.clone()),
                (3, batch2.clone()),
            ],
        )
        .await;
        lab.wait_relayed(3).await;
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.primed_batches,
                counts.unknown_before_start,
                counts.gap_batches_lost,
                counts.publisher_gaps,
            ),
            (3, 0, 0, 0),
            "{counts:?}"
        );
        // Nothing live ever came; the state is served from zero as a snapshot
        // cut at 3 (a window from 1 is not complete from 0).
        let mut fresh = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut fresh).await;
        assert_eq!(chunk.sequence_number, 3);
        assert_eq!(chunk.snapshot.as_ref().map(|chunk| chunk.blocks), Some(3));
        assert_eq!(
            chunk.snapshot.as_ref().map(|chunk| chunk.unknown_before),
            Some(0)
        );
        // Live continues from the publisher's next sequence.
        lab.publish(4, &batch2).await;
        assert_eq!(read(&mut fresh).await.sequence_number, 4);
        let mut resumed = lab.subscribe(2).expect("inside the window");
        for expected in [3, 4] {
            assert_eq!(read(&mut resumed).await.sequence_number, expected);
        }
    }

    /// The start replay answered empty, then the publisher spoke (the mock's
    /// registration batches) without the subscription catching them, and
    /// never again: the first subscriber from zero finds the relay empty,
    /// which makes it ask the replay once more, and receives the batches.
    #[tokio::test]
    async fn a_first_subscriber_makes_an_empty_relay_ask_the_replay_again() {
        let mut lab = start_lab(100, true).await;
        let batch2 = golden::bytes(golden::BATCH2);
        let mut stream = lab.subscribe(0).expect("live, nothing held");
        lab.answer_replay(
            0,
            &[
                (1, batch2.clone()),
                (2, batch2.clone()),
                (3, batch2.clone()),
            ],
        )
        .await;
        for expected in [1, 2, 3] {
            assert_eq!(read(&mut stream).await.sequence_number, expected);
        }
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.primed_batches,
                counts.unknown_before_start,
                counts.relayed
            ),
            (3, 0, 3),
            "{counts:?}"
        );
        lab.publish(4, &batch2).await;
        assert_eq!(read(&mut stream).await.sequence_number, 4);
        // The next subscriber from zero gets the state the relay now holds
        // (a snapshot: the window starts at 1), and asks nothing more.
        let mut next = lab.subscribe(0).expect("a snapshot");
        assert_eq!(read(&mut next).await.sequence_number, 4);
        assert_eq!(lab.relay.counts().served_snapshots, 1);
    }

    /// A replay whose buffer rolled before the relay started marks what it
    /// no longer holds as unknown, as the late-join path does.
    #[tokio::test]
    async fn a_start_replay_that_begins_past_the_publishers_start_marks_the_rest_unknown() {
        let mut lab = start_lab_without_replay_socket_yet().await;
        let (lab, replay_endpoint) = (&mut lab.0, lab.1);
        let batch2 = golden::bytes(golden::BATCH2);
        let mut router = RouterSocket::new();
        router.bind(&replay_endpoint).await.expect("replay binds");
        lab.router = Some(router);
        lab.answer_replay(0, &[(6, batch2.clone()), (7, batch2.clone())])
            .await;
        lab.wait_relayed(2).await;
        let counts = lab.relay.counts();
        assert_eq!(
            (
                counts.primed_batches,
                counts.unknown_before_start,
                counts.gap_batches_lost,
            ),
            (2, 6, 6),
            "{counts:?}"
        );
        let mut fresh = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut fresh).await;
        assert_eq!(chunk.sequence_number, 7);
        assert_eq!(
            chunk.snapshot.as_ref().map(|chunk| chunk.unknown_before),
            Some(6)
        );
    }

    #[tokio::test]
    async fn dropping_the_relay_closes_the_publisher_subscription() {
        let mut lab = start_lab(10, false).await;
        let mut monitor = lab.publisher.monitor();
        lab.prime().await;
        let mut stream = lab.subscribe(0).expect("live");
        assert_eq!(read(&mut stream).await.sequence_number, 0);
        drop(lab.relay);
        read_end(&mut stream).await;
        let disconnected = timeout(Duration::from_secs(5), async {
            while let Some(event) = monitor.next().await {
                if matches!(event, SocketEvent::Disconnected(_)) {
                    return true;
                }
            }
            false
        })
        .await
        .expect("the publisher notices in time");
        assert!(disconnected);
    }
    /// A publisher batch on rank 0 with one `BlockStored`: `hashes` chained
    /// from `parent`, two tokens per block, stamped `ts`.
    fn store_payload(ts: f64, hashes: &[i64], parent: Option<i64>) -> Vec<u8> {
        let tokens: Vec<u32> = (0..hashes.len() as u32 * 2).collect();
        let batch = serde_json::json!([ts, [{
            "type": "BlockStored",
            "block_hashes": hashes,
            "parent_block_hash": parent,
            "token_ids": tokens,
            "block_size": 2,
            "lora_id": null,
            "medium": "GPU",
        }], 0]);
        rmp_serde::to_vec_named(&batch).expect("encodes")
    }

    fn remove_payload(hashes: &[i64]) -> Vec<u8> {
        let batch = serde_json::json!([1700000003.0, [{
            "type": "BlockRemoved",
            "block_hashes": hashes,
            "medium": "GPU",
        }], 0]);
        rmp_serde::to_vec_named(&batch).expect("encodes")
    }

    fn is_cleared(event: &common::KvCacheEvent) -> bool {
        matches!(event.data, Some(kv_cache_event::Data::Cleared(_)))
    }

    /// `(parent, hashes)` of every stored event of a batch, in order.
    fn stored_hashes(batch: &common::KvEventBatch) -> Vec<(Option<i64>, Vec<i64>)> {
        batch
            .events
            .iter()
            .filter_map(|event| match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => Some((
                    stored.parent_block_hash,
                    stored.blocks.iter().map(|block| block.block_hash).collect(),
                )),
                _ => None,
            })
            .collect()
    }

    /// Once the window has rolled, a subscriber without a cursor gets the
    /// live set as a snapshot cut at the newest sequence, then live events
    /// from the next one; a batch published between the subscription and
    /// the first poll follows the snapshot, once.
    #[tokio::test]
    async fn after_the_window_rolled_a_subscriber_without_a_cursor_gets_a_snapshot_then_live() {
        let mut lab = start_lab(3, false).await;
        lab.prime().await; // sequence 0: BATCH1, whose clear leaves nothing live
        lab.publish(1, &store_payload(1.0, &[10, 11], None)).await;
        lab.publish(2, &store_payload(2.0, &[12], Some(11))).await;
        lab.publish(3, &store_payload(3.0, &[20], None)).await;
        lab.publish(4, &remove_payload(&[20])).await;
        lab.publish(5, &store_payload(5.0, &[21, 22], None)).await;
        lab.wait_relayed(6).await;
        let mut stream = lab.subscribe(0).expect("a snapshot");
        // Published before the first poll: must follow the snapshot.
        lab.publish(6, &store_payload(6.0, &[30], Some(22))).await;
        lab.wait_relayed(7).await;
        let chunk = read(&mut stream).await;
        assert_eq!(chunk.sequence_number, 5, "stamped at the cut");
        assert_eq!(
            chunk.snapshot,
            Some(common::KvSnapshotChunk {
                index: 0,
                count: 1,
                blocks: 5,
                unknown_before: 0,
            })
        );
        assert_eq!(chunk.dp_rank, Some(0));
        assert!(is_cleared(&chunk.events[0]));
        assert_eq!(
            stored_hashes(&chunk),
            vec![(None, vec![10, 11, 12]), (None, vec![21, 22])],
            "the live set as the engine stored it, chains merged"
        );
        let live = read(&mut stream).await;
        assert_eq!((live.sequence_number, live.snapshot), (6, None));
        assert_eq!(stored_hashes(&live), vec![(Some(22), vec![30])]);
        lab.publish(7, &store_payload(7.0, &[31], Some(30))).await;
        assert_eq!(read(&mut stream).await.sequence_number, 7);
        let counts = lab.relay.counts();
        assert_eq!(
            (counts.served_snapshots, counts.served_from_history),
            (1, 0)
        );
    }

    /// A cursor below the window is still refused, and the resubscription
    /// from zero the gateway answers with gets the snapshot.
    #[tokio::test]
    async fn a_stale_cursor_below_the_window_is_refused_and_zero_gets_the_snapshot() {
        let mut lab = start_lab(2, false).await;
        lab.prime().await;
        for seq in 1..=4 {
            lab.publish(seq, &store_payload(seq as f64, &[seq as i64 * 10], None))
                .await;
        }
        lab.wait_relayed(5).await;
        // The window is 3..=4; cursor 1 wants 2, gone.
        let status = refused(lab.subscribe(1));
        assert_eq!(status.code(), tonic::Code::OutOfRange);
        assert!(
            status
                .message()
                .contains("resubscribe from zero for a state snapshot"),
            "{}",
            status.message()
        );
        let mut stream = lab.subscribe(0).expect("a snapshot");
        let chunk = read(&mut stream).await;
        assert_eq!(chunk.sequence_number, 4);
        assert_eq!(chunk.snapshot.as_ref().map(|chunk| chunk.blocks), Some(4));
        assert_eq!(
            stored_hashes(&chunk),
            vec![
                (None, vec![10]),
                (None, vec![20]),
                (None, vec![30]),
                (None, vec![40])
            ]
        );
        lab.publish(5, &store_payload(5.0, &[50], None)).await;
        assert_eq!(read(&mut stream).await.sequence_number, 5);
        let counts = lab.relay.counts();
        assert_eq!((counts.out_of_range, counts.served_snapshots), (1, 1));
    }

    /// A relayed batch of `blocks` chained device blocks of 16 tokens on rank 0.
    fn synthetic_batch(seq: u64, blocks: i64) -> common::KvEventBatch {
        let first = seq as i64 * blocks + 1;
        common::KvEventBatch {
            sequence_number: seq,
            timestamp: 1.0,
            events: vec![common::KvCacheEvent {
                event_id: seq,
                data: Some(kv_cache_event::Data::Stored(common::KvBlocksStored {
                    blocks: (first..first + blocks)
                        .map(|hash| common::KvBlock {
                            block_hash: hash,
                            token_ids: (0..16).map(|i| hash as u32 ^ i).collect(),
                            block_size: 16,
                            ..Default::default()
                        })
                        .collect(),
                    parent_block_hash: None,
                    tier: Some(KvCacheTier::Device as i32),
                    medium: Some("GPU".to_string()),
                    ..Default::default()
                })),
            }],
            dp_rank: Some(0),
            snapshot: None,
            load: None,
        }
    }

    fn unix_now() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("after the epoch")
            .as_secs_f64()
    }

    /// Publish `sequences` one at a time, each stamped with its publish time
    /// and read back on `live` before the next: the relay's latency per
    /// batch in milliseconds.
    async fn publish_timed(
        lab: &mut Lab,
        sequences: std::ops::Range<u64>,
        live: &mut BoxStream<common::KvEventBatch>,
    ) -> Vec<f64> {
        let mut latencies = Vec::with_capacity((sequences.end - sequences.start) as usize);
        for seq in sequences {
            lab.publish(seq, &store_payload(unix_now(), &[-(seq as i64)], None))
                .await;
            let batch = read(live).await;
            assert_eq!(batch.sequence_number, seq);
            latencies.push((unix_now() - batch.timestamp) * 1e3);
        }
        latencies
    }

    /// A large worker's pool (676k blocks) behind a rolled window: the
    /// snapshot is cut atomically at the relay's cursor while live batches
    /// stream through to another subscriber, which sees no stall beyond the
    /// one pass under the lock; the snapshot subscriber, not read until the
    /// cut is long past, gets every chunk and then the live stream from the
    /// cut, nothing twice and nothing missing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[expect(
        clippy::print_stderr,
        reason = "the measured pause is this test's report; read it with --nocapture"
    )]
    async fn a_676k_block_snapshot_is_cut_atomically_and_does_not_stall_the_live_stream() {
        const BATCHES: u64 = 10_564;
        const PER_BATCH: i64 = 64;
        let mut lab = start_lab(50, false).await;
        lab.relay
            .preload((0..BATCHES).map(|seq| synthetic_batch(seq, PER_BATCH)));
        let preloaded = BATCHES * PER_BATCH as u64;
        let last = BATCHES - 1;
        let mut live = lab.subscribe(last).expect("at the newest sequence");
        // The SUB connects asynchronously: publish the next sequence until it lands.
        let next = last + 1;
        for _ in 0..250 {
            lab.publish(next, &store_payload(unix_now(), &[-1], None))
                .await;
            if lab.relay.counts().relayed > BATCHES {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(read(&mut live).await.sequence_number, next);
        let control = publish_timed(&mut lab, next + 1..next + 201, &mut live).await;

        // The snapshot is taken while live batches keep flowing to `live`.
        let relay = Arc::clone(&lab.relay);
        let taking = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let stream = relay.subscribe(common::SubscribeKvEventsRequest {
                start_sequence_number: 0,
            });
            (stream, started.elapsed())
        });
        let during = publish_timed(&mut lab, next + 201..next + 1_201, &mut live).await;
        let (snapshot, subscribe_took) = taking.await.expect("the subscribe call ran");
        let mut snapshot = snapshot.expect("a snapshot");
        let max = |values: &[f64]| values.iter().copied().fold(0.0f64, f64::max);
        let median = |values: &[f64]| {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[sorted.len() / 2]
        };
        eprintln!(
            "676k snapshot: subscribe call (lock held for the collection) {subscribe_took:?}; \
             live latency ms: control median {:.3} max {:.3}, during median {:.3} max {:.3}",
            median(&control),
            max(&control),
            median(&during),
            max(&during)
        );
        assert!(
            subscribe_took < Duration::from_secs(2),
            "{subscribe_took:?}"
        );
        assert!(
            max(&during) < 2_000.0,
            "a live batch waited {:.1} ms on the snapshot",
            max(&during)
        );

        // Not read until now: the chunks come whole, then live from the cut.
        let mut chunks = Vec::new();
        let first_live = loop {
            let batch = read(&mut snapshot).await;
            if batch.snapshot.is_none() {
                break batch;
            }
            chunks.push(batch);
        };
        let through = chunks.last().expect("chunks").sequence_number;
        assert!(
            (next..next + 1_201).contains(&through),
            "the cut fell inside the timed publishes: {through}"
        );
        let blocks_at_cut = preloaded + (through - last);
        let count = u32::try_from(chunks.len()).unwrap();
        assert_eq!(
            count as usize,
            (blocks_at_cut as usize).div_ceil(crate::kv_state::CHUNK_BLOCKS)
        );
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.sequence_number,
                through + 1 - u64::from(count) + index as u64,
                "stamps are consecutive up to the cut"
            );
            let marker = chunk.snapshot.as_ref().unwrap();
            assert_eq!((marker.index, marker.count), (index as u32, count));
            // Every live batch up to the cut stored one block: the state and
            // the cut agree, so the collection and the cursor were one guard.
            assert_eq!(marker.blocks, blocks_at_cut);
        }
        assert!(is_cleared(&chunks[0].events[0]));
        let emitted: usize = chunks
            .iter()
            .map(|chunk| {
                stored_hashes(chunk)
                    .iter()
                    .map(|(_, hashes)| hashes.len())
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(emitted as u64, blocks_at_cut);
        assert_eq!(
            first_live.sequence_number,
            through + 1,
            "live continues right after the cut"
        );
        let mut expected = through + 2;
        while expected < next + 1_201 {
            assert_eq!(read(&mut snapshot).await.sequence_number, expected);
            expected += 1;
        }
    }

    /// The normalizer's counters ride on the relay: readable after every
    /// relayed batch and logged with the relay's own counts, so a live run
    /// shows what was forwarded, dropped by reason and, with the engine-hash
    /// check on, verified.
    #[tokio::test]
    async fn the_relays_wire_counts_follow_the_normalizer() {
        let mut lab = start_lab(8, false).await;
        assert_eq!(lab.relay.wire_counts(), WireCounts::default());
        lab.prime().await;
        lab.publish(1, &golden::bytes(golden::BATCH2)).await;
        lab.wait_relayed(2).await;
        let wire = lab.relay.wire_counts();
        assert!(
            wire.forwarded_stored + wire.forwarded_removed + wire.forwarded_cleared > 0,
            "{wire:?}"
        );
        assert_eq!(
            wire.hash_checked, 0,
            "the check is off unless the environment asks"
        );
        assert_eq!(wire.window_only_stores, 0);
    }

    /// Before the stream has sent a batch there is no sequence a heartbeat
    /// could repeat, so a quiet publisher at the start gets none: a gateway
    /// that predates the load field would take the publisher's first batch
    /// for a duplicate of a heartbeat numbered 0. The first real batch
    /// opens the heartbeats.
    #[tokio::test]
    async fn no_heartbeat_goes_out_before_the_streams_first_batch() {
        let loads = StubLoads::new(2, 0);
        let mut lab = start_lab_with_loads_at(
            Arc::clone(&loads),
            Duration::from_millis(20),
            Duration::from_millis(60),
            Duration::from_millis(100),
            false,
        )
        .await;
        let mut stream = lab.subscribe(0).expect("live from the start");
        loads.set(5, 1);
        assert!(
            timeout(Duration::from_millis(300), stream.next())
                .await
                .is_err(),
            "no batch may precede the publisher's first"
        );
        lab.prime().await;
        let first = read(&mut stream).await;
        assert_eq!(first.sequence_number, 0);
        assert!(!load_only_marker(&first));
        let beat = read(&mut stream).await;
        assert!(load_only_marker(&beat));
        assert_eq!(beat.sequence_number, 0);
    }
}
