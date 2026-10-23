//! `SubscribeKvEvents`: an engine's ZMQ KV-cache event publishers relayed into
//! gRPC streams through one [`KvEventRelay`] per engine.
//!
//! An engine runs one publisher per data-parallel rank (rank `r` on the base
//! port plus `r`, its replay socket laid out the same way, its own sequence
//! counter), and the relay follows every rank ([`RankSource`], derived from
//! the engine's data-parallel width as the Python relay's `rank_sources`
//! derives them): one SUB socket, one window and one cursor per rank, and
//! every batch stamped with its socket's rank (the socket's rank wins over
//! what the payload names, as in the Python relay). Each rank's own sequence
//! numbers are kept, which is what the gateway's per-rank admission cursors
//! expect. A subscription's `start_sequence_number` names one rank's
//! sequence (the request carries one cursor): a relay over one rank resumes
//! after it from its window; a relay over several ranks refuses it with
//! `OUT_OF_RANGE`, since the other ranks' cursors are unknown to it and
//! nothing below a cursor the gateway holds may be resent on a fresh stream
//! (the gateway reads that as a publisher restart), so the gateway clears
//! the worker and resubscribes from zero, which is served from every rank's
//! window or from a state snapshot whose chunks are stamped at each rank's
//! own cursor: a resync per reconnect, lossless, until the request carries
//! a cursor per rank.
//!
//! The relay subscribes to the publishers once, for the servicer's lifetime,
//! when the servicer starts serving (so it sees the engine from its first
//! batch, and a servicer that outlives a gateway outage or restarts beside a
//! warm engine is not blind until eviction; `SMG_KV_EVENT_RELAY_START=lazy`
//! defers the subscription to the first `SubscribeKvEvents` for a
//! memory-constrained host). The servicer starts minutes before the engine
//! binds its publisher, so the relay knocks on the endpoint's port every
//! second (`DIAL_INTERVAL`, a plain connect) until it is bound, then joins
//! over ZMQ, and the subscription lands within a second of the bind, before
//! the engine's handshake; the lifecycle owner
//! reads [`KvEventRelay::subscribed`] and holds its planned warm-up until
//! then, so the warm-up's batches are the first the history records. The
//! relay keeps what it relays in a bounded [`History`]
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
//! The publisher's own sequence numbers are kept, per rank. A sequence the
//! relay did not receive is asked of that rank's replay socket (vLLM's and
//! SGLang's ROUTER, the mock engine's too); what the replay cannot give leaves a hole
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
//! state. An `AllBlocksCleared` among the batches the relay records, live
//! or from the replay, ends the unknown stretch: nothing from before it is
//! live, so the snapshots that follow are whole and say so.
//!
//! A publisher restart is read the same way on every wire, from three signs
//! (vLLM and SGLang count from 0 per process; SGLang's first batch after a
//! start carries `AllBlocksCleared`, vLLM's does not): the sequence goes
//! backwards on the same socket; the engine's startup clear arrives under a
//! sequence the relay already passed; the counter is back at 0 or 1 after a
//! cursor above them. Each starts a new incarnation of that rank: its window
//! and its live blocks are cleared, live subscribers end with `DATA_LOSS`, and
//! the gateway clears and resubscribes from zero, where the new incarnation's
//! complete history is waiting for it (the other ranks' windows stay). A
//! repeated sequence without a clear is a duplicate.
//!
//! Every batch a subscriber receives carries the servicer's load record
//! (`LoadSource`, the figures `GetLoads` answers with, as `KvEventBatch.
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
//! arrive: `GetLoads` is the fallback. Before the publisher's first batch
//! there is no sequence a `load_only` batch could repeat: one goes out
//! numbered 0, from the subscription on, only to a subscriber that asked for
//! records before the first batch (`SubscribeKvEventsRequest.
//! load_records_before_first_batch`, set by every gateway that knows
//! `load_only`), so the engine's silence on the first request it ever
//! receives reaches the gateway like any later one; an older subscriber,
//! which would take the publisher's first batch for a duplicate of such a
//! heartbeat, gets its first record with that batch.
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
    collections::{BTreeMap, VecDeque},
    ops::RangeInclusive,
    sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak},
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
    DealerSocket, Endpoint, SocketOptions, SubSocket, ZmqError, ZmqMessage,
};

use crate::{
    engine_link::EngineLink,
    kv_history::{History, Window},
    kv_state::{LiveState, RankCut, Snapshot, SnapshotChunks},
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
/// How often the relay knocks on a publisher's port that nobody has bound
/// yet. The relay dials at the servicer's start, minutes before the engine
/// binds its publisher, and the ZMQ crate retries a refused port with a
/// back-off that grows to 30 s: longer than a planned warm-up, so left to it
/// the subscription would land at a random point of that cadence and miss
/// the warm-up's batches and its closing clear. Knocking every second (a
/// plain connect, no ZMQ handshake) and joining as soon as the port answers,
/// the subscription lands within a second of the bind.
const DIAL_INTERVAL: Duration = Duration::from_secs(1);
/// How long a publisher whose port answered may take to complete the ZMQ
/// handshake (the crate's own default); a slower one is logged and dialled
/// again, so a starved or distant publisher is visible, never silently
/// re-dialled.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// A debug line every this many knocks on a port nobody has bound.
const KNOCKS_PER_LOG: u64 = 60;
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

/// One rank's publisher: where to subscribe and where to ask for replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankSource {
    /// The rank the batches of this source are stamped with.
    pub rank: i32,
    /// The connectable SUB endpoint of the rank's publisher.
    pub endpoint: String,
    /// Its replay ROUTER, when the engine runs one.
    pub replay_endpoint: Option<String>,
}

/// One engine's relay settings.
#[derive(Clone, Debug)]
pub struct RelayConfig {
    /// The publishers to relay, one per data-parallel rank, in rank order.
    pub sources: Vec<RankSource>,
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
    /// Every rank of the publisher at `kv_events_endpoint` (bind wildcards
    /// resolved; see [`rank_sources`]), with the history caps from the
    /// environment.
    pub(crate) fn for_publisher(
        kv_events_endpoint: &str,
        replay_endpoint: Option<&str>,
        topic: &str,
        ranks: u32,
    ) -> Self {
        Self {
            sources: rank_sources(kv_events_endpoint, replay_endpoint, ranks),
            topic: topic.to_string(),
            history_batches: env_usize(HISTORY_BATCHES_ENV, DEFAULT_HISTORY_BATCHES),
            history_bytes: env_usize(HISTORY_BYTES_ENV, DEFAULT_HISTORY_BYTES),
            replay_timeout: DEFAULT_REPLAY_TIMEOUT,
            load_tick: DEFAULT_LOAD_TICK,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_backoff: DEFAULT_HEARTBEAT_BACKOFF,
        }
    }

    /// What log lines and refusals name the relay by: rank 0's endpoint.
    fn label(&self) -> &str {
        self.sources
            .first()
            .map_or("", |source| source.endpoint.as_str())
    }
}

/// The sources of `ranks` publishers laid out from `kv_events_endpoint` and
/// `replay_endpoint` (empty: none): rank `r` on the base port plus `r`, as
/// the engines lay their ranks out and the Python relay's `rank_sources`
/// derives them. Fewer than one rank counts as one.
pub(crate) fn rank_sources(
    kv_events_endpoint: &str,
    replay_endpoint: Option<&str>,
    ranks: u32,
) -> Vec<RankSource> {
    let replay = replay_endpoint.filter(|endpoint| !endpoint.is_empty());
    (0..ranks.max(1))
        .map(|rank| RankSource {
            rank: i32::try_from(rank).unwrap_or(i32::MAX),
            endpoint: endpoint_for_rank(kv_events_endpoint, rank),
            replay_endpoint: replay.map(|endpoint| endpoint_for_rank(endpoint, rank)),
        })
        .collect()
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

/// A servicer state's own `GetLoads` figures, the half of a load record that
/// differs by wire; [`LoadFromState`] adds the half every wire shares.
pub(crate) trait RankLoads: Send + Sync {
    /// The record for `dp_rank` (`None`: a publisher that names no rank)
    /// without the engine link's liveness fields, or `None` while the engine
    /// has reported nothing.
    fn rank_load(&self, dp_rank: Option<i32>) -> Option<common::EngineLoad>;

    /// The engine link whose liveness the record carries.
    fn engine_link(&self) -> &EngineLink;
}

/// The relay's load source over a servicer state, held weakly (the relay
/// outlives the state's last strong holder by nothing it should keep): the
/// state's [`RankLoads::rank_load`] with the engine link's output silence,
/// step reporting and prefill backlog on the record, the same way on every
/// wire.
pub(crate) struct LoadFromState<S>(pub(crate) Weak<S>);

impl<S: RankLoads> LoadSource for LoadFromState<S> {
    fn load(&self, dp_rank: Option<i32>) -> Option<common::EngineLoad> {
        let state = self.0.upgrade()?;
        let mut record = state.rank_load(dp_rank)?;
        let engine = state.engine_link();
        record.engine_silence_ms = engine.output_silence_ms();
        record.engine_reports_steps = engine.reports_steps();
        record.prefill_pending_tokens = engine.prefill_pending_tokens();
        Some(record)
    }
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
        || (last.active_token_usage.unwrap_or(last.token_usage)
            - current.active_token_usage.unwrap_or(current.token_usage))
        .abs()
            > 0.005
        || {
            let delta = (last.gen_throughput - current.gen_throughput).abs();
            delta > 50.0 || delta > 0.05 * last.gen_throughput.max(current.gen_throughput)
        }
        // A growing engine silence goes out once a second, so a subscriber's
        // wedge bound is met within a second of the engine crossing it.
        || last.engine_silence_ms.map(|ms| ms / 1000) != current.engine_silence_ms.map(|ms| ms / 1000)
        || last.engine_reports_steps != current.engine_reports_steps
        || last.prefill_pending_tokens != current.prefill_pending_tokens
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

/// One rank's record: its window and where its publisher's count stands.
struct RankRecord {
    history: History,
    /// The last sequence relayed in this incarnation, or none yet.
    cursor: Option<u64>,
    /// The first sequence relayed in this incarnation: the relay holds
    /// nothing from before it, which is how a relay that started (or
    /// restarted) after the publisher tells a resume from before its time.
    started_at: Option<u64>,
    /// Sequences of this incarnation before the record that even the
    /// engine's replay no longer had: what a snapshot cannot cover, as
    /// counted; [`Self::unknown_before`] is what a snapshot reports.
    unknown_before: u64,
    /// Whether a relayed batch of this incarnation carried the engine's
    /// `AllBlocksCleared`: nothing from before it is live, so the sequences
    /// before the record no longer matter to a snapshot.
    cleared: bool,
}

/// The state the publisher tasks and the subscribers share.
struct Shared {
    /// Per source, in [`RelayConfig::sources`] order.
    ranks: Vec<RankRecord>,
    /// The engine's live blocks as the relayed streams describe them,
    /// current through each rank's cursor.
    state: LiveState,
    /// Counts the publishers' restarts; a rank's sequences compare within one.
    generation: u64,
    counts: RelayCounts,
    /// Per source, its normalizer's counters as of its last relayed batch:
    /// what was forwarded, dropped by reason, and the engine-hash check's
    /// tally; summed for the summary and closing lines.
    wire: Vec<WireCounts>,
    /// Event ids advance once per event over every rank: a publisher task
    /// reserves its batch's run here and numbers the events off the lock.
    event_id: u64,
    /// Why a publisher task gave up, when one did.
    failed: Option<String>,
    /// Ranks whose subscription reached its publisher.
    subscribed: usize,
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

impl RankRecord {
    fn new(history_batches: usize, history_bytes: usize) -> Self {
        Self {
            history: History::new(history_batches, history_bytes),
            cursor: None,
            started_at: None,
            unknown_before: 0,
            cleared: false,
        }
    }

    /// Publisher sequences before the record that a snapshot cannot cover:
    /// what even the engine's replay did not give, until the engine clears
    /// its cache.
    fn unknown_before(&self) -> u64 {
        if self.cleared {
            0
        } else {
            self.unknown_before
        }
    }

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
}

impl Shared {
    /// Start a new incarnation of the rank at `index`: its window and its
    /// live blocks were the old publisher's; the other ranks' stay.
    fn restart_rank(&mut self, index: usize, dp_rank: Option<i32>) -> u64 {
        self.generation += 1;
        let record = &mut self.ranks[index];
        record.history.clear();
        record.cursor = None;
        record.started_at = None;
        record.unknown_before = 0;
        record.cleared = false;
        self.state.clear_rank(dp_rank);
        self.counts.publisher_restarts += 1;
        self.generation
    }

    /// Every source's normalizer counters summed.
    fn wire_counts(&self) -> WireCounts {
        self.wire
            .iter()
            .fold(WireCounts::default(), |mut total, counts| {
                total.add(counts);
                total
            })
    }

    fn relayed_anything(&self) -> bool {
        self.ranks.iter().any(|record| record.cursor.is_some())
    }

    /// Every rank's window holds every batch its publisher ever numbered, and
    /// at least one rank relayed something: a subscriber without a cursor
    /// can take the windows as the engine's whole state.
    fn complete_from_start(&self) -> bool {
        self.relayed_anything()
            && self
                .ranks
                .iter()
                .all(|record| record.history.is_empty() || record.history.complete_from_start())
    }

    fn window(&self) -> usize {
        self.ranks.iter().map(|record| record.history.len()).sum()
    }

    fn holes(&self) -> usize {
        self.ranks.iter().map(|record| record.history.holes()).sum()
    }

    fn bytes(&self) -> usize {
        self.ranks.iter().map(|record| record.history.bytes()).sum()
    }

    fn unknown_before(&self) -> u64 {
        self.ranks.iter().map(RankRecord::unknown_before).sum()
    }

    fn oldest(&self) -> Option<u64> {
        self.ranks
            .iter()
            .filter_map(|record| record.history.oldest())
            .min()
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

/// One engine's KV-event relay: the publisher subscriptions (one per rank),
/// their histories and the live channel its subscribers read.
pub struct KvEventRelay {
    config: Arc<RelayConfig>,
    shared: Arc<Mutex<Shared>>,
    live: broadcast::Sender<Live>,
    /// The publisher tasks, one per source.
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    /// A subscriber came while the relay holds nothing: each publisher task
    /// asks its engine replay for the publisher's start again.
    prime_requests: Vec<Arc<Notify>>,
    /// The servicer's load, attached to every batch sent; none until the
    /// servicer installs it.
    load_source: OnceLock<Arc<dyn LoadSource>>,
}

impl Drop for KvEventRelay {
    fn drop(&mut self) {
        for task in self
            .tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain(..)
        {
            task.abort();
        }
        let counts = self.counts();
        let shared = lock(&self.shared);
        info!(
            endpoint = %self.config.label(),
            ranks = self.config.sources.len(),
            ?counts,
            wire = ?shared.wire_counts(),
            window = shared.window(),
            holes = shared.holes(),
            bytes = shared.bytes(),
            live_blocks = shared.state.blocks(),
            live_entries = shared.state.entries(),
            state = ?shared.state.counts(),
            "KV event relay closed"
        );
    }
}

impl KvEventRelay {
    /// A relay over `config.sources`, at least one (`rank_sources` gives
    /// one for any width); a config without any is a caller bug.
    pub fn new(config: RelayConfig) -> Arc<Self> {
        assert!(
            !config.sources.is_empty(),
            "a KV event relay needs at least one rank source"
        );
        let (live, _) = broadcast::channel(LIVE_CHANNEL);
        let ranks = config
            .sources
            .iter()
            .map(|_| RankRecord::new(config.history_batches, config.history_bytes))
            .collect();
        let prime_requests = config
            .sources
            .iter()
            .map(|_| Arc::new(Notify::new()))
            .collect();
        Arc::new(Self {
            shared: Arc::new(Mutex::new(Shared {
                ranks,
                state: LiveState::new(),
                generation: 0,
                counts: RelayCounts::default(),
                wire: vec![WireCounts::default(); config.sources.len()],
                event_id: 0,
                failed: None,
                subscribed: 0,
            })),
            config: Arc::new(config),
            live,
            tasks: Mutex::new(Vec::new()),
            prime_requests,
            load_source: OnceLock::new(),
        })
    }

    /// Install where the load record on every sent batch comes from (once;
    /// a second call is ignored).
    pub(crate) fn set_load_source(&self, source: Arc<dyn LoadSource>) {
        let _ = self.load_source.set(source);
    }

    /// The relay for an engine's publishers, `ranks` of them laid out from
    /// `kv_events_endpoint` (see [`rank_sources`]), or `None` when events are
    /// off (an empty endpoint).
    pub(crate) fn for_publisher(
        kv_events_endpoint: &str,
        replay_endpoint: Option<&str>,
        topic: &str,
        ranks: u32,
    ) -> Option<Arc<Self>> {
        (!kv_events_endpoint.is_empty()).then(|| {
            Self::new(RelayConfig::for_publisher(
                kv_events_endpoint,
                replay_endpoint,
                topic,
                ranks,
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
        lock(&self.shared).wire_counts()
    }

    /// Ranks holding live blocks in the relay's record.
    #[cfg(test)]
    pub(crate) fn live_ranks(&self) -> usize {
        lock(&self.shared).state.ranks()
    }

    /// Record `batches` as relayed without a publisher, each under the source
    /// of its rank (the first source when none matches): the history, the
    /// live state and the cursor as [`Relaying::relay`] would leave them.
    #[cfg(test)]
    fn preload(&self, batches: impl IntoIterator<Item = common::KvEventBatch>) {
        let mut shared = lock(&self.shared);
        for batch in batches {
            let sequence = batch.sequence_number;
            let index = self
                .config
                .sources
                .iter()
                .position(|source| Some(source.rank) == batch.dp_rank)
                .unwrap_or(0);
            let batch = Arc::new(batch);
            let record = &mut shared.ranks[index];
            record.history.push(sequence, Arc::clone(&batch));
            if record.started_at.is_none() {
                record.started_at = Some(sequence);
            }
            record.cursor = Some(sequence);
            if clears_rank(&batch) {
                record.cleared = true;
            }
            shared.state.apply(&batch);
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
                endpoint = %self.config.label(),
                "{RELAY_START_ENV}=lazy: the KV event relay subscribes at the first SubscribeKvEvents"
            );
        }
    }

    /// Whether every rank's subscription has reached its publisher, or there
    /// is nothing to wait for: a relay not started (`RELAY_START_ENV` is
    /// `lazy`) or one that gave up. The lifecycle owner holds its planned
    /// warm-up until this says so, so the warm-up's batches start the
    /// history instead of being published to nobody.
    pub fn subscribed(&self) -> bool {
        let started = self
            .tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .any(|task| !task.is_finished());
        if !started {
            return true;
        }
        let shared = lock(&self.shared);
        shared.failed.is_some() || shared.subscribed == self.config.sources.len()
    }

    /// Subscribe to every rank's publisher if not yet subscribed. Called at
    /// the servicer's start (`Self::start_at_boot`) and by
    /// [`Self::subscribe`]; needs a Tokio runtime.
    pub fn start(&self) {
        let mut tasks = self.tasks.lock().unwrap_or_else(PoisonError::into_inner);
        if tasks.iter().any(|task| !task.is_finished()) {
            return;
        }
        tasks.clear();
        for (index, prime_request) in self.prime_requests.iter().enumerate() {
            let config = Arc::clone(&self.config);
            let shared = Arc::clone(&self.shared);
            let live = self.live.clone();
            let prime_request = Arc::clone(prime_request);
            #[expect(
                clippy::disallowed_methods,
                reason = "the publisher subscription outlives any one RPC; aborted when the relay drops"
            )]
            let handle = tokio::spawn(run(config, index, shared, live, prime_request));
            tasks.push(handle);
        }
    }

    /// Handle one `SubscribeKvEvents` call: the windows after the cursor, or
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
        let records_before_first_batch = request.load_records_before_first_batch;
        let label = self.config.label();
        let serve = {
            let mut shared = lock(&self.shared);
            if let Some(error) = &shared.failed {
                return Err(Status::internal(format!(
                    "SubscribeKvEvents: the relay for {label} is down: {error}"
                )));
            }
            if cursor == 0 {
                if shared.complete_from_start() {
                    shared.counts.served_from_history += 1;
                    Serve::History {
                        batches: shared
                            .ranks
                            .iter()
                            .flat_map(|record| record.history.all())
                            .collect(),
                        last_sent: BTreeMap::new(),
                    }
                } else if shared.relayed_anything() {
                    // A window no longer reaches back to its publisher's
                    // first batch: the live set as of each rank's cursor,
                    // taken under the guard that admits batches, so every
                    // later batch reaches this subscriber through the live
                    // channel.
                    let started = Instant::now();
                    let snapshot = shared.state.snapshot();
                    let collected = started.elapsed();
                    shared.counts.served_snapshots += 1;
                    let cuts: BTreeMap<Option<i32>, RankCut> = self
                        .config
                        .sources
                        .iter()
                        .zip(&shared.ranks)
                        .filter_map(|(source, record)| {
                            let through = record.cursor?;
                            Some((
                                Some(source.rank),
                                RankCut {
                                    through,
                                    unknown_before: record.unknown_before(),
                                },
                            ))
                        })
                        .collect();
                    Serve::Snapshot {
                        snapshot,
                        cuts,
                        collected,
                        oldest: shared.oldest(),
                        holes: shared.holes(),
                        unknown_before: shared.unknown_before(),
                    }
                } else {
                    // Nothing relayed yet: a publisher may have spoken
                    // before the subscription reached it and after the
                    // start replay answered; ask its replay once more.
                    for (source, prime_request) in
                        self.config.sources.iter().zip(&self.prime_requests)
                    {
                        if source.replay_endpoint.is_some() {
                            prime_request.notify_one();
                        }
                    }
                    Serve::Live
                }
            } else if self.config.sources.len() > 1 {
                // One cursor names one rank's sequence; the other ranks'
                // are the gateway's alone, and nothing below them may be
                // resent on a fresh stream. The gateway clears and comes
                // back from zero for every window or a snapshot.
                shared.counts.out_of_range += 1;
                let message = format!(
                    "SubscribeKvEvents: the relay for {label} follows {} ranks and a cursor \
                     names one rank's sequence; resubscribe from zero for every rank's window \
                     or a state snapshot",
                    self.config.sources.len()
                );
                info!(counts = ?shared.counts, "{message}");
                return Err(Status::out_of_range(message));
            } else {
                let Some((source, record)) = self.config.sources.first().zip(shared.ranks.first())
                else {
                    return Err(Status::internal(
                        "SubscribeKvEvents: the relay has no publisher to resume from",
                    ));
                };
                match record.history.after(cursor) {
                    Ok(batches) => {
                        let last_sent = BTreeMap::from([(Some(source.rank), cursor)]);
                        shared.counts.served_from_history += 1;
                        Serve::History { batches, last_sent }
                    }
                    Err(window) => {
                        let started_at = record.started_at;
                        shared.counts.out_of_range += 1;
                        let endpoint = label;
                        let message = match window {
                            Window::Empty => format!(
                                "SubscribeKvEvents: the relay for {endpoint} holds no history \
                                 yet (it started after the publisher); resubscribe from zero"
                            ),
                            Window::Behind { oldest } => match started_at {
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
                            window = shared.window(),
                            holes = shared.holes(),
                            "{message}"
                        );
                        return Err(Status::out_of_range(message));
                    }
                }
            }
        };
        let (replay, sent, snapshot) = match serve {
            Serve::History { batches, last_sent } => {
                if !batches.is_empty() {
                    debug!(
                        endpoint = %label,
                        cursor,
                        batches = batches.len(),
                        "SubscribeKvEvents: serving from history"
                    );
                }
                (batches, last_sent, SnapshotPhase::None)
            }
            Serve::Snapshot {
                snapshot,
                cuts,
                collected,
                oldest,
                holes,
                unknown_before,
            } => {
                info!(
                    endpoint = %label,
                    through = ?cuts.values().map(|cut| cut.through).collect::<Vec<_>>(),
                    blocks = snapshot.blocks,
                    entries = snapshot.entries(),
                    ranks = snapshot.ranks.len(),
                    collected_us = u64::try_from(collected.as_micros()).unwrap_or(u64::MAX),
                    oldest,
                    holes,
                    unknown_before,
                    "SubscribeKvEvents: a window no longer starts at its publisher's first \
                     batch; serving a state snapshot"
                );
                let timestamp = unix_seconds();
                let sent = cuts
                    .iter()
                    .map(|(rank, cut)| (*rank, cut.through))
                    .collect();
                // Ordering and sizing the chunks is a pass over every live
                // block: off the runtime's workers, and off the lock.
                let ordering = tokio::task::spawn_blocking(move || {
                    SnapshotChunks::new(snapshot, &cuts, timestamp)
                });
                (Vec::new(), sent, SnapshotPhase::Ordering(ordering))
            }
            Serve::Live => (Vec::new(), BTreeMap::new(), SnapshotPhase::None),
        };
        let subscriber = Subscriber {
            snapshot,
            replay: replay.into(),
            rx,
            sent,
            ranks: self
                .config
                .sources
                .iter()
                .map(|source| source.rank)
                .collect(),
            shared: Arc::clone(&self.shared),
            endpoint: label.to_string(),
            done: false,
            source: self.load_source.get().cloned(),
            records_before_first_batch,
            load_tick: self.config.load_tick,
            heartbeat_interval: self.config.heartbeat_interval,
            heartbeat_backoff: self.config.heartbeat_backoff,
            last_rank: None,
            last_sequence: None,
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
        /// Per rank, the sequence the subscriber is known to hold.
        last_sent: BTreeMap<Option<i32>, u64>,
    },
    Snapshot {
        snapshot: Snapshot,
        /// Each rank's cursor and unknown count at the cut.
        cuts: BTreeMap<Option<i32>, RankCut>,
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
/// from the histories, then the live channel, deduplicated by sequence per
/// rank.
struct Subscriber {
    snapshot: SnapshotPhase,
    replay: VecDeque<Arc<common::KvEventBatch>>,
    rx: broadcast::Receiver<Live>,
    /// Per rank, the last sequence sent (or the cursor the subscriber came
    /// with): what a live batch is a duplicate of.
    sent: BTreeMap<Option<i32>, u64>,
    /// The ranks of the relay's sources, in source order.
    ranks: Vec<i32>,
    shared: Arc<Mutex<Shared>>,
    endpoint: String,
    done: bool,
    /// The servicer's load for the record on every batch; `None` sends
    /// batches without one and no `load_only` batches.
    source: Option<Arc<dyn LoadSource>>,
    /// The subscriber takes `load_only` batches before the publisher's
    /// first batch, numbered 0 (`SubscribeKvEventsRequest.
    /// load_records_before_first_batch`); unset, it waits for that batch.
    records_before_first_batch: bool,
    load_tick: Duration,
    heartbeat_interval: Duration,
    heartbeat_backoff: Duration,
    /// The rank and sequence of the last batch sent: what a `load_only`
    /// batch names and repeats.
    last_rank: Option<i32>,
    last_sequence: Option<u64>,
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
        self.last_sequence = Some(batch.sequence_number);
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
        // Nothing sent yet, nothing to repeat: a subscriber that takes
        // records before the publisher's first batch gets them numbered 0
        // (it admits a `load_only` batch to no cursor), so the engine's
        // silence on its first request reaches it in time; a gateway that
        // predates the field would take the publisher's first batch for a
        // duplicate of such a heartbeat, and waits for that batch.
        let sequence_number = match self.last_sequence {
            Some(sequence) => sequence,
            None if self.records_before_first_batch => 0,
            None => return None,
        };
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
                self.sent.insert(batch.dp_rank, batch.sequence_number);
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
                        .sent
                        .get(&batch.dp_rank)
                        .is_some_and(|&last| batch.sequence_number <= last)
                    {
                        continue;
                    }
                    self.sent.insert(batch.dp_rank, batch.sequence_number);
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
                    // Each rank's window after what this stream sent of it.
                    // A rank it sent nothing of yet (one that began publishing
                    // after a subscription from zero: a resumed stream has one
                    // rank, seeded with its cursor) is owed its whole window,
                    // which only a window from the publisher's start can be.
                    let refill = {
                        let mut shared = lock(&self.shared);
                        shared.counts.subscribers_lagged += 1;
                        let mut batches = Vec::new();
                        let mut refilled = Ok(());
                        for (rank, record) in self.ranks.iter().zip(&shared.ranks) {
                            let owed = match self.sent.get(&Some(*rank)) {
                                Some(&last) => record.history.after(last),
                                None if record.history.is_empty() => Ok(Vec::new()),
                                None if record.history.complete_from_start() => {
                                    Ok(record.history.all())
                                }
                                None => Err(Window::Empty),
                            };
                            match owed {
                                Ok(owed) => batches.extend(owed),
                                Err(window) => {
                                    refilled = Err(window);
                                    break;
                                }
                            }
                        }
                        refilled.map(|()| batches)
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

/// The publisher task of the source at `index`: one SUB socket for the
/// relay's lifetime.
async fn run(
    config: Arc<RelayConfig>,
    index: usize,
    shared: Arc<Mutex<Shared>>,
    live: broadcast::Sender<Live>,
    prime_request: Arc<Notify>,
) {
    let source = &config.sources[index];
    let mut socket = match connect(&source.endpoint, &config.topic).await {
        Ok(socket) => {
            lock(&shared).subscribed += 1;
            socket
        }
        Err(status) => {
            lock(&shared).failed = Some(status.message().to_string());
            return;
        }
    };
    let mut relay = Relaying {
        config: &config,
        source,
        index,
        shared: &shared,
        live: &live,
        normalizer: Normalizer::from_env(),
    };
    if index == 0 {
        if let Some(check) = relay.normalizer.hash_check() {
            info!(
                endpoint = %source.endpoint,
                check = check.as_str(),
                "KV event engine-hash check on; its tally is in the relay's summary and closing lines"
            );
        }
    }
    // The publisher may have been counting before the subscription reached
    // it: take what its replay still holds before the first live batch, and
    // keep asking until the engine answers (it may still be starting). A
    // live batch arriving first settles the start on its own path.
    let mut primed = source.replay_endpoint.is_none();
    let mut next_prime = tokio::time::Instant::now();
    loop {
        let received = tokio::select! {
            received = socket.recv() => received,
            () = tokio::time::sleep_until(next_prime), if !primed => {
                primed = relay.prime_from_replay().await;
                next_prime = tokio::time::Instant::now() + PRIME_RETRY;
                continue;
            }
            () = prime_request.notified(), if source.replay_endpoint.is_some() => {
                if lock(&shared).ranks[index].cursor.is_none() {
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
                warn!(
                    endpoint = %source.endpoint,
                    rank = source.rank,
                    %error,
                    "KV event receive failed; retrying"
                );
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
            let record = &shared.ranks[index];
            (
                record.cursor.is_none(),
                record.admit(sequence, startup_clear),
            )
        };
        if first && admission == Admission::Accept && joined_late(sequence, startup_clear) {
            relay.recover_start(sequence).await;
            // The replay may have given this sequence too.
            admission = lock(&shared).ranks[index].admit(sequence, startup_clear);
        }
        if let Admission::Gap { from, to } = admission {
            relay.recover_gap(from..=to).await;
            // The replay may have run past the live sequence.
            admission = lock(&shared).ranks[index].admit(sequence, startup_clear);
        }
        match admission {
            Admission::Accept => relay.relay(sequence, decoded),
            Admission::Gap { .. } | Admission::Duplicate => {}
            Admission::Restart { reason, last } => {
                let (generation, counts) = {
                    let mut shared = lock(&shared);
                    let generation = shared.restart_rank(index, Some(source.rank));
                    (generation, shared.counts.clone())
                };
                warn!(
                    endpoint = %source.endpoint,
                    rank = source.rank,
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
                    if lock(&shared).ranks[index].admit(sequence, startup_clear)
                        != Admission::Accept
                    {
                        continue;
                    }
                }
                relay.relay(sequence, decoded);
            }
        }
    }
}

/// Whether `batch` carries the engine's `AllBlocksCleared`: once it is
/// relayed, nothing stored before it is live on its rank.
fn clears_rank(batch: &common::KvEventBatch) -> bool {
    batch
        .events
        .iter()
        .any(|event| matches!(event.data, Some(common::kv_cache_event::Data::Cleared(_))))
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

/// A publisher task's per-batch work, for the source at `index`.
struct Relaying<'a> {
    config: &'a RelayConfig,
    source: &'a RankSource,
    index: usize,
    shared: &'a Mutex<Shared>,
    live: &'a broadcast::Sender<Live>,
    /// This rank's normalizer (its state is the rank's own).
    normalizer: Normalizer,
}

impl Relaying<'_> {
    /// Normalize, remember and broadcast one batch under `sequence`, stamped
    /// with this source's rank (the socket's rank wins over the payload's,
    /// as in the Python relay: it is what the gateway keys its cursors by).
    fn relay(&mut self, sequence: u64, decoded: Decoded) {
        let (batch, undecodable) = match decoded {
            Decoded::Batch(mut batch) => {
                batch.dp_rank = Some(self.source.rank);
                // The batch's run of event ids, reserved under the lock;
                // the normalization runs off it.
                let mut event_id = {
                    let mut shared = lock(self.shared);
                    let first = shared.event_id;
                    shared.event_id += batch.events.len() as u64;
                    first
                };
                (
                    self.normalizer
                        .normalize_batch(batch, sequence, &mut event_id),
                    false,
                )
            }
            Decoded::Undecodable => (
                common::KvEventBatch {
                    sequence_number: sequence,
                    dp_rank: Some(self.source.rank),
                    ..common::KvEventBatch::default()
                },
                true,
            ),
        };
        let batch = Arc::new(batch);
        {
            let mut shared = lock(self.shared);
            let record = &mut shared.ranks[self.index];
            record.history.push(sequence, Arc::clone(&batch));
            if record.started_at.is_none() {
                record.started_at = Some(sequence);
            }
            record.cursor = Some(sequence);
            if clears_rank(&batch) {
                // The engine's cache is empty: nothing the batches before
                // the record stored is live, so the snapshots cover the
                // rank whole from here on (a flag, not a reset of the count:
                // the replay paths settle the count after relaying what the
                // replay gave, a clear among those batches included).
                record.cleared = true;
            }
            shared.state.apply(&batch);
            shared.counts.relayed += 1;
            if undecodable {
                shared.counts.undecodable_batches += 1;
            }
            shared.wire[self.index] = self.normalizer.counts().clone();
            if shared.counts.relayed.is_multiple_of(SUMMARY_EVERY_BATCHES) {
                info!(
                    endpoint = %self.config.label(),
                    counts = ?shared.counts,
                    wire = ?shared.wire_counts(),
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
            endpoint = %self.source.endpoint,
            rank = self.source.rank,
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
            shared.ranks[self.index].unknown_before = filled.unknown_before;
            shared.counts.unknown_before_start += filled.unknown_before;
            shared.counts.clone()
        };
        if filled.unknown_before == 0 {
            info!(
                endpoint = %self.source.endpoint,
                rank = self.source.rank,
                first_seen,
                recovered = filled.recovered,
                ?counts,
                "KV event relay joined a publisher already counting; its replay covered the \
                 batches before"
            );
        } else {
            warn!(
                endpoint = %self.source.endpoint,
                rank = self.source.rank,
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
        let Some(endpoint) = &self.source.replay_endpoint else {
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
                debug!(endpoint = %self.source.endpoint, %error, "KV event replay not answering yet");
                return false;
            }
            Err(_) => return false,
        };
        if lock(self.shared).ranks[self.index].cursor.is_some() {
            // A live batch got in first and settled the start.
            return true;
        }
        if replies.is_empty() {
            if complete {
                info!(
                    endpoint = %self.source.endpoint,
                    rank = self.source.rank,
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
            shared.ranks[self.index].unknown_before = unknown_before;
            shared.counts.unknown_before_start += unknown_before;
            shared.counts.primed_batches += taken;
            shared.counts.clone()
        };
        if unknown_before == 0 {
            info!(
                endpoint = %self.source.endpoint,
                rank = self.source.rank,
                first,
                last,
                batches = taken,
                ?counts,
                "KV event relay primed from the publisher's replay at start"
            );
        } else {
            warn!(
                endpoint = %self.source.endpoint,
                rank = self.source.rank,
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
        let replies = match &self.source.replay_endpoint {
            Some(endpoint) => match replay(endpoint, from, self.config.replay_timeout).await {
                Ok((replies, _)) => replies,
                Err(error) => {
                    warn!(
                        endpoint = %self.source.endpoint,
                        rank = self.source.rank,
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

    /// `from..=to` passed without batches: holes in the rank's window.
    fn lose(&mut self, from: u64, to: u64) {
        let mut shared = lock(self.shared);
        let record = &mut shared.ranks[self.index];
        record.history.push_lost_range(from, to);
        record.cursor = Some(to);
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

/// Whether `endpoint`'s port (or socket path) answers a plain connect, which
/// is the publisher being bound; no ZMQ handshake, so the knock's cadence
/// alone decides how soon after the bind the join lands.
async fn port_answers(endpoint: &Endpoint) -> bool {
    match endpoint {
        Endpoint::Tcp(host, port) => {
            let connect = tokio::net::TcpStream::connect((host.to_string(), *port));
            tokio::time::timeout(DIAL_INTERVAL, connect)
                .await
                .is_ok_and(|connected| connected.is_ok())
        }
        #[cfg(unix)]
        Endpoint::Ipc(Some(path)) => tokio::net::UnixStream::connect(path).await.is_ok(),
        // Nothing to knock on (the enum is non-exhaustive): the ZMQ connect
        // below says what is wrong.
        _ => true,
    }
}

/// A SUB socket subscribed to `topic` and connected to `endpoint`. The
/// subscription is recorded first and sent on connect (and on the crate's
/// reconnects), as libzmq does. A publisher not bound yet (the engine is
/// still starting) is knocked on every [`DIAL_INTERVAL`] until its port
/// answers, then joined under [`HANDSHAKE_TIMEOUT`], so the join follows the
/// bind within a second instead of the crate's own back-off, which reaches
/// 30 s; a handshake that does not complete in time is logged and the dial
/// starts over; any other failure is final.
async fn connect(endpoint: &str, topic: &str) -> Result<SubSocket, Status> {
    let started = Instant::now();
    let failed = |step: &str, error: ZmqError| {
        Status::internal(format!("SubscribeKvEvents: {step} {endpoint}: {error}"))
    };
    let parsed: Endpoint = endpoint
        .parse()
        .map_err(|error| failed("could not dial", ZmqError::Endpoint(error)))?;
    let mut knocks: u64 = 0;
    let mut handshakes_timed_out: u64 = 0;
    loop {
        while !port_answers(&parsed).await {
            knocks += 1;
            if knocks.is_multiple_of(KNOCKS_PER_LOG) {
                debug!(
                    %endpoint,
                    knocks,
                    waited_secs = started.elapsed().as_secs(),
                    "SubscribeKvEvents: the publisher's port is not bound yet; still dialling"
                );
            }
            tokio::time::sleep(DIAL_INTERVAL).await;
        }
        let mut options = SocketOptions::default();
        options.connect_timeout(HANDSHAKE_TIMEOUT);
        let mut socket = SubSocket::with_options(options);
        socket
            .subscribe(topic)
            .await
            .map_err(|error| failed("could not subscribe to", error))?;
        match socket.connect(endpoint).await {
            Ok(()) => {
                info!(
                    %endpoint,
                    knocks,
                    handshakes_timed_out,
                    dialled_secs = started.elapsed().as_secs_f64(),
                    "SubscribeKvEvents: connected to ZMQ endpoint"
                );
                return Ok(socket);
            }
            Err(ZmqError::ConnectTimeout(budget)) => {
                handshakes_timed_out += 1;
                warn!(
                    %endpoint,
                    ?budget,
                    handshakes_timed_out,
                    "SubscribeKvEvents: the publisher's port answered but the ZMQ handshake did \
                     not complete in time; dialling again"
                );
            }
            Err(error) => return Err(failed("could not connect to", error)),
        }
    }
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
mod tests;
