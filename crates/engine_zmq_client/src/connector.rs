// The connector: the frontend-side request/response layer over a
// [`ConnectedTransport`]. Adapted from the Apache-2.0 reference
// `vllm-engine-core-client` (vllm-project/vllm) client, scoped to SMG's needs:
//
// - DP load balancing for unpinned requests: the connector picks the
//   least-loaded engine from the piggybacked per-rank `scheduler_stats`
//   (mirroring vLLM's frontend DP client). An SMG-stamped
//   `data_parallel_rank` remains authoritative.
// - No DP coordinator process: for a lockstep engine group the connector plays
//   the wake role itself, over the input socket each rank already listens on.
// - Utility RPCs (`call_utility`, vLLM only): one call to one rank, correlated
//   by a call id this client issues. Pending calls live in the request
//   registry, so the one dispatcher resolves their replies and fails them with
//   the requests when the engine dies. Upstream's management wrappers (LoRA
//   add/remove, sleep/wake, profiling, weight updates) stay unported; the
//   prefix-cache reset is built on it by the adapter.

use std::{
    collections::{HashMap, HashSet},
    ops::ControlFlow,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tokio::{
    sync::{mpsc, oneshot, Notify},
    task::JoinHandle,
    time::Instant,
};
use tracing::{trace, warn};
use zeromq::{RouterSendHalf, SocketRecv};

use crate::{
    codec::OpaqueValue,
    error::{Error, Result},
    protocol::{
        sglang::SglangProtocol, tokenspeed::TokenSpeedProtocol, vllm::VllmProtocol, EngineBatch,
        EngineLoad, EngineOutput, EngineProtocol, UtilityReply, WaveEvent,
    },
    transport::{run_output_loop, send_message, ConnectedEngine, ConnectedTransport, EngineId},
};

/// The vLLM EngineCore connector (the original engine surface).
pub type EngineCoreClient = Client<VllmProtocol>;
/// The per-request output stream for the vLLM EngineCore connector.
pub type EngineCoreStream = RequestStream<VllmProtocol>;
/// The TokenSpeed connector.
pub type TokenSpeedClient = Client<TokenSpeedProtocol>;
/// The per-request output stream for the TokenSpeed connector.
pub type TokenSpeedStream = RequestStream<TokenSpeedProtocol>;
/// The SGLang connector.
pub type SglangClient = Client<SglangProtocol>;
/// The per-request output stream for the SGLang connector.
pub type SglangStream = RequestStream<SglangProtocol>;

/// Per-request output channels are unbounded on purpose. The dispatcher fans
/// one engine tick out to every request in it while holding the registry lock,
/// so a bounded channel would let one slow consumer stall delivery for every
/// other request on the transport — and backpressuring further, into the
/// output PULL socket, would wedge the engine itself. Unboundedness is not
/// unbounded memory: a request's queue is capped by its own generation
/// (`max_tokens` outputs), a dropped stream auto-aborts it on the engine, and
/// the buffer is freed with the stream.
type OutputSender<O> = mpsc::UnboundedSender<Result<O>>;
type OutputReceiver<O> = mpsc::UnboundedReceiver<Result<O>>;

/// Routes engine outputs to per-request streams and utility replies to their
/// callers, and tracks what is in flight. Requests are keyed by `request_id`,
/// utility calls by the call id this client issued; closing fails both.
struct RequestRegistry<O> {
    closed: bool,
    requests: HashMap<String, InFlight<O>>,
    utility_calls: HashMap<i64, UtilitySender>,
    /// When `requests` last went from empty to non-empty, while it is
    /// non-empty: the start of the busy period the engine is in.
    in_flight_since: Option<Instant>,
    /// When `requests` last went from non-empty to empty, while it is empty.
    idle_since: Option<Instant>,
    /// The in-flight time of the busy periods that ended since the engine's
    /// last output; with the current one, the engine's unanswered work (see
    /// [`Self::unexplained_for`]).
    unanswered_before: Duration,
    /// The silence watchdog's window ([`ENGINE_SILENCE_DEATH_TIMEOUT`] unless
    /// [`Client::set_silence_death_timeout`] changed it): how long the engine
    /// may be silent past its prefill work in hand ([`Self::work_end`])
    /// before it is dead, and the idle spell after which the count starts
    /// over.
    silence_window: Duration,
    /// The prefill rate the watchdog allows the engine for the prompt tokens
    /// it is handed ([`ENGINE_PREFILL_TOKENS_PER_SEC`] unless
    /// [`Client::set_prefill_tokens_per_sec`] changed it); zero allows
    /// nothing.
    prefill_tokens_per_sec: u64,
    /// When the engine's prefill work in hand is due to be done at the
    /// allowed rate: each prompt handed to the engine queues its allowance
    /// behind the work before it once its frame is accepted
    /// ([`Self::handed`]; an add that fails or is refused before that queues
    /// none), an output restarts the figure from the prompt still pending
    /// then, and a prompt withdrawn before its first output leaves its
    /// allowance in place until it passes (the engine may be mid-way through
    /// it, and nothing on a token-batch wire says otherwise). Silence before
    /// this instant is explained; only the in-flight time past it counts
    /// toward the window (see [`Self::unexplained_for`]).
    work_end: Instant,
    /// Signalled when `requests` goes from empty to non-empty: the dispatcher
    /// re-arms the silence watchdog for the remainder of the window the
    /// engine's unexplained silence has left, not a whole one.
    busy_started: Arc<Notify>,
    /// Prompt tokens of the requests handed to the engine that have produced
    /// no output yet: the engine's prefill work in hand, as the servicer
    /// reports it and as the silence watchdog allows for it. A prompt counts
    /// from the moment its frame is accepted ([`Self::handed`]); one whose
    /// add failed or was refused before that never did.
    prefill_pending: u64,
    /// Registered requests the engine reports on every step it decodes: the
    /// silence [`Client::output_silence`] counts is theirs. A request the
    /// engine reports at intervals is quiet while healthy and counts none.
    every_step: usize,
}

/// One registered request: its output stream, whether the engine reports it
/// on every step ([`EngineProtocol::reported_every_step`]), whether the
/// engine has accepted its add frame ([`RequestRegistry::handed`]) and, until
/// its first output, the prompt the engine still has to prefill for it.
struct InFlight<O> {
    sender: OutputSender<O>,
    prompt_tokens: u64,
    handed: bool,
    started: bool,
    every_step: bool,
}

/// Completes one utility call with its value, or the error that ended it.
type UtilitySender = oneshot::Sender<Result<OpaqueValue>>;

impl<O> Default for RequestRegistry<O> {
    fn default() -> Self {
        Self {
            closed: false,
            requests: HashMap::new(),
            utility_calls: HashMap::new(),
            in_flight_since: None,
            idle_since: None,
            unanswered_before: Duration::ZERO,
            silence_window: ENGINE_SILENCE_DEATH_TIMEOUT,
            prefill_tokens_per_sec: ENGINE_PREFILL_TOKENS_PER_SEC,
            work_end: Instant::now(),
            busy_started: Arc::new(Notify::new()),
            prefill_pending: 0,
            every_step: 0,
        }
    }
}

impl<O: EngineOutput> RequestRegistry<O> {
    /// Register a new request of `prompt_tokens`, reported by the engine on
    /// every step or at intervals (`every_step`), returning the receiver for
    /// its output stream. The busy period and the every-step count start
    /// here; the prompt joins the prefill work in hand when the engine
    /// accepts its frame ([`Self::handed`]).
    fn register(
        &mut self,
        request_id: String,
        prompt_tokens: u64,
        every_step: bool,
    ) -> Result<OutputReceiver<O>> {
        if self.closed {
            return Err(Error::ClientClosed {
                message: "client is shutting down".to_string(),
            });
        }
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }
        let (sender, receiver) = mpsc::unbounded_channel();
        let now = Instant::now();
        if self.requests.is_empty() {
            // A busy period begins. After a whole window idle, the ones before
            // it say nothing about the engine now: the count starts over.
            if self
                .idle_since
                .take()
                .is_some_and(|since| now.saturating_duration_since(since) >= self.silence_window)
            {
                self.unanswered_before = Duration::ZERO;
            }
            self.in_flight_since = Some(now);
            self.busy_started.notify_one();
        }
        self.requests.insert(
            request_id,
            InFlight {
                sender,
                prompt_tokens,
                handed: false,
                started: false,
                every_step,
            },
        );
        if every_step {
            self.every_step += 1;
        }
        Ok(receiver)
    }

    /// The engine accepted the request's add frame: its prompt is prefill
    /// work in hand from now, counted as pending and queued behind the work
    /// before it at the allowed rate. Nothing for a request already answered
    /// (its first output overtook the hand-over), already gone, or handed
    /// before; a registered prompt whose add fails or is refused before this
    /// point (no engine for its rank, a frame refused at a blocked input's
    /// bound) retires with nothing to take out of the bound.
    fn handed(&mut self, request_id: &str) {
        let Some(entry) = self.requests.get_mut(request_id) else {
            return;
        };
        if entry.handed {
            return;
        }
        entry.handed = true;
        if entry.started {
            return;
        }
        let prompt_tokens = entry.prompt_tokens;
        self.prefill_pending = self.prefill_pending.saturating_add(prompt_tokens);
        // The prompt's prefill queues behind the work in hand.
        let allowance = self.prefill_allowance(prompt_tokens);
        self.work_end = deadline(self.work_end.max(Instant::now()), allowance);
    }

    /// The request's prompt is prefilled (its first output came, or it is
    /// gone before one): it no longer counts as pending prefill, if it did
    /// (a prompt the engine was never handed did not).
    fn prefilled(&mut self, entry: &mut InFlight<O>) {
        if !entry.started {
            entry.started = true;
            if entry.handed {
                self.prefill_pending = self.prefill_pending.saturating_sub(entry.prompt_tokens);
            }
        }
    }

    /// Drop a request's entry, settling its pending prefill and its place in
    /// the silence count. A request gone before its first output (aborted,
    /// its stream dropped) leaves its prefill allowance in the work in hand
    /// until it passes: the engine may be mid-way through it, and a client
    /// that gives up on a long prompt mid-prefill must not turn the
    /// allowance it had into a verdict; one the engine was never handed (its
    /// add failed or was refused) queued none. The last one ends the busy
    /// period, whose in-flight time past the work in hand counts as
    /// unexplained until the engine's next output.
    fn retire(&mut self, request_id: &str) -> Option<InFlight<O>> {
        let mut entry = self.requests.remove(request_id)?;
        let now = Instant::now();
        self.prefilled(&mut entry);
        if entry.every_step {
            self.every_step = self.every_step.saturating_sub(1);
        }
        if self.requests.is_empty() {
            if let Some(since) = self.in_flight_since.take() {
                self.unanswered_before += now.saturating_duration_since(since.max(self.work_end));
            }
            self.idle_since = Some(now);
        }
        Some(entry)
    }

    /// The engine answered (an output of any kind): nothing in flight is
    /// unanswered any more, and the busy period, if one is on, counts from
    /// `now`. The work in hand is read again once the batch is routed
    /// ([`Self::work_resumes`]).
    fn output_seen(&mut self, now: Instant) {
        self.unanswered_before = Duration::ZERO;
        if self.in_flight_since.is_some() {
            self.in_flight_since = Some(now);
        }
    }

    /// The engine's prefill work in hand after an output, from `now`: the
    /// allowance for the prompt still pending prefill once the batch that
    /// came is routed (a prompt whose first token it carried is done; one
    /// that got nothing yet may be next). Read after the routing, so the
    /// figure is the pending set the batch left.
    fn work_resumes(&mut self, now: Instant) {
        self.work_end = deadline(now, self.prefill_allowance(self.prefill_pending));
    }

    /// How long the engine has had requests in flight, past its prefill work
    /// in hand, since its last output: the busy period it is in from the
    /// later of its start and the work's end, and the ones that ended since
    /// (their time past the work in hand then), unless a whole window idle
    /// separated them from it. The silence watchdog's measure of the
    /// engine's silence: a request registered just before the watchdog's
    /// tick has a whole window to answer past its prefill, while a core that
    /// reads every request and answers none is given up on even when every
    /// request is abandoned before the window.
    fn unexplained_for(&self, now: Instant) -> Duration {
        self.unanswered_before
            + self.in_flight_since.map_or(Duration::ZERO, |since| {
                now.saturating_duration_since(since.max(self.work_end))
            })
    }

    /// Deliver one output to its request stream; drop the entry when terminal.
    ///
    /// The lookup key stays borrowed from the output, so the hot path — a
    /// non-terminal token batch — never allocates one.
    fn route(&mut self, output: O) {
        if output.finished() {
            // Terminal: retire the entry before handing the output over, while
            // its request id is still borrowable.
            match self.retire(output.request_id()) {
                Some(entry) => {
                    let _ = entry.sender.send(Ok(output));
                }
                None => trace!(
                    request_id = output.request_id(),
                    "output for unknown/finished request; dropping"
                ),
            }
            return;
        }
        let Some(entry) = self.requests.get_mut(output.request_id()) else {
            trace!(
                request_id = output.request_id(),
                "output for unknown/finished request; dropping"
            );
            return;
        };
        if !entry.started {
            // The first output: the prompt is prefilled (one that overtook the
            // hand-over was never counted).
            entry.started = true;
            if entry.handed {
                self.prefill_pending = self.prefill_pending.saturating_sub(entry.prompt_tokens);
            }
        }
        // A failed send hands the output back, so the receiver-gone (stream
        // dropped) path retires the entry without an owned key either.
        if let Err(mpsc::error::SendError(Ok(undelivered))) = entry.sender.send(Ok(output)) {
            self.retire(undelivered.request_id());
        }
    }

    /// Remove finished/aborted request ids reported out-of-band by the engine.
    fn remove_all<'a>(&mut self, request_ids: impl IntoIterator<Item = &'a String>) {
        for request_id in request_ids {
            self.retire(request_id);
        }
    }

    /// Prompt tokens of the in-flight requests without a first output yet.
    fn prefill_pending_tokens(&self) -> u64 {
        self.prefill_pending
    }

    /// How long the engine may take to prefill `tokens` at the allowed rate
    /// (`prefill_tokens_per_sec`); zero when nothing is allowed.
    fn prefill_allowance(&self, tokens: u64) -> Duration {
        if self.prefill_tokens_per_sec == 0 || tokens == 0 {
            return Duration::ZERO;
        }
        let millis = u128::from(tokens) * 1_000 / u128::from(self.prefill_tokens_per_sec);
        Duration::from_millis(u64::try_from(millis).unwrap_or(u64::MAX))
    }

    /// When the silence watchdog checks next: when the unexplained silence
    /// reaches the window, counted from the latest of now, the busy period's
    /// start and the end of the prefill work in hand, and within a window at
    /// most, so the figures are re-read at least once a window.
    fn next_tick(&self, now: Instant) -> Instant {
        let left = self
            .silence_window
            .saturating_sub(self.unexplained_for(now));
        let from = self
            .in_flight_since
            .map_or(now, |since| now.max(since).max(self.work_end));
        deadline(from, left).min(deadline(now, self.silence_window))
    }

    /// In-flight requests the engine reports on every step it decodes.
    fn reported_every_step(&self) -> usize {
        self.every_step
    }

    /// Register a utility call, returning the receiver its reply resolves.
    fn register_utility(&mut self, call_id: i64) -> Result<oneshot::Receiver<Result<OpaqueValue>>> {
        if self.closed {
            return Err(Error::ClientClosed {
                message: "client is shutting down".to_string(),
            });
        }
        let (sender, receiver) = oneshot::channel();
        self.utility_calls.insert(call_id, sender);
        Ok(receiver)
    }

    /// Complete the call waiting on `reply.call_id`. A reply nobody waits for
    /// (a timed-out call, or one of vLLM's unprompted negative-id notices) is
    /// dropped.
    fn resolve_utility(&mut self, reply: UtilityReply) {
        let Some(sender) = self.utility_calls.remove(&reply.call_id) else {
            trace!(
                call_id = reply.call_id,
                "utility reply for no waiting call; dropping"
            );
            return;
        };
        let outcome = reply
            .outcome
            .map_err(|message| Error::UtilityCallFailed { message });
        let _ = sender.send(outcome);
    }

    /// Fail every in-flight request and pending utility call with a shared
    /// error and close the registry.
    fn fail_all(&mut self, error: Arc<Error>) {
        self.closed = true;
        for (_, entry) in self.requests.drain() {
            let _ = entry.sender.send(Err(Error::Shared(error.clone())));
        }
        self.in_flight_since = None;
        self.unanswered_before = Duration::ZERO;
        self.prefill_pending = 0;
        self.every_step = 0;
        for (_, sender) in self.utility_calls.drain() {
            let _ = sender.send(Err(Error::Shared(error.clone())));
        }
    }
}

/// The DP routing signal: what the engines report and what this client knows
/// it put there. Both maps live under one mutex because every reader and every
/// writer touches them together — two mutexes made the acquisition order an
/// invariant nothing enforced, and selection (load then in-flight) taking it
/// opposite to release (in-flight then load) deadlocked the client.
#[derive(Default)]
struct RoutingState {
    /// Latest per-rank load, keyed by engine index. Reported values only —
    /// `select_engine` never mutates this map; its reservations live in
    /// `inflight`.
    load: HashMap<u32, EngineLoad>,
    /// This client's own unfinished request ids, per rank: a floor under the
    /// (possibly stale) reported load, so routing never trusts a snapshot that
    /// says an engine we just loaded is idle. Keyed by id rather than counted,
    /// because a request has several racing ways to finish — the explicit
    /// [`Client::abort`], a dropped stream's auto-abort, and the engine's own
    /// terminal output or finished-id report — and only the first of them may
    /// count. Identity makes every release idempotent; counters would let one
    /// request retire several slots and leave the rank looking emptier than it
    /// is.
    inflight: HashMap<u32, HashSet<String>>,
}

/// One frame queued for the engines' shared input socket.
struct InputFrame {
    engine_id: EngineId,
    request_type: Bytes,
    payload: Bytes,
    aux_frames: Vec<Bytes>,
    /// Where the caller waits for the write's result; `None` for a frame no
    /// caller waits on (the dispatcher's aborts and wakes).
    done: Option<oneshot::Sender<Result<()>>>,
    /// The payload bytes the queue bound counts while the frame waits:
    /// `Some` for an add-request, `None` for a control frame.
    counted: Option<usize>,
}

/// The bound on add-request frames waiting behind a blocked input (see
/// [`ClientInner::admit_add`]).
#[derive(Clone, Copy, Default)]
struct InputQueueLimit {
    frames: usize,
    bytes: usize,
}

/// Logical payloads still owned by zeromq, including its pending writer message.
/// Codec/kernel buffers are separate; an oversized message may occupy an empty
/// budget alone, preserving support for tensors larger than the byte limit.
#[derive(Default)]
struct InputBudget {
    used: Mutex<InputQueueLimit>,
    released: Notify,
}

/// A queue credit lives until the last transport-owned frame reference is dropped,
/// not until send returns, the caller cancels, or the request registry is retired.
struct InputCredit {
    budget: Arc<InputBudget>,
    bytes: usize,
}

impl Drop for InputCredit {
    /// Return both credits exactly once, after every frame owner has gone away.
    fn drop(&mut self) {
        let mut used = self.budget.used.lock();
        used.frames -= 1;
        used.bytes -= self.bytes;
        drop(used);
        self.budget.released.notify_one();
    }
}

/// Carry queue credit through zeromq's Bytes clones without copying the payload.
struct BudgetedBytes {
    bytes: Bytes,
    _credit: Arc<InputCredit>,
}

impl AsRef<[u8]> for BudgetedBytes {
    /// Preserve the original bytes and wire representation.
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl InputCredit {
    /// Attach ownership accounting to one frame; clones share the same credit.
    fn wrap(self: &Arc<Self>, bytes: Bytes) -> Bytes {
        Bytes::from_owner(BudgetedBytes {
            bytes,
            _credit: self.clone(),
        })
    }
}

struct ClientInner<P: EngineProtocol> {
    /// Frames for the shared input ROUTER socket, in submission order. The
    /// input writer ([`run_input_writer`]) owns the socket and writes them one
    /// at a time, so a caller that stopped waiting never cancels a write.
    input_tx: mpsc::UnboundedSender<InputFrame>,
    /// How long a caller waits for local zeromq queue admission:
    /// [`ENGINE_SEND_TIMEOUT`] unless [`Client::set_send_timeout`] changed it.
    send_timeout: Mutex<Duration>,
    /// How long queue admission may stall before the client closes:
    /// [`ENGINE_INPUT_DEATH_TIMEOUT`] unless [`Client::set_input_death_timeout`]
    /// changed it.
    input_death_timeout: Mutex<Duration>,
    /// The bound on waiting add-requests: [`ENGINE_INPUT_QUEUE_FRAMES`] and
    /// [`ENGINE_INPUT_QUEUE_BYTES`] unless [`Client::set_input_queue_limit`]
    /// changed it.
    input_queue_limit: Mutex<InputQueueLimit>,
    /// Separate bounded allowance for payload references handed to zeromq.
    input_budget: Arc<InputBudget>,
    /// Add-request frames waiting in the input queue (not the one being
    /// written) and their payload bytes, for the bound.
    waiting_adds: AtomicUsize,
    waiting_add_bytes: AtomicUsize,
    /// When the input writer began the write it is in, if any: a write
    /// pending for a whole send window is the sign of a blocked input.
    writing_since: Mutex<Option<Instant>>,
    /// Add-requests refused by the bound since the engine last accepted a
    /// frame, for one log line when it does.
    refused: AtomicUsize,
    engines: Vec<ConnectedEngine>,
    registry: Mutex<RequestRegistry<P::Output>>,
    routing: Mutex<RoutingState>,
    /// Rotates the selection scan start so all-zero ties (cold start, fresh
    /// stats) don't systematically favor the first engine.
    scan_start: AtomicUsize,
    /// Wave clock for a lockstep engine group (see [`WaveEvent`] for what a
    /// wave is), mirroring the `current_wave` vLLM's DP coordinator keeps.
    /// `None` when the ranks step independently (dense DP, DP=1) and so never
    /// pause together.
    wave: Option<Mutex<u64>>,
    /// Auto-abort channel fed by dropped streams.
    abort_tx: mpsc::UnboundedSender<(EngineId, String)>,
    /// Issues utility call ids; the engine echoes one on its reply. Positive
    /// only: vLLM reserves negative ids for notices it sends unprompted.
    next_call_id: AtomicI64,
    /// The engine's last output of any kind (a token, a completion, a
    /// stats-only step), in milliseconds on `output_clock`; reset at the
    /// first submission the engine reports every step after a spell without
    /// one, so the silence such a request sees counts from its own dispatch.
    /// Read by [`Client::output_silence`].
    output_clock: Instant,
    last_output_ms: AtomicU64,
    /// The engine has sent a scheduler step without any token (a stats-only
    /// step report): its silence is then the scheduler's own, not a prefill
    /// in progress. Read by [`Client::reports_steps`].
    steps_reported: AtomicBool,
}

impl<P: EngineProtocol> ClientInner<P> {
    /// Pick the engine for a request: by `data_parallel_rank` if set, else the
    /// least-loaded engine. An SMG-pinned rank is authoritative; it is matched
    /// against the engine's ZMQ index (Python `EngineCoreProc`'s 2-byte
    /// identity), which is version-independent — unlike the `data_parallel_rank`
    /// field, which not every vLLM version sends in `EngineCoreReadyResponse`.
    ///
    /// Unpinned requests use vLLM's own frontend DP scoring
    /// (`DPLBAsyncMPClient.get_core_engine_for_request`), single-client form:
    /// an engine's score is the greater of this client's exact in-flight count
    /// and the reported `waiting + running` (the floor survives stale
    /// snapshots), plus a KV-pressure penalty — a queue on a KV-bound engine
    /// drains slowly, so `waiting` is scaled up as `kv_cache_usage` passes
    /// 50%. The scan start rotates so all-zero ties don't always resolve to
    /// the first engine.
    ///
    /// Selection RESERVES the choice: `request_id` is recorded as in flight
    /// here, under the same lock the scores were read with, so a burst between
    /// load reports spreads across ranks instead of dogpiling one snapshot.
    /// Reported load is never mutated — reservations live only in `inflight`,
    /// and every failed submit path must release its reservation via
    /// [`Self::release`]. (vLLM additionally bumps its report cache because
    /// multiple client processes share its engines; this connector is the
    /// sole client of its group, so the in-flight floor already covers it.)
    fn select_engine(&self, data_parallel_rank: Option<u32>, request_id: &str) -> Result<EngineId> {
        match data_parallel_rank {
            Some(rank) => {
                let engine_id = self
                    .engines
                    .iter()
                    .find(|engine| engine.engine_id.engine_index() == Some(rank))
                    .map(|engine| engine.engine_id.clone())
                    .ok_or(Error::InvalidDataParallelRank {
                        rank,
                        num_engines: self.engines.len() as u32,
                    })?;
                self.routing
                    .lock()
                    .inflight
                    .entry(rank)
                    .or_default()
                    .insert(request_id.to_string());
                Ok(engine_id)
            }
            None => {
                if self.engines.is_empty() {
                    return Err(Error::ClientClosed {
                        message: "no engines connected".to_string(),
                    });
                }
                let mut routing = self.routing.lock();
                let RoutingState { load, inflight } = &mut *routing;
                let count = self.engines.len();
                let start = self.scan_start.fetch_add(1, Ordering::Relaxed) % count;
                let mut best: Option<(f64, usize)> = None;
                for offset in 0..count {
                    let position = (start + offset) % count;
                    let index = self.engines[position].engine_id.engine_index();
                    let reported = index
                        .and_then(|i| load.get(&i))
                        .copied()
                        .unwrap_or_default();
                    let own = index.and_then(|i| inflight.get(&i)).map_or(0, HashSet::len) as u64;
                    let mut score = own.max(reported.num_waiting + reported.num_running) as f64;
                    if reported.num_waiting > 0 {
                        score += reported.num_waiting as f64
                            * 6.0
                            * (reported.kv_cache_usage - 0.5).max(0.0);
                    }
                    if best.is_none_or(|(best_score, _)| score < best_score) {
                        best = Some((score, position));
                    }
                }
                let (_, position) = best.unwrap_or((0.0, 0));
                let engine = &self.engines[position];
                if let Some(index) = engine.engine_id.engine_index() {
                    inflight
                        .entry(index)
                        .or_default()
                        .insert(request_id.to_string());
                }
                Ok(engine.engine_id.clone())
            }
        }
    }

    /// Stamp an output from the engine (or a dispatch onto an idle one) now.
    fn note_output(&self) {
        let millis = u64::try_from(self.output_clock.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.last_output_ms.store(millis, Ordering::Relaxed);
    }

    /// Retire in-flight request ids from a rank. Idempotent: an id already
    /// retired by a racing path is simply absent.
    ///
    /// When the retirement empties the rank, its stored queue counts are
    /// zeroed too: engines only report load on output batches (there is no
    /// heartbeat), and the last batch a rank ever sends is sampled before its
    /// final request's finish is committed — so an idle rank's last snapshot
    /// permanently shows that request still resident. This client routed
    /// every request, so an empty in-flight set is ground truth that the
    /// rank's queue is empty; the KV term is left as reported (cache pages
    /// outlive requests).
    fn release<'a>(&self, engine_index: u32, request_ids: impl IntoIterator<Item = &'a String>) {
        let mut routing = self.routing.lock();
        let Some(ids) = routing.inflight.get_mut(&engine_index) else {
            return;
        };
        for request_id in request_ids {
            ids.remove(request_id);
        }
        if ids.is_empty() {
            if let Some(load) = routing.load.get_mut(&engine_index) {
                load.num_running = 0;
                load.num_waiting = 0;
            }
        }
    }

    /// Retire one request from the rank named by `engine_id`, if it has one.
    fn release_one(&self, engine_id: &EngineId, request_id: &String) {
        if let Some(index) = engine_id.engine_index() {
            self.release(index, [request_id]);
        }
    }

    /// Wait for room in the transport-owned payload budget. The single input
    /// writer reserves before enqueueing, so the upstream's 100k-message queue
    /// cannot bypass SMG's frame/byte bound. A single oversized frame is allowed
    /// only while this budget is empty; every following frame waits for release.
    async fn reserve_input(&self, bytes: usize) -> Arc<InputCredit> {
        loop {
            let released = self.input_budget.released.notified();
            {
                let limit = *self.input_queue_limit.lock();
                let mut used = self.input_budget.used.lock();
                if used.frames == 0
                    || (used.frames < limit.frames
                        && used.bytes <= limit.bytes
                        && bytes <= limit.bytes.saturating_sub(used.bytes))
                {
                    used.frames += 1;
                    used.bytes += bytes;
                    return Arc::new(InputCredit {
                        budget: self.input_budget.clone(),
                        bytes,
                    });
                }
            }
            released.await;
        }
    }

    /// Queue a frame in submission order and wait for local zeromq admission.
    /// Completion is not a transport flush or engine acknowledgement. Once the
    /// transport-owned payload budget fills, the send timeout fails this request while the
    /// independent writer continues; the input-death timeout closes the client
    /// if admission remains stalled. Output silence detects admitted requests
    /// that never produce output, and the input monitor detects disconnects.
    /// Cancelling a caller does not retract its frame; its Abort follows its Add.
    async fn send_to_engine(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        self.send_frame(engine_id, request_type, payload, aux_frames, None)
            .await
    }

    /// [`Self::send_to_engine`] for an add-request: admitted against the
    /// queue bound first (see [`Self::admit_add`]) and counted while it waits.
    async fn send_add_to_engine(
        &self,
        engine_id: &EngineId,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        let bytes = payload.len() + aux_frames.iter().map(Bytes::len).sum::<usize>();
        self.admit_add(bytes)?;
        self.send_frame(engine_id, P::add_frame(), payload, aux_frames, Some(bytes))
            .await
    }

    /// Admit an add-request of `bytes` to the input queue. Free while the
    /// zeromq accepts frames locally; once admission has stalled for a whole
    /// send window (the transport-owned payload budget is full), a request that would take the waiting
    /// add-requests past the queue bound is refused at once with
    /// [`Error::EngineInputFull`]: it is better served elsewhere than parked
    /// here for the death window, and the frames already waiting keep their
    /// order behind the blocked one. A request larger than the byte bound on
    /// its own is refused the same way, with nothing waiting: the bound is
    /// what may be parked, and a request refused at once is retried elsewhere
    /// a whole window sooner than one that waits; the error names the
    /// request's size. The check is not synchronized with the
    /// queue, so a burst can overshoot the bound by a few frames; a burst to
    /// an engine that reads is never refused, whatever its size.
    fn admit_add(&self, bytes: usize) -> Result<()> {
        let send_timeout = *self.send_timeout.lock();
        let blocked = self
            .writing_since
            .lock()
            .is_some_and(|since| since.elapsed() >= send_timeout);
        if !blocked {
            return Ok(());
        }
        let limit = *self.input_queue_limit.lock();
        let frames = self.waiting_adds.load(Ordering::Acquire);
        let waiting = self.waiting_add_bytes.load(Ordering::Acquire);
        if frames < limit.frames && waiting.saturating_add(bytes) <= limit.bytes {
            return Ok(());
        }
        if self.refused.fetch_add(1, Ordering::AcqRel) == 0 {
            warn!(
                frames,
                bytes = waiting,
                limit_frames = limit.frames,
                limit_bytes = limit.bytes,
                "engine input blocked and its queue at the bound; refusing new requests \
                 until the engine reads again"
            );
        }
        Err(Error::EngineInputFull {
            frames,
            bytes: waiting,
            request_bytes: bytes,
            limit_frames: limit.frames,
            limit_bytes: limit.bytes,
        })
    }

    /// Queue a frame and wait for local zeromq admission; `counted` is the
    /// payload size the queue bound tracks for an add-request.
    async fn send_frame(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
        counted: Option<usize>,
    ) -> Result<()> {
        let done = self.queue_write(engine_id, request_type, payload, aux_frames, counted)?;
        self.await_writes([done]).await
    }

    /// Queue a frame for the input writer, returning where the write's result
    /// arrives (see [`Self::queue_frame`]).
    fn queue_write(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
        counted: Option<usize>,
    ) -> Result<oneshot::Receiver<Result<()>>> {
        let (done_tx, done_rx) = oneshot::channel();
        self.queue_frame(
            engine_id,
            request_type,
            payload,
            aux_frames,
            Some(done_tx),
            counted,
        )?;
        Ok(done_rx)
    }

    /// Wait at most the send window for the engines to accept the writes
    /// queued as `pending`, all of them. Past the window the caller's request
    /// fails with [`Error::EngineInputBlocked`]; the frames stay queued and
    /// are delivered when the engine reads again.
    async fn await_writes(
        &self,
        pending: impl IntoIterator<Item = oneshot::Receiver<Result<()>>>,
    ) -> Result<()> {
        let timeout = *self.send_timeout.lock();
        let accepted = async {
            for done in pending {
                done.await.map_err(|_writer_gone| Error::ClientClosed {
                    message: "the engine input writer ended".to_string(),
                })??;
            }
            Ok::<(), Error>(())
        };
        match tokio::time::timeout(timeout, accepted).await {
            Ok(result) => result,
            Err(_elapsed) => {
                warn!(
                    ?timeout,
                    "engine did not accept a request frame within the send window; \
                     failing this request, the frame is delivered when the engine reads again"
                );
                Err(Error::EngineInputBlocked { timeout })
            }
        }
    }

    /// Queue a frame for the input writer, behind every frame queued before
    /// it; `done`, if any, receives the result of the write, and `counted`
    /// bytes are held against the queue bound until the writer takes the
    /// frame.
    fn queue_frame(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
        done: Option<oneshot::Sender<Result<()>>>,
        counted: Option<usize>,
    ) -> Result<()> {
        if let Some(bytes) = counted {
            self.waiting_adds.fetch_add(1, Ordering::AcqRel);
            self.waiting_add_bytes.fetch_add(bytes, Ordering::AcqRel);
        }
        self.input_tx
            .send(InputFrame {
                engine_id: engine_id.clone(),
                request_type,
                payload: payload.into(),
                aux_frames,
                done,
                counted,
            })
            .map_err(|_writer_gone| Error::ClientClosed {
                message: "the engine input writer ended".to_string(),
            })
    }

    /// Tell every rank to start `wave`. Does nothing for a protocol without a
    /// wave protocol.
    ///
    /// No rank is excluded, not even the one already holding the request: a
    /// rank that skips an update keeps a lower `current_wave` than its peers,
    /// and the group's wave numbers must stay identical for
    /// [`Self::wake_group`]'s ordering argument to hold.
    async fn broadcast_start_wave(&self, wave: u64) -> Result<()> {
        let Some((frame, payload)) = P::encode_start_wave(wave)? else {
            return Ok(());
        };
        // Every rank's frame is queued before any is waited for: a window that
        // runs out (a frame is blocked ahead in the queue) then never leaves
        // the group half woken; the wake reaches every rank when the engine
        // reads again.
        let pending = self
            .engines
            .iter()
            .map(|engine| {
                self.queue_write(
                    &engine.engine_id,
                    frame.clone(),
                    payload.clone(),
                    Vec::new(),
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.await_writes(pending).await
    }

    /// [`Self::broadcast_start_wave`] without waiting for the engines to
    /// accept the frames: the dispatcher's wake, which must keep routing the
    /// output of every rank while a frame ahead in the queue is blocked.
    fn queue_start_wave(&self, wave: u64) -> Result<()> {
        let Some((frame, payload)) = P::encode_start_wave(wave)? else {
            return Ok(());
        };
        for engine in &self.engines {
            self.queue_frame(
                &engine.engine_id,
                frame.clone(),
                payload.clone(),
                Vec::new(),
                None,
                None,
            )?;
        }
        Ok(())
    }

    /// Wake a paused lockstep group so the rank now holding a request can make
    /// progress: its peers must be stepping too, because every rank joins the
    /// same all-reduce. Upstream vLLM has a DP coordinator process broadcast the
    /// restart; SMG owns rank selection, so it sends the same signal over the
    /// input socket each rank already listens on. A no-op for independent
    /// ranks.
    ///
    /// The wake is UNCONDITIONAL for a lockstep group. Gating it on the
    /// tracked `running` flag raced the drain: between the ranks agreeing to
    /// park and this client processing their `wave_complete`, a submit would
    /// see a stale "running" state, skip the wake, and — without a
    /// coordinator, a parked engine does not self-wake on ADD — strand the
    /// request until the silence watchdog killed the group.
    ///
    /// The wave number is a logical clock this client owns, bumped on every
    /// wake, and an engine only obeys a wave at or above its own. Bumping is
    /// what makes the wake reliable across that same drain window: the ranks
    /// increment `current_wave` themselves as they park, so re-sending the
    /// number we last knew about would land *below* theirs and be dropped on
    /// the floor — a lost wake, whose peers then never join the holder's
    /// all-reduce, hanging the group until vLLM's own RPC timeout kills the
    /// engine. Bumping keeps us ahead: every rank sits at the last number we
    /// broadcast plus at most one self-increment (a second needs another wave,
    /// which needs another wake), so `last + 1` is never stale.
    async fn wake_group(&self) -> Result<()> {
        let Some(wave) = self.next_wave() else {
            return Ok(());
        };
        self.broadcast_start_wave(wave).await
    }

    /// Bump the group's wave clock for a wake; `None` for independent ranks.
    fn next_wave(&self) -> Option<u64> {
        let state = self.wave.as_ref()?;
        let mut current = state.lock();
        *current = current.saturating_add(1);
        Some(*current)
    }

    /// Fold a wave notification from an engine into the group's clock. Never
    /// waits on the engine: the dispatcher calls this between outputs.
    fn observe_wave(&self, event: WaveEvent, engine_index: u32) {
        let Some(state) = self.wave.as_ref() else {
            warn!(
                ?event,
                engine_index, "wave notification from independent ranks; ignoring"
            );
            return;
        };
        match event {
            // The group drained the wave and parked itself, incrementing its
            // own `current_wave` on the way out; track that so the next wake
            // names a wave the ranks will still accept.
            WaveEvent::Complete(wave) => {
                let mut current = state.lock();
                *current = (*current).max(wave.saturating_add(1));
            }
            // A rank took a request for an already-drained wave and is asking
            // for the rest of the group to catch up.
            WaveEvent::Start(wave) => {
                {
                    let mut current = state.lock();
                    *current = (*current).max(wave);
                }
                // Queued for every rank, not awaited: a frame blocked ahead in
                // the input queue must not park the dispatcher (the output of
                // every rank keeps flowing), and the wake reaches the whole
                // group when the engine reads again.
                let queued = match self.next_wave() {
                    Some(next) => self.queue_start_wave(next),
                    None => Ok(()),
                };
                if let Err(error) = queued {
                    warn!(%error, wave, "failed to queue the requested wave");
                }
            }
        }
    }

    /// Send an Abort for one request id to its engine.
    async fn abort(&self, engine_id: &EngineId, request_id: &str) -> Result<()> {
        let payload = P::encode_abort(request_id)?;
        self.send_to_engine(engine_id, P::abort_frame(), payload, Vec::new())
            .await
    }

    /// Queue the abort of `request_id` without waiting for the engine to
    /// accept it: the frame goes out in order behind whatever the engine has
    /// not read yet, and the input writer reports a write that failed.
    fn queue_abort(&self, engine_id: &EngineId, request_id: &str) -> Result<()> {
        let payload = P::encode_abort(request_id)?;
        self.queue_frame(engine_id, P::abort_frame(), payload, Vec::new(), None, None)
    }
}

/// Direct ZMQ connection to a same-host engine (one or more DP ranks behind one
/// shared transport), generic over the engine's wire protocol `P`.
pub struct Client<P: EngineProtocol> {
    inner: Arc<ClientInner<P>>,
    tasks: Vec<JoinHandle<()>>,
}

impl<P: EngineProtocol> Client<P> {
    /// Build a client over a connected transport, spawning the input writer,
    /// the output loop and the dispatcher. Background tasks are aborted when
    /// the client is dropped, but already admitted zeromq frames may still arrive.
    /// Transport-owned payloads have a separate allowance with the same limits.
    /// Codec and kernel buffers are not included in either allowance.
    pub fn new(transport: ConnectedTransport) -> Self {
        let ConnectedTransport {
            engines,
            input_send,
            mut input_events,
            mut input_recv,
            output_socket,
            ..
        } = transport;

        // Only a group whose ranks step in lockstep pauses as a unit and needs
        // waking. The engines report which they are at handshake: a lockstep
        // group keeps its data-parallel size, while independent ranks are
        // reconfigured to a size of one before they answer.
        let lockstep = engines
            .iter()
            .any(|engine| engine.ready_response.data_parallel_size > 1);

        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let inner = Arc::new(ClientInner {
            input_tx,
            send_timeout: Mutex::new(ENGINE_SEND_TIMEOUT),
            input_death_timeout: Mutex::new(ENGINE_INPUT_DEATH_TIMEOUT),
            input_queue_limit: Mutex::new(InputQueueLimit {
                frames: ENGINE_INPUT_QUEUE_FRAMES,
                bytes: ENGINE_INPUT_QUEUE_BYTES,
            }),
            input_budget: Arc::new(InputBudget::default()),
            waiting_adds: AtomicUsize::new(0),
            waiting_add_bytes: AtomicUsize::new(0),
            writing_since: Mutex::new(None),
            refused: AtomicUsize::new(0),
            engines,
            registry: Mutex::new(RequestRegistry::default()),
            routing: Mutex::new(RoutingState::default()),
            scan_start: AtomicUsize::new(0),
            wave: lockstep.then(|| Mutex::new(0)),
            abort_tx,
            next_call_id: AtomicI64::new(1),
            output_clock: Instant::now(),
            last_output_ms: AtomicU64::new(0),
            steps_reported: AtomicBool::new(false),
        });

        // Input writer: the one owner of the input socket's send half.
        let writer_inner = inner.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "background tasks are aborted on client drop"
        )]
        let input_task = tokio::spawn(async move {
            tokio::select! {
                () = run_input_writer::<P>(input_send, input_rx, writer_inner.clone()) => {}
                () = async {
                    // Polling the receive half drives read-side EOF detection even
                    // when no more writes are sent to a disconnected engine.
                    loop {
                        match input_recv.recv().await {
                            Ok(message) => trace!(
                                frames = message.len(),
                                "ignoring unexpected post-registration input message"
                            ),
                            Err(error) => {
                                warn!(%error, "input receiver ended; idle disconnect detection stopped");
                                break;
                            }
                        }
                    }
                    std::future::pending::<()>().await;
                } => {}

                () = async {
                    while let Some(event) = input_events.next().await {
                        if let zeromq::SocketEvent::Disconnected(peer) = event {
                            warn!(?peer, "engine input transport disconnected after queue admission");
                            writer_inner.registry.lock().fail_all(Arc::new(Error::EngineCoreDead));
                            writer_inner.routing.lock().inflight.clear();
                            return;
                        }
                    }
                    // Monitor closure alone does not prove engine death.
                    std::future::pending::<()>().await;
                } => {}
            }
        });
        // Transport output loop: decode raw frames -> EngineBatch channel.
        let (out_tx, out_rx) = mpsc::channel(256);
        #[expect(
            clippy::disallowed_methods,
            reason = "background tasks are aborted on client drop"
        )]
        let output_task = tokio::spawn(run_output_loop::<P>(output_socket, out_tx));
        #[expect(
            clippy::disallowed_methods,
            reason = "background tasks are aborted on client drop"
        )]
        let dispatch_task = tokio::spawn(run_dispatcher::<P>(out_rx, abort_rx, inner.clone()));

        Self {
            inner,
            tasks: vec![input_task, output_task, dispatch_task],
        }
    }

    /// The engines connected on this transport.
    pub fn engines(&self) -> &[ConnectedEngine] {
        &self.inner.engines
    }

    /// Whether the connection is still live. Becomes false once the dispatcher
    /// observes `ENGINE_CORE_DEAD`, a transport failure, or the output stream
    /// closing, or once local queue admission has stalled for
    /// `ENGINE_INPUT_DEATH_TIMEOUT`. No RPC — this is a local liveness flag.
    pub fn is_alive(&self) -> bool {
        !self.inner.registry.lock().closed
    }

    /// Bound how long a submit, abort or utility call waits for zeromq to
    /// accept its frame in the local peer queue (`ENGINE_SEND_TIMEOUT` by
    /// default). Past it the call fails with [`Error::EngineInputBlocked`];
    /// the frame is still delivered when the engine reads again.
    pub fn set_send_timeout(&self, timeout: Duration) {
        *self.inner.send_timeout.lock() = timeout;
    }

    /// Bound how long admission to a full zeromq peer queue may stall before
    /// the engine is treated as dead (`ENGINE_INPUT_DEATH_TIMEOUT` by
    /// default): the requests in flight fail and the client closes.
    pub fn set_input_death_timeout(&self, timeout: Duration) {
        *self.inner.input_death_timeout.lock() = timeout;
    }

    /// Bound how long the engine may have requests in flight, in total since
    /// its last output and past the prefill allowance for the prompt it was
    /// handed (the rate [`Self::set_prefill_tokens_per_sec`] allows), before
    /// it is treated as dead ([`ENGINE_SILENCE_DEATH_TIMEOUT`] by default).
    pub fn set_silence_death_timeout(&self, timeout: Duration) {
        self.inner.registry.lock().silence_window = timeout;
    }

    /// The prefill rate the silence watchdog allows the engine
    /// ([`ENGINE_PREFILL_TOKENS_PER_SEC`] by default): the prompt tokens the
    /// engine is handed queue up as prefill work in hand at this rate, from
    /// the moment it accepts a request's frame (an add it never accepted
    /// queues nothing), and only its silence past that work counts toward
    /// the window, so a batch
    /// of long prompts on a wire that says nothing through a prefill is not
    /// read as a dead engine. A prompt withdrawn before its first output
    /// leaves its allowance in hand until it passes. Zero allows nothing:
    /// the bound is the silence window alone.
    pub fn set_prefill_tokens_per_sec(&self, tokens_per_sec: u64) {
        self.inner.registry.lock().prefill_tokens_per_sec = tokens_per_sec;
    }

    /// Bound the add-requests that may wait behind a blocked input before new
    /// ones are refused at once with [`Error::EngineInputFull`]
    /// (`ENGINE_INPUT_QUEUE_FRAMES` frames and `ENGINE_INPUT_QUEUE_BYTES`
    /// payload bytes by default). The same limits independently bound frames
    /// still owned by zeromq. That budget admits one oversized message alone;
    /// it is released by transport ownership, never by caller cancellation.
    pub fn set_input_queue_limit(&self, frames: usize, bytes: usize) {
        *self.inner.input_queue_limit.lock() = InputQueueLimit { frames, bytes };
        self.inner.input_budget.released.notify_one();
    }

    /// The latest per-rank load for one engine index (DP routing signal), if
    /// any batch has been seen from it yet.
    pub fn engine_load(&self, engine_index: u32) -> Option<EngineLoad> {
        self.inner.routing.lock().load.get(&engine_index).copied()
    }

    /// How long the engine has produced nothing while this client holds
    /// requests in flight on it that the engine reports on every step
    /// ([`EngineProtocol::reported_every_step`]): the time since its last
    /// output batch of any kind (a token, a completion, a stats-only step),
    /// counted from the dispatch of the first such request after a spell
    /// without one; `None` while none is in flight. A request the engine
    /// reports at intervals (one that does not stream, one streamed every so
    /// many tokens) is quiet while healthy, so it counts no silence of its
    /// own, though its prompt counts in [`Self::prefill_pending_tokens`]
    /// (its prefill silences the rest). A frozen engine keeps its connection
    /// open and its last load report standing, so this is the first sign of
    /// it: the servicer puts it on the load record it pushes
    /// (`EngineLoad.engine_silence_ms`) for the gateway's wedge rule, with
    /// [`Self::reports_steps`] beside it to say whether the silence is the
    /// scheduler's own.
    pub fn output_silence(&self) -> Option<Duration> {
        if self.inner.registry.lock().reported_every_step() == 0 {
            return None;
        }
        let since = Duration::from_millis(self.inner.last_output_ms.load(Ordering::Relaxed));
        Some(self.inner.output_clock.elapsed().saturating_sub(since))
    }

    /// Whether the engine's wire reports every scheduler step: it has shown
    /// a scheduler step without any token (a stats-only step report, which
    /// vLLM sends every step while its stats logging is on), or the launcher
    /// that started the engine said so ([`Self::note_reports_steps`]). From
    /// then on [`Self::output_silence`] is the scheduler's silence. A wire
    /// that carries token batches only (SGLang, TokenSpeed) never sets it,
    /// and its silence may be a prefill in progress, which the gateway
    /// allows for.
    pub fn reports_steps(&self) -> bool {
        self.inner.steps_reported.load(Ordering::Relaxed)
    }

    /// The engine's wire reports every scheduler step, as the launcher that
    /// started the engine knows (its scheduler statistics are on): hold
    /// [`Self::reports_steps`] from now rather than from the first stats-only
    /// step batch, so an engine that freezes before its first batch after
    /// the servicer connected is judged on the step-reporting path too. A
    /// client told nothing is classified by what its wire shows.
    pub fn note_reports_steps(&self) {
        self.inner.steps_reported.store(true, Ordering::Relaxed);
    }

    /// Prompt tokens of the requests the engine has accepted (their add
    /// frames handed over) that have produced no output yet: the engine's
    /// prefill work in hand, which the servicer puts on its
    /// load record (`EngineLoad.prefill_pending_tokens`) so a gateway can
    /// allow for it when it judges the engine's silence, whatever it has in
    /// flight on the engine itself.
    pub fn prefill_pending_tokens(&self) -> u64 {
        self.inner.registry.lock().prefill_pending_tokens()
    }

    /// Submit a request and return a stream of its outputs. The request is
    /// routed by `data_parallel_rank` (SMG-pinned) or, unpinned, to the
    /// least-loaded engine (see `ClientInner::select_engine`).
    /// Dropping the returned stream before it finishes aborts the request.
    pub async fn submit(&self, request: P::Request) -> Result<RequestStream<P>> {
        self.submit_with_aux(request, Vec::new()).await
    }

    /// [`submit`](Self::submit) with `aux_frames` appended after the payload:
    /// vLLM's zero-copy tensor frames, which the payload references by index
    /// (the payload is frame 0, so the first of these is frame 1). The
    /// request's own encoding must not reference aux frames of its own.
    pub async fn submit_with_aux(
        &self,
        request: P::Request,
        extra_aux_frames: Vec<Bytes>,
    ) -> Result<RequestStream<P>> {
        P::validate(&request)?;
        let request_id = P::request_id(&request).to_string();
        // Register before selecting: registration is the admission gate (it
        // rejects a duplicate id), and in-flight slots are held by id, so a
        // reservation taken before that gate could collide with the live
        // request's own slot and release it on rollback.
        let every_step = P::reported_every_step(&request);
        let receiver = {
            let mut registry = self.inner.registry.lock();
            let quiet = registry.reported_every_step() == 0;
            let receiver =
                registry.register(request_id.clone(), P::prompt_tokens(&request), every_step)?;
            // The silence a request sees counts from its dispatch: an engine
            // with nothing it reports every step had nothing to say until
            // now.
            if quiet && every_step {
                self.inner.note_output();
            }
            receiver
        };

        // Selection reserves the engine's in-flight slot; every failure path
        // from here to a successful hand-off must release it.
        let engine_id = match self
            .inner
            .select_engine(P::data_parallel_rank(&request), &request_id)
        {
            Ok(engine_id) => engine_id,
            Err(error) => {
                self.inner.registry.lock().remove_all([&request_id]);
                return Err(error);
            }
        };

        // From here on the submit holds the entry and the slot; the guard
        // gives both back, and aborts the request on the engine, whenever the
        // submit does not complete — including a caller that stops polling
        // while the frame waits for the engine.
        let mut guard = SubmitGuard {
            inner: &self.inner,
            engine_id,
            request_id,
            armed: true,
        };
        let (payload, mut aux_frames) = match P::encode_add(&request) {
            Ok(encoded) => encoded,
            Err(error) => {
                guard.take_back(false);
                return Err(error);
            }
        };
        aux_frames.extend(extra_aux_frames);
        if let Err(error) = self
            .inner
            .send_add_to_engine(&guard.engine_id, payload, aux_frames)
            .await
        {
            // A frame the engine has not accepted is still queued for it: the
            // abort queues behind it, so the engine drops the request on
            // arrival instead of serving it for nobody. A frame refused at the
            // queue's bound, or by the socket, never reached the engine, so
            // nothing is there to abort.
            guard.take_back(matches!(error, Error::EngineInputBlocked { .. }));
            return Err(error);
        }
        // The engine has the prompt: its prefill is work in hand from here,
        // whatever becomes of the submit (a request taken back from the engine
        // keeps its allowance like any withdrawn prompt).
        self.inner.registry.lock().handed(&guard.request_id);

        // The rank now holds the request; its peers must be awake for it to
        // step. Waking after the send keeps the group parked (and so unable to
        // report another drained wave) for the whole window.
        if let Err(error) = self.inner.wake_group().await {
            // The request can never make progress with its peers asleep, so
            // take it back rather than leaving it stranded on the engine.
            guard.take_back(true);
            return Err(error);
        }

        let (engine_id, request_id) = guard.complete();
        Ok(RequestStream {
            request_id,
            engine_id,
            receiver,
            abort_tx: self.inner.abort_tx.clone(),
            finished: false,
        })
    }

    /// Explicitly abort an in-flight request.
    pub async fn abort(&self, engine_id: &EngineId, request_id: &str) -> Result<()> {
        let request_id = request_id.to_string();
        self.inner.registry.lock().remove_all([&request_id]);
        self.inner.release_one(engine_id, &request_id);
        self.inner.abort(engine_id, &request_id).await
    }

    /// Hand one engine a notice under the protocol's add frame: a message of
    /// the engine's own request vocabulary that names a request already
    /// submitted to that engine and that the engine consumes itself (nothing
    /// answers it, no stream is opened, no registration or in-flight slot is
    /// taken). The caller encodes it; the frame is queued in order behind
    /// the frames before it, so a notice sent after a request's submit
    /// reaches the engine after that request.
    pub async fn notify(&self, engine_id: &EngineId, payload: Vec<u8>) -> Result<()> {
        self.inner
            .send_to_engine(engine_id, P::add_frame(), payload, Vec::new())
            .await
    }

    /// [`notify`](Self::notify) under a request-type frame of the caller's
    /// choosing, with `aux_frames` behind the payload: a message of the
    /// engine's vocabulary outside the add/abort pair that the engine consumes
    /// itself (an encode worker's encode request, the largest payload this
    /// wire carries). Nothing answers it, no stream is opened, no registration
    /// or in-flight slot is taken; the frame is admitted against the input
    /// queue bound and counted while it waits, as an add frame is (see
    /// [`ClientInner::admit_add`]): once the engine has stopped reading for
    /// a whole send window, a frame that would take the waiting payload past
    /// the bound is refused at once rather than parked and delivered later.
    pub async fn send_typed(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        let bytes = payload.len() + aux_frames.iter().map(Bytes::len).sum::<usize>();
        self.inner.admit_add(bytes)?;
        self.inner
            .send_frame(engine_id, request_type, payload, aux_frames, Some(bytes))
            .await
    }
}

impl<P: EngineProtocol> Drop for Client<P> {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

impl<P: EngineProtocol> Client<P> {
    /// Run `method(*args)` on one rank and return its result (vLLM's
    /// `AsyncMPClient._call_utility_async`; the SGLang plugin's control
    /// call): the protocol frames the call and the reply resolves by call id.
    /// Waits at most `timeout`; the engine still runs a call whose reply comes
    /// late, that reply is just dropped. A protocol without control messages
    /// answers [`Error::UtilityUnsupported`] without touching the wire.
    pub async fn call_utility(
        &self,
        engine_id: &EngineId,
        method: &str,
        args: Vec<OpaqueValue>,
        timeout: Duration,
    ) -> Result<OpaqueValue> {
        let call_id = self.inner.next_call_id.fetch_add(1, Ordering::Relaxed);
        let Some((frame, payload)) = P::encode_utility(call_id, method, &args)? else {
            return Err(Error::UtilityUnsupported {
                method: method.to_string(),
            });
        };
        let receiver = self.inner.registry.lock().register_utility(call_id)?;
        // Retires the registration on every exit (timeout, cancellation, a
        // failed send); a resolved call has already left the map.
        let _pending = PendingUtilityCall {
            inner: &self.inner,
            call_id,
        };
        self.inner
            .send_to_engine(engine_id, frame, payload, Vec::new())
            .await?;
        match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(outcome)) => outcome,
            // The registry closed and dropped the sender without an error item.
            Ok(Err(_closed)) => Err(Error::ClientClosed {
                message: "client closed while a utility call was pending".to_string(),
            }),
            Err(_elapsed) => Err(Error::UtilityTimeout {
                method: method.to_string(),
                timeout,
            }),
        }
    }
}

/// Removes a utility call's registry entry once its caller stops waiting.
struct PendingUtilityCall<'a, P: EngineProtocol> {
    inner: &'a ClientInner<P>,
    call_id: i64,
}

impl<P: EngineProtocol> Drop for PendingUtilityCall<'_, P> {
    fn drop(&mut self) {
        self.inner
            .registry
            .lock()
            .utility_calls
            .remove(&self.call_id);
    }
}

/// How long a caller waits for local zeromq admission of one request on the
/// input socket. A healthy engine drains its input continuously; a frame not
/// accepted in this window is behind input the engine is not reading, and the
/// caller's request fails with [`Error::EngineInputBlocked`] while the frame
/// stays queued for the engine's return (see [`ClientInner::send_to_engine`]).
/// The client itself stays up: its death is decided on the output side.
const ENGINE_SEND_TIMEOUT: Duration = Duration::from_secs(10);

/// If the engine has had requests in flight for this long, in total, since its
/// last output (one request's whole window, or a run of shorter ones it read
/// and never answered, with no window-long idle spell between them), with
/// nothing pending prefill, the dispatcher treats it as dead. A hard engine
/// death (SIGKILL/OOM) sends no `ENGINE_CORE_DEAD` sentinel, and a bound PULL
/// socket never errors when its PUSH peer vanishes. Input monitoring catches
/// disconnects, while silence also catches a connected engine that stalls.
/// Generous so a slow first token never trips it, while still bounding an
/// otherwise infinite hang.
///
/// The window is counted past the engine's prefill work in hand: the prompt
/// tokens it is handed queue up as work at [`ENGINE_PREFILL_TOKENS_PER_SEC`]
/// ([`Client::set_prefill_tokens_per_sec`]), and only the silence past that
/// work counts. A wire that reports every scheduler step is never silent
/// through a prefill; one that carries token batches only says nothing until
/// a prompt's first token, and a batch of million-token prompts is silent for
/// as long as their prefill takes, which no fixed window covers. Both figures
/// are per client ([`Client::set_silence_death_timeout`]); neither removes
/// the watchdog: without it a killed engine hangs its requests forever.
pub const ENGINE_SILENCE_DEATH_TIMEOUT: Duration = Duration::from_secs(300);

/// `at + after`, or an instant that never comes when the sum does not fit
/// (a window or an allowance of thousands of years means "no bound", not a
/// panic in the dispatcher).
fn deadline(at: Instant, after: Duration) -> Instant {
    at.checked_add(after).unwrap_or_else(far_future)
}

/// Thirty years out: past any deployment, within what every platform's
/// `Instant` holds.
fn far_future() -> Instant {
    Instant::now() + Duration::from_secs(30 * 365 * 24 * 60 * 60)
}

/// The prefill rate the silence watchdog allows the engine for the prompt
/// tokens pending prefill, by default: a bound on how slowly a healthy engine
/// may prefill, not an estimate of how fast it does. Deliberately low, a fifth
/// of the cold prior a gateway starts its own prefill allowance from (which it
/// then corrects by measuring; the connector measures nothing): at a million
/// tokens of context, attention costs many times what it does at a few
/// thousand, and chunked prefill interleaves a batch of such prompts so that
/// none of them produces a token until most of the batch is prefilled. The
/// cost of the guess is a slower verdict on a frozen engine, in proportion to
/// the prompt it was handed: 500 s per million pending prompt tokens on top of
/// the window, nothing with nothing pending.
pub const ENGINE_PREFILL_TOKENS_PER_SEC: u64 = 2_000;

/// How long local admission to a full zeromq peer queue may stall before the
/// engine is treated as dead. A pause the send window outlasts fails the
/// requests that arrive during it, and only them; a core that has read nothing
/// for this long is hung, not pausing, and the client closes so health checks
/// evict the worker. The input side's counterpart of the silence watchdog
/// above, with the same span, and like it independent of traffic: the output
/// watchdog needs requests in flight, which the requests a blocked input
/// failed are not.
const ENGINE_INPUT_DEATH_TIMEOUT: Duration = ENGINE_SILENCE_DEATH_TIMEOUT;

/// The bound on add-requests waiting behind a blocked input, in frames and in
/// payload bytes: once a send window has elapsed with a frame unaccepted, a
/// submission that would take the waiting add-requests past either is refused
/// at once instead of parking its payload for the death window (a gateway that
/// retries every send window would otherwise grow the queue for as long as the
/// engine does not read). A separate allowance with these same limits covers
/// payload references still owned by zeromq, including control messages. It
/// admits an oversized message only when empty. Codec/kernel buffers remain
/// outside these logical budgets; this is not an allocator/RSS limit. Control
/// frames bypass the waiting-Add limit but not the transport-owned allowance.
const ENGINE_INPUT_QUEUE_FRAMES: usize = 256;
const ENGINE_INPUT_QUEUE_BYTES: usize = 32 << 20;

/// Consecutive undecodable output messages tolerated before the client is
/// declared dead. One corrupt message is survivable, but an engine whose wire
/// schema this client cannot read never recovers: every dropped batch strands
/// the requests it carried, and the silence watchdog never fires because other
/// traffic keeps arriving. Failing those requests is strictly better than
/// hanging them.
const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 3;

/// The dispatcher's state between two arms of its loop ([`run_dispatcher`]):
/// the client it serves and the run of undecodable outputs. Each arm of the
/// loop is a method here, with the arm's statements in the arm's order; the
/// loop joins them with the silence watchdog they re-arm.
struct Dispatcher<P: EngineProtocol> {
    inner: Arc<ClientInner<P>>,
    /// Undecodable outputs in a row; a decoded batch starts the count over
    /// (see [`MAX_CONSECUTIVE_DECODE_ERRORS`]).
    consecutive_decode_errors: u32,
}

/// The silence watchdog's verdict at a tick ([`Dispatcher::watchdog_verdict`]).
enum Verdict {
    /// The engine is idle or its silence is explained: tick again at this
    /// instant.
    Keep(Instant),
    /// A whole window of unexplained silence: the in-flight requests are
    /// failed and the dispatcher ends.
    Dead,
}

impl<P: EngineProtocol> Dispatcher<P> {
    /// A busy period began. The engine has what is left of a window of
    /// unexplained silence to answer in, past its prefill work in hand, not a
    /// whole one: a tick that landed in the gap before this period waited a
    /// whole window, and the sum of the periods before it must not be waited
    /// out again. Returns the watchdog's next tick.
    fn arm_window(&self, now: Instant) -> Instant {
        self.inner.registry.lock().next_tick(now)
    }

    /// The engine sent something, a batch or bytes this client cannot read:
    /// its silence ends at `now`. Returns the watchdog's next tick.
    fn output_seen(&self, now: Instant) -> Instant {
        let mut registry = self.inner.registry.lock();
        registry.output_seen(now);
        registry.next_tick(now)
    }

    /// Route a decoded batch: the rank's load and wave event to the client,
    /// a utility reply to its caller, every output to its request's stream,
    /// and the finished requests out of the registry and the rank's in-flight
    /// set. Returns the watchdog's next tick, counted from the end of the
    /// prefill work the batch left pending.
    fn route_batch(&mut self, batch: EngineBatch<P::Output>) -> Instant {
        self.consecutive_decode_errors = 0;
        let inner = &self.inner;
        inner.note_output();
        if batch.outputs.is_empty() && batch.finished_request_ids.is_empty() && batch.load.is_some()
        {
            inner.steps_reported.store(true, Ordering::Relaxed);
        }
        if let Some(load) = batch.load {
            inner.routing.lock().load.insert(batch.engine_index, load);
        }
        if let Some(event) = batch.wave {
            inner.observe_wave(event, batch.engine_index);
        }
        // Unique finished ids: a terminal output and the out-of-band finished
        // list may name the same request.
        let mut finished: HashSet<String> = batch.finished_request_ids.into_iter().collect();
        let mut registry = inner.registry.lock();
        if let Some(reply) = batch.utility {
            registry.resolve_utility(reply);
        }
        for output in batch.outputs {
            if output.finished() {
                finished.insert(output.request_id().to_string());
            }
            registry.route(output);
        }
        registry.remove_all(&finished);
        // The work in hand is what the batch left pending; the watchdog counts
        // from its end.
        let now = Instant::now();
        registry.work_resumes(now);
        let next = registry.next_tick(now);
        drop(registry);
        inner.release(batch.engine_index, &finished);
        next
    }

    /// The output loop reported an error. A transport failure or the dead
    /// sentinel ends the client at once. A single undecodable output is
    /// non-fatal; a run of [`MAX_CONSECUTIVE_DECODE_ERRORS`] means this client
    /// cannot read the engine, so it fails the in-flight requests rather than
    /// silently drop every output they wait for. `Break` once they are failed
    /// and the dispatcher ends.
    fn output_error(&mut self, error: Error) -> ControlFlow<()> {
        if matches!(error, Error::EngineCoreDead | Error::Transport(_)) {
            warn!(%error, "engine transport failed; failing all in-flight requests");
            self.fail_all(error);
            return ControlFlow::Break(());
        }
        self.consecutive_decode_errors += 1;
        if self.consecutive_decode_errors >= MAX_CONSECUTIVE_DECODE_ERRORS {
            warn!(
                %error,
                consecutive_decode_errors = self.consecutive_decode_errors,
                "engine output undecodable repeatedly; failing all in-flight requests"
            );
            self.fail_all(error);
            return ControlFlow::Break(());
        }
        warn!(%error, "ignoring undecodable engine output");
        ControlFlow::Continue(())
    }

    /// A dropped stream's abort: retire the request here and forward the
    /// abort to its engine. Queued, never awaited: the frame goes out in order
    /// behind whatever the engine has not read yet, while the loop keeps
    /// routing the output of every rank. The input writer reports a write
    /// that failed.
    fn forward_abort(&self, engine_id: EngineId, request_id: String) {
        self.inner.registry.lock().remove_all([&request_id]);
        self.inner.release_one(&engine_id, &request_id);
        if let Err(error) = self.inner.queue_abort(&engine_id, &request_id) {
            warn!(%error, %request_id, "failed to queue abort");
        }
    }

    /// The watchdog's tick. The engine's unexplained silence is its in-flight
    /// time since its last output, past the prefill work it was handed, over
    /// the busy periods since. Short of a window it is a request still in its
    /// first step (one registered just before this tick, say) or a prefill
    /// still within its allowance; a whole window of it is a dead engine,
    /// however many requests it was spread over and whether or not one is in
    /// flight at this tick.
    fn watchdog_verdict(&self, now: Instant) -> Verdict {
        let mut registry = self.inner.registry.lock();
        let window = registry.silence_window;
        if registry.requests.is_empty() && registry.unanswered_before < window {
            // Idle, the busy periods since the last output short of a window:
            // silence is expected, keep waiting. The next busy period re-arms
            // this tick for what is left of the window (`busy_started`), so a
            // whole one here defers nothing.
            return Verdict::Keep(now + window);
        }
        let unexplained = registry.unexplained_for(now);
        if unexplained < window {
            return Verdict::Keep(registry.next_tick(now));
        }
        self.inner.routing.lock().inflight.clear();
        warn!(
            prefill_pending_tokens = registry.prefill_pending_tokens(),
            "no engine output through {unexplained:?} of in-flight requests past \
             the prefill allowance for the prompt it was handed (the window: \
             {window:?}); treating engine as dead"
        );
        registry.fail_all(Arc::new(Error::EngineCoreDead));
        Verdict::Dead
    }

    /// The output stream ended: the output loop is gone, and with it every
    /// answer the in-flight requests wait for.
    fn stream_ended(&self) {
        self.fail_all(Error::ClientClosed {
            message: "engine output stream ended".to_string(),
        });
    }

    /// Fail every in-flight request and pending utility call with `error`,
    /// closing the registry, and forget the ranks' in-flight sets.
    fn fail_all(&self, error: Error) {
        self.inner.registry.lock().fail_all(Arc::new(error));
        self.inner.routing.lock().inflight.clear();
    }
}

/// Route decoded outputs to per-request streams and forward auto-abort requests
/// to their engines. Runs until the output channel closes or the engine dies.
async fn run_dispatcher<P: EngineProtocol>(
    mut out_rx: mpsc::Receiver<Result<EngineBatch<P::Output>>>,
    mut abort_rx: mpsc::UnboundedReceiver<(EngineId, String)>,
    inner: Arc<ClientInner<P>>,
) {
    // Positive-liveness watchdog: only output from the engine resets it. The
    // aborts this loop also handles are the client's own events and say
    // nothing about the engine; a steady trickle of them (a gateway retrying
    // into a stalled engine) must not keep a silent engine alive.
    let (window, busy_started) = {
        let registry = inner.registry.lock();
        (registry.silence_window, registry.busy_started.clone())
    };
    let watchdog = tokio::time::sleep_until(Instant::now() + window);
    tokio::pin!(watchdog);
    let mut dispatcher = Dispatcher {
        inner,
        consecutive_decode_errors: 0,
    };
    loop {
        tokio::select! {
            () = busy_started.notified() => {
                let next = dispatcher.arm_window(Instant::now());
                watchdog.as_mut().reset(next);
            }
            output = out_rx.recv() => {
                let Some(output) = output else { break };
                let next = dispatcher.output_seen(Instant::now());
                watchdog.as_mut().reset(next);
                match output {
                    Ok(batch) => {
                        let next = dispatcher.route_batch(batch);
                        watchdog.as_mut().reset(next);
                    }
                    Err(error) => match dispatcher.output_error(error) {
                        ControlFlow::Continue(()) => {}
                        ControlFlow::Break(()) => return,
                    },
                }
            }
            abort = abort_rx.recv() => {
                if let Some((engine_id, request_id)) = abort {
                    dispatcher.forward_abort(engine_id, request_id);
                }
            }
            () = &mut watchdog => {
                match dispatcher.watchdog_verdict(Instant::now()) {
                    Verdict::Keep(next) => watchdog.as_mut().reset(next),
                    Verdict::Dead => return,
                }
            }
        }
    }
    dispatcher.stream_ended();
}

/// Admit frames to zeromq's local peer queue in FIFO order. Accepted frames
/// can remain in its background writer after this task or the client stops.
/// The input-death timeout bounds stalled admission to a full peer queue;
/// the input monitor separately detects asynchronous transport failures.
async fn run_input_writer<P: EngineProtocol>(
    mut input_send: RouterSendHalf,
    mut frames: mpsc::UnboundedReceiver<InputFrame>,
    inner: Arc<ClientInner<P>>,
) {
    while let Some(frame) = frames.recv().await {
        if let Some(bytes) = frame.counted {
            inner.waiting_adds.fetch_sub(1, Ordering::AcqRel);
            inner.waiting_add_bytes.fetch_sub(bytes, Ordering::AcqRel);
        }
        let death_timeout = *inner.input_death_timeout.lock();
        *inner.writing_since.lock() = Some(Instant::now());
        let write = async {
            let bytes =
                frame.payload.len() + frame.aux_frames.iter().map(Bytes::len).sum::<usize>();
            let credit = inner.reserve_input(bytes).await;
            send_message(
                &mut input_send,
                &frame.engine_id,
                credit.wrap(frame.request_type),
                credit.wrap(frame.payload),
                frame
                    .aux_frames
                    .into_iter()
                    .map(|bytes| credit.wrap(bytes))
                    .collect(),
            )
            .await
        };
        let result = tokio::time::timeout(death_timeout, write).await;
        *inner.writing_since.lock() = None;
        let result = match result {
            Ok(result) => result,
            Err(_elapsed) => {
                warn!(
                    timeout = ?death_timeout,
                    "local zeromq admission stalled for the whole death window; \
                     treating engine as dead"
                );
                inner
                    .registry
                    .lock()
                    .fail_all(Arc::new(Error::EngineCoreDead));
                inner.routing.lock().inflight.clear();
                if let Some(done) = frame.done {
                    let _ = done.send(Err(Error::EngineCoreDead));
                }
                return;
            }
        };
        if result.is_ok() {
            // Local queue admission resumed: an episode of refusals is over.
            let refused = inner.refused.swap(0, Ordering::AcqRel);
            if refused > 0 {
                warn!(
                    refused,
                    "zeromq accepted a frame locally again; requests were refused while admission was \
                     blocked and the queue at the bound"
                );
            }
        }
        match frame.done {
            // A caller that stopped waiting has failed its request already; a
            // write that then failed is still an operator's business.
            Some(done) => {
                if let Err(Err(error)) = done.send(result) {
                    warn!(%error, "engine input write failed after its caller stopped waiting");
                }
            }
            None => {
                if let Err(error) = result {
                    warn!(%error, "failed to write a queued abort or wake");
                    if matches!(error, Error::EngineCoreDead | Error::Transport(_)) {
                        // The socket to the engine is gone, and with it the way
                        // to stop the requests it holds: fail what is in flight,
                        // as the output side does on a transport failure.
                        inner.registry.lock().fail_all(Arc::new(error));
                        inner.routing.lock().inflight.clear();
                        return;
                    }
                }
            }
        }
    }
}

/// What a submit holds for its request until it completes: the registry entry
/// and the engine's in-flight slot, with the add frame queued or delivered
/// once the send began. A submit that does not complete gives them back here
/// — explicitly on its failure paths, and from `Drop` when its caller stopped
/// polling it (the request's handler was cancelled while the frame waited for
/// the engine) — and, when the engine may hold or later read the add, aborts
/// the request on the engine so it is not served for nobody.
struct SubmitGuard<'a, P: EngineProtocol> {
    inner: &'a ClientInner<P>,
    engine_id: EngineId,
    request_id: String,
    armed: bool,
}

impl<P: EngineProtocol> SubmitGuard<'_, P> {
    /// Give the entry and the slot back; `abort` when the engine may hold or
    /// later read the add frame.
    fn take_back(&mut self, abort: bool) {
        self.armed = false;
        self.inner.registry.lock().remove_all([&self.request_id]);
        self.inner.release_one(&self.engine_id, &self.request_id);
        if abort {
            let _ = self
                .inner
                .abort_tx
                .send((self.engine_id.clone(), self.request_id.clone()));
        }
    }

    /// The submit completed: the stream takes the request over.
    fn complete(mut self) -> (EngineId, String) {
        self.armed = false;
        (self.engine_id.clone(), std::mem::take(&mut self.request_id))
    }
}

impl<P: EngineProtocol> Drop for SubmitGuard<'_, P> {
    fn drop(&mut self) {
        if self.armed {
            // Dropped at an await point, so the frame is queued or delivered.
            self.take_back(true);
        }
    }
}

/// Stream of outputs for one submitted request. Dropping it before the terminal
/// output aborts the request on the engine.
pub struct RequestStream<P: EngineProtocol> {
    request_id: String,
    engine_id: EngineId,
    receiver: OutputReceiver<P::Output>,
    abort_tx: mpsc::UnboundedSender<(EngineId, String)>,
    finished: bool,
}

impl<P: EngineProtocol> RequestStream<P> {
    /// The request id this stream tracks.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// The data-parallel rank the request went to (the engine's index, see
    /// [`EngineId::engine_index`]); `None` for an engine whose identity
    /// names no index.
    pub fn engine_index(&self) -> Option<u32> {
        self.engine_id.engine_index()
    }
}

impl<P: EngineProtocol> Stream for RequestStream<P> {
    type Item = Result<P::Output>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        if self.finished {
            return Poll::Ready(None);
        }
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(Ok(output))) => {
                if output.finished() {
                    self.finished = true;
                }
                Poll::Ready(Some(Ok(output)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.finished = true;
                Poll::Ready(Some(Err(error)))
            }
            // The sender was dropped without a terminal output — the client was
            // dropped out from under the stream, or the dispatcher stopped
            // tracking the request. Surface it as an error: a clean end here is
            // indistinguishable from a completed generation, and a truncated
            // response must not look complete to the caller.
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(Some(Err(Error::RequestStreamClosed {
                    request_id: self.request_id.clone(),
                })))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<P: EngineProtocol> Drop for RequestStream<P> {
    fn drop(&mut self) {
        if !self.finished {
            // Best-effort auto-abort; the dispatcher sends the Abort frame.
            let _ = self
                .abort_tx
                .send((self.engine_id.clone(), self.request_id.clone()));
        }
    }
}

#[cfg(test)]
mod tests;
