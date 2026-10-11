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
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::Bytes;
use futures::Stream;
use parking_lot::Mutex;
use tokio::{
    sync::{mpsc, oneshot, Notify},
    task::JoinHandle,
    time::Instant,
};
use tracing::{trace, warn};
use zeromq::RouterSendHalf;

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
    /// [`Self::unanswered_for`]).
    unanswered_before: Duration,
    /// The silence watchdog's window ([`ENGINE_SILENCE_DEATH_TIMEOUT`] unless
    /// [`Client::set_silence_death_timeout`] changed it): the dispatcher's
    /// deadline, and the idle spell after which the count starts over.
    silence_window: Duration,
    /// Signalled when `requests` goes from empty to non-empty: the dispatcher
    /// re-arms the silence watchdog for the remainder of the window the
    /// engine's unanswered work has left, not a whole one.
    busy_started: Arc<Notify>,
    /// Prompt tokens of the registered requests that have produced no output
    /// yet: the engine's prefill work in hand, as the servicer reports it.
    prefill_pending: u64,
    /// Registered requests the engine reports on every step it decodes: the
    /// silence [`Client::output_silence`] counts is theirs. A request the
    /// engine reports at intervals is quiet while healthy and counts none.
    every_step: usize,
}

/// One registered request: its output stream, whether the engine reports it
/// on every step ([`EngineProtocol::reported_every_step`]) and, until its
/// first output, the prompt the engine still has to prefill for it.
struct InFlight<O> {
    sender: OutputSender<O>,
    prompt_tokens: u64,
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
            busy_started: Arc::new(Notify::new()),
            prefill_pending: 0,
            every_step: 0,
        }
    }
}

impl<O: EngineOutput> RequestRegistry<O> {
    /// Register a new request of `prompt_tokens`, reported by the engine on
    /// every step or at intervals (`every_step`), returning the receiver for
    /// its output stream.
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
        if self.requests.is_empty() {
            let now = Instant::now();
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
                started: false,
                every_step,
            },
        );
        self.prefill_pending = self.prefill_pending.saturating_add(prompt_tokens);
        if every_step {
            self.every_step += 1;
        }
        Ok(receiver)
    }

    /// The request's prompt is prefilled (its first output came, or it is
    /// gone before one): it no longer counts as pending prefill.
    fn prefilled(&mut self, entry: &mut InFlight<O>) {
        if !entry.started {
            entry.started = true;
            self.prefill_pending = self.prefill_pending.saturating_sub(entry.prompt_tokens);
        }
    }

    /// Drop a request's entry, settling its pending prefill and its place in
    /// the silence count. The last one ends the busy period, whose in-flight
    /// time counts as unanswered until the engine's next output.
    fn retire(&mut self, request_id: &str) -> Option<InFlight<O>> {
        let mut entry = self.requests.remove(request_id)?;
        self.prefilled(&mut entry);
        if entry.every_step {
            self.every_step = self.every_step.saturating_sub(1);
        }
        if self.requests.is_empty() {
            let now = Instant::now();
            if let Some(since) = self.in_flight_since.take() {
                self.unanswered_before += now.saturating_duration_since(since);
            }
            self.idle_since = Some(now);
        }
        Some(entry)
    }

    /// The engine answered (an output of any kind): nothing in flight is
    /// unanswered any more, and the busy period, if one is on, counts from
    /// `now`.
    fn output_seen(&mut self, now: Instant) {
        self.unanswered_before = Duration::ZERO;
        if self.in_flight_since.is_some() {
            self.in_flight_since = Some(now);
        }
    }

    /// How long the engine has had requests in flight since its last output:
    /// the busy period it is in and the ones that ended since, unless a whole
    /// window idle separated them from it. The silence watchdog's measure of
    /// the engine's silence: a request registered just before the watchdog's
    /// tick has a whole window to answer, while a core that reads every
    /// request and answers none is given up on even when every request is
    /// abandoned before the window.
    fn unanswered_for(&self, now: Instant) -> Duration {
        self.unanswered_before
            + self
                .in_flight_since
                .map_or(Duration::ZERO, |since| now.saturating_duration_since(since))
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
            // The first output: the prompt is prefilled.
            entry.started = true;
            self.prefill_pending = self.prefill_pending.saturating_sub(entry.prompt_tokens);
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
#[derive(Clone, Copy)]
struct InputQueueLimit {
    frames: usize,
    bytes: usize,
}

struct ClientInner<P: EngineProtocol> {
    /// Frames for the shared input ROUTER socket, in submission order. The
    /// input writer ([`run_input_writer`]) owns the socket and writes them one
    /// at a time, so a caller that stopped waiting never cancels a write.
    input_tx: mpsc::UnboundedSender<InputFrame>,
    /// How long a caller waits for the engine to accept one frame:
    /// [`ENGINE_SEND_TIMEOUT`] unless [`Client::set_send_timeout`] changed it.
    send_timeout: Mutex<Duration>,
    /// How long one frame may stay unaccepted before the engine is dead:
    /// [`ENGINE_INPUT_DEATH_TIMEOUT`] unless [`Client::set_input_death_timeout`]
    /// changed it.
    input_death_timeout: Mutex<Duration>,
    /// The bound on waiting add-requests: [`ENGINE_INPUT_QUEUE_FRAMES`] and
    /// [`ENGINE_INPUT_QUEUE_BYTES`] unless [`Client::set_input_queue_limit`]
    /// changed it.
    input_queue_limit: Mutex<InputQueueLimit>,
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

    /// Send an encoded request frame to one engine over the shared input socket.
    ///
    /// The frame joins the input queue at once, in submission order, and the
    /// caller waits at most the send timeout ([`ENGINE_SEND_TIMEOUT`] by
    /// default) for the engine to accept it. A healthy engine drains its
    /// input continuously, so a frame not accepted in that window is behind
    /// input the engine is not reading (a frozen or stalled core: the socket
    /// buffer absorbs small frames meanwhile, but not a large prompt), and
    /// the caller's request fails with [`Error::EngineInputBlocked`]. That is
    /// the failure of one request, not of the client: the requests the engine
    /// already holds keep their streams, and the client closes only on the
    /// signs of the engine's death the output side sees, or once a frame has
    /// stayed unaccepted for the whole of [`ENGINE_INPUT_DEATH_TIMEOUT`].
    ///
    /// The write is the input writer's, not the caller's, so a caller that
    /// stopped waiting never cancels a write half-way through the socket's
    /// buffer: the frame stays queued, whole and in order, and is delivered
    /// when the engine reads again. A caller that gave up on an add-request
    /// therefore aborts it on the engine as well (see
    /// [`Client::submit_with_aux`]).
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
    /// engine accepts frames; once a write has been pending for a whole send
    /// window (the input is blocked, and the writer's frame is the one the
    /// engine is not reading), a request that would take the waiting
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

    /// Queue a frame and wait for the engine to accept it; `counted` is the
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
    /// the client is dropped.
    pub fn new(transport: ConnectedTransport) -> Self {
        let ConnectedTransport {
            engines,
            input_send,
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
        #[expect(
            clippy::disallowed_methods,
            reason = "background tasks are aborted on client drop"
        )]
        let input_task = tokio::spawn(run_input_writer::<P>(input_send, input_rx, inner.clone()));
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
    /// closing, or once the engine has left a frame unaccepted on its input for
    /// [`ENGINE_INPUT_DEATH_TIMEOUT`]. No RPC — this is a local liveness flag.
    pub fn is_alive(&self) -> bool {
        !self.inner.registry.lock().closed
    }

    /// Bound how long a submit, abort or utility call waits for the engine to
    /// accept its frame on the input socket ([`ENGINE_SEND_TIMEOUT`] by
    /// default). Past it the call fails with [`Error::EngineInputBlocked`];
    /// the frame is still delivered when the engine reads again.
    pub fn set_send_timeout(&self, timeout: Duration) {
        *self.inner.send_timeout.lock() = timeout;
    }

    /// Bound how long one frame may stay unaccepted on the input socket before
    /// the engine is treated as dead ([`ENGINE_INPUT_DEATH_TIMEOUT`] by
    /// default): the requests in flight fail and the client closes.
    pub fn set_input_death_timeout(&self, timeout: Duration) {
        *self.inner.input_death_timeout.lock() = timeout;
    }

    /// Bound how long the engine may have requests in flight, in total since
    /// its last output, before it is treated as dead
    /// ([`ENGINE_SILENCE_DEATH_TIMEOUT`] by default).
    pub fn set_silence_death_timeout(&self, timeout: Duration) {
        self.inner.registry.lock().silence_window = timeout;
    }

    /// Bound the add-requests that may wait behind a blocked input before new
    /// ones are refused at once with [`Error::EngineInputFull`]
    /// ([`ENGINE_INPUT_QUEUE_FRAMES`] frames and [`ENGINE_INPUT_QUEUE_BYTES`]
    /// payload bytes by default).
    pub fn set_input_queue_limit(&self, frames: usize, bytes: usize) {
        *self.inner.input_queue_limit.lock() = InputQueueLimit { frames, bytes };
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

    /// Prompt tokens of the in-flight requests that have produced no output
    /// yet: the engine's prefill work in hand, which the servicer puts on its
    /// load record (`EngineLoad.prefill_pending_tokens`) so a gateway can
    /// allow for it when it judges a token-batch wire's silence, whatever it
    /// has in flight on the engine itself.
    pub fn prefill_pending_tokens(&self) -> u64 {
        self.inner.registry.lock().prefill_pending_tokens()
    }

    /// Submit a request and return a stream of its outputs. The request is
    /// routed by `data_parallel_rank` (SMG-pinned) or, unpinned, to the
    /// least-loaded engine (see [`ClientInner::select_engine`]).
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

/// How long a caller waits for the engine to accept one request frame on the
/// input socket. A healthy engine drains its input continuously; a frame not
/// accepted in this window is behind input the engine is not reading, and the
/// caller's request fails with [`Error::EngineInputBlocked`] while the frame
/// stays queued for the engine's return (see [`ClientInner::send_to_engine`]).
/// The client itself stays up: its death is decided on the output side.
const ENGINE_SEND_TIMEOUT: Duration = Duration::from_secs(10);

/// If the engine has had requests in flight for this long, in total, since its
/// last output (one request's whole window, or a run of shorter ones it read
/// and never answered, with no window-long idle spell between them), the
/// dispatcher treats it as dead. A hard engine death (SIGKILL/OOM) sends no
/// `ENGINE_CORE_DEAD` sentinel, and a bound PULL socket never errors when its
/// PUSH peer vanishes — so silence is the only available death signal. Generous
/// so a slow first token never trips it, while still bounding an otherwise
/// infinite hang.
///
/// The window is deliberately fixed rather than derived from context length:
/// any single request whose first token takes longer than this — a lone
/// million-token prefill on an otherwise idle transport — is failed as if the
/// engine had died. Deployments that can prefill for minutes with no other
/// traffic need this raised (per-worker configuration), not the watchdog
/// removed: without it a killed engine hangs its requests forever.
const ENGINE_SILENCE_DEATH_TIMEOUT: Duration = Duration::from_secs(300);

/// How long one frame may stay unaccepted on the engine's input before the
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
/// engine does not read). Sized for a stall, not for throughput: the socket
/// buffer absorbs about 200 KiB, a long text prompt is under a MiB, and the
/// bound never applies while the engine reads, so a burst to a healthy engine,
/// which takes frames in microseconds, is never refused. Aborts and other
/// control frames are never bounded.
const ENGINE_INPUT_QUEUE_FRAMES: usize = 256;
const ENGINE_INPUT_QUEUE_BYTES: usize = 32 << 20;

/// Consecutive undecodable output messages tolerated before the client is
/// declared dead. One corrupt message is survivable, but an engine whose wire
/// schema this client cannot read never recovers: every dropped batch strands
/// the requests it carried, and the silence watchdog never fires because other
/// traffic keeps arriving. Failing those requests is strictly better than
/// hanging them.
const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 3;

/// Route decoded outputs to per-request streams and forward auto-abort requests
/// to their engines. Runs until the output channel closes or the engine dies.
async fn run_dispatcher<P: EngineProtocol>(
    mut out_rx: mpsc::Receiver<Result<EngineBatch<P::Output>>>,
    mut abort_rx: mpsc::UnboundedReceiver<(EngineId, String)>,
    inner: Arc<ClientInner<P>>,
) {
    let mut consecutive_decode_errors = 0;
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
    loop {
        tokio::select! {
            () = busy_started.notified() => {
                // A busy period began. The engine has what is left of a window
                // of unanswered work to answer in, not a whole one: a tick that
                // landed in the gap before this period waited a whole window,
                // and the sum of the periods before it must not be waited out
                // again.
                let now = Instant::now();
                let (unanswered, window) = {
                    let registry = inner.registry.lock();
                    (registry.unanswered_for(now), registry.silence_window)
                };
                watchdog.as_mut().reset(now + window.saturating_sub(unanswered));
            }
            output = out_rx.recv() => {
                if output.is_some() {
                    let now = Instant::now();
                    let mut registry = inner.registry.lock();
                    registry.output_seen(now);
                    watchdog.as_mut().reset(now + registry.silence_window);
                }
                match output {
                    Some(Ok(batch)) => {
                        consecutive_decode_errors = 0;
                        inner.note_output();
                        if batch.outputs.is_empty()
                            && batch.finished_request_ids.is_empty()
                            && batch.load.is_some()
                        {
                            inner.steps_reported.store(true, Ordering::Relaxed);
                        }
                        if let Some(load) = batch.load {
                            inner.routing.lock().load.insert(batch.engine_index, load);
                        }
                        if let Some(event) = batch.wave {
                            inner.observe_wave(event, batch.engine_index);
                        }
                        // Unique finished ids: a terminal output and the
                        // out-of-band finished list may name the same request.
                        let mut finished: HashSet<String> =
                            batch.finished_request_ids.into_iter().collect();
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
                        drop(registry);
                        inner.release(batch.engine_index, &finished);
                    }
                    Some(Err(error)) => {
                        if matches!(error, Error::EngineCoreDead | Error::Transport(_)) {
                            warn!(%error, "engine transport failed; failing all in-flight requests");
                            inner.registry.lock().fail_all(Arc::new(error));
                            inner.routing.lock().inflight.clear();
                            return;
                        }
                        // A single decode error is non-fatal; a run of them means
                        // this client cannot read the engine, so fail rather than
                        // silently drop every output the requests are waiting for.
                        consecutive_decode_errors += 1;
                        if consecutive_decode_errors >= MAX_CONSECUTIVE_DECODE_ERRORS {
                            warn!(
                                %error,
                                consecutive_decode_errors,
                                "engine output undecodable repeatedly; failing all in-flight requests"
                            );
                            inner.registry.lock().fail_all(Arc::new(error));
                            inner.routing.lock().inflight.clear();
                            return;
                        }
                        warn!(%error, "ignoring undecodable engine output");
                    }
                    None => break,
                }
            }
            abort = abort_rx.recv() => {
                if let Some((engine_id, request_id)) = abort {
                    inner.registry.lock().remove_all([&request_id]);
                    inner.release_one(&engine_id, &request_id);
                    // Queued, never awaited: the frame goes out in order behind
                    // whatever the engine has not read yet, while this loop
                    // keeps routing the output of every rank. The input writer
                    // reports a write that failed.
                    if let Err(error) = inner.queue_abort(&engine_id, &request_id) {
                        warn!(%error, %request_id, "failed to queue abort");
                    }
                }
            }
            () = &mut watchdog => {
                let now = Instant::now();
                let mut registry = inner.registry.lock();
                let window = registry.silence_window;
                if registry.requests.is_empty() && registry.unanswered_before < window {
                    // Idle, the busy periods since the last output short of a
                    // window: silence is expected, keep waiting. The next busy
                    // period re-arms this tick for what is left of the window
                    // (`busy_started`), so a whole one here defers nothing.
                    drop(registry);
                    watchdog.as_mut().reset(now + window);
                    continue;
                }
                // The engine's unanswered work: its in-flight time since its
                // last output, over the busy periods since. Short of a window
                // it is a request still in its first step (one registered just
                // before this tick, say); a whole window of it is a dead
                // engine, however many requests it was spread over and whether
                // or not one is in flight at this tick.
                let unanswered = registry.unanswered_for(now);
                if unanswered < window {
                    drop(registry);
                    watchdog.as_mut().reset(now + (window - unanswered));
                    continue;
                }
                inner.routing.lock().inflight.clear();
                warn!(
                    "no engine output through {window:?} of in-flight requests; \
                     treating engine as dead"
                );
                registry.fail_all(Arc::new(Error::EngineCoreDead));
                return;
            }
        }
    }
    inner
        .registry
        .lock()
        .fail_all(Arc::new(Error::ClientClosed {
            message: "engine output stream ended".to_string(),
        }));
    inner.routing.lock().inflight.clear();
}

/// Write queued frames to the engines' shared input socket, in order, for as
/// long as the client lives. One frame at a time: a frame the engine does not
/// read blocks the socket once its buffer is full, and every frame behind it
/// waits, whole and in order, for the engine to read again. A frame unaccepted
/// for the input death timeout is a hung engine, not a pausing one: the
/// requests in flight fail, the client closes, and the writer stops (the
/// cancelled write left the socket mid-frame; the frames still queued fail
/// with the client).
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
        let write = send_message(
            &mut input_send,
            &frame.engine_id,
            frame.request_type,
            frame.payload,
            frame.aux_frames,
        );
        let result = tokio::time::timeout(death_timeout, write).await;
        *inner.writing_since.lock() = None;
        let result = match result {
            Ok(result) => result,
            Err(_elapsed) => {
                warn!(
                    timeout = ?death_timeout,
                    "engine has not accepted a request frame for the whole death window; \
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
            // The engine reads again: an episode of refusals, if any, is over.
            let refused = inner.refused.swap(0, Ordering::AcqRel);
            if refused > 0 {
                warn!(
                    refused,
                    "engine accepted a frame again; requests were refused while its input was \
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
mod tests {
    use std::time::Duration;

    use futures::StreamExt;

    use super::*;
    use crate::{
        codec::{decode_msgpack, encode_msgpack},
        mock_engine::{
            connect_to_frontend, default_ready_response, EngineInbound, IpcNamespace, MockEngine,
        },
        protocol::{
            handshake::EngineCoreReadyResponse,
            vllm::{
                output::{
                    DpControlMessage, DpControlOutput, EngineCoreFinishReason, EngineCoreOutput,
                    EngineCoreOutputs, RequestBatchOutputs,
                },
                request::{EngineCoreRequest, UtilityCall},
                stats::SchedulerStats,
            },
        },
        transport::{connect_handshake, ENGINE_CORE_DEAD_SENTINEL},
    };

    const TIMEOUT: Duration = Duration::from_secs(10);

    /// The returned [`IpcNamespace`] owns the socket tempdir; hold it for the
    /// test duration so the ipc files outlive the client and are cleaned up.
    async fn connect() -> (EngineCoreClient, MockEngine, IpcNamespace) {
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        let (transport, engine) = tokio::join!(
            connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        (
            EngineCoreClient::new(transport.unwrap()),
            engine.unwrap(),
            ns,
        )
    }

    /// Connect `engine_count` mock ranks behind one transport. `lockstep`
    /// controls the data-parallel size they report at handshake, which is how
    /// the client tells a group that pauses as a unit (vLLM MoE DP) from ranks
    /// that step independently.
    async fn connect_ranks(
        engine_count: usize,
        lockstep: bool,
    ) -> (EngineCoreClient, Vec<MockEngine>, IpcNamespace) {
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        let ready = |rank: u32| EngineCoreReadyResponse {
            data_parallel_size: if lockstep { engine_count as u64 } else { 1 },
            data_parallel_rank: rank,
            ..default_ready_response()
        };
        let engines = (0..engine_count as u32).map(|rank| {
            connect_to_frontend(&handshake, EngineId::from_engine_index(rank), ready(rank))
        });
        let (transport, engines) = tokio::join!(
            connect_handshake(&handshake, engine_count, &input, &output, TIMEOUT),
            futures::future::join_all(engines),
        );
        let engines = engines.into_iter().map(Result::unwrap).collect();
        (EngineCoreClient::new(transport.unwrap()), engines, ns)
    }

    /// An add-request pinned to one rank.
    fn request_for(request_id: &str, rank: u32) -> EngineCoreRequest {
        EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            data_parallel_rank: Some(rank),
            ..EngineCoreRequest::default()
        }
    }

    fn wave_control(engine_index: u32, control: DpControlMessage) -> Vec<Bytes> {
        let outputs = EngineCoreOutputs::DpControl(DpControlOutput {
            engine_index,
            timestamp: 0.0,
            control,
        });
        vec![Bytes::from(encode_msgpack(&outputs).unwrap())]
    }

    fn batch(engine_index: u32, output: EngineCoreOutput) -> Vec<Bytes> {
        let finished = output.finish_reason.map(|_| {
            let mut set = std::collections::BTreeSet::new();
            set.insert(output.request_id.clone());
            set
        });
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index,
            outputs: vec![output],
            finished_requests: finished,
            ..RequestBatchOutputs::default()
        });
        vec![Bytes::from(encode_msgpack(&outputs).unwrap())]
    }

    #[tokio::test]
    async fn submit_streams_tokens_until_finish() {
        let (client, mut engine, _ns) = connect().await;

        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let mut stream = client.submit(request).await.unwrap();

        // Engine receives the Add and streams two chunks then a terminal output.
        let frames = engine.recv_request().await.unwrap();
        assert_eq!(frames[0].as_ref(), b"\x00");
        let received: EngineCoreRequest = decode_msgpack(frames[1].as_ref()).unwrap();
        assert_eq!(received.request_id, "req-1");

        engine
            .send_output(batch(
                0,
                EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![10],
                    ..Default::default()
                },
            ))
            .await
            .unwrap();
        engine
            .send_output(batch(
                0,
                EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![11],
                    finish_reason: Some(EngineCoreFinishReason::Stop),
                    ..Default::default()
                },
            ))
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.new_token_ids, vec![10]);
        assert!(!first.finished());
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.new_token_ids, vec![11]);
        assert!(second.finished());
        // Terminal output ends the stream.
        assert!(stream.next().await.is_none());
    }

    /// The silence the servicer reports for a frozen engine: nothing while
    /// idle, counted from the dispatch, reset by any output batch (a
    /// stats-only step included), gone with the last in-flight request.
    #[tokio::test]
    async fn output_silence_counts_from_the_dispatch_and_ends_with_the_requests() {
        let (client, mut engine, _ns) = connect().await;
        assert_eq!(client.output_silence(), None, "nothing in flight");
        let mut stream = client.submit(request_for("req-1", 0)).await.unwrap();
        engine.recv_request().await.unwrap();
        let at_dispatch = client.output_silence().expect("a request is in flight");
        assert!(at_dispatch < Duration::from_millis(500), "{at_dispatch:?}");
        assert!(
            !client.reports_steps(),
            "no step without a token seen yet: a token-batch wire until shown otherwise"
        );
        assert_eq!(
            client.prefill_pending_tokens(),
            3,
            "the request's three prompt tokens are the engine's prefill in hand"
        );
        tokio::time::sleep(Duration::from_millis(80)).await;
        let grown = client.output_silence().expect("still in flight");
        assert!(
            grown >= Duration::from_millis(70),
            "grows while the engine is silent: {grown:?}"
        );

        // A stats-only step is output: the engine is alive, if quiet.
        let stats_only = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            scheduler_stats: Some(Box::new(SchedulerStats {
                num_running_reqs: 1,
                ..Default::default()
            })),
            ..Default::default()
        });
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&stats_only).unwrap())])
            .await
            .unwrap();
        let deadline = Instant::now() + TIMEOUT;
        while client.engine_load(0).is_none() {
            assert!(Instant::now() < deadline, "stats never arrived");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let after_stats = client.output_silence().expect("still in flight");
        assert!(
            after_stats < grown,
            "reset by the step: {after_stats:?} after {grown:?}"
        );
        assert!(
            client.reports_steps(),
            "a stats-only step shows the wire reports every step"
        );
        assert_eq!(
            client.prefill_pending_tokens(),
            3,
            "a step without an output for the request leaves its prompt pending"
        );

        // The terminal output ends the request: nothing in flight, no silence.
        engine
            .send_output(batch(
                0,
                EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![11],
                    finish_reason: Some(EngineCoreFinishReason::Stop),
                    ..Default::default()
                },
            ))
            .await
            .unwrap();
        let last = stream.next().await.unwrap().unwrap();
        assert!(last.finished());
        assert!(stream.next().await.is_none());
        assert_eq!(client.output_silence(), None, "nothing in flight again");
        assert_eq!(
            client.prefill_pending_tokens(),
            0,
            "its first output settled the prompt"
        );
    }

    /// A stream that outlives its client is truncated, not complete: it must
    /// report the truncation rather than end like a finished generation.
    #[tokio::test]
    async fn dropping_the_client_errors_live_streams() {
        let (client, mut engine, _ns) = connect().await;
        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "orphan".into(),
                prompt_token_ids: Some(vec![1, 2, 3]),
                ..EngineCoreRequest::default()
            })
            .await
            .unwrap();
        engine.recv_request().await.unwrap();

        drop(client);

        let error = stream
            .next()
            .await
            .expect("a terminal error item")
            .expect_err("truncation is not a clean end of stream");
        assert!(
            matches!(&error, Error::RequestStreamClosed { request_id } if request_id == "orphan"),
            "unexpected error: {error}"
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn scheduler_stats_surface_as_load_signal() {
        let (client, mut engine, _ns) = connect().await;
        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "r".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        engine.recv_request().await.unwrap();

        // Stats ride a mid-stream chunk: the request is still in flight, so
        // the snapshot is current and must be stored verbatim. (A terminal
        // batch's snapshot predates its own finish; once the rank empties,
        // the queue counts are zeroed — covered separately.)
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r".into(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            scheduler_stats: Some(Box::new(SchedulerStats {
                num_running_reqs: 3,
                num_waiting_reqs: 5,
                ..Default::default()
            })),
            ..Default::default()
        });
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
            .await
            .unwrap();

        let chunk = stream.next().await.expect("mid-stream chunk").unwrap();
        assert!(!chunk.finished());
        let load = client.engine_load(0).expect("load recorded");
        assert_eq!(load.num_running, 3);
        assert_eq!(load.num_waiting, 5);
    }

    #[tokio::test]
    async fn unpinned_requests_prefer_the_least_loaded_engine() {
        let (client, mut engines, _ns) = connect_ranks(2, false).await;

        // Make engine 0 report load via a pinned warm-up request that stays
        // in flight (a finished-and-emptied rank has its queue counts zeroed,
        // because a terminal batch's snapshot predates its own finish);
        // engine 1 never reports and therefore scores zero.
        let mut stream = client.submit(request_for("warm", 0)).await.unwrap();
        engines[0].recv_request().await.unwrap();
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "warm".into(),
                new_token_ids: vec![7],
                ..Default::default()
            }],
            scheduler_stats: Some(Box::new(SchedulerStats {
                num_running_reqs: 3,
                num_waiting_reqs: 5,
                ..Default::default()
            })),
            ..Default::default()
        });
        engines[0]
            .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
            .await
            .unwrap();
        let chunk = stream.next().await.expect("warm chunk").unwrap();
        assert!(!chunk.finished());
        assert!(client.engine_load(0).is_some(), "warm-up load recorded");

        // An unpinned request must now land on the idle engine 1, not
        // engines.first().
        let _stream = client
            .submit(EngineCoreRequest {
                request_id: "cold".into(),
                prompt_token_ids: Some(vec![1, 2, 3]),
                ..EngineCoreRequest::default()
            })
            .await
            .unwrap();
        let inbound = tokio::time::timeout(TIMEOUT, engines[1].recv())
            .await
            .expect("unpinned request should route to the least-loaded engine")
            .unwrap();
        match inbound {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "cold"),
            other => panic!("expected Add on engine 1, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_cold_burst_spreads_across_engines() {
        // No engine has reported load yet: the in-flight floor plus the
        // rotating scan start must spread a burst instead of dogpiling the
        // first engine (which is what a pure snapshot-min would do).
        let (client, mut engines, _ns) = connect_ranks(2, false).await;
        let unpinned = |id: &str| EngineCoreRequest {
            request_id: id.into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let _first = client.submit(unpinned("burst-0")).await.unwrap();
        let _second = client.submit(unpinned("burst-1")).await.unwrap();

        for engine in &mut engines {
            let inbound = tokio::time::timeout(TIMEOUT, engine.recv())
                .await
                .expect("each engine should receive exactly one of the burst")
                .unwrap();
            match inbound {
                EngineInbound::Add(request) => {
                    assert!(request.request_id.starts_with("burst-"));
                }
                other => panic!("expected Add, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn failed_submit_releases_its_reservation() {
        // A rejected submit (duplicate request id) must release the in-flight
        // reservation selection took, or the failed attempt permanently
        // penalizes that engine's score.
        let (client, mut engines, _ns) = connect_ranks(2, false).await;
        let unpinned = |id: &str| EngineCoreRequest {
            request_id: id.into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let _held = client.submit(unpinned("dup")).await.unwrap();
        engines[0].recv().await.unwrap();
        // Second submit with the same id fails at registration; its
        // reservation must roll back, leaving the counts (1, 0).
        assert!(client.submit(unpinned("dup")).await.is_err());

        // With counts (1, 0) the next request must land on engine 1. If the
        // failed attempt leaked its reservation the counts would read (1, 1)
        // and rotation could send this to engine 0.
        let _next = client.submit(unpinned("after")).await.unwrap();
        match tokio::time::timeout(TIMEOUT, engines[1].recv())
            .await
            .expect("request should route to the unloaded engine")
            .unwrap()
        {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "after"),
            other => panic!("expected Add on engine 1, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn retiring_one_request_twice_frees_only_its_own_slot() {
        // A request has several racing ways to finish: the explicit abort API,
        // the dropped stream's auto-abort, and the engine's own finished
        // report. Counting each one would retire slots the request never held
        // and leave the rank looking emptier than it is; slots are keyed by
        // request id so only the first retirement lands.
        let (client, mut engines, _ns) = connect_ranks(2, false).await;
        let pinned = |id: &str| EngineCoreRequest {
            request_id: id.into(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            data_parallel_rank: Some(0),
            ..EngineCoreRequest::default()
        };
        let doomed = client.submit(pinned("doomed")).await.unwrap();
        let _held = client.submit(pinned("held")).await.unwrap();
        engines[0].recv().await.unwrap();
        engines[0].recv().await.unwrap();

        // Abort retires "doomed"; dropping its unfinished stream sends the
        // auto-abort for the same id behind it.
        client
            .abort(&engines_id(&client, 0), "doomed")
            .await
            .unwrap();
        drop(doomed);
        for _ in 0..2 {
            tokio::time::timeout(TIMEOUT, engines[0].recv())
                .await
                .expect("both aborts should reach the engine")
                .unwrap();
        }

        // Engine 0 still holds "held". An unpinned request must therefore go
        // to the idle engine 1 — it would go to 0 if the second retirement had
        // freed "held"'s slot too.
        let _next = client
            .submit(EngineCoreRequest {
                request_id: "next".into(),
                prompt_token_ids: Some(vec![1, 2, 3]),
                ..EngineCoreRequest::default()
            })
            .await
            .unwrap();
        match tokio::time::timeout(TIMEOUT, engines[1].recv())
            .await
            .expect("request should route to the idle engine")
            .unwrap()
        {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "next"),
            other => panic!("expected Add on engine 1, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_emptied_rank_sheds_its_stale_queue_counts() {
        // Engines report load only on output batches, and a terminal batch's
        // snapshot is sampled before its own finish commits — so the last
        // thing an idle rank ever says is "still busy". The client routed
        // every request: once a rank's in-flight set empties, its stored
        // queue counts must be zeroed or the rank is shunned forever.
        let (client, mut engines, _ns) = connect_ranks(2, false).await;

        // Rank 0 runs one request to completion; its terminal batch carries
        // the stale pre-commit snapshot (3 running, 5 waiting).
        let mut stream = client.submit(request_for("last", 0)).await.unwrap();
        engines[0].recv_request().await.unwrap();
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "last".into(),
                new_token_ids: vec![7],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            }],
            scheduler_stats: Some(Box::new(SchedulerStats {
                num_running_reqs: 3,
                num_waiting_reqs: 5,
                ..Default::default()
            })),
            finished_requests: Some(std::collections::BTreeSet::from(["last".to_string()])),
            ..Default::default()
        });
        engines[0]
            .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let load = client.engine_load(0).expect("snapshot stored");
        assert_eq!((load.num_running, load.num_waiting), (0, 0));

        // Rank 1 holds one live request. The next unpinned request must go to
        // the genuinely idle rank 0 — trusting the stale snapshot would send
        // it behind rank 1's real work.
        let _held = client.submit(request_for("held", 1)).await.unwrap();
        engines[1].recv().await.unwrap();
        let _next = client
            .submit(EngineCoreRequest {
                request_id: "next".into(),
                prompt_token_ids: Some(vec![1, 2, 3]),
                ..EngineCoreRequest::default()
            })
            .await
            .unwrap();
        match tokio::time::timeout(TIMEOUT, engines[0].recv())
            .await
            .expect("request should route to the emptied rank")
            .unwrap()
        {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "next"),
            other => panic!("expected Add on engine 0, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn selection_and_release_race_without_deadlocking() {
        // Selection reads the reported load and the in-flight sets together;
        // release retires an id and, when that empties the rank, zeroes its
        // queue counts. While those maps had a mutex each, the two paths took
        // them in opposite orders, so a submit racing the release of a rank's
        // last request wedged both threads forever (parking_lot: no timeout,
        // no poisoning). Under one lock the inversion is unexpressible; this
        // stress test trips its deadline against the two-mutex version.
        const THREADS: usize = 8;
        const ITERATIONS: usize = 2_000;

        let (client, _engines, _ns) = connect_ranks(4, false).await;
        let inner = client.inner.clone();
        let completed = Arc::new(AtomicUsize::new(0));

        // Plain threads, not tasks: the racing methods are synchronous, and a
        // regression must not be able to wedge the runtime this test is
        // waiting on.
        let workers: Vec<_> = (0..THREADS)
            .map(|thread| {
                let inner = inner.clone();
                let completed = completed.clone();
                std::thread::spawn(move || {
                    for iteration in 0..ITERATIONS {
                        let request_id = format!("race-{thread}-{iteration}");
                        let engine_id = inner.select_engine(None, &request_id).unwrap();
                        inner.release_one(&engine_id, &request_id);
                    }
                    completed.fetch_add(1, Ordering::Release);
                })
            })
            .collect();

        let deadline = Instant::now() + TIMEOUT;
        while completed.load(Ordering::Acquire) < THREADS {
            assert!(
                Instant::now() < deadline,
                "selection racing release deadlocked"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for worker in workers {
            worker.join().unwrap();
        }

        // Every reservation was released, so no rank carries a phantom floor.
        let routing = inner.routing.lock();
        assert!(routing.inflight.values().all(HashSet::is_empty));
    }

    /// The `EngineId` of one connected rank, by index.
    fn engines_id(client: &EngineCoreClient, engine_index: u32) -> EngineId {
        client
            .engines()
            .iter()
            .find(|engine| engine.engine_id.engine_index() == Some(engine_index))
            .expect("connected rank")
            .engine_id
            .clone()
    }

    #[tokio::test]
    async fn dropping_stream_sends_abort() {
        let (client, mut engine, _ns) = connect().await;
        let stream = client
            .submit(EngineCoreRequest {
                request_id: "req-abort".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        // Consume the Add frame.
        let add = engine.recv_request().await.unwrap();
        assert_eq!(add[0].as_ref(), b"\x00");

        drop(stream); // not finished -> auto-abort

        let abort = engine.recv_request().await.unwrap();
        assert_eq!(abort[0].as_ref(), b"\x01"); // Abort
        let ids: Vec<String> = decode_msgpack(abort[1].as_ref()).unwrap();
        assert_eq!(ids, vec!["req-abort".to_string()]);
    }

    #[tokio::test]
    async fn engine_dead_fails_inflight_requests() {
        let (client, mut engine, _ns) = connect().await;
        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "r".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        engine.recv_request().await.unwrap();

        engine
            .send_output(vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
            .await
            .unwrap();

        let result = stream.next().await.expect("an error item");
        assert!(matches!(result, Err(Error::Shared(_))));
    }

    #[tokio::test]
    async fn persistent_decode_errors_fail_inflight_requests() {
        let (client, mut engine, _ns) = connect().await;
        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "r".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        engine.recv_request().await.unwrap();

        // One bad message is tolerated; a run of them is a schema mismatch the
        // client cannot recover from, so the request fails instead of hanging.
        for _ in 0..MAX_CONSECUTIVE_DECODE_ERRORS {
            engine
                .send_output(vec![Bytes::from_static(b"\xc1")])
                .await
                .unwrap();
        }

        let result = stream.next().await.expect("an error item");
        assert!(matches!(result, Err(Error::Shared(_))), "{result:?}");
    }

    /// The generic connector drives the TokenSpeed protocol over the same shared
    /// transport: it frames a tagged `TokenizedGenerateReqInput` and streams
    /// `BatchTokenIDOutSlim` batches back until a finish reason is seen.
    #[tokio::test]
    async fn tokenspeed_client_submits_and_streams() {
        use crate::protocol::tokenspeed::{
            output::BatchTokenIDOutSlim,
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            sampling::SamplingParams,
        };

        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        // Held for the test duration so the ipc files outlive the client.
        let _ns = ns;
        let (transport, engine) = tokio::join!(
            connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = TokenSpeedClient::new(transport.unwrap());
        let mut engine = engine.unwrap();

        let request = TokenizedGenerateReqInput {
            rid: "ts-1".to_string(),
            input_ids: vec![1, 2, 3],
            sampling_params: SamplingParams {
                max_new_tokens: Some(4),
                ..SamplingParams::default()
            },
            stream: true,
            ..TokenizedGenerateReqInput::default()
        };
        let mut stream = client.submit(request).await.unwrap();

        // Engine sees the Add frame + the positional-array payload.
        let frames = engine.recv_request().await.unwrap();
        assert_eq!(
            TokenSpeedRequestType::from_frame(frames[0].as_ref()),
            Some(TokenSpeedRequestType::Add)
        );
        let received: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
        assert_eq!(received.rid, "ts-1");
        assert_eq!(received.input_ids, vec![1, 2, 3]);

        let chunk = BatchTokenIDOutSlim {
            rids: vec!["ts-1".into()],
            output_ids: vec![vec![10]],
            finished_reasons: vec![String::new()],
            prompt_tokens: vec![3],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            ..Default::default()
        };
        let done = BatchTokenIDOutSlim {
            rids: vec!["ts-1".into()],
            output_ids: vec![vec![11]],
            finished_reasons: vec!["stop".into()],
            prompt_tokens: vec![3],
            completion_tokens: vec![2],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            ..Default::default()
        };
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&chunk).unwrap())])
            .await
            .unwrap();
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&done).unwrap())])
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.output_ids, vec![10]);
        assert!(!first.finished());
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.output_ids, vec![11]);
        assert_eq!(second.finish_reason.as_deref(), Some("stop"));
        assert!(second.finished());
        // Terminal output ends the stream.
        assert!(stream.next().await.is_none());
    }

    /// A launcher that knows the engine's scheduler statistics are on tells
    /// the client so: the wire counts as step-reporting from the start, before
    /// any batch, and the first request's silence counts from its dispatch;
    /// a client told nothing waits for a stats-only step batch to say so.
    #[tokio::test]
    async fn a_client_told_its_engine_reports_steps_is_step_reporting_before_any_batch() {
        let (client, mut engine, _ns) = connect().await;
        assert!(
            !client.reports_steps(),
            "told nothing and shown nothing: a token-batch wire until shown otherwise"
        );
        client.note_reports_steps();
        assert!(
            client.reports_steps(),
            "the launcher's word holds before any batch"
        );
        let _stream = client.submit(request_for("req-1", 0)).await.unwrap();
        engine.recv_request().await.unwrap();
        tokio::time::sleep(Duration::from_millis(80)).await;
        let silence = client.output_silence().expect("a request is in flight");
        assert!(
            silence >= Duration::from_millis(70) && client.reports_steps(),
            "the first request's silence is the scheduler's own from its dispatch: {silence:?}"
        );
        assert_eq!(client.prefill_pending_tokens(), 3);
    }

    /// The silence the servicer reports is counted over the requests the
    /// engine reports every step. A request it reports at intervals (one that
    /// does not stream, one streamed every so many tokens) is quiet while
    /// healthy: it counts no silence however long it runs, though its prompt
    /// counts as pending prefill; a streaming request beside it counts from
    /// its own dispatch, and takes the silence with it when it ends.
    #[tokio::test]
    async fn a_request_reported_at_intervals_counts_no_silence() {
        use crate::protocol::tokenspeed::{
            output::BatchTokenIDOutSlim, request::TokenizedGenerateReqInput,
            sampling::SamplingParams,
        };
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        let _ns = ns;
        let (transport, engine) = tokio::join!(
            connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = TokenSpeedClient::new(transport.unwrap());
        let mut engine = engine.unwrap();
        let request = |rid: &str, input_ids: Vec<u32>, stream: bool, stream_interval| {
            TokenizedGenerateReqInput {
                rid: rid.to_string(),
                input_ids,
                sampling_params: SamplingParams {
                    max_new_tokens: Some(4),
                    stream_interval,
                    ..SamplingParams::default()
                },
                stream,
                ..TokenizedGenerateReqInput::default()
            }
        };
        let done = |rid: &str, prompt_tokens: u32| BatchTokenIDOutSlim {
            rids: vec![rid.to_string()],
            output_ids: vec![vec![11]],
            finished_reasons: vec!["stop".into()],
            prompt_tokens: vec![prompt_tokens],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            ..Default::default()
        };

        // A request that does not stream is reported at its end: it is owed
        // nothing in between, so it counts no silence, however long it runs.
        let mut unstreamed = client
            .submit(request("ts-end", vec![1, 2, 3], false, None))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(client.output_silence(), None, "reported at its end only");
        assert_eq!(
            client.prefill_pending_tokens(),
            3,
            "its prompt is pending prefill all the same"
        );
        // So is one streamed every eight tokens.
        let mut spaced = client
            .submit(request("ts-spaced", vec![4, 5], true, Some(8)))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        assert_eq!(client.output_silence(), None, "streamed at an interval");
        assert_eq!(client.prefill_pending_tokens(), 5);
        // A streaming request is reported every step: the silence counts
        // from its dispatch, not from the quiet requests' earlier ones.
        let mut streamed = client
            .submit(request("ts-step", vec![6], true, None))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        let at_dispatch = client
            .output_silence()
            .expect("a request reported every step is in flight");
        assert!(
            at_dispatch < Duration::from_millis(150),
            "counted from its own dispatch: {at_dispatch:?}"
        );
        assert_eq!(client.prefill_pending_tokens(), 6);
        // Its end takes the silence with it, though the other two go on.
        engine
            .send_output(vec![Bytes::from(
                encode_msgpack(&done("ts-step", 1)).unwrap(),
            )])
            .await
            .unwrap();
        assert!(streamed.next().await.unwrap().unwrap().finished());
        assert!(streamed.next().await.is_none());
        assert_eq!(
            client.output_silence(),
            None,
            "the requests left are reported at intervals"
        );
        assert_eq!(
            client.prefill_pending_tokens(),
            5,
            "the ended request's prompt settled; the others' stand"
        );
        // The other two end: nothing pending, nothing in flight.
        for (rid, stream, prompt_tokens) in [
            ("ts-end", &mut unstreamed, 3),
            ("ts-spaced", &mut spaced, 2),
        ] {
            engine
                .send_output(vec![Bytes::from(
                    encode_msgpack(&done(rid, prompt_tokens)).unwrap(),
                )])
                .await
                .unwrap();
            assert!(stream.next().await.unwrap().unwrap().finished());
            assert!(stream.next().await.is_none());
        }
        assert_eq!(client.prefill_pending_tokens(), 0);
        assert_eq!(client.output_silence(), None);
    }

    /// The same on the SGLang wire, embeddings included: a generation that
    /// does not stream, or streams every so many tokens, counts no silence;
    /// an embedding (one forward, one output) and a streaming generation at
    /// an interval of one count from their own dispatch and take the silence
    /// with them when they end.
    #[tokio::test]
    async fn an_sglang_request_reported_at_intervals_counts_no_silence() {
        use crate::protocol::sglang::{
            output::{BatchEmbeddingSlimOutput, BatchTokenIDSlimOutput},
            request::{SglangRequest, TokenizedEmbeddingReqInput, TokenizedGenerateReqInput},
            sampling::SamplingParams,
        };
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        let _ns = ns;
        let (transport, engine) = tokio::join!(
            connect_handshake(&handshake, 1, &input, &output, TIMEOUT),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = SglangClient::new(transport.unwrap());
        let mut engine = engine.unwrap();
        let generate = |rid: &str, input_ids: Vec<u32>, stream: bool, stream_interval| {
            SglangRequest::Generate(TokenizedGenerateReqInput {
                rid: rid.to_string(),
                input_ids,
                sampling_params: SamplingParams {
                    max_new_tokens: Some(4),
                    stream_interval,
                    ..SamplingParams::default()
                },
                stream,
                ..TokenizedGenerateReqInput::default()
            })
        };
        let done = |rid: &str, prompt_tokens: u32| BatchTokenIDSlimOutput {
            rids: vec![rid.to_string()],
            output_ids: vec![vec![11]],
            finished_reasons: vec!["stop".into()],
            finished_messages: vec![None],
            finished_matched: vec![None],
            prompt_tokens: vec![prompt_tokens],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
            finished_status: vec![None],
            ..Default::default()
        };

        // A generation that does not stream is reported at its end: no
        // silence, however long it runs; its prompt is pending all the same.
        let mut unstreamed = client
            .submit(generate("sg-end", vec![1, 2, 3], false, None))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(client.output_silence(), None, "reported at its end only");
        assert_eq!(client.prefill_pending_tokens(), 3);
        // So is one streamed every eight tokens.
        let mut spaced = client
            .submit(generate("sg-spaced", vec![4, 5], true, Some(8)))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        assert_eq!(client.output_silence(), None, "streamed at an interval");
        assert_eq!(client.prefill_pending_tokens(), 5);
        // An embedding is owed its one output: it counts from its dispatch.
        let mut embed = client
            .submit(SglangRequest::Embed(TokenizedEmbeddingReqInput {
                rid: "sg-embed".to_string(),
                input_ids: vec![6, 7],
                ..TokenizedEmbeddingReqInput::default()
            }))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        let at_dispatch = client
            .output_silence()
            .expect("an embedding in flight is owed its output");
        assert!(
            at_dispatch < Duration::from_millis(150),
            "counted from its own dispatch: {at_dispatch:?}"
        );
        assert_eq!(client.prefill_pending_tokens(), 7);
        // Its result ends it, and the silence with it.
        let result = BatchEmbeddingSlimOutput {
            rids: vec!["sg-embed".to_string()],
            embeddings: vec![vec![0.5]],
            prompt_tokens: vec![2],
            cached_tokens: vec![0],
            finished_reasons: vec!["stop".into()],
            finished_messages: vec![None],
            finished_status: vec![None],
            ..Default::default()
        };
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&result).unwrap())])
            .await
            .unwrap();
        assert!(embed.next().await.unwrap().unwrap().finished());
        assert!(embed.next().await.is_none());
        assert_eq!(
            client.output_silence(),
            None,
            "the generations left are reported at intervals"
        );
        assert_eq!(client.prefill_pending_tokens(), 5);
        // A streaming generation at the default interval is reported every
        // step: it counts from its own dispatch, not from the embedding's.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut streamed = client
            .submit(generate("sg-step", vec![8], true, None))
            .await
            .unwrap();
        engine.recv_request().await.unwrap();
        let at_dispatch = client
            .output_silence()
            .expect("a streaming generation at an interval of one is owed every step");
        assert!(
            at_dispatch < Duration::from_millis(150),
            "counted from its own dispatch: {at_dispatch:?}"
        );
        assert_eq!(client.prefill_pending_tokens(), 6);
        engine
            .send_output(vec![Bytes::from(
                encode_msgpack(&done("sg-step", 1)).unwrap(),
            )])
            .await
            .unwrap();
        assert!(streamed.next().await.unwrap().unwrap().finished());
        assert!(streamed.next().await.is_none());
        assert_eq!(client.output_silence(), None);
        assert_eq!(client.prefill_pending_tokens(), 5);
        // The quiet two end: nothing pending, nothing in flight.
        for (rid, stream, prompt_tokens) in [
            ("sg-end", &mut unstreamed, 3),
            ("sg-spaced", &mut spaced, 2),
        ] {
            engine
                .send_output(vec![Bytes::from(
                    encode_msgpack(&done(rid, prompt_tokens)).unwrap(),
                )])
                .await
                .unwrap();
            assert!(stream.next().await.unwrap().unwrap().finished());
            assert!(stream.next().await.is_none());
        }
        assert_eq!(client.prefill_pending_tokens(), 0);
        assert_eq!(client.output_silence(), None);
    }

    /// A lockstep group is paused when the client connects, so the first submit
    /// must wake every rank — including the one taking the request, so the
    /// whole group shares one wave number.
    #[tokio::test]
    async fn first_submit_wakes_the_paused_lockstep_group() {
        let (client, mut engines, _ns) = connect_ranks(2, true).await;

        let _stream = client.submit(request_for("req-1", 1)).await.unwrap();

        match engines[1].recv().await.unwrap() {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "req-1"),
            other => panic!("rank 1 expected the Add, got {other:?}"),
        }
        assert_eq!(next_wake(&mut engines[0]).await, 1);
        assert_eq!(next_wake(&mut engines[1]).await, 1);
    }

    /// Every submit to a lockstep group wakes it with a FRESH wave number,
    /// even submits that race the group's drain.
    ///
    /// This is the load-bearing property. The ranks bump their own
    /// `current_wave` as they park, and an engine drops any wave below its
    /// own, so a wake that reuses the last number the client saw is silently
    /// discarded — the holder's peers stay parked, never join its all-reduce,
    /// and vLLM kills the engine on its own RPC timeout. The client's clock is
    /// therefore monotonic and independent of when the `wave_complete`
    /// notifications arrive.
    #[tokio::test]
    async fn every_wake_names_a_wave_the_ranks_will_accept() {
        let (client, mut engines, _ns) = connect_ranks(2, true).await;

        let _first = client.submit(request_for("req-1", 1)).await.unwrap();
        let first = next_wake(&mut engines[0]).await;
        assert_eq!(next_wake(&mut engines[1]).await, first);

        // The group drains the wave and parks, moving on to `first + 1` — but
        // the client has not heard about it yet. Its next wake must still
        // outrank the engines' self-increment.
        let _second = client.submit(request_for("req-2", 0)).await.unwrap();
        let second = next_wake(&mut engines[0]).await;
        assert!(
            second > first,
            "wake {second} does not outrank the parked ranks at {}",
            first + 1
        );
        assert_eq!(next_wake(&mut engines[1]).await, second);

        // A rank can also report a wave AHEAD of the client's clock — it kept
        // its `current_wave` across a gateway restart, say. Folding that in is
        // the other half of the same invariant: the next wake must clear the
        // highest wave any rank is known to hold.
        let ahead = second + 5;
        engines[0]
            .send_output(wave_control(0, DpControlMessage::WaveComplete(ahead)))
            .await
            .unwrap();
        tokio::time::timeout(TIMEOUT, async {
            while *client.inner.wave.as_ref().unwrap().lock() <= ahead {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the reported wave never reached the clock");

        let _third = client.submit(request_for("req-3", 1)).await.unwrap();
        let third = next_wake(&mut engines[0]).await;
        assert!(
            third > ahead,
            "wake {third} does not clear the rank already on {ahead}"
        );
    }

    /// Read one engine's inbound queue up to its next wake, returning the wave
    /// it names.
    async fn next_wake(engine: &mut MockEngine) -> u64 {
        loop {
            match engine.recv().await.unwrap() {
                EngineInbound::StartDpWave {
                    wave,
                    exclude_engine_index,
                } => {
                    // No rank is excluded: the whole group shares one wave.
                    assert_eq!(exclude_engine_index, None);
                    return wave;
                }
                EngineInbound::Add(_) => continue,
                other => panic!("expected a wake or an Add, got {other:?}"),
            }
        }
    }

    /// Independent ranks (dense DP) never pause as a group, so they are never
    /// woken — each rank only ever sees its own requests.
    #[tokio::test]
    async fn independent_ranks_are_never_woken() {
        let (client, mut engines, _ns) = connect_ranks(2, false).await;

        let _first = client.submit(request_for("req-1", 1)).await.unwrap();
        let _second = client.submit(request_for("req-2", 0)).await.unwrap();

        // Rank 0's first message is its own Add, not a wake for rank 1's.
        match engines[0].recv().await.unwrap() {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "req-2"),
            other => panic!("independent rank expected only its Add, got {other:?}"),
        }
    }

    /// A rank that took a request for an already-drained wave asks the client
    /// to start the next one (vLLM `start_wave`). Without a DP coordinator
    /// nobody else answers that request, so the client must broadcast the wake
    /// itself — to EVERY rank, at a wave the asking rank will still accept.
    #[tokio::test]
    async fn a_ranks_start_wave_request_wakes_the_whole_group() {
        let (_client, mut engines, _ns) = connect_ranks(2, true).await;

        // The rank is ahead of the client's clock (it self-incremented on its
        // way out of the last wave), which is exactly the stale-wake trap.
        let requested = 7;
        engines[0]
            .send_output(wave_control(0, DpControlMessage::StartWave(requested)))
            .await
            .unwrap();

        assert!(next_wake(&mut engines[0]).await >= requested);
        assert!(next_wake(&mut engines[1]).await >= requested);
    }

    /// A drain notification is bookkeeping, not a wake: `wave_complete` folds
    /// into the clock but must NOT restart the group, or the ranks would never
    /// stay parked and an idle group would spin.
    #[tokio::test]
    async fn wave_complete_updates_the_clock_without_waking() {
        let (client, mut engines, _ns) = connect_ranks(2, true).await;

        engines[0]
            .send_output(wave_control(0, DpControlMessage::WaveComplete(3)))
            .await
            .unwrap();
        tokio::time::timeout(TIMEOUT, async {
            while *client.inner.wave.as_ref().unwrap().lock() <= 3 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the drained wave never reached the clock");

        // The next submit is the first thing that may wake the group, and it
        // must clear the wave the ranks parked on.
        let _stream = client.submit(request_for("req-1", 1)).await.unwrap();
        assert!(next_wake(&mut engines[0]).await > 3);
    }

    /// Independent ranks keep no wave clock, so a stray wave notification (a
    /// mislabeled engine, say) must be ignored rather than panic or start
    /// waking ranks that never park.
    #[tokio::test]
    async fn wave_notifications_from_independent_ranks_are_ignored() {
        let (client, mut engines, _ns) = connect_ranks(2, false).await;

        engines[0]
            .send_output(wave_control(0, DpControlMessage::StartWave(2)))
            .await
            .unwrap();

        let _stream = client.submit(request_for("req-1", 0)).await.unwrap();
        match engines[0].recv().await.unwrap() {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "req-1"),
            other => panic!("independent rank expected only its Add, got {other:?}"),
        }
        assert!(
            client.inner.wave.is_none(),
            "no clock for independent ranks"
        );
    }

    /// The utility RPC: the call is framed as vLLM's client frames it, goes to
    /// the named rank, and its reply resolves the caller by call id.
    #[tokio::test]
    async fn call_utility_resolves_the_reply_by_call_id() {
        let (client, mut engine, _ns) = connect().await;
        let engine_id = engines_id(&client, 0);
        let args = vec![OpaqueValue::from(false), OpaqueValue::from(false)];
        let (result, call) = tokio::join!(
            client.call_utility(&engine_id, "reset_prefix_cache", args.clone(), TIMEOUT),
            async {
                let frames = engine.recv_request().await.unwrap();
                assert_eq!(frames[0].as_ref(), b"\x03");
                let call: UtilityCall = decode_msgpack(frames[1].as_ref()).unwrap();
                engine
                    .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(true)))
                    .await
                    .unwrap();
                call
            }
        );
        assert_eq!(result.unwrap(), OpaqueValue::from(true));
        assert_eq!(call.client_index, 0);
        assert_eq!(call.method, "reset_prefix_cache");
        assert_eq!(call.args, args);
        assert!(client.inner.registry.lock().utility_calls.is_empty());

        // Calls are told apart by id: the next one gets a fresh id and its own
        // reply.
        let (result, next) = tokio::join!(
            client.call_utility(&engine_id, "is_sleeping", Vec::new(), TIMEOUT),
            async {
                let EngineInbound::Utility(next) = engine.recv().await.unwrap() else {
                    panic!("expected a utility call");
                };
                engine
                    .send_utility_reply(0, next.call_id, Ok(OpaqueValue::from(false)))
                    .await
                    .unwrap();
                next
            }
        );
        assert_ne!(next.call_id, call.call_id);
        assert_eq!(result.unwrap(), OpaqueValue::from(false));
    }

    /// The engine's failure message is the call's error.
    #[tokio::test]
    async fn call_utility_surfaces_the_engine_failure_message() {
        let (client, mut engine, _ns) = connect().await;
        let engine_id = engines_id(&client, 0);
        let failure = "Call to reset_prefix_cache method failed: boom";
        let (result, ()) = tokio::join!(
            client.call_utility(&engine_id, "reset_prefix_cache", Vec::new(), TIMEOUT),
            async {
                let EngineInbound::Utility(call) = engine.recv().await.unwrap() else {
                    panic!("expected a utility call");
                };
                engine
                    .send_utility_reply(0, call.call_id, Err(failure.to_string()))
                    .await
                    .unwrap();
            }
        );
        assert!(
            matches!(result, Err(Error::UtilityCallFailed { ref message }) if message == failure),
            "{result:?}"
        );
    }

    /// An unanswered call times out and leaves nothing behind; the reply that
    /// arrives late is dropped without disturbing later calls.
    #[tokio::test]
    async fn call_utility_times_out_and_drops_a_late_reply() {
        let (client, mut engine, _ns) = connect().await;
        let engine_id = engines_id(&client, 0);
        let result = client
            .call_utility(
                &engine_id,
                "reset_prefix_cache",
                Vec::new(),
                Duration::from_millis(200),
            )
            .await;
        assert!(
            matches!(result, Err(Error::UtilityTimeout { .. })),
            "{result:?}"
        );
        assert!(client.inner.registry.lock().utility_calls.is_empty());

        let EngineInbound::Utility(stale) = engine.recv().await.unwrap() else {
            panic!("expected a utility call");
        };
        engine
            .send_utility_reply(0, stale.call_id, Ok(OpaqueValue::from(true)))
            .await
            .unwrap();

        let (result, ()) = tokio::join!(
            client.call_utility(&engine_id, "reset_prefix_cache", Vec::new(), TIMEOUT),
            async {
                let EngineInbound::Utility(call) = engine.recv().await.unwrap() else {
                    panic!("expected a utility call");
                };
                assert_ne!(call.call_id, stale.call_id);
                engine
                    .send_utility_reply(0, call.call_id, Ok(OpaqueValue::from(false)))
                    .await
                    .unwrap();
            }
        );
        assert_eq!(result.unwrap(), OpaqueValue::from(false));
    }

    /// A pending call fails with the requests when the engine dies.
    #[tokio::test]
    async fn engine_dead_fails_pending_utility_calls() {
        let (client, mut engine, _ns) = connect().await;
        let engine_id = engines_id(&client, 0);
        let (result, ()) = tokio::join!(
            client.call_utility(&engine_id, "reset_prefix_cache", Vec::new(), TIMEOUT),
            async {
                engine.recv().await.unwrap();
                engine
                    .send_output(vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
                    .await
                    .unwrap();
            }
        );
        assert!(matches!(result, Err(Error::Shared(_))), "{result:?}");
        assert!(!client.is_alive());
    }
    /// The window a caller waits for the engine to accept a frame in the
    /// frozen-core cases below: short, so they run in milliseconds.
    const SEND_WINDOW: Duration = Duration::from_millis(300);

    /// An add-request no socket buffer absorbs: a 64 MiB aux frame stands in
    /// for the huge prompt of production.
    fn large_request(request_id: &str) -> (EngineCoreRequest, Vec<Bytes>) {
        (
            request_for(request_id, 0),
            vec![Bytes::from(vec![0u8; 64 << 20])],
        )
    }

    /// A finished one-token output for `request_id` on rank 0.
    fn finished_output(request_id: &str) -> Vec<Bytes> {
        batch(
            0,
            EngineCoreOutput {
                request_id: request_id.into(),
                new_token_ids: vec![7],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        )
    }

    /// Read the engine's input until the abort of `request_id` arrives;
    /// everything received, in order, as (kind, request id).
    async fn drain_until_aborted(
        engine: &mut MockEngine,
        request_id: &str,
    ) -> Vec<(&'static str, String)> {
        let mut seen = Vec::new();
        loop {
            match engine.recv().await.unwrap() {
                EngineInbound::Add(request) => seen.push(("add", request.request_id)),
                EngineInbound::Abort(ids) => {
                    let done = ids.iter().any(|id| id == request_id);
                    seen.extend(ids.into_iter().map(|id| ("abort", id)));
                    if done {
                        return seen;
                    }
                }
                other => panic!("unexpected inbound {other:?}"),
            }
        }
    }

    /// A small request to an engine that is not reading its input (a frozen
    /// core) fits the socket buffer and simply waits: the engine serves it
    /// when it reads again, and the client never closed.
    #[tokio::test]
    async fn a_small_request_to_a_frozen_engine_waits_and_is_served_on_resume() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        // The engine reads nothing until told to below.
        let mut stream = client.submit(request_for("req-small", 0)).await.unwrap();
        tokio::time::sleep(SEND_WINDOW).await;
        assert!(
            client.is_alive(),
            "the frame fit the socket buffer: nothing failed"
        );

        // The engine resumes: the frame is there and the request is served.
        let inbound = engine.recv().await.unwrap();
        assert!(
            matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-small"),
            "{inbound:?}"
        );
        engine
            .send_output(finished_output("req-small"))
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().unwrap().finished());
        assert!(stream.next().await.is_none());
        assert!(client.is_alive());
    }

    /// A frame the engine is not reading and the socket buffer cannot absorb
    /// fails its own request once the send window passes, and nothing else:
    /// the client stays up, the request already in flight keeps its stream,
    /// a request behind the blocked one fails within the window too instead
    /// of hanging, and every queued frame still reaches the engine when it
    /// reads again, each abandoned add-request followed by its abort.
    #[tokio::test]
    async fn a_large_request_a_frozen_engine_does_not_read_fails_alone() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        let mut in_flight = client.submit(request_for("req-small", 0)).await.unwrap();

        let (request, aux) = large_request("req-large");
        let started = std::time::Instant::now();
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        let waited = started.elapsed();
        assert!(
            matches!(error, Error::EngineInputBlocked { timeout } if timeout == SEND_WINDOW),
            "{error:?}"
        );
        assert!(
            waited >= SEND_WINDOW && waited < TIMEOUT,
            "waited {waited:?}"
        );
        assert!(client.is_alive(), "one request failed, not the client");

        let error = client
            .submit(request_for("req-behind", 0))
            .await
            .err()
            .expect("the input is still blocked");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        assert!(client.is_alive());

        let seen = drain_until_aborted(&mut engine, "req-behind").await;
        let position = |kind: &str, id: &str| {
            seen.iter()
                .position(|(k, i)| *k == kind && i == id)
                .unwrap_or_else(|| panic!("no {kind} of {id} in {seen:?}"))
        };
        assert_eq!(position("add", "req-small"), 0, "{seen:?}");
        assert!(
            position("add", "req-large") < position("abort", "req-large"),
            "{seen:?}"
        );
        assert!(
            position("add", "req-behind") < position("abort", "req-behind"),
            "{seen:?}"
        );

        engine
            .send_output(finished_output("req-small"))
            .await
            .unwrap();
        assert!(in_flight.next().await.unwrap().unwrap().finished());
        assert!(client.is_alive());
    }

    /// Once the engine reads again, the client (never closed) serves the next
    /// request as before.
    #[tokio::test]
    async fn a_request_after_the_engine_resumes_is_served() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );

        // The engine resumes and drains what was queued: the add and its abort.
        let seen = drain_until_aborted(&mut engine, "req-large").await;
        assert_eq!(
            seen.iter().filter(|(kind, _)| *kind == "add").count(),
            1,
            "{seen:?}"
        );

        let mut stream = client.submit(request_for("req-next", 0)).await.unwrap();
        let inbound = engine.recv().await.unwrap();
        assert!(
            matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-next"),
            "{inbound:?}"
        );
        engine
            .send_output(finished_output("req-next"))
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().unwrap().finished());
        assert!(client.is_alive());
    }

    /// While a frame the engine is not reading blocks the input, the aborts
    /// queued behind it (a timed-out submit's, a dropped stream's) never park
    /// the dispatcher: the output the engine still produces for the requests
    /// it holds is routed at once.
    #[tokio::test]
    async fn output_is_routed_while_aborts_wait_on_a_blocked_input() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        let mut in_flight = client.submit(request_for("req-small", 0)).await.unwrap();
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        // The abort of the abandoned request is with the dispatcher by now,
        // queued behind the frame the engine is not reading.
        tokio::time::sleep(SEND_WINDOW / 10).await;
        engine
            .send_output(finished_output("req-small"))
            .await
            .unwrap();
        let output = tokio::time::timeout(SEND_WINDOW / 2, in_flight.next())
            .await
            .expect("output routed while the abort waits on the blocked input")
            .unwrap()
            .unwrap();
        assert!(output.finished());
        assert!(client.is_alive());
    }

    /// An engine that reads nothing for the input death window is dead, not
    /// pausing: the requests it holds fail, the client closes, and a request
    /// after that is refused instead of queued for nobody.
    #[tokio::test]
    async fn an_input_blocked_for_the_death_window_ends_the_client() {
        let (client, _engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        client.set_input_death_timeout(SEND_WINDOW * 3);
        let mut in_flight = client.submit(request_for("req-small", 0)).await.unwrap();
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        assert!(client.is_alive(), "one send window is a pause, not a death");
        let error = tokio::time::timeout(TIMEOUT, in_flight.next())
            .await
            .expect("the engine is given up on within the death window")
            .expect("a terminal error item")
            .expect_err("the in-flight request fails with the engine");
        assert!(
            matches!(&error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead)),
            "{error:?}"
        );
        assert!(!client.is_alive());
        let error = client
            .submit(request_for("req-after", 0))
            .await
            .err()
            .expect("the client is closed");
        assert!(matches!(error, Error::ClientClosed { .. }), "{error:?}");
    }

    /// A frame the engine had not read when the client was dropped is never
    /// delivered: the input writer goes with the client, so an engine that
    /// reads again does not receive add-requests whose streams, and whose
    /// aborts, no longer exist.
    #[tokio::test]
    async fn dropping_the_client_withdraws_the_frames_the_engine_has_not_read() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        drop(client);
        // The engine reads again: nothing of the dropped client's arrives.
        match tokio::time::timeout(SEND_WINDOW, engine.recv()).await {
            // Nothing arrived, or the peer is gone with the client.
            Err(_) | Ok(Err(_)) => {}
            Ok(Ok(inbound)) => panic!("delivered after the client was dropped: {inbound:?}"),
        }
    }

    /// A submit its caller stops polling while the frame waits for the engine
    /// (the request's handler was cancelled) still aborts the request on the
    /// engine, which reads the add later, and gives its registration back:
    /// the same id can be submitted again.
    #[tokio::test]
    async fn an_abandoned_submit_aborts_its_request_and_frees_its_id() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        let (request, aux) = large_request("req-large");
        let abandoned =
            tokio::time::timeout(SEND_WINDOW / 3, client.submit_with_aux(request, aux)).await;
        assert!(
            abandoned.is_err(),
            "the engine is not reading a frame this large"
        );
        assert!(
            !client
                .inner
                .routing
                .lock()
                .inflight
                .values()
                .any(|ids| ids.contains("req-large")),
            "the abandoned request still holds its in-flight slot"
        );
        // The engine reads again: the queued add, then its abort.
        let seen = tokio::time::timeout(TIMEOUT, drain_until_aborted(&mut engine, "req-large"))
            .await
            .expect("the abort of the abandoned add follows it");
        assert_eq!(
            seen,
            vec![
                ("add", "req-large".to_string()),
                ("abort", "req-large".to_string())
            ]
        );
        // Its registration is gone too: the id is free, and the request is
        // served.
        let mut stream = client.submit(request_for("req-large", 0)).await.unwrap();
        let inbound = engine.recv().await.unwrap();
        assert!(
            matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-large"),
            "{inbound:?}"
        );
        engine
            .send_output(finished_output("req-large"))
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().unwrap().finished());
        assert!(client.is_alive());
    }

    /// Once the engine has left a frame unaccepted for a send window, the
    /// add-requests waiting behind it are bounded: the ones within the bound
    /// wait their window as before, the next is refused at once without
    /// joining the queue, and when the engine reads again the queue drains in
    /// order, each abandoned add followed by its abort, the refused request
    /// nowhere in it.
    #[tokio::test]
    async fn a_blocked_input_refuses_add_requests_past_the_frame_bound() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        client.set_input_queue_limit(2, usize::MAX);
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        // Two requests fit the bound and wait behind the blocked frame; a
        // third, submitted while they wait, is refused on the spot.
        let (behind_1, behind_2, refused) = tokio::join!(
            client.submit(request_for("req-behind-1", 0)),
            client.submit(request_for("req-behind-2", 0)),
            async {
                tokio::time::sleep(SEND_WINDOW / 3).await;
                let started = std::time::Instant::now();
                let result = client.submit(request_for("req-refused", 0)).await;
                (result, started.elapsed())
            },
        );
        for behind in [behind_1, behind_2] {
            let error = behind
                .err()
                .expect("the frame waits behind the blocked one");
            assert!(
                matches!(error, Error::EngineInputBlocked { .. }),
                "{error:?}"
            );
        }
        let (refused, waited) = refused;
        let error = refused.err().expect("the queue is at its bound");
        assert!(
            matches!(error, Error::EngineInputFull { frames: 2, .. }),
            "{error:?}"
        );
        assert!(
            waited < SEND_WINDOW / 2,
            "refused at once, not after a window: {waited:?}"
        );
        assert!(client.is_alive());
        // The engine reads again.
        let seen = drain_until_aborted(&mut engine, "req-behind-2").await;
        let adds: Vec<&str> = seen
            .iter()
            .filter(|(kind, _)| *kind == "add")
            .map(|(_, id)| id.as_str())
            .collect();
        assert_eq!(
            adds,
            ["req-large", "req-behind-1", "req-behind-2"],
            "{seen:?}"
        );
        let position = |kind: &str, id: &str| {
            seen.iter()
                .position(|(k, i)| *k == kind && i == id)
                .unwrap_or_else(|| panic!("no {kind} of {id} in {seen:?}"))
        };
        for id in ["req-large", "req-behind-1", "req-behind-2"] {
            assert!(position("add", id) < position("abort", id), "{seen:?}");
        }
        assert!(
            !seen.iter().any(|(_, id)| id == "req-refused"),
            "a refused request never reaches the engine: {seen:?}"
        );
        // The refused request gave its id back: it can be submitted now.
        let mut stream = client.submit(request_for("req-refused", 0)).await.unwrap();
        let inbound = engine.recv().await.unwrap();
        assert!(
            matches!(inbound, EngineInbound::Add(ref request) if request.request_id == "req-refused"),
            "{inbound:?}"
        );
        engine
            .send_output(finished_output("req-refused"))
            .await
            .unwrap();
        assert!(stream.next().await.unwrap().unwrap().finished());
    }

    /// The bound in bytes: behind a blocked frame a small add-request within
    /// it still waits its window (and reaches the engine on resume), while
    /// one that would take the waiting payload past it is refused at once.
    #[tokio::test]
    async fn a_blocked_input_refuses_add_requests_past_the_byte_bound() {
        let (client, mut engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        client.set_input_queue_limit(usize::MAX, 1 << 20);
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        let started = std::time::Instant::now();
        let error = client
            .submit(request_for("req-small", 0))
            .await
            .err()
            .expect("the small frame waits behind the blocked one");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        assert!(
            started.elapsed() >= SEND_WINDOW,
            "a frame within the bound waits its window"
        );
        let started = std::time::Instant::now();
        let error = client
            .submit_with_aux(
                request_for("req-big", 0),
                vec![Bytes::from(vec![0u8; 2 << 20])],
            )
            .await
            .err()
            .expect("the waiting payload would pass the bound");
        assert!(matches!(error, Error::EngineInputFull { .. }), "{error:?}");
        assert!(started.elapsed() < SEND_WINDOW / 2, "refused at once");
        assert!(client.is_alive());
        let seen = drain_until_aborted(&mut engine, "req-small").await;
        let adds: Vec<&str> = seen
            .iter()
            .filter(|(kind, _)| *kind == "add")
            .map(|(_, id)| id.as_str())
            .collect();
        assert_eq!(adds, ["req-large", "req-small"], "{seen:?}");
    }

    /// The silence watchdog counts from the moment work became in flight when
    /// no output came since: a request registered just before its tick, whose
    /// first output takes longer than the tick's remainder, is served, not
    /// failed as the engine's death.
    #[tokio::test]
    async fn a_request_registered_just_before_the_watchdog_tick_is_served() {
        let (client, mut engine, _ns) = connect().await;
        let window = Duration::from_millis(600);
        client.set_silence_death_timeout(window);
        // Idle through most of the first window, then submit.
        tokio::time::sleep(window - Duration::from_millis(100)).await;
        let mut stream = client.submit(request_for("req-late", 0)).await.unwrap();
        engine.recv().await.unwrap();
        // Past the tick, within the window counted from the registration.
        tokio::time::sleep(Duration::from_millis(300)).await;
        engine
            .send_output(finished_output("req-late"))
            .await
            .unwrap();
        let output = stream
            .next()
            .await
            .expect("an item")
            .expect("served, not failed by the watchdog");
        assert!(output.finished());
        assert!(client.is_alive());
    }

    /// A wake a rank asks for while a frame is blocked ahead in the input
    /// queue (a lockstep group, one rank not reading) neither parks the
    /// dispatcher nor skips a rank: the output of the rank that reads is
    /// routed at once, and when the engine reads again every rank gets the
    /// wake, the one whose frame was blocked included.
    #[tokio::test]
    async fn a_wake_requested_while_a_frame_is_blocked_parks_no_output_and_skips_no_rank() {
        let (client, mut engines, _ns) = connect_ranks(2, true).await;
        client.set_send_timeout(SEND_WINDOW);
        // Rank 1 holds a request and reads its input; rank 0 reads nothing.
        let mut in_flight = client.submit(request_for("req-1", 1)).await.unwrap();
        match engines[1].recv().await.unwrap() {
            EngineInbound::Add(request) => assert_eq!(request.request_id, "req-1"),
            other => panic!("rank 1 expected the Add, got {other:?}"),
        }
        assert_eq!(next_wake(&mut engines[1]).await, 1);
        // A frame rank 0 does not read blocks the input queue.
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("rank 0 is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        // Rank 1 took a request for a drained wave and asks for the next one.
        let requested = 7;
        engines[1]
            .send_output(wave_control(1, DpControlMessage::StartWave(requested)))
            .await
            .unwrap();
        // The wake waits on the blocked input; the dispatcher does not.
        engines[1]
            .send_output(batch(
                1,
                EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![7],
                    finish_reason: Some(EngineCoreFinishReason::Stop),
                    ..Default::default()
                },
            ))
            .await
            .unwrap();
        let output = tokio::time::timeout(SEND_WINDOW / 2, in_flight.next())
            .await
            .expect("output routed while the wake waits on the blocked input")
            .unwrap()
            .unwrap();
        assert!(output.finished());
        // Rank 0 reads again: its queue drains, the requested wake in it.
        let woken = tokio::time::timeout(TIMEOUT, async {
            loop {
                match engines[0].recv().await.unwrap() {
                    EngineInbound::StartDpWave { wave, .. } if wave >= requested => return wave,
                    EngineInbound::StartDpWave { .. }
                    | EngineInbound::Add(_)
                    | EngineInbound::Abort(_) => continue,
                    EngineInbound::Utility(call) => panic!("unexpected utility call {call:?}"),
                }
            }
        })
        .await
        .expect("rank 0 is woken for the requested wave once it reads again");
        assert!(woken >= requested);
        // And rank 1, whose frame was behind the blocked one, has its wake too:
        // the broadcast was queued whole, not cut at the window.
        let woken = tokio::time::timeout(TIMEOUT, next_wake(&mut engines[1]))
            .await
            .expect("rank 1 is woken for the requested wave");
        assert!(woken >= requested);
        assert!(client.is_alive());
    }

    /// A core that reads every request and answers none (its input thread
    /// alive, its step hung) is given up on once it has had requests in
    /// flight for a whole window in total, even when every request is
    /// abandoned before the window: the busy periods add up until an output,
    /// or a whole window idle, starts the count over.
    #[tokio::test]
    async fn a_core_that_reads_every_request_and_answers_none_is_given_up_on() {
        let (client, mut engine, _ns) = connect().await;
        let window = Duration::from_millis(600);
        client.set_silence_death_timeout(window);
        // One answered request arms the watchdog with that window.
        let mut warm = client.submit(request_for("req-warm", 0)).await.unwrap();
        engine.recv().await.unwrap();
        engine
            .send_output(finished_output("req-warm"))
            .await
            .unwrap();
        assert!(warm.next().await.unwrap().unwrap().finished());
        // From here the engine reads its input, adds and aborts alike, and
        // answers nothing. Requests abandoned after less than half a window
        // each, with a short gap between them: no busy period lasts a window,
        // together they do.
        let (busy, gap) = (Duration::from_millis(250), Duration::from_millis(70));
        let started = std::time::Instant::now();
        let mut given_up = None;
        for n in 0..12 {
            let request_id = format!("req-{n}");
            let mut stream = match client.submit(request_for(&request_id, 0)).await {
                Ok(stream) => stream,
                Err(error) => {
                    given_up = Some(error);
                    break;
                }
            };
            match engine.recv().await.unwrap() {
                EngineInbound::Add(request) => assert_eq!(request.request_id, request_id),
                other => panic!("expected the Add, got {other:?}"),
            }
            match tokio::time::timeout(busy, stream.next()).await {
                Err(_abandoned) => drop(stream),
                Ok(Some(Err(error))) => {
                    given_up = Some(error);
                    break;
                }
                Ok(other) => panic!("the engine answers nothing: {other:?}"),
            }
            // The abandoned request's abort, read like everything else.
            match engine.recv().await.unwrap() {
                EngineInbound::Abort(ids) => assert_eq!(ids, [request_id]),
                other => panic!("expected the Abort, got {other:?}"),
            }
            tokio::time::sleep(gap).await;
        }
        let error = given_up.expect("the engine is given up on within a few abandoned requests");
        assert!(
            matches!(&error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead))
                || matches!(error, Error::ClientClosed { .. }),
            "{error:?}"
        );
        assert!(!client.is_alive());
        assert!(
            started.elapsed() >= window,
            "given up on no sooner than a window of unanswered work"
        );
    }

    /// A request larger than the byte bound on its own is refused while the
    /// input is blocked, with nothing waiting (the bound is what may be
    /// parked), and the error names the request's own size against the bound.
    #[tokio::test]
    async fn a_request_past_the_byte_bound_on_its_own_is_refused_with_its_size_named() {
        let (client, _engine, _ns) = connect().await;
        client.set_send_timeout(SEND_WINDOW);
        client.set_input_queue_limit(usize::MAX, 1 << 20);
        let (request, aux) = large_request("req-large");
        let error = client
            .submit_with_aux(request, aux)
            .await
            .err()
            .expect("the engine is not reading a frame this large");
        assert!(
            matches!(error, Error::EngineInputBlocked { .. }),
            "{error:?}"
        );
        let started = std::time::Instant::now();
        let error = client
            .submit_with_aux(
                request_for("req-big", 0),
                vec![Bytes::from(vec![0u8; 2 << 20])],
            )
            .await
            .err()
            .expect("the request alone passes the bound");
        assert!(started.elapsed() < SEND_WINDOW / 2, "refused at once");
        match &error {
            Error::EngineInputFull {
                frames: 0,
                bytes: 0,
                request_bytes,
                limit_frames: usize::MAX,
                limit_bytes,
            } => {
                assert!(*request_bytes >= 2 << 20, "{request_bytes}");
                assert_eq!(*limit_bytes, 1 << 20);
            }
            other => panic!("{other:?}"),
        }
        let text = error.to_string();
        assert!(
            text.contains("this request's") && text.contains("would pass the bound"),
            "{text}"
        );
        assert!(client.is_alive());
    }

    /// The watchdog's tick landing in a gap between two abandoned requests
    /// does not push the give-up back a whole window: the engine is given up
    /// on during the request in which its unanswered work reaches the window,
    /// not at a later tick that happens to land while one is in flight.
    #[tokio::test]
    async fn a_tick_in_a_gap_between_requests_does_not_push_the_give_up_back() {
        let (client, mut engine, _ns) = connect().await;
        let window = Duration::from_millis(600);
        client.set_silence_death_timeout(window);
        // One answered request arms the watchdog with that window.
        let mut warm = client.submit(request_for("req-warm", 0)).await.unwrap();
        engine.recv().await.unwrap();
        engine
            .send_output(finished_output("req-warm"))
            .await
            .unwrap();
        assert!(warm.next().await.unwrap().unwrap().finished());
        // Requests of 250 ms with 150 ms gaps: the first tick (600 ms) lands
        // in the second request and re-arms for the 150 ms left (750 ms),
        // which is the gap before the third; the work reaches the window
        // 100 ms into the third request.
        let (busy, gap) = (Duration::from_millis(250), Duration::from_millis(150));
        let started = std::time::Instant::now();
        let mut given_up = None;
        let mut submitted = 0;
        for n in 0..12 {
            let request_id = format!("req-{n}");
            let mut stream = match client.submit(request_for(&request_id, 0)).await {
                Ok(stream) => stream,
                Err(error) => {
                    given_up = Some(error);
                    break;
                }
            };
            submitted += 1;
            match engine.recv().await.unwrap() {
                EngineInbound::Add(request) => assert_eq!(request.request_id, request_id),
                other => panic!("expected the Add, got {other:?}"),
            }
            match tokio::time::timeout(busy, stream.next()).await {
                Err(_abandoned) => drop(stream),
                Ok(Some(Err(error))) => {
                    given_up = Some(error);
                    break;
                }
                Ok(other) => panic!("the engine answers nothing: {other:?}"),
            }
            match engine.recv().await.unwrap() {
                EngineInbound::Abort(ids) => assert_eq!(ids, [request_id]),
                other => panic!("expected the Abort, got {other:?}"),
            }
            tokio::time::sleep(gap).await;
        }
        let elapsed = started.elapsed();
        let error = given_up.expect("the engine is given up on within a few abandoned requests");
        assert!(
            matches!(&error, Error::Shared(shared) if matches!(**shared, Error::EngineCoreDead))
                || matches!(error, Error::ClientClosed { .. }),
            "{error:?}"
        );
        assert!(!client.is_alive());
        assert!(
            elapsed >= window,
            "given up on no sooner than a window of unanswered work"
        );
        assert_eq!(
            submitted, 3,
            "given up on during the request in which the work reached the window \
             (after {elapsed:?}), not at a tick a whole window after the gap"
        );
    }
}
