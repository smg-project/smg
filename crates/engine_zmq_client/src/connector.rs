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
// - No utility RPC (deferred).

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use bytes::Bytes;
use futures::Stream;
use parking_lot::Mutex;
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{trace, warn};
use zeromq::RouterSendHalf;

use crate::{
    error::{Error, Result},
    protocol::{
        tokenspeed::TokenSpeedProtocol, vllm::VllmProtocol, EngineBatch, EngineLoad, EngineOutput,
        EngineProtocol, WaveEvent,
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

type OutputSender<O> = mpsc::UnboundedSender<Result<O>>;
type OutputReceiver<O> = mpsc::UnboundedReceiver<Result<O>>;

/// Routes engine outputs to per-request streams and tracks in-flight requests.
/// Keyed by `request_id`; the value is that request's output channel sender.
struct RequestRegistry<O> {
    closed: bool,
    requests: HashMap<String, OutputSender<O>>,
}

impl<O> Default for RequestRegistry<O> {
    fn default() -> Self {
        Self {
            closed: false,
            requests: HashMap::new(),
        }
    }
}

impl<O: EngineOutput> RequestRegistry<O> {
    /// Register a new request, returning the receiver for its output stream.
    fn register(&mut self, request_id: String) -> Result<OutputReceiver<O>> {
        if self.closed {
            return Err(Error::ClientClosed {
                message: "client is shutting down".to_string(),
            });
        }
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }
        let (sender, receiver) = mpsc::unbounded_channel();
        self.requests.insert(request_id, sender);
        Ok(receiver)
    }

    /// Deliver one output to its request stream; drop the entry when terminal.
    fn route(&mut self, output: O) {
        let request_id = output.request_id().to_string();
        let Some(sender) = self.requests.get(&request_id) else {
            trace!(%request_id, "output for unknown/finished request; dropping");
            return;
        };
        let finished = output.finished();
        if sender.send(Ok(output)).is_err() || finished {
            // Receiver gone (stream dropped) or request finished — stop tracking.
            self.requests.remove(&request_id);
        }
    }

    /// Remove finished/aborted request ids reported out-of-band by the engine.
    fn remove_all<'a>(&mut self, request_ids: impl IntoIterator<Item = &'a String>) {
        for request_id in request_ids {
            self.requests.remove(request_id);
        }
    }

    /// Fail every in-flight request with a shared error and close the registry.
    fn fail_all(&mut self, error: Arc<Error>) {
        self.closed = true;
        for (_, sender) in self.requests.drain() {
            let _ = sender.send(Err(Error::Shared(error.clone())));
        }
    }
}

struct ClientInner<P: EngineProtocol> {
    /// Shared input ROUTER send half (serialized across concurrent submits).
    input_send: tokio::sync::Mutex<RouterSendHalf>,
    engines: Vec<ConnectedEngine>,
    registry: Mutex<RequestRegistry<P::Output>>,
    /// Latest per-rank load, keyed by engine index. SMG's DP load signal.
    /// Reported values only — `select_engine` never mutates this map; its
    /// reservations live in `inflight`.
    load: Mutex<HashMap<u32, EngineLoad>>,
    /// This client's own unfinished request ids, per rank: a floor under the
    /// (possibly stale) reported load, so routing never trusts a snapshot that
    /// says an engine we just loaded is idle. Keyed by id rather than counted,
    /// because a request has several racing ways to finish — the explicit
    /// [`Client::abort`], a dropped stream's auto-abort, and the engine's own
    /// terminal output or finished-id report — and only the first of them may
    /// count. Identity makes every release idempotent; counters would let one
    /// request retire several slots and leave the rank looking emptier than it
    /// is.
    inflight: Mutex<HashMap<u32, HashSet<String>>>,
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
                self.inflight
                    .lock()
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
                let load = self.load.lock();
                let mut inflight = self.inflight.lock();
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
        let mut inflight = self.inflight.lock();
        let Some(ids) = inflight.get_mut(&engine_index) else {
            return;
        };
        for request_id in request_ids {
            ids.remove(request_id);
        }
        if ids.is_empty() {
            if let Some(load) = self.load.lock().get_mut(&engine_index) {
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
    /// The send is bounded by [`ENGINE_SEND_TIMEOUT`]: a healthy engine drains its
    /// input continuously, so a send that cannot complete in that window means the
    /// engine's event loop is wedged. Because the timeout cancels the send
    /// mid-frame (leaving the shared socket unusable), the timeout is fatal — it
    /// fails every in-flight request and closes the client so health checks evict
    /// this worker, instead of the untimed send freezing the dispatcher (starving
    /// the death watchdog) and every concurrent submit behind the shared lock.
    async fn send_to_engine(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        let mut input_send = self.input_send.lock().await;
        let send = send_message(
            &mut input_send,
            engine_id,
            request_type,
            payload.into(),
            aux_frames,
        );
        match tokio::time::timeout(ENGINE_SEND_TIMEOUT, send).await {
            Ok(result) => result,
            Err(_elapsed) => {
                warn!(
                    timeout = ?ENGINE_SEND_TIMEOUT,
                    "engine input send timed out; treating engine as dead"
                );
                self.registry
                    .lock()
                    .fail_all(Arc::new(Error::EngineCoreDead));
                Err(Error::EngineCoreDead)
            }
        }
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
        for engine in &self.engines {
            self.send_to_engine(
                &engine.engine_id,
                frame.clone(),
                payload.clone(),
                Vec::new(),
            )
            .await?;
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
        let Some(state) = self.wave.as_ref() else {
            return Ok(());
        };
        let wave = {
            let mut current = state.lock();
            *current = current.saturating_add(1);
            *current
        };
        self.broadcast_start_wave(wave).await
    }

    /// Fold a wave notification from an engine into the group's clock.
    async fn observe_wave(&self, event: WaveEvent, engine_index: u32) {
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
                if let Err(error) = self.wake_group().await {
                    warn!(%error, wave, "failed to start the requested wave");
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
}

/// Direct ZMQ connection to a same-host engine (one or more DP ranks behind one
/// shared transport), generic over the engine's wire protocol `P`.
pub struct Client<P: EngineProtocol> {
    inner: Arc<ClientInner<P>>,
    tasks: Vec<JoinHandle<()>>,
}

impl<P: EngineProtocol> Client<P> {
    /// Build a client over a connected transport, spawning the output loop and
    /// dispatcher. Background tasks are aborted when the client is dropped.
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
        let inner = Arc::new(ClientInner {
            input_send: tokio::sync::Mutex::new(input_send),
            engines,
            registry: Mutex::new(RequestRegistry::default()),
            load: Mutex::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
            scan_start: AtomicUsize::new(0),
            wave: lockstep.then(|| Mutex::new(0)),
            abort_tx,
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
            tasks: vec![output_task, dispatch_task],
        }
    }

    /// The engines connected on this transport.
    pub fn engines(&self) -> &[ConnectedEngine] {
        &self.inner.engines
    }

    /// Whether the connection is still live. Becomes false once the dispatcher
    /// observes `ENGINE_CORE_DEAD`, a transport failure, or the output stream
    /// closing. No RPC — this is a local liveness flag.
    pub fn is_alive(&self) -> bool {
        !self.inner.registry.lock().closed
    }

    /// The latest per-rank load for one engine index (DP routing signal), if
    /// any batch has been seen from it yet.
    pub fn engine_load(&self, engine_index: u32) -> Option<EngineLoad> {
        self.inner.load.lock().get(&engine_index).copied()
    }

    /// Submit a request and return a stream of its outputs. The request is
    /// routed by `data_parallel_rank` (SMG-pinned) or to the sole engine.
    /// Dropping the returned stream before it finishes aborts the request.
    pub async fn submit(&self, request: P::Request) -> Result<RequestStream<P>> {
        P::validate(&request)?;
        let request_id = P::request_id(&request).to_string();
        // Register before selecting: registration is the admission gate (it
        // rejects a duplicate id), and in-flight slots are held by id, so a
        // reservation taken before that gate could collide with the live
        // request's own slot and release it on rollback.
        let receiver = self.inner.registry.lock().register(request_id.clone())?;

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

        let (payload, aux_frames) = match P::encode_add(&request) {
            Ok(encoded) => encoded,
            Err(error) => {
                self.inner.registry.lock().remove_all([&request_id]);
                self.inner.release_one(&engine_id, &request_id);
                return Err(error);
            }
        };
        if let Err(error) = self
            .inner
            .send_to_engine(&engine_id, P::add_frame(), payload, aux_frames)
            .await
        {
            // Roll back the registry entry so a failed send doesn't leak it.
            self.inner.registry.lock().remove_all([&request_id]);
            self.inner.release_one(&engine_id, &request_id);
            return Err(error);
        }

        // The rank now holds the request; its peers must be awake for it to
        // step. Waking after the send keeps the group parked (and so unable to
        // report another drained wave) for the whole window.
        if let Err(error) = self.inner.wake_group().await {
            // The request can never make progress with its peers asleep, so
            // take it back rather than leaving it stranded on the engine.
            self.inner.registry.lock().remove_all([&request_id]);
            self.inner.release_one(&engine_id, &request_id);
            let _ = self.inner.abort(&engine_id, &request_id).await;
            return Err(error);
        }

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

/// If the engine emits no output for this long while requests are in flight, the
/// dispatcher treats it as dead. A hard engine death (SIGKILL/OOM) sends no
/// `ENGINE_CORE_DEAD` sentinel, and a bound PULL socket never errors when its
/// PUSH peer vanishes — so silence is the only available death signal. Generous
/// so a slow first token never trips it, while still bounding an otherwise
/// infinite hang.
/// Maximum time to wait for a single request frame to be accepted by an engine's
/// input socket before treating the engine as wedged. A healthy engine drains its
/// input continuously; a send that blocks this long means its event loop is stuck.
/// Bounds an otherwise unbounded await under the shared input lock (see
/// [`ClientInner::send_to_engine`]).
const ENGINE_SEND_TIMEOUT: Duration = Duration::from_secs(10);

const ENGINE_SILENCE_DEATH_TIMEOUT: Duration = Duration::from_secs(300);

/// Route decoded outputs to per-request streams and forward auto-abort requests
/// to their engines. Runs until the output channel closes or the engine dies.
async fn run_dispatcher<P: EngineProtocol>(
    mut out_rx: mpsc::Receiver<Result<EngineBatch<P::Output>>>,
    mut abort_rx: mpsc::UnboundedReceiver<(EngineId, String)>,
    inner: Arc<ClientInner<P>>,
) {
    loop {
        tokio::select! {
            output = out_rx.recv() => {
                match output {
                    Some(Ok(batch)) => {
                        if let Some(load) = batch.load {
                            inner.load.lock().insert(batch.engine_index, load);
                        }
                        if let Some(event) = batch.wave {
                            inner.observe_wave(event, batch.engine_index).await;
                        }
                        // Unique finished ids: a terminal output and the
                        // out-of-band finished list may name the same request.
                        let mut finished: HashSet<String> = batch
                            .finished_request_ids
                            .iter()
                            .cloned()
                            .collect();
                        let mut registry = inner.registry.lock();
                        for output in batch.outputs {
                            if output.finished() {
                                finished.insert(output.request_id().to_string());
                            }
                            registry.route(output);
                        }
                        registry.remove_all(&batch.finished_request_ids);
                        drop(registry);
                        inner.release(batch.engine_index, &finished);
                    }
                    Some(Err(error)) => {
                        if matches!(error, Error::EngineCoreDead | Error::Transport(_)) {
                            warn!(%error, "engine transport failed; failing all in-flight requests");
                            inner.registry.lock().fail_all(Arc::new(error));
                            inner.inflight.lock().clear();
                            return;
                        }
                        // A per-message decode error is non-fatal; keep going.
                        warn!(%error, "ignoring undecodable engine output");
                    }
                    None => break,
                }
            }
            abort = abort_rx.recv() => {
                if let Some((engine_id, request_id)) = abort {
                    inner.registry.lock().remove_all([&request_id]);
                    inner.release_one(&engine_id, &request_id);
                    if let Err(error) = inner.abort(&engine_id, &request_id).await {
                        warn!(%error, %request_id, "failed to send abort");
                        if matches!(error, Error::EngineCoreDead | Error::Transport(_)) {
                            // The engine is dead — send_to_engine already failed all
                            // in-flight requests; stop dispatching.
                            return;
                        }
                    }
                }
            }
            // Positive-liveness watchdog. select! rebuilds this sleep every
            // iteration, so any output or abort resets it; it only fires after a
            // full window of pure silence.
            () = tokio::time::sleep(ENGINE_SILENCE_DEATH_TIMEOUT) => {
                let mut registry = inner.registry.lock();
                if !registry.requests.is_empty() {
                    inner.inflight.lock().clear();
                    warn!(
                        "no engine output for {ENGINE_SILENCE_DEATH_TIMEOUT:?} with in-flight \
                         requests; treating engine as dead"
                    );
                    registry.fail_all(Arc::new(Error::EngineCoreDead));
                    return;
                }
                // Idle with nothing in flight: silence is expected, keep waiting.
            }
        }
    }
    inner
        .registry
        .lock()
        .fail_all(Arc::new(Error::ClientClosed {
            message: "engine output stream ended".to_string(),
        }));
    inner.inflight.lock().clear();
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
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(None)
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
                request::EngineCoreRequest,
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
            connect_handshake(
                &handshake,
                1,
                "127.0.0.1",
                Some(&input),
                Some(&output),
                TIMEOUT
            ),
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
            connect_handshake(
                &handshake,
                engine_count,
                "127.0.0.1",
                Some(&input),
                Some(&output),
                TIMEOUT
            ),
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
        std::mem::forget(ns);
        let (transport, engine) = tokio::join!(
            connect_handshake(
                &handshake,
                1,
                "127.0.0.1",
                Some(&input),
                Some(&output),
                TIMEOUT
            ),
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
                    assert_eq!(exclude_engine_index, u32::MAX);
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
}
