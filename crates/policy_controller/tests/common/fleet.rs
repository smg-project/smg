//! The end-to-end harness: a controller started on ephemeral loopback ports
//! in the case's own process, a fleet of mock engines beside it, and one
//! handle, [`Harness`], through which a case registers the fleet, feeds the
//! store and asks the service through the crate's reference client.
//! Included by path beside `common/engines.rs` and `common/net.rs` (the
//! including binary names them `engines` and `net`).
//!
//! Every fact the store holds of an engine came from that engine, through a
//! sink. Loads are the controller's own load driver's: the harness writes no
//! load row. Health has no driver in the crate yet, so the harness stands in
//! for one ([`probe`]): once an interval it asks each engine its wire's
//! health check and writes what it heard, never what the case knows. A dial
//! is bounded, so an engine that stops answering costs a turn that bound and
//! no more. KV events have no driver either: a second stand-in
//! ([`subscribe`]) holds one event stream an engine, decodes each batch into
//! the events an index applies and hands it to the sink. Each stand-in leaves
//! the day a driver feeds its sink.
//!
//! An engine speaks the wire of its kind ([`Shape::runtime`]): the fleet's
//! first wire, or the one whose load report counts resident tokens. The
//! controller's load driver polls each through that kind's own arm, the
//! prober asks each its own health check, and the subscriber speaks the
//! first wire alone.
//!
//! A fleet registers as an operator's does ([`Harness::register`]): an
//! engine of one rank is an add on the control listener, under the name the
//! harness states there. The call makes one rank, so a rank of a wider
//! engine is still the store's membership write. The view a case reads
//! ([`Harness::view`]) is the store's handle too, and gives way to the
//! decisions the service answers. A case calls the handle alone, so neither
//! move rewrites a case.

use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use kv_index::compute_content_hash;
use mock_worker::{
    config::{Config as EngineConfig, SchedulerWire},
    engine::{EngineParams, LoadsLike, TimingModel},
};
use openai_protocol::worker::RuntimeType;
use parking_lot::Mutex;
use smg_grpc_client::{
    common_proto::{kv_cache_event::Data, KvEventBatch},
    tokenspeed_proto, vllm_proto, TokenSpeedSchedulerClient, VllmEngineClient,
};
use smg_policy_controller::{
    client::Client,
    controller::{
        Config, Controller, Election, Error, LeasePlace, Listener, Readiness, Scope, State,
    },
    proto::{
        self, completed::Class, policy_controller_client::PolicyControllerClient,
        register_worker_request::Mode, worker_event::Event, Completed, Dispatched, LeaderInfo,
        Side, Welcome,
    },
    store::{
        Ask, BatchFacts, BatchOutcome, GatewayId, HealthPolicy, HealthSink, KvCursor, KvEvent,
        KvEventSink, LaneCounts, LeaseState, MemoryStore, SequenceHash, Signals, StoredBlock,
        ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
    },
    view::{fingerprint, LoadRow, Role, WireModes, WorkerId},
    AppliedSpec, LeaseId, PolicySpec,
};
use tokio::{
    io::{copy, sink},
    net::{TcpListener, TcpStream},
    sync::{oneshot, watch},
    task::JoinSet,
    time::MissedTickBehavior,
};
use tonic::{
    codegen::tokio_stream::StreamExt, metadata::AsciiMetadataValue, transport::Channel, Request,
};

use crate::{
    engines::{engine_config, eventually},
    net::loopback,
};

/// How often the stand-in prober asks every engine: the check interval of
/// every registration.
pub const PROBE_INTERVAL: Duration = Duration::from_millis(500);

/// The bound on one dial of an engine, connect included: the timeout of
/// every registration.
pub const DIAL_TIMEOUT: Duration = Duration::from_secs(1);

/// How often the controller's load driver polls an engine.
pub const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Room for a loaded host on top of a wait's own timing.
pub const ROOM: Duration = Duration::from_secs(30);

/// The metadata entry a caller states its name in.
const PRINCIPAL: &str = "x-smg-principal";

/// The name the harness states on the control listener.
const OPERATOR: &str = "fleet-operator";

/// The tokens a held request asks for: at the engines' decode step it
/// outlasts any case.
const HELD_TOKENS: u32 = 1 << 20;

/// The tokens a sent request asks for: a few, read to the stream's end.
const SENT_TOKENS: u32 = 4;

/// The name the harness's client states as its gateway id.
pub const GATEWAY: &str = "fleet-gateway";

/// One kind of engine in a fleet's table: a model, a role and its
/// data-parallel ranks.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub model: &'static str,
    pub role: Role,
    pub ranks: u16,
    /// The engine's kind, as its registration states it: the wire it speaks.
    pub runtime: RuntimeType,
}

/// One row of a pool as a view reads it, named by the engine that
/// registered it.
#[derive(Clone, Copy, Debug)]
pub struct Seen {
    /// The engine's index in the harness, in registration order.
    pub engine: usize,
    pub rank: u16,
    pub healthy: bool,
    /// Whether the row reads overloaded: its engine's reports against the
    /// thresholds that stand for it.
    pub overloaded: bool,
    pub load: Option<LoadRow>,
}

/// One status the stand-in prober moved.
#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    pub engine: usize,
    pub from: WorkerStatus,
    pub to: WorkerStatus,
    /// The probe turn that wrote it, counted from one.
    pub turn: u64,
    pub at: Instant,
}

/// One rank's KV events: what its stream delivered beside what the store
/// holds of it, read at one instant.
#[derive(Clone, Copy, Debug)]
pub struct Heard {
    /// The engine's data-parallel rank the row stands for.
    pub rank: u16,
    /// The last sequence the engine's stream delivered for the rank; `0`
    /// when none came.
    pub last: u64,
    /// The batches the sink answered applied.
    pub applied: u64,
    /// The blocks the applied events left the engine holding.
    pub held: usize,
    /// The blocks the store's index holds for the row.
    pub indexed: usize,
    /// What the row's lane counted beside its blocks.
    pub counts: Option<LaneCounts>,
    /// The rank's cursor in the store, when the rank has been heard from.
    pub cursor: Option<KvCursor>,
}

/// An engine's death.
#[derive(Clone, Copy, Debug)]
pub struct Death {
    /// When the kill was asked: no socket of the engine had closed yet.
    pub at: Instant,
    /// The probe turns begun when the kill was asked: a dial of a turn
    /// numbered below it heard a live engine.
    pub asked: u64,
    /// The probe turns begun once every socket of the engine had closed: a
    /// turn numbered above it dialed a dead engine.
    pub turn: u64,
}

/// A mock engine on a runtime and a thread of its own. Its kill drops that
/// runtime, so every socket of the engine closes at once, as a dead
/// process's do.
struct Engine {
    endpoint: String,
    /// Its kind: the wire a request the harness holds on it is sent on.
    runtime: RuntimeType,
    /// Where its listener is bound.
    addr: SocketAddr,
    kill: Option<oneshot::Sender<()>>,
    /// Its sender drops when the engine's thread ends, the runtime gone: a
    /// wait on it is bounded, where a join of the thread is not.
    ended: Option<oneshot::Receiver<()>>,
    /// A connection of the harness's own, opened at the start and never
    /// written: it reads its end when the engine's sockets close.
    witness: Option<TcpStream>,
    /// Hangs the engine: its thread waits on the receiver it is sent, so
    /// nothing of the engine answers while its sockets stay open.
    hang: Option<oneshot::Sender<mpsc::Receiver<()>>>,
    /// Held while the engine hangs: its drop lets the thread go on to its
    /// end.
    release: Option<mpsc::Sender<()>>,
}

/// What the stand-in prober counts and writes.
#[derive(Default)]
struct Turns {
    begun: AtomicU64,
    /// What each dial heard, in the order dialed.
    dialed: Mutex<Vec<Dialed>>,
    verdicts: Mutex<Vec<Verdict>>,
}

/// How one dial of an engine ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialEnd {
    /// The engine's wire answered: healthy or not.
    Answered(bool),
    /// The dial failed inside its bound: a refused connection, a broken
    /// one.
    Failed,
    /// Nothing came within [`DIAL_TIMEOUT`]: the bound ended the dial.
    TimedOut,
}

impl DialEnd {
    /// Whether the engine answered healthy.
    fn healthy(self) -> bool {
        self == Self::Answered(true)
    }
}

/// One dial of an engine: the turn that dialed it and how it ended.
#[derive(Clone, Copy, Debug)]
struct Dialed {
    engine: usize,
    turn: u64,
    end: DialEnd,
}

/// What the stand-in subscribers heard, a row at a time, and the gaps the
/// sink named. The rows' one lock is held across a batch's hand-over, so a
/// read under it sees the store and every row in step.
#[derive(Default)]
struct Events {
    rows: Mutex<HashMap<WorkerId, Stream>>,
    /// What refuses each engine's stream, by the engine's index: a dial or
    /// a subscription that failed or outlasted its bound, a stream that
    /// broke. An engine whose stream stands open has none.
    refused: Mutex<HashMap<usize, String>>,
    gaps: AtomicU64,
    /// The snapshot chunks that came: an engine's own stream carries none.
    snapshots: AtomicU64,
}

/// One row's stream as its stand-in subscriber heard it.
#[derive(Default)]
struct Stream {
    /// The last sequence the stream delivered.
    last: u64,
    /// The batches the sink answered applied.
    applied: u64,
    /// The blocks the applied events left the engine holding, by its hashes.
    /// An engine's own stream carries no snapshot chunk (a relay's does), so
    /// stores, removals and clears are the whole of it. A stream opened
    /// again past the engine's replay buffer would part it from the index:
    /// the sink reads a restarted publisher there and empties the rank,
    /// and this set keeps what it had. No engine of the fleet opens one.
    held: HashSet<SequenceHash>,
    /// The first chain the stream stored, in the chain's order: the blocks of
    /// the first store under no parent, then of each store whose parent is
    /// the chain's last block. A prompt's head as the engine holds it.
    head: Vec<StoredBlock>,
}

/// An engine as the stand-in prober dials it: its address and the rows its
/// one answer speaks for.
#[derive(Clone)]
struct Probed {
    engine: usize,
    runtime: RuntimeType,
    endpoint: String,
    ids: Vec<WorkerId>,
}

/// A controller in the binary's default shape, served on ephemeral loopback
/// ports in the case's own process. Its gateway listener takes a gateway's
/// name as told: the reference client states its own on every call. Its
/// control listener takes an operator's the same way.
pub struct Replica {
    gateway: SocketAddr,
    control: SocketAddr,
    store: Arc<MemoryStore>,
    readiness: Readiness,
    stop: Option<oneshot::Sender<()>>,
    serve: JoinSet<Result<(), Error>>,
}

/// The one handle a case holds: register, feed, ask.
pub struct Harness {
    replica: Replica,
    store: Arc<MemoryStore>,
    client: Client,
    operator: PolicyControllerClient<Channel>,
    engines: Vec<Engine>,
    members: HashMap<WorkerId, (usize, u16)>,
    probed: Arc<Mutex<Vec<Probed>>>,
    turns: Arc<Turns>,
    events: Arc<Events>,
    /// Whether the feeds run: an engine registered after their start gets
    /// its subscriber at its registration.
    feeding: bool,
    held: Vec<Box<dyn std::any::Any>>,
    /// The requests sent so far: the next one's lease id.
    sent: u64,
    /// The spec generations applied so far.
    generation: u64,
    buffers: ViewBuffers,
    stop_feeds: watch::Sender<bool>,
    feeds: JoinSet<()>,
}

/// Waits for `check` under the bound named `name`; the error names the
/// bound that ran out. Returns how long the wait took.
pub async fn within(
    name: &str,
    bound: Duration,
    check: impl FnMut() -> bool,
) -> Result<Duration, String> {
    let started = Instant::now();
    if eventually(bound, check).await {
        Ok(started.elapsed())
    } else {
        Err(format!("{name}: not within {bound:?}"))
    }
}

/// `step` under the bound named `name`; the error names the bound that ran
/// out.
async fn bounded<T>(
    name: &str,
    bound: Duration,
    step: impl std::future::Future<Output = T>,
) -> Result<T, String> {
    tokio::time::timeout(bound, step)
        .await
        .map_err(|_| format!("{name}: not within {bound:?}"))
}

fn millis(duration: Duration) -> u32 {
    u32::try_from(duration.as_millis()).unwrap_or(u32::MAX)
}

/// The health policy every registration states: one probe turn moves a
/// rank between ready and unhealthy, at the stand-in prober's interval and
/// under its dial bound.
fn health_policy() -> HealthPolicy {
    HealthPolicy {
        failure_threshold: 1,
        success_threshold: 1,
        check_interval_ms: millis(PROBE_INTERVAL),
        timeout_ms: millis(DIAL_TIMEOUT),
        probes: true,
    }
}

/// The policy in the contract's form, for an operator's add.
fn wire_health(policy: HealthPolicy) -> proto::HealthPolicy {
    proto::HealthPolicy {
        failure_threshold: policy.failure_threshold,
        success_threshold: policy.success_threshold,
        check_interval_ms: policy.check_interval_ms,
        timeout_ms: policy.timeout_ms,
        disabled: !policy.probes,
    }
}

/// `role` as the contract names it.
fn wire_role(role: Role) -> proto::Role {
    match role {
        Role::Single => proto::Role::Single,
        Role::Prefill => proto::Role::Prefill,
        Role::Decode => proto::Role::Decode,
        Role::Encode => proto::Role::Encode,
    }
}

/// A fleet engine's parameters: a decode step of 10 ms whatever the batch,
/// so a held request stays in the running set for as long as it is held.
fn engine_params() -> EngineParams {
    EngineParams {
        timing: TimingModel::Linear {
            prefill_tps: 8000.0,
            decode_base_ms: 10.0,
            decode_per_req_ms: 0.0,
        },
        ..EngineParams::default()
    }
}

/// The mock engine's configuration at `port` for an engine of kind
/// `runtime`: the fleet's first wire, or the wire whose load report counts
/// resident tokens, served with its report in full.
fn wire_config(runtime: RuntimeType, port: u16, ranks: u16) -> EngineConfig {
    let config = engine_config(port, ranks);
    if counts_resident(runtime) {
        EngineConfig {
            grpc_wire: SchedulerWire::TokenSpeed,
            loads_like: LoadsLike::Mock,
            ..config
        }
    } else {
        config
    }
}

/// Whether an engine of kind `runtime` speaks the wire whose load report
/// counts resident tokens. The subscriber stand-in speaks the first wire
/// alone: an engine of this kind states no events signal, and none of its
/// streams is opened.
fn counts_resident(runtime: RuntimeType) -> bool {
    runtime == RuntimeType::TokenSpeed
}

/// One mock engine of kind `kind` with `ranks` ranks, served on a runtime of this thread
/// until `killed` resolves; the runtime is dropped at the return, and every
/// task of the engine with it. `bound` takes the listener's address. A
/// receiver sent on `hung` stops the thread on it: nothing of the engine
/// answers and its sockets stay open, until that receiver's sender drops.
fn serve_engine(
    ranks: u16,
    kind: RuntimeType,
    bound: oneshot::Sender<Result<SocketAddr, String>>,
    killed: oneshot::Receiver<()>,
    hung: oneshot::Receiver<mpsc::Receiver<()>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = bound.send(Err(error.to_string()));
            return;
        }
    };
    runtime.block_on(async move {
        let listener = match TcpListener::bind(loopback()).await {
            Ok(listener) => listener,
            Err(error) => {
                let _ = bound.send(Err(error.to_string()));
                return;
            }
        };
        let addr = match listener.local_addr() {
            Ok(addr) => addr,
            Err(error) => {
                let _ = bound.send(Err(error.to_string()));
                return;
            }
        };
        let config = Arc::new(EngineConfig {
            engine: engine_params(),
            ..wire_config(kind, addr.port(), ranks)
        });
        if bound.send(Ok(addr)).is_err() {
            return;
        }
        // Held outside the select: a branch's future is dropped before its
        // arm runs, and this one owns the listener, which a hang keeps.
        let mut serving = std::pin::pin!(mock_worker::grpc::serve_with_listener_until(
            config,
            listener,
            std::future::pending(),
        ));
        tokio::select! {
            () = &mut serving => {}
            _ = killed => {}
            Ok(release) = hung => {
                // The thread stops here, and the runtime with it; the
                // serving future, the listener in it, is still held.
                let _ = release.recv();
            }
        }
    });
}

/// Starts one mock engine of kind `runtime` with `ranks` ranks on a runtime and a thread of
/// its own.
async fn start_engine(ranks: u16, runtime: RuntimeType) -> Result<Engine, String> {
    let (bound, address) = oneshot::channel::<Result<SocketAddr, String>>();
    let (kill, killed) = oneshot::channel::<()>();
    let (ending, ended) = oneshot::channel::<()>();
    let (hang, hung) = oneshot::channel::<mpsc::Receiver<()>>();
    thread::Builder::new()
        .name("mock-engine".to_owned())
        .spawn(move || {
            serve_engine(ranks, runtime, bound, killed, hung);
            drop(ending);
        })
        .map_err(|error| error.to_string())?;
    let addr = bounded("an engine's listener is bound", ROOM, address)
        .await?
        .map_err(|_| "an engine's thread ended before its listener was bound".to_owned())??;
    let witness = bounded(
        "an engine accepts a connection",
        ROOM,
        TcpStream::connect(addr),
    )
    .await?
    .map_err(|error| error.to_string())?;
    Ok(Engine {
        endpoint: format!("http://{addr}"),
        runtime,
        addr,
        kill: Some(kill),
        ended: Some(ended),
        witness: Some(witness),
        hang: Some(hang),
        release: None,
    })
}

/// An engine's client on the wire its kind speaks.
#[derive(Clone)]
enum EngineClient {
    First(VllmEngineClient),
    Counting(TokenSpeedSchedulerClient),
}

impl EngineClient {
    /// Dials an engine of kind `runtime`.
    async fn connect(runtime: RuntimeType, endpoint: &str) -> Option<Self> {
        if counts_resident(runtime) {
            let client = TokenSpeedSchedulerClient::connect(endpoint).await.ok()?;
            Some(Self::Counting(client))
        } else {
            let client = VllmEngineClient::connect(endpoint).await.ok()?;
            Some(Self::First(client))
        }
    }

    /// The wire's health answer, when one came.
    async fn healthy(&self) -> Option<bool> {
        match self {
            Self::First(client) => Some(client.health_check().await.ok()?.healthy),
            Self::Counting(client) => Some(client.health_check().await.ok()?.healthy),
        }
    }
}

/// How one dial of an engine ended: its wire's health answer, a failure,
/// or nothing within [`DIAL_TIMEOUT`]. The connection is kept across turns
/// and dropped when no answer came, so the next turn dials anew.
async fn dial(
    clients: &mut HashMap<String, EngineClient>,
    runtime: RuntimeType,
    endpoint: &str,
) -> DialEnd {
    let kept = clients.get(endpoint).cloned();
    let asked = tokio::time::timeout(DIAL_TIMEOUT, async {
        let client = match kept {
            Some(client) => client,
            None => EngineClient::connect(runtime, endpoint).await?,
        };
        let healthy = client.healthy().await?;
        Some((client, healthy))
    })
    .await;
    match asked {
        Ok(Some((client, healthy))) => {
            clients.insert(endpoint.to_owned(), client);
            DialEnd::Answered(healthy)
        }
        Ok(None) => {
            clients.remove(endpoint);
            DialEnd::Failed
        }
        Err(_) => {
            clients.remove(endpoint);
            DialEnd::TimedOut
        }
    }
}

/// The case's stand-in for the health prober: every [`PROBE_INTERVAL`] one
/// turn asks each engine once and writes what the dial heard for each of
/// its rows, under the row's revision. A status that moved is kept with the
/// turn that moved it.
async fn probe(
    store: Arc<MemoryStore>,
    probed: Arc<Mutex<Vec<Probed>>>,
    turns: Arc<Turns>,
    mut stop: watch::Receiver<bool>,
) {
    let mut clients = HashMap::new();
    let mut tick = tokio::time::interval(PROBE_INTERVAL);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            _ = stop.wait_for(|stop| *stop) => return,
        }
        let turn = turns.begun.fetch_add(1, Ordering::SeqCst) + 1;
        let engines = probed.lock().clone();
        for Probed {
            engine,
            runtime,
            endpoint,
            ids,
        } in engines
        {
            let end = dial(&mut clients, runtime, &endpoint).await;
            let healthy = end.healthy();
            turns.dialed.lock().push(Dialed { engine, turn, end });
            for id in ids {
                let Some(row) = store.get(id) else { continue };
                // A verdict is pushed under the lock its write was made
                // under: a reader that saw the write reads the verdict.
                let mut verdicts = turns.verdicts.lock();
                if let Some((from, to)) = store.probe_outcome(id, healthy, row.revision) {
                    verdicts.push(Verdict {
                        engine,
                        from,
                        to,
                        turn,
                        at: Instant::now(),
                    });
                }
            }
        }
    }
}

/// One engine batch as the events an index applies: every block under the
/// engine's own hash, its content hash made of its token ids. The harness's
/// engines publish one cache group in one tier, so this decode chooses
/// nothing: which blocks reach an index is a collector's decision.
fn decode(batch: &KvEventBatch) -> Vec<KvEvent> {
    batch
        .events
        .iter()
        .filter_map(|event| {
            Some(match event.data.as_ref()? {
                Data::Stored(stored) => KvEvent::Stored {
                    parent: stored.parent_block_hash.map(SequenceHash::from),
                    blocks: stored
                        .blocks
                        .iter()
                        .map(|block| StoredBlock {
                            seq_hash: SequenceHash::from(block.block_hash),
                            content_hash: compute_content_hash(&block.token_ids),
                        })
                        .collect(),
                },
                Data::Removed(removed) => KvEvent::Removed(
                    removed
                        .block_hashes
                        .iter()
                        .copied()
                        .map(SequenceHash::from)
                        .collect(),
                ),
                Data::Cleared(_) => KvEvent::Cleared,
            })
        })
        .collect()
}

/// One batch, decoded, through its rank's cursor into its row's index, with
/// the header the batch carries. A batch that carries a load record alone is
/// not handed over, and its record is not written here: loads are the
/// driver's. A snapshot chunk is a relay's: its start is a collector's word
/// to every row of the engine, which this stand-in does not say, so a chunk
/// is counted and not handed over. Returns `false` at a gap.
fn admit(store: &MemoryStore, engine: &Probed, events: &Events, batch: &KvEventBatch) -> bool {
    if batch.load.as_ref().is_some_and(|load| load.load_only) {
        return true;
    }
    if batch.snapshot.is_some() {
        events.snapshots.fetch_add(1, Ordering::SeqCst);
        return true;
    }
    let rank = u16::try_from(batch.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX);
    let Some(&id) = engine.ids.get(usize::from(rank)) else {
        return true;
    };
    let decoded = decode(batch);
    let facts = BatchFacts {
        sequence: batch.sequence_number,
        clears: decoded
            .iter()
            .any(|event| matches!(event, KvEvent::Cleared)),
        snapshot: None,
    };
    let mut rows = events.rows.lock();
    let row = rows.entry(id).or_default();
    row.last = batch.sequence_number;
    match store.admit(id, rank, facts, &decoded) {
        BatchOutcome::Applied => {
            row.applied += 1;
            for event in decoded {
                match event {
                    KvEvent::Stored { parent, blocks } => {
                        if parent == row.head.last().map(|block| block.seq_hash) {
                            row.head.extend(&blocks);
                        }
                        row.held.extend(blocks.iter().map(|block| block.seq_hash));
                    }
                    KvEvent::Removed(hashes) => {
                        for hash in hashes {
                            row.held.remove(&hash);
                        }
                    }
                    KvEvent::Cleared => row.held.clear(),
                }
            }
            true
        }
        BatchOutcome::Skipped => true,
        BatchOutcome::Gap { .. } => {
            events.gaps.fetch_add(1, Ordering::SeqCst);
            false
        }
    }
}

/// The case's stand-in for the KV event subscription: one stream an engine,
/// every rank's batches on it, each handed to the sink as it arrives. A
/// stream that ends or names a gap is opened again after one probe interval:
/// from the rank's cursor for an engine of one rank, from zero for an engine
/// of several (one cursor names one rank's sequence). An engine asked from
/// zero replays what its buffer still holds. A row with no cursor is
/// unheard: its stream is asked from zero, so what the engine stored before
/// the row was made is fed too. What refuses a dial or a stream is kept
/// until a stream opens, so a wait that runs out can say why.
async fn subscribe(
    store: Arc<MemoryStore>,
    engine: Probed,
    events: Arc<Events>,
    mut stop: watch::Receiver<bool>,
) {
    loop {
        let resume = match engine.ids.as_slice() {
            [only] => store.cursor(*only, 0).map_or(0, |cursor| cursor.applied),
            _ => 0,
        };
        let opened = tokio::time::timeout(DIAL_TIMEOUT, async {
            let client = VllmEngineClient::connect(&engine.endpoint)
                .await
                .map_err(|error| format!("the dial: {error}"))?;
            client
                .subscribe_kv_events(resume)
                .await
                .map_err(|status| format!("the subscription: {status}"))
        })
        .await
        .unwrap_or_else(|_| Err(format!("no stream within {DIAL_TIMEOUT:?}")));
        let refusal = match opened {
            Ok(mut stream) => {
                events.refused.lock().remove(&engine.engine);
                for id in &engine.ids {
                    store.reconnected(*id);
                }
                loop {
                    let batch = tokio::select! {
                        batch = stream.message() => batch,
                        _ = stop.wait_for(|stop| *stop) => return,
                    };
                    match batch {
                        Ok(Some(batch)) if admit(&store, &engine, &events, &batch) => {}
                        Ok(_) => break None,
                        Err(status) => break Some(format!("the stream: {status}")),
                    }
                }
            }
            Err(refusal) => Some(refusal),
        };
        if let Some(refusal) = refusal {
            events.refused.lock().insert(engine.engine, refusal);
        }
        tokio::select! {
            () = tokio::time::sleep(PROBE_INTERVAL) => {}
            _ = stop.wait_for(|stop| *stop) => return,
        }
    }
}

impl Replica {
    /// Binds and serves a controller named `id`. Replicas given one `lease`
    /// file elect over it; with none the controller leads itself.
    pub async fn start(id: &str, lease: Option<&Path>) -> Result<Self, String> {
        let listener = |name: &str, scopes: &[Scope]| Listener {
            name: name.to_owned(),
            addr: loopback(),
            scopes: scopes.to_vec(),
            tls: None,
        };
        let config = Config {
            controller_id: id.to_owned(),
            listeners: vec![
                listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
                listener("control", &[Scope::Control]),
                listener("replication", &[Scope::Replication]),
            ],
            admin: loopback(),
            decide_shards: Config::default_decide_shards(),
            shutdown_settle: Duration::ZERO,
            shutdown_grace: Duration::from_secs(5),
            load_poll_interval: POLL_INTERVAL,
            election: Election {
                lease: lease.map_or(LeasePlace::Process, |file| {
                    LeasePlace::File(file.to_owned())
                }),
                ..Election::default()
            },
        };
        let mut controller = bounded(
            "the controller binds its listeners",
            ROOM,
            Controller::bind(config),
        )
        .await?
        .map_err(|error| error.to_string())?;
        controller
            .identity_as_told("gateway")
            .map_err(|error| error.to_string())?;
        // The registration calls open to a caller with a principal.
        controller
            .identity_as_told("control")
            .map_err(|error| error.to_string())?;
        let addresses = controller.addresses();
        let gateway = addresses
            .of(Scope::Gateway)
            .ok_or("the controller has no gateway listener")?;
        let control = addresses
            .of(Scope::Control)
            .ok_or("the controller has no control listener")?;
        let store = controller.store();
        let readiness = controller.readiness();
        let (stop, stopped) = oneshot::channel::<()>();
        let mut serve = JoinSet::new();
        serve.spawn(async move {
            controller
                .serve(async move {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Self {
            gateway,
            control,
            store,
            readiness,
            stop: Some(stop),
            serve,
        })
    }

    /// Waits, under a bound, until the controller leads or follows. A serve
    /// that ended first fails the wait with the serve's own end.
    pub async fn reaches(&mut self, state: State) -> Result<(), String> {
        let name = if state == State::Leader {
            "the controller leads"
        } else {
            "the controller follows"
        };
        let reached = bounded(name, ROOM, self.readiness.wait_for(state)).await?;
        if reached == Some(state) {
            return Ok(());
        }
        let ended = self.ended().await;
        Err(format!(
            "the controller ended as {reached:?} before it was {state:?}: {ended}"
        ))
    }

    /// How the serve ended, read under a bound: a serve that ends before
    /// its stop is asked holds the reason in its own result.
    async fn ended(&mut self) -> String {
        match bounded("the serve ends", ROOM, self.serve.join_next()).await {
            Ok(Some(Ok(Ok(())))) => "the serve ended clean".to_owned(),
            Ok(Some(Ok(Err(error)))) => format!("the serve ended with: {error}"),
            Ok(Some(Err(error))) => format!("the serve's task ended with: {error}"),
            Ok(None) => "the serve's task is gone".to_owned(),
            Err(bound) => bound,
        }
    }

    /// The gateway listener's address.
    pub fn gateway(&self) -> SocketAddr {
        self.gateway
    }

    /// Stops the serve under a bound; it must end clean. A serve that
    /// ended before its stop fails with its own end.
    pub async fn stop(mut self) -> Result<(), String> {
        let stop = self.stop.take().ok_or("the serve was stopped already")?;
        if stop.send(()).is_err() {
            let ended = self.ended().await;
            return Err(format!("the serve ended before its stop: {ended}"));
        }
        bounded("the serve ends", ROOM, self.serve.join_next())
            .await?
            .ok_or("the serve's task is gone")?
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())
    }
}

impl Harness {
    /// Starts a lone controller, waits until it leads and connects the
    /// reference client to its gateway listener and the generated client to
    /// its control listener, which takes the name its caller states.
    pub async fn start() -> Result<Self, String> {
        let mut replica = Replica::start("fleet-controller", None).await?;
        replica.reaches(State::Leader).await?;
        let gateway = replica.gateway();
        let client = bounded(
            "the client connects to the gateway listener",
            ROOM,
            Client::connect(&format!("http://{gateway}"), GATEWAY),
        )
        .await?
        .map_err(|error| error.to_string())?;
        let operator = bounded(
            "the operator connects to the control listener",
            ROOM,
            PolicyControllerClient::connect(format!("http://{}", replica.control)),
        )
        .await?
        .map_err(|error| error.to_string())?;
        let (stop_feeds, _) = watch::channel(false);
        Ok(Self {
            store: Arc::clone(&replica.store),
            replica,
            client,
            operator,
            engines: Vec::new(),
            members: HashMap::new(),
            probed: Arc::default(),
            turns: Arc::default(),
            events: Arc::default(),
            feeding: false,
            held: Vec::new(),
            sent: 0,
            generation: 0,
            buffers: ViewBuffers::default(),
            stop_feeds,
            feeds: JoinSet::new(),
        })
    }

    /// Ask: who leads, as the reference client hears it on the gateway
    /// listener.
    pub async fn leader(&mut self) -> Result<LeaderInfo, String> {
        bounded("the leader call answers", ROOM, self.client.leader())
            .await?
            .map_err(|error| error.to_string())
    }

    /// The welcome that answered the harness's client.
    pub fn welcome(&self) -> &Welcome {
        self.client.welcome()
    }

    /// The term floor the harness's client holds.
    pub fn term_floor(&self) -> u64 {
        self.client.term_floor()
    }

    /// The gateway listener's address.
    pub fn gateway(&self) -> SocketAddr {
        self.replica.gateway()
    }

    /// Applies the preset named `preset` as the store's next generation,
    /// in the case's process: no call of the controller applies a spec yet.
    pub fn apply(&mut self, preset: &str) -> Result<(), String> {
        let document = PolicySpec::preset(preset).map_err(|error| error.to_string())?;
        self.apply_document(document)
    }

    /// Applies `document` as the store's next generation, in the case's
    /// process: a form of a preset that the catalog spells under no name.
    pub fn apply_document(&mut self, mut document: PolicySpec) -> Result<(), String> {
        self.generation += 1;
        document.generation = self.generation;
        let spec = AppliedSpec::validate(&document).map_err(|error| error.to_string())?;
        self.store.apply(spec).map_err(|error| error.to_string())
    }

    /// The engine and the rank of the row a decision names by `id`, where
    /// it names a row of the fleet.
    pub fn named(&self, id: u64) -> Option<(usize, u16)> {
        self.members.get(&WorkerId(id)).copied()
    }

    /// Register: starts one mock engine per shape and registers each of its
    /// ranks, pending: only a probe's verdict makes a rank ready. The check
    /// interval and the timeout are the stand-in prober's own, and both
    /// thresholds are one, so one probe turn moves a rank between ready and
    /// not ready; the store steps a rank that stays not ready on to failed
    /// three failures later. Returns the engines' indexes, in the shapes'
    /// order.
    pub async fn register(&mut self, shapes: &[Shape]) -> Result<Vec<usize>, String> {
        let mut started = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let running = start_engine(shape.ranks, shape.runtime).await?;
            let endpoint = running.endpoint.clone();
            let engine = self.engines.len();
            let mut ids = Vec::with_capacity(usize::from(shape.ranks));
            for rank in 0..shape.ranks {
                let id = self.member(&endpoint, shape, rank).await?;
                self.members.insert(id, (engine, rank));
                ids.push(id);
            }
            let probed = Probed {
                engine,
                runtime: shape.runtime,
                endpoint,
                ids,
            };
            if self.feeding && !counts_resident(shape.runtime) {
                self.subscriber(probed.clone());
            }
            self.probed.lock().push(probed);
            self.engines.push(running);
            started.push(engine);
        }
        Ok(started)
    }

    /// One rank's registration. An engine of one rank is an operator's add
    /// on the control listener. The call makes one rank, so a rank of a
    /// wider engine is the store's membership write.
    async fn member(
        &mut self,
        endpoint: &str,
        shape: &Shape,
        rank: u16,
    ) -> Result<WorkerId, String> {
        if shape.ranks == 1 {
            return self.add(endpoint, shape).await;
        }
        let id = self
            .store
            .upsert(WorkerFacts {
                key: format!("{endpoint}@{rank}"),
                engine: endpoint.to_owned(),
                incarnation: 1,
                role: shape.role,
                wire_modes: WireModes::GRPC,
                engine_kind: shape.runtime.as_str().to_owned(),
                models: vec![shape.model.to_owned()],
                dp_rank: rank,
                dp_size: shape.ranks,
                signals: Signals {
                    events: !counts_resident(shape.runtime),
                    ..Signals::default()
                },
                health: health_policy(),
                ..WorkerFacts::default()
            })
            .id();
        Ok(id)
    }

    /// An operator's add of a one-rank engine. The answer is the row the
    /// store holds: registered and pending, under the id a sink names it by.
    async fn add(&mut self, endpoint: &str, shape: &Shape) -> Result<WorkerId, String> {
        let mut request = Request::new(proto::RegisterWorkerRequest {
            spec: Some(proto::WorkerSpec {
                url: endpoint.to_owned(),
                wire: proto::WireMode::Grpc.into(),
                role: wire_role(shape.role).into(),
                runtime: shape.runtime.as_str().to_owned(),
                models: vec![shape.model.to_owned()],
                health_policy: Some(wire_health(health_policy())),
                signals: Some(proto::Signals {
                    events: !counts_resident(shape.runtime),
                    ..proto::Signals::default()
                }),
                ..proto::WorkerSpec::default()
            }),
            mode: Mode::Add.into(),
            wait_ready: false,
        });
        request
            .metadata_mut()
            .insert(PRINCIPAL, AsciiMetadataValue::from_static(OPERATOR));
        let answer = bounded(
            "the registration call answers",
            ROOM,
            self.operator.register_worker(request),
        )
        .await?
        .map_err(|status| {
            format!(
                "the add of {endpoint} was refused: {:?}: {}",
                status.code(),
                status.message()
            )
        })?
        .into_inner();
        match answer.event {
            Some(Event::Registered(row)) if row.status() == proto::WorkerStatus::Pending => {
                Ok(WorkerId(row.id))
            }
            other => Err(format!("the add of {endpoint} answered {other:?}")),
        }
    }

    /// Feed: starts the stand-in prober and one stand-in subscriber an
    /// engine, over every engine registered now and later. Loads need no
    /// start: the controller's own driver polls every ready engine.
    pub fn feed(&mut self) {
        self.feeds.spawn(probe(
            Arc::clone(&self.store),
            Arc::clone(&self.probed),
            Arc::clone(&self.turns),
            self.stop_feeds.subscribe(),
        ));
        let engines = self.probed.lock().clone();
        for engine in engines {
            if !counts_resident(engine.runtime) {
                self.subscriber(engine);
            }
        }
        self.feeding = true;
    }

    /// Starts the stand-in subscriber of one engine.
    fn subscriber(&mut self, engine: Probed) {
        self.feeds.spawn(subscribe(
            Arc::clone(&self.store),
            engine,
            Arc::clone(&self.events),
            self.stop_feeds.subscribe(),
        ));
    }

    /// The first chain the first rank of the engine at `engine` stored, in
    /// the chain's order, each block by the hash of its tokens: what a
    /// request that brings the same prompt states of its head.
    pub fn head(&self, engine: usize) -> Vec<u64> {
        let streams = self.events.rows.lock();
        self.members
            .iter()
            .find(|(_, (of, rank))| *of == engine && *rank == 0)
            .and_then(|(id, _)| streams.get(id))
            .map(|stream| {
                stream
                    .head
                    .iter()
                    .map(|block| block.content_hash.0)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The KV events of the engine at `engine`, a rank a row: what each
    /// stream delivered beside what the store holds of it.
    pub fn heard(&self, engine: usize) -> Vec<Heard> {
        let streams = self.events.rows.lock();
        let mut rows: Vec<Heard> = self
            .members
            .iter()
            .filter(|(_, (of, _))| *of == engine)
            .map(|(id, (_, rank))| {
                let stream = streams.get(id);
                Heard {
                    rank: *rank,
                    last: stream.map_or(0, |stream| stream.last),
                    applied: stream.map_or(0, |stream| stream.applied),
                    held: stream.map_or(0, |stream| stream.held.len()),
                    indexed: self.store.indexed_blocks(*id, *rank),
                    counts: self.store.lane_counts(*id, *rank),
                    cursor: self.store.cursor(*id, *rank),
                }
            })
            .collect();
        rows.sort_unstable_by_key(|row| row.rank);
        rows
    }

    /// What refuses the event stream of the engine at `engine`, in the
    /// subscriber's words; `None` when nothing refuses it.
    pub fn refused(&self, engine: usize) -> Option<String> {
        self.events.refused.lock().get(&engine).cloned()
    }

    /// The gaps the KV sink named on any engine's stream.
    pub fn gaps(&self) -> u64 {
        self.events.gaps.load(Ordering::SeqCst)
    }

    /// The snapshot chunks any engine's stream carried.
    pub fn snapshot_chunks(&self) -> u64 {
        self.events.snapshots.load(Ordering::SeqCst)
    }

    /// Patches the overload thresholds of every rank of the engine at
    /// `engine` on the control listener, each under the revision its row
    /// stands at: an operator's patch. The engine's next overload verdict
    /// reads the thresholds it left.
    pub async fn patch_overload(
        &mut self,
        engine: usize,
        overload: proto::OverloadThresholds,
    ) -> Result<(), String> {
        let ids: Vec<WorkerId> = self
            .members
            .iter()
            .filter(|(_, (of, _))| *of == engine)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let row = self
                .store
                .get(id)
                .ok_or("a rank of the engine holds no row")?;
            let mut request = Request::new(proto::PatchWorkerRequest {
                worker_id: id.0,
                overload: Some(overload),
                revision: row.revision,
                ..proto::PatchWorkerRequest::default()
            });
            request
                .metadata_mut()
                .insert(PRINCIPAL, AsciiMetadataValue::from_static(OPERATOR));
            bounded(
                "the patch call answers",
                ROOM,
                self.operator.patch_worker(request),
            )
            .await?
            .map_err(|status| {
                format!(
                    "the patch was refused: {:?}: {}",
                    status.code(),
                    status.message()
                )
            })?;
        }
        Ok(())
    }

    /// The pool serving `model` in `role` as a decision's view reads it at
    /// this instant, each row named by its engine. Until the service
    /// answers decisions this is what a case reads back.
    pub fn view(&mut self, model: &str, role: Role) -> Result<Vec<Seen>, String> {
        let ask = Ask {
            gateway: GatewayId::LOCAL,
            now: Instant::now(),
            pinned: None,
            affinity: false,
        };
        if self
            .store
            .build_view(model, role, &ask, &mut self.buffers)
            .is_none()
        {
            return Ok(Vec::new());
        }
        self.buffers
            .candidates
            .iter()
            .map(|row| {
                let (engine, rank) = *self.members.get(&row.id).ok_or_else(|| {
                    format!("the view holds a row no registration made: {:?}", row.id)
                })?;
                Ok(Seen {
                    engine,
                    rank,
                    healthy: row.health.healthy,
                    overloaded: row.health.overloaded,
                    load: row.load,
                })
            })
            .collect()
    }

    /// Holds `count` requests in flight on the engine at `engine`, on the
    /// engine's own wire: its reports then say so by themselves. A held
    /// stream is never read and is dropped with the harness.
    pub async fn hold(&mut self, engine: usize, count: usize) -> Result<(), String> {
        let (endpoint, runtime) = self
            .engines
            .get(engine)
            .map(|engine| (engine.endpoint.clone(), engine.runtime))
            .ok_or_else(|| format!("no engine {engine}"))?;
        if counts_resident(runtime) {
            return self.hold_counting(engine, &endpoint, count).await;
        }
        let client = bounded(
            "the holder connects to its engine",
            ROOM,
            VllmEngineClient::connect(&endpoint),
        )
        .await?
        .map_err(|error| error.to_string())?;
        for request in 0..count {
            let asked = client.generate(vllm_proto::GenerateRequest {
                request_id: format!("held-{engine}-{request}"),
                input: Some(vllm_proto::generate_request::Input::Tokenized(
                    vllm_proto::TokenizedInput {
                        original_text: String::new(),
                        input_ids: vec![1; 64],
                    },
                )),
                sampling_params: Some(vllm_proto::SamplingParams {
                    max_tokens: Some(HELD_TOKENS),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            });
            let stream = bounded("an engine accepts a held request", ROOM, asked)
                .await?
                .map_err(|status| status.to_string())?;
            self.held.push(Box::new(stream));
        }
        Ok(())
    }

    /// [`Harness::hold`] on the wire whose load report counts resident
    /// tokens: the same requests in that wire's form.
    async fn hold_counting(
        &mut self,
        engine: usize,
        endpoint: &str,
        count: usize,
    ) -> Result<(), String> {
        let client = bounded(
            "the holder connects to its engine",
            ROOM,
            TokenSpeedSchedulerClient::connect(endpoint),
        )
        .await?
        .map_err(|error| error.to_string())?;
        for request in 0..count {
            let asked = client.generate_framed(tokenspeed_proto::GenerateRequest {
                request_id: format!("held-{engine}-{request}"),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: vec![1; 64],
                    ..Default::default()
                }),
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    max_new_tokens: Some(HELD_TOKENS),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            });
            let stream = bounded("an engine accepts a held request", ROOM, asked)
                .await?
                .map_err(|status| status.to_string())?;
            self.held.push(Box::new(stream));
        }
        Ok(())
    }

    /// Sends one short request to the engine at `engine` on the engine's own
    /// wire and reads it to its end, as a gateway that decided by itself
    /// sends one, the whole under [`DIAL_TIMEOUT`]. Then it reports the
    /// request on the client's session: a `Dispatched`, and a `Completed`
    /// for the engine's first rank under the class the transport read: a
    /// stream read to its end, a connection refused, or the bound run out.
    /// The lease is the harness's own number: no leader issued it. Returns
    /// the class.
    pub async fn send(&mut self, engine: usize) -> Result<Class, String> {
        let endpoint = self
            .engines
            .get(engine)
            .map(|engine| engine.endpoint.clone())
            .ok_or_else(|| format!("no engine {engine}"))?;
        let served = self.worker(engine)?;
        self.sent += 1;
        let lease_id = self.sent;
        let started = Instant::now();
        let request = vllm_proto::GenerateRequest {
            request_id: format!("sent-{engine}-{lease_id}"),
            input: Some(vllm_proto::generate_request::Input::Tokenized(
                vllm_proto::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1; 8],
                },
            )),
            sampling_params: Some(vllm_proto::SamplingParams {
                max_tokens: Some(SENT_TOKENS),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let asked = tokio::time::timeout(DIAL_TIMEOUT, async {
            let Ok(client) = VllmEngineClient::connect(&endpoint).await else {
                return (Class::ConnectFailed, 0);
            };
            let Ok(mut stream) = client.generate(request).await else {
                return (Class::ServerError, 0);
            };
            let mut frames = 0;
            loop {
                match stream.next().await {
                    Some(Ok(_)) => frames += 1,
                    Some(Err(_)) => return (Class::ServerError, frames),
                    None => return (Class::Ok, frames),
                }
            }
        })
        .await;
        let (class, frames) = asked.unwrap_or((Class::Timeout, 0));
        let side = Side::Single.into();
        let dispatched = Dispatched {
            lease_id,
            target_index: 0,
            side,
            since_decision_micros: 0,
        };
        bounded(
            "the client reports a dispatch",
            ROOM,
            self.client.dispatched(dispatched),
        )
        .await?
        .map_err(|error| error.to_string())?;
        let completed = Completed {
            lease_id,
            side,
            served: served.0,
            // The gRPC code the request ended with.
            status: match class {
                Class::Ok => 0,
                Class::Timeout => 4,
                Class::ConnectFailed => 14,
                _ => 2,
            },
            class: class.into(),
            ttft_micros: 0,
            tokens_in: 8,
            tokens_out: frames,
            duration_micros: u32::try_from(started.elapsed().as_micros()).unwrap_or(u32::MAX),
        };
        bounded(
            "the client reports a completion",
            ROOM,
            self.client.completed(completed),
        )
        .await?
        .map_err(|error| error.to_string())?;
        Ok(class)
    }

    /// The client: a case reports what the harness has no form for.
    pub fn client(&mut self) -> &mut Client {
        &mut self.client
    }

    /// The leases the store holds for the gateway named `gateway`, issued
    /// or dispatched: each books a request in flight on its worker.
    pub fn leases(&self, gateway: &str) -> usize {
        let gateway = GatewayId(fingerprint(gateway.as_bytes()));
        // By the trait's path: in scope, its `get` would meet the rows' own.
        smg_policy_controller::store::LeasesStore::live(self.store.as_ref(), gateway)
    }

    /// Names the dispatch of the client's lease `lease` on its session, to
    /// the pick's first target, and waits until the store reads the lease
    /// dispatched: no dispatch deadline ends it then, so it books its
    /// worker until its gateway gives it up.
    pub async fn dispatch(&mut self, lease: u64) -> Result<(), String> {
        self.dispatch_at(lease, 0).await
    }

    /// As [`Harness::dispatch`], on the target at `target` of the lease's
    /// pick, best first: a gateway that sent its request past the first
    /// target names the one it sent to.
    pub async fn dispatch_at(&mut self, lease: u64, target: u32) -> Result<(), String> {
        let dispatched = Dispatched {
            lease_id: lease,
            target_index: target,
            side: Side::Single.into(),
            since_decision_micros: 0,
        };
        bounded(
            "the client reports a dispatch",
            ROOM,
            self.client.dispatched(dispatched),
        )
        .await?
        .map_err(|error| error.to_string())?;
        let gateway = GatewayId(fingerprint(GATEWAY.as_bytes()));
        let store = self.store.as_ref();
        within("the store reads the lease dispatched", ROOM, || {
            // By the trait's path, as `leases` reads the file.
            smg_policy_controller::store::LeasesStore::get(store, gateway, LeaseId(lease))
                .is_some_and(|record| matches!(record.state, LeaseState::Dispatched { .. }))
        })
        .await
        .map(|_| ())
    }

    /// The row the store's pin of record names for `session`, the value a
    /// lease states of its request's session; `None` where the store pins
    /// the session nowhere.
    pub fn pinned(&self, session: u64) -> Option<(usize, u16)> {
        // By the trait's path, as `leases` reads the file.
        let pin =
            smg_policy_controller::store::SessionsStore::lookup(self.store.as_ref(), session)?;
        self.named(pin.worker.0)
    }

    /// Whether the store holds a completion of class ok by the gateway named
    /// `gateway` on the first rank of the engine at `engine`, at or after
    /// `since`: the read a decision's precedence rule makes of one
    /// gateway's own dispatches.
    pub fn completed_ok(&self, gateway: &str, engine: usize, since: Instant) -> bool {
        self.worker(engine).is_ok_and(|served| {
            let gateway = GatewayId(fingerprint(gateway.as_bytes()));
            self.store.completed_ok_since(gateway, served, since)
        })
    }

    /// The row id of the first rank of the engine at `engine`.
    pub fn worker(&self, engine: usize) -> Result<WorkerId, String> {
        self.members
            .iter()
            .find(|(_, member)| **member == (engine, 0))
            .map(|(id, _)| *id)
            .ok_or_else(|| format!("engine {engine} has no registered first rank"))
    }

    /// Kills the engine at `engine`: its runtime is dropped and the call
    /// returns once its thread has ended and a connection the harness held
    /// to it read its end, so every socket of the engine closed. A hung
    /// engine is let go first. The thread is never joined: a thread that
    /// did not end would hold the test process at its exit.
    pub async fn kill(&mut self, engine: usize) -> Result<Death, String> {
        let slot = self
            .engines
            .get_mut(engine)
            .ok_or_else(|| format!("no engine {engine}"))?;
        let (Some(kill), Some(ended), Some(mut witness)) =
            (slot.kill.take(), slot.ended.take(), slot.witness.take())
        else {
            return Err(format!("engine {engine} is dead already"));
        };
        // Both are read before a hung engine is let go: its release runs
        // the thread to its end.
        let at = Instant::now();
        let asked = self.turns.begun.load(Ordering::SeqCst);
        slot.hang = None;
        slot.release = None;
        drop(kill);
        // The sender's drop is the thread's end: the wait ends on the error.
        let _ = bounded("a killed engine's thread ends", ROOM, ended).await?;
        // The proof of death is a socket of the harness's own: it reads to
        // its end once the engine's side closed. A reset is an end too.
        let _ = bounded(
            "a killed engine's sockets close",
            ROOM,
            copy(&mut witness, &mut sink()),
        )
        .await?;
        Ok(Death {
            at,
            asked,
            turn: self.turns.begun.load(Ordering::SeqCst),
        })
    }

    /// Hangs the engine at `engine`: its thread stops, so nothing of it
    /// answers, and every socket of it stays open. A kill ends the hang.
    pub fn hang(&mut self, engine: usize) -> Result<(), String> {
        let slot = self
            .engines
            .get_mut(engine)
            .ok_or_else(|| format!("no engine {engine}"))?;
        let hang = slot
            .hang
            .take()
            .ok_or_else(|| format!("engine {engine} hangs already, or is dead"))?;
        let (release, released) = mpsc::channel();
        hang.send(released)
            .map_err(|_| format!("engine {engine} ended before it hung"))?;
        slot.release = Some(release);
        Ok(())
    }

    /// The first turn, from `turn` on, whose dial of the engine at `engine`
    /// heard it unhealthy.
    pub fn first_unhealthy(&self, engine: usize, turn: u64) -> Option<u64> {
        self.turns
            .dialed
            .lock()
            .iter()
            .find(|dial| dial.engine == engine && dial.turn >= turn && !dial.end.healthy())
            .map(|dial| dial.turn)
    }

    /// The dials of the turn numbered `turn` so far, in the order dialed:
    /// each engine, and how its dial ended.
    pub fn dialed(&self, turn: u64) -> Vec<(usize, DialEnd)> {
        self.turns
            .dialed
            .lock()
            .iter()
            .filter(|dial| dial.turn == turn)
            .map(|dial| (dial.engine, dial.end))
            .collect()
    }

    /// How the dials of the engine at `engine` ended, from the turn
    /// numbered `turn` on, in the order dialed.
    pub fn ends(&self, engine: usize, turn: u64) -> Vec<DialEnd> {
        self.turns
            .dialed
            .lock()
            .iter()
            .filter(|dial| dial.engine == engine && dial.turn >= turn)
            .map(|dial| dial.end)
            .collect()
    }

    /// Opens a connection to the engine at `engine` and closes it: an
    /// engine that listens takes it, though it answers nothing.
    pub async fn connect(&self, engine: usize) -> Result<(), String> {
        let slot = self
            .engines
            .get(engine)
            .ok_or_else(|| format!("no engine {engine}"))?;
        bounded(
            "an engine takes a connection",
            ROOM,
            TcpStream::connect(slot.addr),
        )
        .await?
        .map(drop)
        .map_err(|error| error.to_string())
    }

    /// The statuses the stand-in prober moved so far, in the order written.
    pub fn verdicts(&self) -> Vec<Verdict> {
        self.turns.verdicts.lock().clone()
    }

    /// Ends the feeds, the held requests, the engines and the controller,
    /// each under a bound; the serve must end clean.
    pub async fn stop(mut self) -> Result<(), String> {
        self.stop_feeds.send_replace(true);
        while let Some(joined) = bounded("a feed ends", ROOM, self.feeds.join_next()).await? {
            joined.map_err(|error| error.to_string())?;
        }
        self.held.clear();
        for engine in 0..self.engines.len() {
            if self
                .engines
                .get(engine)
                .is_some_and(|slot| slot.kill.is_some())
            {
                self.kill(engine).await?;
            }
        }
        self.replica.stop().await
    }
}
