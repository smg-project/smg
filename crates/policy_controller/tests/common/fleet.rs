//! The end-to-end harness: a controller started on ephemeral loopback ports
//! in the case's own process, a fleet of mock engines beside it, and one
//! handle, [`Harness`], through which a case registers the fleet, feeds the
//! store and asks the service through the crate's reference client.
//! Included by path beside `common/engines.rs` and `common/net.rs` (the
//! including binary names them `engines` and `net`).
//!
//! Every fact the store holds of an engine came from that engine, through a
//! sink. Loads are the controller's own load monitor's: the harness writes
//! none. Health and KV events are the controller's own where its build
//! has the `worker-host` feature: its prober and its feed are then the
//! store's one writer of each, and the harness only listens. Its dialer
//! ([`probe`]) asks each engine its wire's health check once an interval
//! and keeps how the dial ended, and a listener ([`subscribe`]) holds one
//! event stream an engine and keeps what the stream said. A build without
//! that feature has no driver of either, so there each stands in for one.
//! The dialer writes what it heard, never what the case knows, and the
//! listener decodes each batch into the events an index applies and hands
//! it to the sink. A dial is bounded, so an engine that stops answering
//! costs a turn that bound and no more.
//!
//! An engine speaks the wire of its kind ([`Shape::runtime`]): the fleet's
//! first wire, or the one whose load report counts resident tokens. The
//! controller's load monitor polls each through that kind's own arm, the
//! prober asks each its own health check, and the subscriber speaks the
//! first wire alone.
//!
//! A fleet registers as an operator's does ([`Harness::register`]): an
//! engine of one rank is an add on the control listener, under the name the
//! harness states there. The call makes one rank, so a rank of a wider
//! engine is still the store's membership write. The rows a case reads
//! ([`Harness::pool`]) are the store's too: its workers' table and its load
//! table, as the controller's own writers left them. A case calls the
//! handle alone, so a move of either rewrites no case.

use std::{
    collections::{HashMap, HashSet},
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, SocketAddr},
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
    document::{AppliedSpec, PolicySpec},
    engine::{fingerprint, LoadRow, Role, SignalSource, WireModes, WorkerId},
    load::load_row,
    service::{
        client::Client,
        controller::{
            Config, Controller, ElectionConfig, Error, LeasePlace, Listener, Readiness, Scope,
            State,
        },
        proto::{
            self, completed::Class, policy_controller_client::PolicyControllerClient,
            register_worker_request::Mode, worker_event::Event, Completed, Dispatched, LeaderInfo,
            Side, Welcome,
        },
    },
    stages::StageRegistry,
    store::{
        BatchFacts, BatchOutcome, GatewayId, HealthPolicy, HealthSink, KvCursor, KvEvent,
        KvEventSink, LaneCounts, LeaseState, MembershipChange, MembershipTail, MemoryStore,
        ResyncReason, SequenceHash, Signals, StoredBlock, WorkerFacts, WorkerStatus, WorkersStore,
    },
    tracking::LeaseId,
};
use smg_worker::load::state::LoadReceiver;
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
    pair,
};

/// The leader's address and the standby's of a pair: one host, two
/// addresses, so both listen on one replication port as replicas
/// configured alike do. Linux routes the second with no alias; another
/// host refuses its bind, and a case that binds it is ignored there.
const LEADER: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const STANDBY: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));

/// How often the stand-in prober asks every engine: the check interval of
/// every registration.
pub const PROBE_INTERVAL: Duration = Duration::from_millis(500);

/// The bound on one dial of an engine, connect included: the timeout of
/// every registration.
pub const DIAL_TIMEOUT: Duration = Duration::from_secs(1);

/// How often the controller's load monitor polls an engine.
pub const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Room for a loaded host on top of a wait's own timing.
pub const ROOM: Duration = Duration::from_secs(30);

/// Whether the controller under test feeds its own cache index: a build
/// with the `worker-host` feature streams each engine's events into its
/// store, and the harness then only listens.
const PRODUCT_FEEDS: bool = cfg!(feature = "worker-host");

/// Whether the controller under test probes its own workers: a build with
/// the `worker-host` feature probes each while it leads, and the harness's
/// dialer then writes nothing.
pub const PRODUCT_PROBES: bool = cfg!(feature = "worker-host");

/// Every reason the store counts a resync under.
const RESYNCS: [ResyncReason; 9] = [
    ResyncReason::OutOfRange,
    ResyncReason::DataLoss,
    ResyncReason::StreamEnded,
    ResyncReason::SnapshotAbandoned,
    ResyncReason::RelayRestart,
    ResyncReason::EngineReplaced,
    ResyncReason::PublisherRestart,
    ResyncReason::GapCleared,
    ResyncReason::Snapshot,
];

/// A reason's place in [`RESYNCS`]. The match names every reason, so a new
/// one fails to build here, beside the list that must name it.
const fn place(reason: ResyncReason) -> usize {
    match reason {
        ResyncReason::OutOfRange => 0,
        ResyncReason::DataLoss => 1,
        ResyncReason::StreamEnded => 2,
        ResyncReason::SnapshotAbandoned => 3,
        ResyncReason::RelayRestart => 4,
        ResyncReason::EngineReplaced => 5,
        ResyncReason::PublisherRestart => 6,
        ResyncReason::GapCleared => 7,
        ResyncReason::Snapshot => 8,
    }
}

// Every place of the list holds its reason, read at the build.
const _: () = {
    let mut at = 0;
    while at < RESYNCS.len() {
        assert!(
            place(RESYNCS[at]) == at,
            "a resync reason is out of its place"
        );
        at += 1;
    }
};

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

/// One row of a pool as the store holds it, named by the engine that
/// registered it.
#[derive(Clone, Copy, Debug)]
pub struct Seen {
    /// The engine's index in the harness, in registration order.
    pub engine: usize,
    pub rank: u16,
    pub healthy: bool,
    /// The load monitor's newest report of the row, where it has one.
    pub load: Option<LoadRow>,
}

/// One status the store moved, whoever probed: the engine of the row, where
/// its status came from and where it went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Moved {
    pub engine: usize,
    pub from: WorkerStatus,
    pub to: WorkerStatus,
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

/// One rank's KV events: what the listener's stream delivered beside what
/// the store holds of it, read at one instant. Where the controller feeds
/// itself the two come by streams of their own, so they agree only at rest.
#[derive(Clone, Copy, Debug)]
pub struct Heard {
    /// The engine's data-parallel rank the row stands for.
    pub rank: u16,
    /// The last sequence the engine's stream delivered for the rank; `0`
    /// when none came.
    pub last: u64,
    /// The blocks the stream's events left the engine holding.
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

impl Engine {
    /// Kills the engine: its runtime is dropped, and the call returns once
    /// its thread has ended and the connection the harness held to it read
    /// its end, so every socket of the engine closed. A hung engine is let
    /// go first. `false` for an engine that is dead already.
    async fn die(&mut self) -> Result<bool, String> {
        let (Some(kill), Some(ended), Some(mut witness)) =
            (self.kill.take(), self.ended.take(), self.witness.take())
        else {
            return Ok(false);
        };
        self.hang = None;
        self.release = None;
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
        Ok(true)
    }
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

/// What the listeners heard, a row at a time, and the gaps the sink named
/// to a listener that stands in for a driver. There the rows' one lock is
/// held across a batch's hand-over, so a read under it sees the store and
/// every row in step. Where the controller feeds itself the store has its
/// own writer, and a read sees the two in step only at rest.
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
    /// The blocks the stream's events left the engine holding, by its hashes:
    /// every batch's where the controller feeds itself, the batches the sink
    /// applied otherwise.
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
    replication: SocketAddr,
    store: Arc<MemoryStore>,
    /// The channel of the controller's load monitor: what a decision is
    /// given of its workers' load.
    loads: LoadReceiver,
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
    stop_feeds: watch::Sender<bool>,
    feeds: JoinSet<()>,
    /// Whether the controller the harness asks elects over a lease file: a
    /// leader that elects can stop leading for a moment.
    elects: bool,
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
/// under its dial bound. A controller that probes by itself counts both in
/// whole seconds: it asks an engine once a second.
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

/// One rank written into the store by its membership write, pending: a
/// rank of a wider engine under a controller.
fn written(store: &MemoryStore, endpoint: &str, shape: &Shape, rank: u16) -> WorkerId {
    store
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
        .id()
}

/// Sends one short request named `id` to the engine at `endpoint` on the
/// fleet's first wire and reads it to its end, the whole under
/// [`DIAL_TIMEOUT`]: the class the transport read (a stream read to its
/// end, a connection refused or the bound run out) and the frames read.
async fn sent(endpoint: &str, id: String) -> (Class, u32) {
    let request = vllm_proto::GenerateRequest {
        request_id: id,
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
        let Ok(client) = VllmEngineClient::connect(endpoint).await else {
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
    asked.unwrap_or((Class::Timeout, 0))
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

/// The harness's dialer: every [`PROBE_INTERVAL`] one turn asks each engine
/// once and keeps how the dial ended. Where it `writes` it stands in for
/// the health prober: it writes what the dial heard for each of the
/// engine's rows, under the row's revision, and a status that moved is kept
/// with the turn that moved it.
async fn probe(
    store: Arc<MemoryStore>,
    probed: Arc<Mutex<Vec<Probed>>>,
    turns: Arc<Turns>,
    mut stop: watch::Receiver<bool>,
    writes: bool,
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
            if !writes {
                continue;
            }
            for id in ids {
                let Some(row) = store.get(id) else { continue };
                // A verdict is pushed under the lock its write was made
                // under: a reader that saw the write reads the verdict.
                let mut verdicts = turns.verdicts.lock();
                if let Some((from, to)) = store
                    .probe_outcome(id, healthy, row.revision, &|| true)
                    .moved()
                {
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

/// One batch as its rank's row keeps it: the last sequence, and the blocks
/// its events leave the engine holding. Where the controller feeds itself
/// the store is handed nothing here. Otherwise the batch goes, decoded,
/// through its rank's cursor into its row's index, with the header the
/// batch carries, and a batch the sink did not apply leaves the blocks as
/// they were. A batch that carries a load record alone is passed by, and
/// its record is not written here: loads are the monitor's. A snapshot chunk
/// is a relay's: its start is a collector's word to every row of the
/// engine, which this listener does not say, so a chunk is counted and
/// passed by. Returns `false` at a gap the sink named.
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
    if !PRODUCT_FEEDS {
        match store.admit(id, rank, facts, &decoded) {
            BatchOutcome::Applied => {}
            BatchOutcome::Skipped => return true,
            BatchOutcome::Gap { .. } => {
                events.gaps.fetch_add(1, Ordering::SeqCst);
                return false;
            }
        }
    }
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

/// The case's listener on the KV events: one stream an engine, every rank's
/// batches on it, each kept as it arrives ([`admit`]). Where the controller
/// feeds itself the stream is asked from zero each time it opens, and the
/// engine's rows start over. Where the listener stands in for a driver, a
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
            [only] if !PRODUCT_FEEDS => store.cursor(*only, 0).map_or(0, |cursor| cursor.applied),
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
                if PRODUCT_FEEDS {
                    // The engine replays from zero: its rows start over,
                    // and the store is told nothing.
                    let mut rows = events.rows.lock();
                    for id in &engine.ids {
                        rows.remove(id);
                    }
                } else {
                    for id in &engine.ids {
                        store.reconnected(*id);
                    }
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
        Self::start_over(id, lease, StageRegistry::builtin(), None).await
    }

    /// As [`Replica::start`], over the stages `catalog` holds, a host's own
    /// among them. With `spec` the controller takes its policy document
    /// from that file and applies it itself, as a host's own build does: a
    /// document names a host's stage where the controller bound over that
    /// catalog validates it, and nowhere else.
    pub async fn start_over(
        id: &str,
        lease: Option<&Path>,
        catalog: StageRegistry,
        spec: Option<&Path>,
    ) -> Result<Self, String> {
        Self::serve(id, lease, loopback().ip(), 0, false, (Some(catalog), spec)).await
    }

    /// As [`Replica::start`], with every listener on `ip` and the
    /// replication listener at `port`. Replicas of one lease listen on one
    /// replication port, each at an address of its own, as replicas
    /// configured alike do: the one that follows finds its leader's stream
    /// there, and the leader takes its name as told. `0` takes a number of
    /// the pairs' own ([`pair::ports`]), the next one free on `ip` and, at
    /// the leader's address, on the standby's too ([`pair::free`]): a bind
    /// of port `0` is handed none of them, so the number stays free on the
    /// address a standby binds it at.
    pub async fn start_at(
        id: &str,
        lease: Option<&Path>,
        ip: IpAddr,
        port: u16,
    ) -> Result<Self, String> {
        Self::serve(id, lease, ip, port, true, (None, None)).await
    }

    /// Binds and serves the controller over the stages `catalog` holds,
    /// the stock build's where it names none, and on the `spec` file where
    /// one is given; with `standbys` its replication listener takes a
    /// standby's name as told.
    async fn serve(
        id: &str,
        lease: Option<&Path>,
        ip: IpAddr,
        port: u16,
        standbys: bool,
        (catalog, spec): (Option<StageRegistry>, Option<&Path>),
    ) -> Result<Self, String> {
        let listener = |name: &str, scopes: &[Scope], port: u16| Listener {
            name: name.to_owned(),
            addr: SocketAddr::new(ip, port),
            scopes: scopes.to_vec(),
            tls: None,
        };
        let config = |port: u16| Config {
            controller_id: id.to_owned(),
            listeners: vec![
                listener("gateway", &[Scope::Gateway, Scope::Ingestion], 0),
                listener("control", &[Scope::Control], 0),
                listener("replication", &[Scope::Replication], port),
            ],
            admin: SocketAddr::new(ip, 0),
            decide_shards: Config::default_decide_shards(),
            shutdown_settle: Duration::ZERO,
            shutdown_grace: Duration::from_secs(5),
            load_poll_interval: POLL_INTERVAL,
            discovery: None,
            election: ElectionConfig {
                lease: lease.map_or(LeasePlace::Process, |file| {
                    LeasePlace::File(file.to_owned())
                }),
                ..ElectionConfig::default()
            },
        };
        // A pair's first replica draws its replication port from the pairs'
        // own numbers and takes the next one free on its address. A bind
        // takes its catalog: the stock build's is made again for each
        // number, and a host's own is bound once, at the port its start
        // names.
        let drawn = standbys && port == 0 && catalog.is_none();
        let mut numbers = pair::ports(ip);
        let mut host = catalog;
        let mut controller = loop {
            let port = if drawn {
                numbers
                    .next()
                    .ok_or("no port among the pairs' own is free")?
            } else {
                port
            };
            let catalog = host.take().unwrap_or_else(StageRegistry::builtin);
            let bound = bounded(
                "the controller binds its listeners",
                ROOM,
                Controller::bind_with(config(port), catalog),
            )
            .await?;
            match bound {
                // In use on this address: the next number.
                Err(Error::Bind { source, .. })
                    if drawn && source.kind() == ErrorKind::AddrInUse => {}
                // Held on the standby's address by a standby that outlived
                // its leader: the next number, and this bind is let go.
                Ok(_)
                    if drawn
                        && ip == LEADER
                        && !pair::free(STANDBY, port).map_err(|error| error.to_string())? => {}
                bound => break bound.map_err(|error| error.to_string())?,
            }
        };
        controller
            .identity_as_told("gateway")
            .map_err(|error| error.to_string())?;
        // The registration calls open to a caller with a principal.
        controller
            .identity_as_told("control")
            .map_err(|error| error.to_string())?;
        // A standby states its name on its stream, as a gateway does.
        if standbys {
            controller
                .identity_as_told("replication")
                .map_err(|error| error.to_string())?;
        }
        if let Some(spec) = spec {
            bounded(
                "the controller takes its spec file",
                ROOM,
                controller.spec_file(spec),
            )
            .await?
            .map_err(|error| error.to_string())?;
        }
        let addresses = controller.addresses();
        let gateway = addresses
            .of(Scope::Gateway)
            .ok_or("the controller has no gateway listener")?;
        let control = addresses
            .of(Scope::Control)
            .ok_or("the controller has no control listener")?;
        let replication = addresses
            .of(Scope::Replication)
            .ok_or("the controller has no replication listener")?;
        let store = controller.store();
        let loads = controller.loads();
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
            replication,
            store,
            loads,
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
        let replica = Replica::start("fleet-controller", None).await?;
        Self::around(replica).await
    }

    /// Starts a lone controller of a host's own build: over the stages
    /// `catalog` holds and on the policy document at `spec`, which the
    /// controller reads and applies itself. The case applies no generation
    /// in its own process beside it.
    #[cfg(feature = "wasm")]
    pub async fn start_hosting(catalog: StageRegistry, spec: &Path) -> Result<Self, String> {
        let replica = Replica::start_over("fleet-controller", None, catalog, Some(spec)).await?;
        Self::around(replica).await
    }

    /// The harness around a controller once it leads: the reference client
    /// at its gateway listener and the generated client at its control
    /// listener.
    async fn around(mut replica: Replica) -> Result<Self, String> {
        replica.reaches(State::Leader).await?;
        let (client, operator) = Self::clients(&replica).await?;
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
            stop_feeds,
            feeds: JoinSet::new(),
            elects: false,
        })
    }

    /// Starts two controllers over the lease file `lease`, each at an
    /// address of its own on one replication port, waits until the first
    /// leads and the second follows, and returns the harness around the
    /// leader with its standby.
    pub async fn start_pair(lease: &Path) -> Result<(Self, Replica), String> {
        let mut leader = Replica::start_at("fleet-leader", Some(lease), LEADER, 0).await?;
        leader.reaches(State::Leader).await?;
        let port = leader.replication.port();
        let mut standby = Replica::start_at("fleet-standby", Some(lease), STANDBY, port).await?;
        standby.reaches(State::Follower).await?;
        let mut harness = Self::around(leader).await?;
        harness.elects = true;
        Ok((harness, standby))
    }

    /// The reference client at `replica`'s gateway listener and the
    /// generated client at its control listener.
    async fn clients(
        replica: &Replica,
    ) -> Result<(Client, PolicyControllerClient<Channel>), String> {
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
        Ok((client, operator))
    }

    /// How many of the fleet's ranks `standby`'s own store holds ready:
    /// what a controller that follows has of its leader's log. A row no
    /// registration made is an error, and so is a load report: a leader
    /// sends a standby none, and a standby polls nobody.
    pub fn followed(&self, standby: &Replica) -> Result<usize, String> {
        let cut = standby.store.checkpoint();
        let reports = standby.loads.borrow();
        let mut held = 0;
        for row in &cut.rows {
            if reports.report(&row.facts.key).is_some() {
                return Err(format!("the standby holds a load report: {:?}", row.id));
            }
            if !self.members.contains_key(&row.id) {
                return Err(format!(
                    "the standby holds a row no registration made: {:?}",
                    row.id
                ));
            }
            if row.status == WorkerStatus::Ready {
                held += 1;
            }
        }
        Ok(held)
    }

    /// Stops the controller the harness asks, the planned stop that
    /// releases the lease, and turns the harness to `standby` once it
    /// leads: the two clients connect to its listeners and the store read
    /// from here on is its own. The registrations stand, since a standby
    /// holds its leader's rows under their ids. A stand-in prober goes on
    /// writing to the stopped controller's store, so there the promoted
    /// one's rows keep the status its leader's log told. A controller that
    /// probes by itself probes its rows from its take on.
    pub async fn hand_over(&mut self, standby: Replica) -> Result<(), String> {
        let stopped = std::mem::replace(&mut self.replica, standby);
        stopped.stop().await?;
        self.replica.reaches(State::Leader).await?;
        let (client, operator) = Self::clients(&self.replica).await?;
        self.store = Arc::clone(&self.replica.store);
        self.client = client;
        self.operator = operator;
        Ok(())
    }

    /// Whether the controller the harness asks elects over a lease file.
    pub const fn elects(&self) -> bool {
        self.elects
    }

    /// Opens the harness's session again at the controller it asks, as a
    /// gateway does whose session that controller ended. The dial is tried
    /// until the controller welcomes it, inside the room.
    pub async fn redial(&mut self) -> Result<(), String> {
        let gateway = format!("http://{}", self.replica.gateway());
        let asked = Instant::now();
        loop {
            match Client::connect(&gateway, GATEWAY).await {
                Ok(client) => {
                    self.client = client;
                    return Ok(());
                }
                Err(_) if asked.elapsed() < ROOM => {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                Err(refused) => return Err(format!("no session opens inside the room: {refused}")),
            }
        }
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
                // The build this registration owes, asked: a decision asked
                // right behind it reads the row.
                self.store.publish_now();
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
        Ok(written(&self.store, endpoint, shape, rank))
    }

    /// An operator's add of a one-rank engine. The answer is the row the
    /// store holds, under the id a sink names it by: registered and pending.
    /// The call reads its answer behind its write, so a controller that
    /// probes by itself may have readied the row by then.
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
            Some(Event::Registered(row))
                if row.status() == proto::WorkerStatus::Pending
                    || (PRODUCT_PROBES && row.status() == proto::WorkerStatus::Ready) =>
            {
                Ok(WorkerId(row.id))
            }
            other => Err(format!("the add of {endpoint} answered {other:?}")),
        }
    }

    /// Feed: starts the dialer and one subscriber an engine, over every
    /// engine registered now and later. Each stands in for a driver the
    /// controller's build does not have. Loads need no start: the
    /// controller's own load monitor polls every ready engine.
    pub fn feed(&mut self) {
        self.feeds.spawn(probe(
            Arc::clone(&self.store),
            Arc::clone(&self.probed),
            Arc::clone(&self.turns),
            self.stop_feeds.subscribe(),
            !PRODUCT_PROBES,
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

    /// The gaps on any engine's stream: the store's own resyncs where the
    /// controller feeds itself, the gaps the sink named to the stand-in
    /// otherwise.
    pub fn gaps(&self) -> u64 {
        if PRODUCT_FEEDS {
            RESYNCS
                .iter()
                .map(|reason| self.store.resyncs(*reason))
                .sum()
        } else {
            self.events.gaps.load(Ordering::SeqCst)
        }
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
        if ids.is_empty() {
            return Err(format!("engine {engine} holds no rank to patch"));
        }
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

    /// Whether the controller's load monitor has published, for every row
    /// of each of `engines`, a report sampled at or after `since`. A
    /// decision is given the monitor's snapshot, and [`Self::pool`] reads
    /// the same one: the monitor polls on its own clock.
    pub fn monitored(&self, engines: &[usize], since: Instant) -> bool {
        let snapshot = self.replica.loads.borrow();
        let mut rows = self.members.iter();
        rows.all(|(id, (engine, _))| {
            if !engines.contains(engine) {
                return true;
            }
            let row = WorkersStore::get(self.store.as_ref(), *id);
            row.is_some_and(|row| {
                let report = snapshot.report(&row.facts.key);
                report.is_some_and(|(_, sampled)| sampled >= since)
            })
        })
    }

    /// The pool serving `model` in `role` as the store holds it at this
    /// instant, each row named by its engine: the rows of the workers'
    /// table that serve the model in that role. A row is healthy where its
    /// status is ready, and its load is the load monitor's newest report
    /// of it, as a decision's record states it. The rows come in the
    /// table's order, by registration key: no case reads their order. The
    /// overload verdict is no part of a row here: a case reads a veto in
    /// the picks the service answers.
    pub fn pool(&self, model: &str, role: Role) -> Result<Vec<Seen>, String> {
        let (_, mut rows) = self.store.all_rows();
        rows.retain(|row| {
            row.facts.role == role && row.facts.models.iter().any(|served| served == model)
        });
        let reports = self.replica.loads.borrow();
        rows.iter()
            .map(|row| {
                let (engine, rank) = *self.members.get(&row.id).ok_or_else(|| {
                    format!("the store holds a row no registration made: {:?}", row.id)
                })?;
                let load = reports.report(&row.facts.key).map(|(report, sampled)| {
                    load_row(report, row.facts.role, sampled, sampled, SignalSource::Poll)
                });
                Ok(Seen {
                    engine,
                    rank,
                    healthy: row.status == WorkerStatus::Ready,
                    load,
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

    /// Sends a request on the worker's own wire and reads it to its end.
    pub async fn serve_request(&self, engine: usize, id: String) -> Result<(Class, u32), String> {
        let worker = self
            .engines
            .get(engine)
            .ok_or_else(|| format!("no engine {engine}"))?;
        Ok(sent(&worker.endpoint, id).await)
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
        let served = self.worker(engine)?;
        self.sent += 1;
        let lease_id = self.sent;
        let started = Instant::now();
        let (class, frames) = self
            .serve_request(engine, format!("sent-{engine}-{lease_id}"))
            .await?;
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
        within("the store reads the lease dispatched", ROOM, || {
            self.dispatched(GATEWAY, lease)
        })
        .await
        .map(|_| ())
    }

    /// Whether the store reads the lease `lease` of the gateway named
    /// `gateway` dispatched: no dispatch deadline ends it then.
    pub fn dispatched(&self, gateway: &str, lease: u64) -> bool {
        let gateway = GatewayId(fingerprint(gateway.as_bytes()));
        let store = self.store.as_ref();
        // By the trait's path, as `leases` reads the file.
        smg_policy_controller::store::LeasesStore::get(store, gateway, LeaseId(lease))
            .is_some_and(|record| matches!(record.state, LeaseState::Dispatched { .. }))
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
        // Both are read before a hung engine is let go: its release runs
        // the thread to its end.
        let at = Instant::now();
        let asked = self.turns.begun.load(Ordering::SeqCst);
        if !slot.die().await? {
            return Err(format!("engine {engine} is dead already"));
        }
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

    /// The statuses the stand-in prober moved so far, in the order written:
    /// none where the controller probes by itself.
    pub fn verdicts(&self) -> Vec<Verdict> {
        self.turns.verdicts.lock().clone()
    }

    /// A place in the store's membership, for [`Harness::moved`].
    pub fn mark(&self) -> u64 {
        self.store.epoch()
    }

    /// Each status of a fleet's row the store moved since `mark`, in the
    /// order written: the store's own word of it, whoever probes.
    pub fn moved(&self, mark: u64) -> Result<Vec<Moved>, String> {
        let MembershipTail::Changes(changes) = self.store.changes_since(mark) else {
            return Err(format!("the store's tail no longer reaches {mark}"));
        };
        Ok(changes
            .into_iter()
            .filter_map(|(_, change)| match change {
                MembershipChange::StatusChanged { id, from, to } => {
                    let &(engine, _) = self.members.get(&id)?;
                    Some(Moved { engine, from, to })
                }
                _ => None,
            })
            .collect())
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
