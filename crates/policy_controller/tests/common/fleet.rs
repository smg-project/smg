//! The end-to-end harness: a controller started on ephemeral loopback ports
//! in the case's own process, a fleet of mock engines beside it, and one
//! handle, [`Harness`], through which a case registers the fleet, feeds the
//! store and asks the service. Included by path beside `common/engines.rs`
//! and `common/net.rs` (the including binary names them `engines` and
//! `net`).
//!
//! Every fact the store holds of an engine came from that engine, through a
//! sink. Loads are the controller's own load driver's: the harness writes no
//! load row. Health has no driver in the crate yet, so the harness stands in
//! for one ([`probe`]): once an interval it asks each engine its wire's
//! health check and writes what it heard, never what the case knows. A dial
//! is bounded, so an engine that stops answering costs a turn that bound and
//! no more. The stand-in leaves the day a driver feeds that sink.
//!
//! What is the store's handle today is the wire tomorrow: a rank's
//! registration ([`Harness::register`]) becomes the control scope's call when
//! it is served, and the view a case reads ([`Harness::view`]) gives way to
//! the decisions the service answers. A case calls the handle alone, so
//! neither move rewrites a case.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use mock_worker::{
    config::Config as EngineConfig,
    engine::{EngineParams, TimingModel},
};
use openai_protocol::worker::RuntimeType;
use parking_lot::Mutex;
use smg_grpc_client::{vllm_engine::AbortOnDropStream, vllm_proto, VllmEngineClient};
use smg_policy_controller::{
    controller::{Config, Controller, Election, Error, Listener, Scope, State},
    proto::{policy_controller_client::PolicyControllerClient, LeaderInfo, LeaderRequest},
    store::{
        Ask, GatewayId, HealthPolicy, HealthSink, MemoryStore, ViewBuffers, WorkerFacts,
        WorkerStatus, WorkersStore,
    },
    view::{LoadRow, Role, WireModes, WorkerId},
};
use tokio::{
    io::{copy, sink},
    net::{TcpListener, TcpStream},
    sync::{oneshot, watch},
    task::JoinSet,
    time::MissedTickBehavior,
};
use tonic::transport::Channel;

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

/// The tokens a held request asks for: at the engines' decode step it
/// outlasts any case.
const HELD_TOKENS: u32 = 1 << 20;

/// One kind of engine in a fleet's table: a model, a role and its
/// data-parallel ranks.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub model: &'static str,
    pub role: Role,
    pub ranks: u16,
}

/// One row of a pool as a view reads it, named by the engine that
/// registered it.
#[derive(Clone, Copy, Debug)]
pub struct Seen {
    /// The engine's index in the harness, in registration order.
    pub engine: usize,
    pub rank: u16,
    pub healthy: bool,
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

/// An engine as the stand-in prober dials it: its address and the rows its
/// one answer speaks for.
#[derive(Clone)]
struct Probed {
    engine: usize,
    endpoint: String,
    ids: Vec<WorkerId>,
}

/// The one handle a case holds: register, feed, ask.
pub struct Harness {
    store: Arc<MemoryStore>,
    client: PolicyControllerClient<Channel>,
    engines: Vec<Engine>,
    members: HashMap<WorkerId, (usize, u16)>,
    probed: Arc<Mutex<Vec<Probed>>>,
    turns: Arc<Turns>,
    held: Vec<AbortOnDropStream>,
    buffers: ViewBuffers,
    stop_feeds: watch::Sender<bool>,
    feeds: JoinSet<()>,
    stop_serve: Option<oneshot::Sender<()>>,
    serve: JoinSet<Result<(), Error>>,
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

/// One mock engine with `ranks` ranks, served on a runtime of this thread
/// until `killed` resolves; the runtime is dropped at the return, and every
/// task of the engine with it. `bound` takes the listener's address. A
/// receiver sent on `hung` stops the thread on it: nothing of the engine
/// answers and its sockets stay open, until that receiver's sender drops.
fn serve_engine(
    ranks: u16,
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
            ..engine_config(addr.port(), ranks)
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

/// Starts one mock engine with `ranks` ranks on a runtime and a thread of
/// its own.
async fn start_engine(ranks: u16) -> Result<Engine, String> {
    let (bound, address) = oneshot::channel::<Result<SocketAddr, String>>();
    let (kill, killed) = oneshot::channel::<()>();
    let (ending, ended) = oneshot::channel::<()>();
    let (hang, hung) = oneshot::channel::<mpsc::Receiver<()>>();
    thread::Builder::new()
        .name("mock-engine".to_owned())
        .spawn(move || {
            serve_engine(ranks, bound, killed, hung);
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
        addr,
        kill: Some(kill),
        ended: Some(ended),
        witness: Some(witness),
        hang: Some(hang),
        release: None,
    })
}

/// How one dial of an engine ended: its wire's health answer, a failure,
/// or nothing within [`DIAL_TIMEOUT`]. The connection is kept across turns
/// and dropped when no answer came, so the next turn dials anew.
async fn dial(clients: &mut HashMap<String, VllmEngineClient>, endpoint: &str) -> DialEnd {
    let kept = clients.get(endpoint).cloned();
    let asked = tokio::time::timeout(DIAL_TIMEOUT, async {
        let client = match kept {
            Some(client) => client,
            None => VllmEngineClient::connect(endpoint).await.ok()?,
        };
        let answer = client.health_check().await.ok()?;
        Some((client, answer.healthy))
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
            endpoint,
            ids,
        } in engines
        {
            let end = dial(&mut clients, &endpoint).await;
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

impl Harness {
    /// Starts a controller in the binary's default shape on ephemeral
    /// loopback ports, waits until it leads and connects the generated
    /// client to its gateway listener.
    pub async fn start() -> Result<Self, String> {
        let listener = |name: &str, scopes: &[Scope]| Listener {
            name: name.to_owned(),
            addr: loopback(),
            scopes: scopes.to_vec(),
            tls: None,
        };
        let config = Config {
            controller_id: "fleet-controller".to_owned(),
            listeners: vec![
                listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
                listener("control", &[Scope::Control]),
                listener("replication", &[Scope::Replication]),
            ],
            admin: loopback(),
            shutdown_settle: Duration::ZERO,
            shutdown_grace: Duration::from_secs(5),
            load_poll_interval: POLL_INTERVAL,
            election: Election::default(),
        };
        let controller = bounded(
            "the controller binds its listeners",
            ROOM,
            Controller::bind(config),
        )
        .await?
        .map_err(|error| error.to_string())?;
        let gateway = controller
            .addresses()
            .of(Scope::Gateway)
            .ok_or("the controller has no gateway listener")?;
        let store = controller.store();
        let mut readiness = controller.readiness();
        let (stop_serve, stopped) = oneshot::channel::<()>();
        let mut serve = JoinSet::new();
        serve.spawn(async move {
            controller
                .serve(async move {
                    let _ = stopped.await;
                })
                .await
        });
        let state = bounded(
            "the controller leads",
            ROOM,
            readiness.wait_for(State::Leader),
        )
        .await?;
        if state != Some(State::Leader) {
            return Err(format!("the controller ended as {state:?} before it led"));
        }
        let client = bounded(
            "the client connects to the gateway listener",
            ROOM,
            PolicyControllerClient::connect(format!("http://{gateway}")),
        )
        .await?
        .map_err(|error| error.to_string())?;
        let (stop_feeds, _) = watch::channel(false);
        Ok(Self {
            store,
            client,
            engines: Vec::new(),
            members: HashMap::new(),
            probed: Arc::default(),
            turns: Arc::default(),
            held: Vec::new(),
            buffers: ViewBuffers::default(),
            stop_feeds,
            feeds: JoinSet::new(),
            stop_serve: Some(stop_serve),
            serve,
        })
    }

    /// Ask: who leads, as the generated client hears it on the gateway
    /// listener.
    pub async fn leader(&mut self) -> Result<LeaderInfo, String> {
        let asked = self.client.leader(LeaderRequest {
            model: String::new(),
        });
        bounded("the leader call answers", ROOM, asked)
            .await?
            .map(tonic::Response::into_inner)
            .map_err(|status| status.to_string())
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
            let running = start_engine(shape.ranks).await?;
            let endpoint = running.endpoint.clone();
            let engine = self.engines.len();
            let ids: Vec<WorkerId> = (0..shape.ranks)
                .map(|rank| {
                    let id = self.member(&endpoint, shape, rank);
                    self.members.insert(id, (engine, rank));
                    id
                })
                .collect();
            self.probed.lock().push(Probed {
                engine,
                endpoint,
                ids,
            });
            self.engines.push(running);
            started.push(engine);
        }
        Ok(started)
    }

    /// One rank's registration: the store's membership write today, the
    /// control scope's call once it is served.
    fn member(&self, endpoint: &str, shape: &Shape, rank: u16) -> WorkerId {
        let key = if shape.ranks == 1 {
            endpoint.to_owned()
        } else {
            format!("{endpoint}@{rank}")
        };
        self.store
            .upsert(WorkerFacts {
                key,
                engine: endpoint.to_owned(),
                incarnation: 1,
                role: shape.role,
                wire_modes: WireModes::GRPC,
                engine_kind: RuntimeType::Vllm.as_str().to_owned(),
                models: vec![shape.model.to_owned()],
                dp_rank: rank,
                dp_size: shape.ranks,
                health: HealthPolicy {
                    failure_threshold: 1,
                    success_threshold: 1,
                    check_interval_ms: millis(PROBE_INTERVAL),
                    timeout_ms: millis(DIAL_TIMEOUT),
                    probes: true,
                },
                ..WorkerFacts::default()
            })
            .id()
    }

    /// Feed: starts the stand-in prober over every engine registered now
    /// and later. Loads need no start: the controller's own driver polls
    /// every ready engine.
    pub fn feed(&mut self) {
        self.feeds.spawn(probe(
            Arc::clone(&self.store),
            Arc::clone(&self.probed),
            Arc::clone(&self.turns),
            self.stop_feeds.subscribe(),
        ));
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
                    load: row.load,
                })
            })
            .collect()
    }

    /// Holds `count` requests in flight on the engine at `engine`, on the
    /// engine's own wire: its reports then say so by themselves. A held
    /// stream is never read and is dropped with the harness.
    pub async fn hold(&mut self, engine: usize, count: usize) -> Result<(), String> {
        let endpoint = self
            .engines
            .get(engine)
            .map(|engine| engine.endpoint.clone())
            .ok_or_else(|| format!("no engine {engine}"))?;
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
            self.held.push(stream);
        }
        Ok(())
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
        let stop = self
            .stop_serve
            .take()
            .ok_or("the serve was stopped already")?;
        stop.send(())
            .map_err(|()| "the serve ended before its stop".to_owned())?;
        bounded("the serve ends", ROOM, self.serve.join_next())
            .await?
            .ok_or("the serve's task is gone")?
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())
    }
}
