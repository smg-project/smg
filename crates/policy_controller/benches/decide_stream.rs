//! A decision's round trip over the wire, as a gateway on the controller's
//! host pays it: a served controller (`Controller::bind` and `serve`, its
//! gateway listener on loopback), one `Session`, and warm `Decide` streams,
//! each on a client thread of its own with that thread's connection (the
//! design's one stream a gateway runtime thread). Not a criterion target: a
//! round trip under a fixed count in flight is a distribution read by its
//! tail, and a rate is one number a row.
//!
//! - `policy_decide_stream/round_trip/<in flight>`: one stream holding 1, 8
//!   and 64 decisions in flight over 64 members: the median, the 99th and
//!   the 99.9th percentile of a decision's round trip, from the request's
//!   hand-over to the client's stream to its answer's receipt.
//! - `policy_decide_stream/offered/<rate>/<streams>`: 45,000 and 100,000
//!   decisions a second, one deciding thread's load, offered by an open
//!   loop over one stream and spread over four, whatever the answers do.
//!   A request is due at the start plus its number over its stream's
//!   rate. Due with credit, it is handed over, and its round trip is read
//!   from the due instant to its answer's receipt, so the client's own lag
//!   counts. Due with no credit, it is the gateway's own decision at once,
//!   as the design's gateway decides: counted `local`, never sent. A row
//!   with a fixed count in flight and no pause reads the stream's ceiling;
//!   this one reads a latency at a rate, and the share that went local.
//! - `policy_decide_stream/rate/<members>/<deciders>`: decisions a second
//!   over 8, 64 and 512 members with four streams of 32 in flight each: on
//!   one decider (one model) and on eight (64 models).
//! - `policy_decide_stream/count/<mode>/<members>`, one cell a run, named by
//!   `DECIDE_STREAM_COUNT=<mode>:<members>:<decisions>`: the whole process's
//!   allocator calls a decision, `in_process` (the store's deciding door, a
//!   view build and the engine's decide on this thread) or `stream` (the
//!   same decision asked over one stream, one at a time so every decision
//!   wakes the same threads whatever its fleet, the client's own calls
//!   included). A run's instructions are read from outside it
//!   (`perf stat -e instructions:u`): two runs of different length give a
//!   decision's by their difference.
//!
//! Every wire row prints the `Busy` answers it met (none is expected: the
//! rows stay inside a stream's window and a decider's queue), the process's
//! CPU seconds per million decisions, client and controller together, and
//! its resident size. `DECIDE_STREAM_ROWS=round_trip`, `offered` or `rate`
//! runs one kind alone, and `--test` runs every row for a moment.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a controller that fails to serve is a bug the tests catch"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a decision allocated"
)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    env, fs,
    hint::black_box,
    io::{self, Write},
    net::{Ipv4Addr, SocketAddr},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc as std_mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

use smg_policy_controller::{
    controller::{Config, Controller, Election, Listener, Scope, State},
    proto::{
        controller_frame, decide_answer::Answer, gateway_frame::Frame,
        policy_controller_client::PolicyControllerClient, ControllerFrame, DecideRequest,
        GatewayFrame, Hello, RequestFacts as WireFacts,
    },
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts, WorkerStatus,
        WorkersStore,
    },
    AppliedSpec, Constraints, LoadRow, PolicySpec, PoolFacts, RequestFacts, Role, Scratch,
};
use tokio::{
    runtime,
    sync::{mpsc, oneshot},
};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue,
    transport::Channel, Request, Streaming,
};

/// Whether the allocator counts: on for a count row's measured decisions.
static COUNTING: AtomicBool = AtomicBool::new(false);
/// The allocation and reallocation calls of every thread while it counts.
static CALLS: AtomicU64 = AtomicU64::new(0);

/// The system allocator, counting its calls while [`COUNTING`] is on. The
/// rows that read time pay one read of a flag nobody writes.
struct Counted;

fn count_call() {
    if COUNTING.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_call();
        // SAFETY: `ptr` came from this allocator with this layout; the caller upholds `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counted = Counted;

const GATEWAY: &str = "bench-gateway";
const PRESET: &str = "least_load";
/// The controller's runtime threads.
const SERVER_THREADS: usize = 2;
/// The slots a stream keeps its requests' send instants in: above every
/// row's count in flight.
const RING: u64 = 1_024;
/// The budget every request brings.
const BUDGET_MICROS: u32 = 5_000_000;
/// A stream's window of credit, as the `Welcome` states it.
const WINDOW: u64 = 128;
/// An offered request handed over this long after it was due is counted
/// late: the client's own lag.
const LATE: Duration = Duration::from_micros(100);

/// One row's shape.
#[derive(Clone, Copy)]
struct Shape {
    members: u32,
    deciders: usize,
    models: u32,
    streams: u32,
    in_flight: u32,
    /// The decisions a second an open loop offers each stream; zero keeps
    /// `in_flight` requests in flight instead.
    offered: u64,
    /// The decisions a stream discards, then the ones it measures.
    warm: u64,
    measured: u64,
    /// The allocator counts over the measured decisions.
    counted: bool,
}

/// A controller served on its own thread and runtime.
struct Served {
    addr: SocketAddr,
    store: Arc<MemoryStore>,
    stop: oneshot::Sender<()>,
    thread: thread::JoinHandle<()>,
}

impl Served {
    fn stop(self) {
        // The serve hears the stop unless it already ended, which its
        // thread's end tells.
        drop(self.stop);
        self.thread
            .join()
            .expect("the controller's thread ends cleanly");
    }
}

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

/// The binary's default shape on ephemeral loopback ports.
fn config(deciders: usize) -> Config {
    Config {
        controller_id: "bench-controller".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(10),
        shutdown_grace: Duration::from_secs(5),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        decide_shards: deciders,
        election: Election::default(),
    }
}

fn model(index: u32) -> String {
    format!("model-{index}")
}

/// `members` ready members that serve every one of `models` models, each
/// with a load report, under the row's preset as generation 1.
fn fill(store: &MemoryStore, members: u32, models: u32) {
    let models: Vec<String> = (0..models).map(model).collect();
    let now = Instant::now();
    for index in 0..members {
        let key = format!("http://worker-{index}.fleet:8000");
        let id = store
            .upsert(WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: models.clone(),
                dp_size: 1,
                max_running: Some(256),
                ..WorkerFacts::default()
            })
            .id();
        let revision = WorkersStore::get(store, id).expect("the row").revision;
        store.set_status(id, WorkerStatus::Ready, revision);
        store.record(
            id,
            0,
            LoadRow {
                running: index % 64,
                queued: index % 8,
                max_running: 256,
                ..LoadRow::default()
            },
            now,
            LoadSource::Stream,
        );
    }
    let mut document = PolicySpec::preset(PRESET).expect("a preset of this build");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the preset validates");
    store.apply(spec).expect("above the unbound start");
}

/// Serves a controller over the shape's fleet; returns once it leads.
fn serve(shape: Shape) -> Served {
    let (ready, bound) = std_mpsc::channel();
    let thread = thread::Builder::new()
        .name("controller".to_owned())
        .spawn(move || {
            let runtime = runtime::Builder::new_multi_thread()
                .worker_threads(SERVER_THREADS)
                .enable_all()
                .build()
                .expect("the controller's runtime");
            runtime.block_on(async move {
                let mut controller = Controller::bind(config(shape.deciders))
                    .await
                    .expect("the listeners bind");
                controller
                    .identity_as_told("gateway")
                    .expect("a listener of the configuration");
                let store = controller.store();
                fill(&store, shape.members, shape.models);
                let addr = controller
                    .addresses()
                    .of(Scope::Gateway)
                    .expect("the gateway scope is mounted");
                let mut readiness = controller.readiness();
                let (stop, stopped) = oneshot::channel::<()>();
                let serving = controller.serve(async {
                    // The bench's drop of its end is the stop.
                    let _ = stopped.await;
                });
                let told = async {
                    assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
                    ready
                        .send((addr, store, stop))
                        .expect("the bench waits for the address");
                };
                let (served, ()) = tokio::join!(serving, told);
                served.expect("a clean stop");
            });
        })
        .expect("the controller's thread");
    let (addr, store, stop) = bound.recv().expect("the controller serves");
    Served {
        addr,
        store,
        stop,
        thread,
    }
}

/// A call that states the bench's gateway on the listener that takes
/// identities as told.
fn stated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(GATEWAY));
    request
}

async fn connect(addr: SocketAddr) -> PolicyControllerClient<Channel> {
    PolicyControllerClient::connect(format!("http://{addr}"))
        .await
        .expect("the listener accepts")
}

/// The gateway's session: its `Hello` is welcomed, and the stream stands
/// for as long as the returned ends are held.
async fn hello(addr: SocketAddr) -> (mpsc::Sender<GatewayFrame>, Streaming<ControllerFrame>) {
    let mut client = connect(addr).await;
    let (frames, sent) = mpsc::channel(4);
    let mut answers = client
        .session(stated(ReceiverStream::new(sent)))
        .await
        .expect("the stream opens")
        .into_inner();
    let hello = Hello {
        gateway_id: GATEWAY.to_owned(),
        protocol_min: 1,
        protocol_max: 1,
        ..Hello::default()
    };
    frames
        .send(GatewayFrame {
            seq: 0,
            term_floor: 0,
            frame: Some(Frame::Hello(hello)),
        })
        .await
        .expect("the stream takes the Hello");
    let answer = answers
        .message()
        .await
        .expect("a frame, not a status")
        .expect("a frame, not the stream's end");
    let window = match &answer.frame {
        Some(controller_frame::Frame::Welcome(welcome)) => u64::from(welcome.decide_window),
        _ => 0,
    };
    assert_eq!(
        window, WINDOW,
        "a Welcome answers a Hello, with the window the open loop counts by: {answer:?}"
    );
    (frames, answers)
}

/// What one stream measured.
struct Tally {
    /// Each measured decision's round trip, in nanoseconds.
    nanos: Vec<u64>,
    busy: u64,
    /// From the first measured answer to the last.
    window: Duration,
    /// The allocator's calls over the measured decisions, where it counted.
    calls: u64,
    /// The measured requests an open loop handed over late.
    late: u64,
    /// The measured requests an open loop decided itself: due with no
    /// credit.
    local: u64,
}

fn slot(id: u64) -> usize {
    usize::try_from(id % RING).expect("a slot of the ring")
}

/// Request `id` of a row, for one of its models.
fn request(id: u64, models: &[String]) -> DecideRequest {
    let name = models
        .get(slot(id) % models.len().max(1))
        .expect("a model of the row");
    DecideRequest {
        request_id: id.to_be_bytes().to_vec(),
        facts: Some(WireFacts {
            model: name.clone(),
            seed: id,
            budget_micros: BUDGET_MICROS,
            ..WireFacts::default()
        }),
        constraints: None,
    }
}

/// Hands the stream its next request and keeps the instant.
async fn hand(
    requests: &mpsc::Sender<DecideRequest>,
    sent_at: &mut [Instant],
    next: &mut u64,
    models: &[String],
) {
    let id = *next;
    *next += 1;
    let frame = request(id, models);
    if let Some(at) = sent_at.get_mut(slot(id)) {
        *at = Instant::now();
    }
    requests
        .send(frame)
        .await
        .expect("the stream takes the request");
}

/// One stream of the shape: `in_flight` requests kept in flight until the
/// warm and the measured decisions are answered.
async fn ask(addr: SocketAddr, shape: Shape) -> Tally {
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let mut sent_at = vec![Instant::now(); slot(RING - 1) + 1];
    let mut next = 0;
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        busy: 0,
        window: Duration::ZERO,
        calls: 0,
        late: 0,
        local: 0,
    };
    for _ in 0..shape.in_flight {
        hand(&requests, &mut sent_at, &mut next, &models).await;
    }
    let mut done = 0;
    let mut opened = Instant::now();
    while done < shape.warm + shape.measured {
        let answer = answers
            .message()
            .await
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        match answer.answer {
            Some(Answer::Decision(_)) => {}
            Some(Answer::Busy(_)) => tally.busy += 1,
            Some(Answer::Credit(_)) | None => continue,
        }
        if done == shape.warm {
            opened = received;
            if shape.counted {
                CALLS.store(0, Ordering::Relaxed);
                COUNTING.store(true, Ordering::Relaxed);
            }
        }
        if done >= shape.warm {
            let id: [u8; 8] = answer
                .request_id
                .as_slice()
                .try_into()
                .expect("the request's eight bytes");
            let at = sent_at
                .get(slot(u64::from_be_bytes(id)))
                .expect("a slot of the ring");
            let took = received.saturating_duration_since(*at);
            tally
                .nanos
                .push(u64::try_from(took.as_nanos()).expect("a round trip under centuries"));
        }
        done += 1;
        hand(&requests, &mut sent_at, &mut next, &models).await;
    }
    if shape.counted {
        COUNTING.store(false, Ordering::Relaxed);
        tally.calls = CALLS.load(Ordering::Relaxed);
    }
    tally.window = opened.elapsed();
    tally
}

/// One stream of the shape under an open loop, as the design's gateway
/// runs one: `shape.offered` requests a second, whatever the answers do.
/// Request `i` is due at the start plus `i` over the rate. Due while the
/// stream has credit, it is handed over, and its round trip is read from
/// the due instant; due with no credit, it is the gateway's own decision
/// at once, counted and never sent. The client spins between instants,
/// since a timer's millisecond is coarser than the gaps, and yields to its
/// runtime each turn so the connection is read and written.
async fn offer(addr: SocketAddr, shape: Shape) -> Tally {
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let total = shape.warm + shape.measured;
    let gap = Duration::from_nanos(1_000_000_000 / shape.offered.max(1));
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        busy: 0,
        window: Duration::ZERO,
        calls: 0,
        late: 0,
        local: 0,
    };
    let start = Instant::now();
    let due = |id: u64| start + gap.saturating_mul(u32::try_from(id).unwrap_or(u32::MAX));
    let (mut next, mut done, mut credit) = (0, 0, WINDOW);
    let mut opened = start;
    while done < total {
        let now = Instant::now();
        while next < total && due(next) <= now {
            if credit == 0 {
                if next >= shape.warm {
                    tally.local += 1;
                }
                done += 1;
            } else {
                if next >= shape.warm && now.saturating_duration_since(due(next)) > LATE {
                    tally.late += 1;
                }
                requests
                    .try_send(request(next, &models))
                    .expect("a window of requests fits the stream's channel");
                credit -= 1;
            }
            next += 1;
        }
        if done >= total {
            break;
        }
        let answer = tokio::select! {
            biased;
            answer = answers.message() => answer,
            () = tokio::task::yield_now() => continue,
        };
        let answer = answer
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        match answer.answer {
            Some(Answer::Decision(_)) => {}
            Some(Answer::Busy(_)) => tally.busy += 1,
            Some(Answer::Credit(grant)) => {
                credit += u64::from(grant.window);
                continue;
            }
            None => continue,
        }
        let id: [u8; 8] = answer
            .request_id
            .as_slice()
            .try_into()
            .expect("the request's eight bytes");
        let id = u64::from_be_bytes(id);
        if id >= shape.warm {
            if tally.nanos.is_empty() {
                opened = received;
            }
            let took = received.saturating_duration_since(due(id));
            tally
                .nanos
                .push(u64::try_from(took.as_nanos()).expect("a round trip under centuries"));
        }
        done += 1;
    }
    tally.window = opened.elapsed();
    tally
}

fn current_thread() -> runtime::Runtime {
    runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a client's runtime")
}

/// The process's CPU time so far, in the kernel's ticks of ten
/// milliseconds; zero where the kernel's file does not read.
fn cpu_ticks() -> u64 {
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // Past the command's closing parenthesis the state is the first field,
    // the user time the twelfth and the system time the thirteenth.
    let fields = stat.rsplit_once(')').map_or("", |(_, fields)| fields);
    fields
        .split_whitespace()
        .skip(11)
        .take(2)
        .filter_map(|field| field.parse::<u64>().ok())
        .sum()
}

/// The process's resident size in MiB; zero where the kernel's file does
/// not read.
fn rss_mib() -> u64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .map_or(0, |kib| kib / 1_024)
}

/// What a wire row measured, over all its streams.
struct Row {
    /// Every measured round trip, sorted.
    nanos: Vec<u64>,
    busy: u64,
    per_second: u64,
    /// CPU milliseconds of the whole process per million decisions.
    cpu_millis_per_million: u64,
    rss_mib: u64,
    calls: u64,
    late: u64,
    local: u64,
}

/// Runs one wire row: a served controller, the gateway's session, and the
/// shape's streams, each on its own thread.
fn wire(shape: Shape) -> Row {
    let served = serve(shape);
    let addr = served.addr;
    let session_runtime = current_thread();
    let session = session_runtime.block_on(hello(addr));
    let before = cpu_ticks();
    let streams: Vec<_> = (0..shape.streams)
        .map(|index| {
            thread::Builder::new()
                .name(format!("client-{index}"))
                .spawn(move || {
                    current_thread().block_on(async move {
                        if shape.offered == 0 {
                            ask(addr, shape).await
                        } else {
                            offer(addr, shape).await
                        }
                    })
                })
                .expect("a client's thread")
        })
        .collect();
    let tallies: Vec<Tally> = streams
        .into_iter()
        .map(|stream| stream.join().expect("a client's thread ends cleanly"))
        .collect();
    let ticks = cpu_ticks().saturating_sub(before);
    let rss_mib = rss_mib();
    drop(session);
    drop(session_runtime);
    served.stop();

    let decisions = u64::from(shape.streams) * (shape.warm + shape.measured);
    let mut row = Row {
        nanos: Vec::new(),
        busy: 0,
        per_second: 0,
        cpu_millis_per_million: (ticks * 10 * 1_000_000).checked_div(decisions).unwrap_or(0),
        rss_mib,
        calls: 0,
        late: 0,
        local: 0,
    };
    for tally in tallies {
        let window = u64::try_from(tally.window.as_nanos()).expect("a window under centuries");
        let answered = u64::try_from(tally.nanos.len()).expect("a count that fits");
        row.per_second += (answered * 1_000_000_000).checked_div(window).unwrap_or(0);
        row.busy += tally.busy;
        row.calls += tally.calls;
        row.late += tally.late;
        row.local += tally.local;
        row.nanos.extend(tally.nanos);
    }
    row.nanos.sort_unstable();
    row
}

/// The value `numerator` in `denominator` of the sorted values are at or
/// under, as microseconds and tenths.
fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> String {
    let rank = (sorted.len() * numerator).div_ceil(denominator);
    micros(sorted.get(rank.saturating_sub(1)).copied().unwrap_or(0))
}

fn micros(nanos: u64) -> String {
    format!("{}.{}", nanos / 1_000, nanos % 1_000 / 100)
}

fn seconds(millis: u64) -> String {
    format!("{}.{:02}", millis / 1_000, millis % 1_000 / 10)
}

/// The engine's own decision, in process: the store's deciding door, a view
/// build and the engine's decide on this thread, over buffers kept across
/// decisions. Returns the allocator's calls over `decisions` of them.
fn in_process(store: &MemoryStore, decisions: u64) -> u64 {
    let name = model(0);
    let mut view = ViewBuffers::default();
    let mut scratch = Scratch::new();
    let mut decide = |seed: u64| {
        let ask = Ask {
            gateway: GatewayId(1),
            now: Instant::now(),
            pinned: None,
            affinity: false,
        };
        let decision = store.deciding(|world| {
            let built = world.build(&name, Role::Single, &ask, &mut view);
            let facts = RequestFacts {
                model: &name,
                seed,
                budget_micros: BUDGET_MICROS,
                now_micros: world.micros(ask.now),
                ..RequestFacts::default()
            };
            let pool = built.map_or_else(
                || PoolFacts {
                    generation: world.spec().generation(),
                    ..PoolFacts::default()
                },
                |built| built.pool_facts(),
            );
            world.spec().engine().decide(
                &facts,
                &pool,
                &view.candidates,
                &Constraints::default(),
                &mut scratch,
            )
        });
        black_box(&decision);
    };
    for seed in 0..2_000 {
        decide(seed);
    }
    CALLS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for seed in 0..decisions {
        decide(2_000 + seed);
    }
    COUNTING.store(false, Ordering::Relaxed);
    CALLS.load(Ordering::Relaxed)
}

/// One count row, `<mode>:<members>:<decisions>`.
fn count(cell: &str, out: &mut impl Write) -> io::Result<()> {
    let mut parts = cell.split(':');
    let mode = parts.next().unwrap_or_default();
    let mut number = || parts.next().and_then(|part| part.parse::<u64>().ok());
    let (Some(members), Some(decisions)) = (number(), number()) else {
        return writeln!(out, "DECIDE_STREAM_COUNT is <mode>:<members>:<decisions>");
    };
    let shape = Shape {
        members: u32::try_from(members).expect("a fleet that fits"),
        deciders: 1,
        models: 1,
        streams: 1,
        in_flight: 1,
        offered: 0,
        warm: 2_000,
        measured: decisions,
        counted: true,
    };
    let (calls, busy) = match mode {
        "in_process" => {
            let served = serve(shape);
            let calls = in_process(&served.store, decisions);
            served.stop();
            (calls, 0)
        }
        "stream" => {
            let row = wire(shape);
            (row.calls, row.busy)
        }
        _ => return writeln!(out, "a count row's mode is in_process or stream"),
    };
    writeln!(
        out,
        "policy_decide_stream/count/{mode}/{members} decisions={decisions} allocator_calls={calls} calls_per_decision={}.{:02} busy={busy}",
        calls / decisions.max(1),
        calls % decisions.max(1) * 100 / decisions.max(1),
    )
}

fn main() -> io::Result<()> {
    let quick = env::args().any(|arg| arg == "--test");
    let mut out = io::stdout().lock();
    if let Ok(cell) = env::var("DECIDE_STREAM_COUNT") {
        return count(&cell, &mut out);
    }
    let rows = env::var("DECIDE_STREAM_ROWS").unwrap_or_default();
    let (warm, measured) = if quick {
        (200, 2_000)
    } else {
        (20_000, 200_000)
    };
    let all = rows.is_empty();
    if all || rows == "round_trip" {
        for in_flight in [1, 8, 64] {
            let row = wire(Shape {
                members: 64,
                deciders: 2,
                models: 1,
                streams: 1,
                in_flight,
                offered: 0,
                warm,
                measured,
                counted: false,
            });
            writeln!(
                out,
                "policy_decide_stream/round_trip/{in_flight} members=64 deciders=2 decisions={} p50_us={} p99_us={} p999_us={} max_us={} per_s={} busy={} cpu_s_per_million={} rss_mib={}",
                row.nanos.len(),
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                percentile(&row.nanos, 999, 1_000),
                percentile(&row.nanos, 1, 1),
                row.per_second,
                row.busy,
                seconds(row.cpu_millis_per_million),
                row.rss_mib,
            )?;
            out.flush()?;
        }
    }
    if all || rows == "offered" {
        // One model, so one deciding thread takes the whole rate; one
        // stream reads a stream's ceiling, and four spread the rate as the
        // rate rows' gateway does, a stream a runtime thread.
        for (offered, streams) in [(45_000, 1), (100_000, 1), (45_000, 4), (100_000, 4)] {
            let row = wire(Shape {
                members: 64,
                deciders: 2,
                models: 1,
                streams,
                in_flight: 0,
                offered: offered / u64::from(streams),
                warm: warm / u64::from(streams),
                measured: measured / u64::from(streams),
                counted: false,
            });
            writeln!(
                out,
                "policy_decide_stream/offered/{offered}/{streams} members=64 deciders=2 decisions={} p50_us={} p99_us={} p999_us={} max_us={} per_s={} busy={} local={} late={} rss_mib={}",
                row.nanos.len(),
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                percentile(&row.nanos, 999, 1_000),
                percentile(&row.nanos, 1, 1),
                row.per_second,
                row.busy,
                row.local,
                row.late,
                row.rss_mib,
            )?;
            out.flush()?;
        }
    }
    if all || rows == "rate" {
        for (deciders, models) in [(1, 1), (8, 64)] {
            for members in [8, 64, 512] {
                let row = wire(Shape {
                    members,
                    deciders,
                    models,
                    streams: 4,
                    in_flight: 32,
                    offered: 0,
                    warm,
                    measured,
                    counted: false,
                });
                writeln!(
                    out,
                    "policy_decide_stream/rate/{members}/{deciders} models={models} streams=4 in_flight=32 decisions={} per_s={} p50_us={} p99_us={} busy={} cpu_s_per_million={} rss_mib={}",
                    row.nanos.len(),
                    row.per_second,
                    percentile(&row.nanos, 1, 2),
                    percentile(&row.nanos, 99, 100),
                    row.busy,
                    seconds(row.cpu_millis_per_million),
                    row.rss_mib,
                )?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
