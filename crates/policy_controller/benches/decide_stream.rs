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
//!   one decider (one model) and on eight (64 models). The rate is the
//!   streams' measured answers together over the one interval they share,
//!   from the first of them to the last: a stream that ends early idles
//!   in it, so the row reads no higher than the streams did together.
//! - `policy_decide_stream/gateway/64/<deciders>`: the rate rows over 64
//!   members, on one decider and on eight, with each answered request told
//!   on the gateway's `Session` as a gateway tells one served to its end: a
//!   `Dispatched` and a `Completed`, sent by the thread that holds the
//!   session. The frames name the lease the answer carried; where it
//!   carried none they name no lease, which a session takes. A stream
//!   hands the session its reports sixteen at a time: a hand-over wakes
//!   the session's thread, and one a request would time the wake.
//! - `policy_decide_stream/count/<mode>/<members>`, one cell a run, named by
//!   `DECIDE_STREAM_COUNT=<mode>:<members>:<decisions>`: the whole process's
//!   allocator calls a decision, `in_process` (the store's deciding door, a
//!   view as the row's decider reads it and the engine's decide on this
//!   thread), `commit` (the same with the pick's lease issued at the
//!   store's leases door and ended there, as a host that embeds the engine
//!   commits a decision) or `stream` (the decision asked over one stream,
//!   one at a time so every decision wakes the same threads whatever its
//!   fleet, the client's own calls included). A run's instructions are read
//!   from outside it (`perf stat -e instructions:u`): two runs of different
//!   length give a
//!   decision's by their difference. A counted run's window holds whole
//!   exchanges: it opens before the first measured request is handed over
//!   and closes at the last measured answer, which no request follows. A
//!   counted run keeps no round trip and
//!   sorts none: a sort grows faster than its length, and the difference
//!   of two runs would keep part of it.
//!
//! Every wire row prints the `Busy` answers it met (none is expected: the
//! rows stay inside a stream's window and a decider's queue) and the
//! process's resident size. A closed-loop row also prints its CPU seconds
//! per million decisions, client and controller together; an open-loop
//! client spins between instants, so its row prints none. A reading the
//! kernel's file did not give prints `n/a`. A closed-loop row prints the
//! answers it could not time too (`overtaken`, none expected): an answer
//! whose slot of the ring a later request took while it waited. Every
//! decision of a row or a count cell is a pick over its fleet, or the
//! bench stops: a setup that drifted to another outcome would print
//! another path's numbers. Outside the gateway rows a stream ends the
//! lease an answer carries through its hand at the leases' file as it
//! reads the answer, as a session hands a gateway's frame, so a wire row
//! holds as many leases as it has decisions in flight (a gateway row,
//! those of its frames on their way too) and its numbers hold a lease's
//! issue and its end. A build that
//! files no lease answers with none: its rows run the same, and every wire
//! row prints the leases its measured picks carried (`leases`). A row
//! whose picks carry a lease for some and none for others stops the bench.
//! `DECIDE_STREAM_ROWS=round_trip`, `offered`, `rate` or `gateway` runs one
//! kind alone, and `--test` runs every row for a moment. A run with no
//! `--bench` is a test too, as criterion reads one: `cargo bench` passes
//! `--bench`, and a test of every target passes neither.
//!
//! A row's members report their loads in bands, by one seeded rule
//! (`bands`): dealt in a fixed shuffled order, one member stands alone at
//! the least load, the next two report one request more and score alike,
//! the next one a request more again, and so on by ones and twos; over 512
//! members by eights and sixteens, so the least band holds more rows than
//! a pick has places. A pick's places are then drawn as leases in flight
//! make the least loaded rows tie: over 8 and 64 members one row alone, a
//! band of two and the next band's row; over 512 members from one band.
//! Every member reports 2,097,152 requests beneath its band, more than a
//! row has decisions, so no count of leases on file reaches it and a
//! lease on file moves no row's cost: a build that files leases and one
//! that files none pick over the same costs. Every wire row prints how the three best rows of
//! its measured decisions scored (`ties`, four counts): each alone, the
//! best alone and the next two alike, the best two alike, all three
//! alike. Two builds are compared where they print one tally.
//!
//! Every row runs on the allocator the binary registers, under a copy of
//! its settings (`MALLOC_CONF`). A served decision's frame holds lists
//! made on a deciding thread and freed on a stream's, and what a list
//! freed on another thread costs is the allocator's own: over the
//! system's a row reads a rate the binary does not serve. A run that is
//! not on that allocator and those settings is refused as it starts: the
//! allocator is asked for each setting, and for the arena an oversize
//! request of the global allocator must have stood up. Two runs of one
//! count cell are up to a hundred thousand instructions or so apart under
//! it, whatever their length: a decision's instructions are read by the
//! difference of two lengths, as above, never from one run.
//!
//! The process installs a tracing subscriber that takes nothing, as the
//! binary installs one of its own: with none, each span of the transport
//! takes the tracing crate's log path, a cost no served controller pays.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a controller that fails to serve is a bug the tests catch"
)]
#![expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a decision allocated, and the allocator reads its settings from an exported symbol and answers for them through its control call"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    alloc::{GlobalAlloc, Layout},
    env, fs,
    hint::black_box,
    io::{self, Write},
    mem,
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
        completed::Class, controller_frame, decide_answer::Answer, decision::Outcome,
        gateway_frame::Frame, policy_controller_client::PolicyControllerClient, Completed,
        ControllerFrame, DecideRequest, Decision, Dispatched, GatewayFrame, Hello,
        RequestFacts as WireFacts, Side,
    },
    store::{
        Ask, GatewayId, Hand, LeaseRecord, LeaseState, LeasesStore, LoadSink, LoadSource,
        MemoryStore, Picked, Said, ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore,
    },
    AppliedSpec, Constraints, Lease, LeaseId, LoadRow, PolicySpec, PoolFacts, ReleaseReason,
    RequestFacts, Role, Scratch,
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

/// The allocator every row runs on: the one the binary registers
/// (`src/bin/smg-policy-controller.rs`), on the targets it registers it. A
/// served decision's lists are made on one thread and freed on another,
/// and what that costs is the allocator's: a row read over another one is
/// not the binary's row.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
static SHIPPED: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The system allocator, on a target where the binary registers none.
#[cfg(any(target_env = "msvc", target_env = "musl"))]
static SHIPPED: std::alloc::System = std::alloc::System;

/// The binary's settings of the shipped allocator and the guard that holds
/// a run to them, where the binary runs on them. An Apple target keeps the
/// allocator's own settings: it compiles no background thread there, and
/// no row read there is the binary's.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
mod shipped {
    use std::{
        ffi::{c_char, c_int, c_uint, c_void, CStr, CString},
        hint::black_box,
        io, ptr,
    };

    /// The settings, in the symbol the allocator reads them from as it
    /// starts: a copy of the binary's line, which a bench cannot import. A
    /// case reads both sources by path (`tests/binary.rs`): the copy moves
    /// with the binary's, or that case fails.
    #[export_name = "_rjem_malloc_conf"]
    pub static MALLOC_CONF: &[u8; 100] =
        b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

    extern "C" {
        /// The allocator's control call, under the prefix its build gives
        /// every name of its own: the one [`MALLOC_CONF`] is exported
        /// under.
        #[link_name = "_rjem_mallctl"]
        fn mallctl(
            name: *const c_char,
            old: *mut c_void,
            old_len: *mut usize,
            new: *mut c_void,
            new_len: usize,
        ) -> c_int;
    }

    /// What the allocator answers for `name`, where the answer is a `T`:
    /// `None` for a name it does not know or an answer of another size.
    fn asked<T: Copy + Default>(name: &CStr) -> Option<T> {
        let mut answer = T::default();
        let mut len = size_of::<T>();
        // SAFETY: `name` ends in NUL, and the allocator writes `len` bytes
        // at most into `answer`, a plain value of that size. Nothing is
        // written to the allocator.
        let refused = unsafe {
            mallctl(
                name.as_ptr(),
                (&raw mut answer).cast(),
                &raw mut len,
                ptr::null_mut(),
                0,
            )
        };
        (refused == 0 && len == size_of::<T>()).then_some(answer)
    }

    /// Refuses a run that is not on the binary's allocator under the
    /// binary's settings: its rows would pass for the binary's. The
    /// allocator answers for itself. Each entry of [`MALLOC_CONF`] must
    /// read back as the option it sets, whatever the environment adds to
    /// the line. And the global allocator must be this one: asked for as
    /// much as its oversize threshold, it stands up the arena it keeps for
    /// such requests, the one past the `narenas` it spreads threads over,
    /// which a request to another allocator does not. That request is one
    /// mapping, made and returned before a row starts.
    pub(super) fn stands() -> io::Result<()> {
        let refused = |why: String| {
            io::Error::other(format!(
                "this run is not on the binary's allocator and settings: {why}"
            ))
        };
        let line = MALLOC_CONF
            .strip_suffix(b"\0")
            .and_then(|line| str::from_utf8(line).ok())
            .ok_or_else(|| refused("the settings line is no text that ends in NUL".into()))?;
        for entry in line.split(',') {
            let (key, value) = entry.split_once(':').unwrap_or((entry, ""));
            let read = match key {
                "background_thread" => {
                    asked::<bool>(c"opt.background_thread").map(|on| on.to_string())
                }
                "dirty_decay_ms" => asked::<isize>(c"opt.dirty_decay_ms").map(|ms| ms.to_string()),
                "muzzy_decay_ms" => asked::<isize>(c"opt.muzzy_decay_ms").map(|ms| ms.to_string()),
                "narenas" => asked::<c_uint>(c"opt.narenas").map(|arenas| arenas.to_string()),
                "oversize_threshold" => {
                    asked::<usize>(c"opt.oversize_threshold").map(|bytes| bytes.to_string())
                }
                // A setting the guard cannot ask for is one it cannot hold.
                _ => None,
            };
            if read.as_deref() != Some(value) {
                let read = read.as_deref().unwrap_or("nothing");
                return Err(refused(format!(
                    "`{key}` reads back as {read}, the line says {value}"
                )));
            }
        }
        let asked_by =
            asked::<usize>(c"opt.oversize_threshold").zip(asked::<c_uint>(c"opt.narenas"));
        let Some((threshold, arenas)) = asked_by else {
            return Err(refused("it names no oversize threshold".into()));
        };
        drop(black_box(Vec::<u8>::with_capacity(threshold)));
        // The allocator's answers on its arenas are of its last epoch.
        let mut epoch = 1_u64;
        // SAFETY: the name ends in NUL, nothing is read back, and the
        // allocator reads the eight bytes of `epoch`.
        let stale = unsafe {
            mallctl(
                c"epoch".as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                (&raw mut epoch).cast(),
                size_of::<u64>(),
            )
        };
        let arena = CString::new(format!("arena.{arenas}.initialized"))
            .map_err(|_| refused("an arena's name holds a NUL".into()))?;
        if stale != 0 || asked::<bool>(&arena) != Some(true) {
            return Err(refused(format!(
                "a request of {threshold} bytes did not reach it"
            )));
        }
        Ok(())
    }
}

/// The shipped allocator, counting its calls while [`COUNTING`] is on. The
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
        // SAFETY: the caller's layout contract, forwarded to the shipped allocator.
        unsafe { SHIPPED.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_call();
        // SAFETY: the caller's layout contract, forwarded to the shipped allocator.
        unsafe { SHIPPED.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with this layout.
        unsafe { SHIPPED.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_call();
        // SAFETY: `ptr` came from this allocator with this layout; the caller upholds `new_size`.
        unsafe { SHIPPED.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counted = Counted;

const GATEWAY: &str = "bench-gateway";
const PRESET: &str = "least_load";
/// The controller's runtime threads.
const SERVER_THREADS: usize = 2;
/// The slots a stream keeps its requests' send instants in, each beside
/// its request's id: above every row's count in flight. An answer whose
/// slot a later request took is counted, not timed.
const RING: u64 = 1_024;
/// The budget every request brings.
const BUDGET_MICROS: u32 = 5_000_000;
/// The dispatch deadline an in-process commit files its lease with.
const DISPATCH: Duration = Duration::from_secs(5);
/// A stream's window of credit, as the `Welcome` states it.
const WINDOW: u64 = 128;
/// An offered request handed over this long after it was due is counted
/// late: the client's own lag.
const LATE: Duration = Duration::from_micros(100);
/// The requests every member reports beneath its band: more than a row has
/// decisions ([`wire`] checks each row's count), so no count of leases on
/// file reaches it and a lease on file moves no row's cost.
const HELD: u32 = 1 << 21;
/// The reports a stream hands the session together: a hand-over wakes the
/// session's thread, and one a request would time the wake.
const REPORTS: usize = 16;
/// The seed of the order the members are dealt to their bands in.
const DEAL: u64 = 0x0B_A4D5;

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
    /// The streams tell each answered request on the gateway's session, as
    /// a gateway does; otherwise they end its lease at the store's door.
    frames: bool,
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

/// The band of each of `members` rows, a band a request above the one
/// before it. The rows are dealt in an order shuffled by a fixed seed to
/// bands of one row and of two by turns; over 512 members to bands of
/// eight and of sixteen, a sixty-fourth of the rows a unit.
fn bands(members: u32) -> Vec<u32> {
    let rows = usize::try_from(members).expect("a count that fits");
    let mut order: Vec<usize> = (0..rows).collect();
    let mut state = DEAL;
    for last in (1..rows).rev() {
        // One value of SplitMix64 a swap: the same order on every build.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut drawn = state;
        drawn = (drawn ^ (drawn >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        drawn = (drawn ^ (drawn >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        drawn ^= drawn >> 31;
        let below = u64::try_from(last).expect("an index that fits") + 1;
        order.swap(last, usize::try_from(drawn % below).expect("an index"));
    }
    let unit = (rows / 64).max(1);
    let mut band_of = vec![0; rows];
    let (mut band, mut left) = (0, unit);
    for row in order {
        if left == 0 {
            band += 1;
            left = if band % 2 == 0 { unit } else { 2 * unit };
        }
        if let Some(its) = band_of.get_mut(row) {
            *its = band;
        }
        left -= 1;
    }
    band_of
}

/// `members` ready members that serve every one of `models` models, each
/// with a load report in its band ([`bands`]), under the row's preset as
/// generation 1, their rows built.
fn fill(store: &MemoryStore, members: u32, models: u32) {
    let models: Vec<String> = (0..models).map(model).collect();
    let now = Instant::now();
    for (index, band) in (0..members).zip(bands(members)) {
        let key = format!("http://worker-{index}.fleet:8000");
        let id = store
            .upsert(WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: models.clone(),
                dp_size: 1,
                max_running: Some(2 * HELD),
                ..WorkerFacts::default()
            })
            .id();
        let revision = WorkersStore::get(store, id).expect("the row").revision;
        store.set_status(id, WorkerStatus::Ready, revision);
        store.record(
            id,
            0,
            LoadRow {
                running: HELD + band,
                max_running: 2 * HELD,
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
    store.publish_now();
    filled::filled(store, &model(0), Role::Single, members as usize);
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
    // Room for the frames of one hand-over of reports: they leave together.
    let (frames, sent) = mpsc::channel(2 * REPORTS);
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
    /// Each measured decision's round trip, in nanoseconds; none of a
    /// counted run's.
    nanos: Vec<u64>,
    /// The measured answers, whether their round trips were kept.
    answered: u64,
    busy: u64,
    /// The first measured answer's receipt, and the last one's.
    opened: Instant,
    closed: Instant,
    /// The allocator's calls over the measured decisions, where it counted.
    calls: u64,
    /// The measured requests an open loop handed over late.
    late: u64,
    /// The measured requests an open loop decided itself: due with no
    /// credit.
    local: u64,
    /// The measured answers a closed loop could not time: a later request
    /// took their slot of the ring.
    overtaken: u64,
    /// The measured decisions by how their three best rows scored: each
    /// alone, the best alone and the next two alike, the best two alike,
    /// all three alike.
    ties: [u64; 4],
    /// The measured picks whose answer carried a lease.
    leases: u64,
}

impl Tally {
    /// Counts a measured pick: under its place of [`Self::ties`], and its
    /// lease where its answer carried one.
    fn picked(&mut self, pick: Option<&Answered>) {
        let Some(pick) = pick else {
            return;
        };
        if let Some(count) = self.ties.get_mut(pick.place) {
            *count += 1;
        }
        self.leases += u64::from(pick.lease.is_some());
    }
}

/// The bench's gateway as the store keys its leases: by the fingerprint of
/// the subject its calls state.
fn gateway() -> GatewayId {
    GatewayId(smg_policy_controller::fingerprint(GATEWAY.as_bytes()))
}

/// What a stream reads of a pick's answer.
struct Answered {
    /// Its place in the stream's tally of ties ([`Tally::ties`]).
    place: usize,
    /// The lease the answer carried.
    lease: Option<u64>,
    /// The pick's first target: the member a gateway dispatches to.
    served: u64,
}

/// A request that is over, as its stream tells the gateway's session: the
/// lease its answer carried, zero where it carried none (a lease the store
/// does not hold is no error to a session), and the member that served it.
#[derive(Clone, Copy)]
struct Over {
    lease: u64,
    served: u64,
}

/// How a stream reports an answered request as over.
enum Ends {
    /// Through the stream's own hand at the leases' file, in process: the
    /// hand is made at the stream's first report.
    Door(Arc<MemoryStore>, Option<Hand>),
    /// As a gateway does: a `Dispatched` and a `Completed` on its session,
    /// whose thread takes the stream's reports [`REPORTS`] at a time.
    Session {
        reports: mpsc::Sender<Vec<Over>>,
        held: Vec<Over>,
    },
}

impl Clone for Ends {
    /// A stream's own end: a door's hand is not shared.
    fn clone(&self) -> Self {
        match self {
            Self::Door(store, _) => Self::Door(Arc::clone(store), None),
            Self::Session { reports, held } => Self::Session {
                reports: reports.clone(),
                held: held.clone(),
            },
        }
    }
}

impl Ends {
    /// Reports a picked request as over: its lease, where its answer
    /// carried one, ends at the store's door, or the gateway's session is
    /// told. A row holds as many leases as it has decisions in flight and
    /// frames on their way.
    async fn over(&mut self, gateway: GatewayId, pick: &Answered) {
        match self {
            Self::Door(store, hand) => {
                if let Some(lease) = pick.lease {
                    let hand = hand.get_or_insert_with(|| store.handover(gateway));
                    let ended = Said::Released(None, ReleaseReason::ClientGone);
                    store.hand(hand, LeaseId(lease), ended);
                }
            }
            Self::Session { reports, held } => {
                held.push(Over {
                    lease: pick.lease.unwrap_or(0),
                    served: pick.served,
                });
                if held.len() >= REPORTS {
                    let together = mem::replace(held, Vec::with_capacity(REPORTS));
                    reports
                        .send(together)
                        .await
                        .expect("the session takes a stream's reports");
                }
            }
        }
    }

    /// Hands the session the reports the stream still holds, as it ends.
    async fn rest(&mut self) {
        if let Self::Session { reports, held } = self {
            if !held.is_empty() {
                reports
                    .send(mem::take(held))
                    .await
                    .expect("the session takes a stream's last reports");
            }
        }
    }
}

/// Every decision of a row is a pick over its fleet. Another outcome means
/// the setup drifted, and the row would print another path's numbers.
/// Reads the decision's place in a stream's tally of ties, by its trace's
/// three best rows (two score alike where their scores are one value, bit
/// for bit), the lease its answer carried and its first target.
fn picked(decision: &Decision) -> Answered {
    let first = match &decision.outcome {
        Some(Outcome::Pick(pick)) => pick.targets.first(),
        _ => None,
    };
    assert!(first.is_some(), "a pick over the fleet: {decision:?}");
    let top = decision.trace.as_ref().map_or(&[][..], |trace| &trace.top);
    assert!(
        top.len() >= 3,
        "a pick's trace holds its three best rows: {decision:?}"
    );
    let place = match top {
        [first, second, third, ..] => {
            let alike = |one: f32, other: f32| usize::from(one.to_bits() == other.to_bits());
            2 * alike(first.score, second.score) + alike(second.score, third.score)
        }
        _ => 0,
    };
    Answered {
        place,
        lease: decision.leases.first().map(|lease| lease.id),
        served: first.map_or(0, |target| target.id),
    }
}

/// Opens the allocator's count.
fn counting() {
    CALLS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
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
    sent_at: &mut [(u64, Instant)],
    next: &mut u64,
    models: &[String],
) {
    let id = *next;
    *next += 1;
    let frame = request(id, models);
    if let Some(at) = sent_at.get_mut(slot(id)) {
        *at = (id, Instant::now());
    }
    requests
        .send(frame)
        .await
        .expect("the stream takes the request");
}

/// One stream of the shape: `in_flight` requests kept in flight until the
/// warm and the measured decisions are answered.
async fn ask(addr: SocketAddr, shape: Shape, ends: &mut Ends) -> Tally {
    let gateway = gateway();
    let models: Vec<String> = (0..shape.models).map(model).collect();
    let mut client = connect(addr).await;
    let (requests, sent) = mpsc::channel(256);
    let mut answers = client
        .decide(stated(ReceiverStream::new(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    let mut sent_at = vec![(u64::MAX, Instant::now()); slot(RING - 1) + 1];
    let mut next = 0;
    let mut tally = Tally {
        nanos: Vec::with_capacity(usize::try_from(shape.measured).expect("a count that fits")),
        answered: 0,
        busy: 0,
        opened: Instant::now(),
        closed: Instant::now(),
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        leases: 0,
    };
    if shape.counted && shape.warm == 0 {
        counting();
    }
    for _ in 0..shape.in_flight {
        hand(&requests, &mut sent_at, &mut next, &models).await;
    }
    let mut done = 0;
    loop {
        let answer = answers
            .message()
            .await
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        let received = Instant::now();
        let pick = match &answer.answer {
            Some(Answer::Decision(decision)) => Some(picked(decision)),
            Some(Answer::Busy(_)) => {
                tally.busy += 1;
                None
            }
            Some(Answer::Credit(_)) | None => continue,
        };
        if let Some(pick) = &pick {
            ends.over(gateway, pick).await;
        }
        if done == shape.warm {
            tally.opened = received;
        }
        if done >= shape.warm {
            tally.answered += 1;
            tally.picked(pick.as_ref());
        }
        // A counted run keeps no round trip: a sort of them would stay in
        // the difference of two runs.
        if done >= shape.warm && !shape.counted {
            let id: [u8; 8] = answer
                .request_id
                .as_slice()
                .try_into()
                .expect("the request's eight bytes");
            let id = u64::from_be_bytes(id);
            match sent_at.get(slot(id)) {
                Some((held, at)) if *held == id => {
                    let took = received.saturating_duration_since(*at);
                    tally.nanos.push(
                        u64::try_from(took.as_nanos()).expect("a round trip under centuries"),
                    );
                }
                // A later request took the slot while this one waited:
                // its instant is not this request's.
                _ => tally.overtaken += 1,
            }
        }
        done += 1;
        // No request follows the last measured answer, and a counted run
        // opens its window before the first measured request is handed
        // over: the window holds whole exchanges.
        if done == shape.warm + shape.measured {
            break;
        }
        if shape.counted && done == shape.warm {
            counting();
        }
        hand(&requests, &mut sent_at, &mut next, &models).await;
    }
    if shape.counted {
        COUNTING.store(false, Ordering::Relaxed);
        tally.calls = CALLS.load(Ordering::Relaxed);
    }
    tally.closed = Instant::now();
    ends.rest().await;
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
async fn offer(addr: SocketAddr, shape: Shape, ends: &mut Ends) -> Tally {
    let gateway = gateway();
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
        answered: 0,
        busy: 0,
        opened: Instant::now(),
        closed: Instant::now(),
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        leases: 0,
    };
    let start = Instant::now();
    let due = |id: u64| start + gap.saturating_mul(u32::try_from(id).unwrap_or(u32::MAX));
    let (mut next, mut done, mut credit) = (0, 0, WINDOW);
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
        let pick = match &answer.answer {
            Some(Answer::Decision(decision)) => Some(picked(decision)),
            Some(Answer::Busy(_)) => {
                tally.busy += 1;
                None
            }
            Some(Answer::Credit(grant)) => {
                credit += u64::from(grant.window);
                continue;
            }
            None => continue,
        };
        if let Some(pick) = &pick {
            ends.over(gateway, pick).await;
        }
        let id: [u8; 8] = answer
            .request_id
            .as_slice()
            .try_into()
            .expect("the request's eight bytes");
        let id = u64::from_be_bytes(id);
        if id >= shape.warm {
            if tally.answered == 0 {
                tally.opened = received;
            }
            tally.answered += 1;
            tally.picked(pick.as_ref());
            let took = received.saturating_duration_since(due(id));
            tally
                .nanos
                .push(u64::try_from(took.as_nanos()).expect("a round trip under centuries"));
        }
        done += 1;
    }
    tally.closed = Instant::now();
    ends.rest().await;
    tally
}

/// The gateway's session on a thread of its own: its `Hello` is welcomed,
/// then each request a stream reports as over is told as a gateway tells
/// one served to its end, a `Dispatched` and a `Completed`. Once every
/// reporting end is dropped the thread ends its side of the stream and
/// reads the controller's to its end: every frame was applied by then.
/// Returns the frames it sent.
fn session(addr: SocketAddr) -> (mpsc::Sender<Vec<Over>>, thread::JoinHandle<u64>) {
    let (ready, opened) = std_mpsc::channel();
    let thread = thread::Builder::new()
        .name("session".to_owned())
        .spawn(move || {
            current_thread().block_on(async move {
                let (frames, mut answers) = hello(addr).await;
                let (reports, mut over) = mpsc::channel::<Vec<Over>>(64);
                ready.send(reports).expect("the row waits for the session");
                let mut seq = 0;
                while let Some(together) = over.recv().await {
                    for Over { lease, served } in together {
                        let dispatched = Dispatched {
                            lease_id: lease,
                            target_index: 0,
                            side: Side::Single.into(),
                            since_decision_micros: 0,
                        };
                        let completed = Completed {
                            lease_id: lease,
                            side: Side::Single.into(),
                            served,
                            status: 200,
                            class: Class::Ok.into(),
                            ..Completed::default()
                        };
                        for frame in [Frame::Dispatched(dispatched), Frame::Completed(completed)] {
                            seq += 1;
                            let taken = frames
                                .send(GatewayFrame {
                                    seq,
                                    term_floor: 0,
                                    frame: Some(frame),
                                })
                                .await;
                            // A session the controller ended takes no
                            // frame, and its status says why.
                            let ended = if taken.is_err() {
                                answers.message().await.err()
                            } else {
                                None
                            };
                            assert!(taken.is_ok(), "the controller ended the session: {ended:?}");
                        }
                    }
                }
                drop(frames);
                while let Ok(Some(_)) = answers.message().await {}
                seq
            })
        })
        .expect("the session's thread");
    (opened.recv().expect("the session opens"), thread)
}

fn current_thread() -> runtime::Runtime {
    runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a client's runtime")
}

/// The process's CPU time so far, in the kernel's ticks of ten
/// milliseconds; none where the kernel's file does not read.
fn cpu_ticks() -> Option<u64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    // Past the command's closing parenthesis the state is the first field,
    // the user time the twelfth and the system time the thirteenth.
    let (_, fields) = stat.rsplit_once(')')?;
    let mut times = fields.split_whitespace().skip(11).map(str::parse::<u64>);
    let (Some(Ok(user)), Some(Ok(system))) = (times.next(), times.next()) else {
        return None;
    };
    Some(user + system)
}

/// The process's resident size in MiB; none where the kernel's file does
/// not read.
fn rss_mib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .map(|kib| kib / 1_024)
}

/// What a wire row measured, over all its streams.
struct Row {
    /// Every measured round trip, sorted.
    nanos: Vec<u64>,
    busy: u64,
    per_second: u64,
    /// CPU milliseconds of the whole process per million decisions, and
    /// its resident size: none where the kernel's file did not read.
    cpu_millis_per_million: Option<u64>,
    rss_mib: Option<u64>,
    calls: u64,
    late: u64,
    local: u64,
    overtaken: u64,
    /// The streams' tallies of ties together ([`Tally::ties`]).
    ties: [u64; 4],
    /// The leases the streams' measured picks carried.
    leases: u64,
    /// The frames the gateway's session sent for the streams' reports.
    frames: u64,
}

/// Runs one wire row: a served controller, the gateway's session, and the
/// shape's streams, each on its own thread.
fn wire(shape: Shape) -> Row {
    // The floor under the bands stands above the row's whole count of
    // decisions: whatever leases its streams and its session's frames hold
    // at once, a member's leases stay beneath what it reports.
    let decisions = u64::from(shape.streams) * (shape.warm + shape.measured);
    assert!(
        decisions < u64::from(HELD),
        "a row of {decisions} decisions reaches the {HELD} requests beneath a member's band"
    );
    let served = serve(shape);
    let addr = served.addr;
    let (reports, session) = session(addr);
    let ends = if shape.frames {
        Ends::Session {
            reports: reports.clone(),
            held: Vec::with_capacity(REPORTS),
        }
    } else {
        Ends::Door(Arc::clone(&served.store), None)
    };
    let before = cpu_ticks();
    let streams: Vec<_> = (0..shape.streams)
        .map(|index| {
            let mut ends = ends.clone();
            thread::Builder::new()
                .name(format!("client-{index}"))
                .spawn(move || {
                    current_thread().block_on(async move {
                        if shape.offered == 0 {
                            ask(addr, shape, &mut ends).await
                        } else {
                            offer(addr, shape, &mut ends).await
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
    // The row's own reporting end kept the session standing for the whole
    // row, whoever reported on it: a gateway whose session ended is let go
    // after its grace period, its streams and its leases with it. With the
    // last end dropped the session tells what is left, ends its stream and
    // is read to its end, so the row's CPU holds every frame.
    drop(ends);
    drop(reports);
    let frames = session.join().expect("the session's thread ends cleanly");
    let ticks = cpu_ticks()
        .zip(before)
        .map(|(after, before)| after.saturating_sub(before));
    let rss_mib = rss_mib();
    served.stop();

    let mut row = Row {
        nanos: Vec::new(),
        busy: 0,
        per_second: 0,
        cpu_millis_per_million: ticks
            .and_then(|ticks| (ticks * 10 * 1_000_000).checked_div(decisions)),
        rss_mib,
        calls: 0,
        late: 0,
        local: 0,
        overtaken: 0,
        ties: [0; 4],
        leases: 0,
        frames,
    };
    // The streams' answers together over the one interval they share, from
    // the first measured answer of any to the last of all: a sum of each
    // stream's own rate would read high by the stagger of their ends.
    let opened = tallies.iter().map(|tally| tally.opened).min();
    let closed = tallies.iter().map(|tally| tally.closed).max();
    let shared = opened
        .zip(closed)
        .map_or(Duration::ZERO, |(opened, closed)| {
            closed.saturating_duration_since(opened)
        });
    let shared = u64::try_from(shared.as_nanos()).expect("an interval under centuries");
    let answered: u64 = tallies.iter().map(|tally| tally.answered).sum();
    row.per_second = (answered * 1_000_000_000).checked_div(shared).unwrap_or(0);
    for tally in tallies {
        row.busy += tally.busy;
        row.calls += tally.calls;
        row.late += tally.late;
        row.local += tally.local;
        row.overtaken += tally.overtaken;
        for (sum, count) in row.ties.iter_mut().zip(tally.ties) {
            *sum += count;
        }
        row.leases += tally.leases;
        row.nanos.extend(tally.nanos);
    }
    // A build files a lease for every pick or for none: a row that met both
    // would print numbers of neither.
    let picks: u64 = row.ties.iter().sum();
    assert!(
        row.leases == 0 || row.leases == picks,
        "{} of the row's {picks} measured picks carried a lease",
        row.leases
    );
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

/// A reading of the process as a row prints it: `n/a` where the kernel's
/// file gave none.
fn read<T>(reading: Option<T>, print: impl FnOnce(T) -> String) -> String {
    reading.map_or_else(|| "n/a".to_owned(), print)
}

/// A row's tally of ties as it prints: the four counts in their order.
fn ties(counts: [u64; 4]) -> String {
    let [apart, one_then_two, two_then_one, alike] = counts;
    format!("{apart}/{one_then_two}/{two_then_one}/{alike}")
}

/// The engine's own decision, in process: the store's deciding door, the
/// view as the row's decider reads it and the engine's decide on this
/// thread, over buffers kept across decisions. With `commit` the pick's
/// lease is issued at the store's leases door and ended there, as a host
/// that embeds the engine commits a decision and reports its end: one lease
/// on file at a time. Returns the allocator's calls over `decisions` of
/// them.
fn in_process(store: &MemoryStore, decisions: u64, commit: bool) -> u64 {
    let name = model(0);
    let mut view = ViewBuffers::default();
    let mut scratch = Scratch::new();
    let gateway = GatewayId(1);
    let term = store.term();
    let mut decide = |seed: u64| {
        let ask = Ask {
            gateway,
            now: Instant::now(),
            pinned: None,
            affinity: false,
        };
        let decision = store.deciding(|world| {
            // The door a decider reads a plain request's pool by: the
            // publication's own rows, the asking gateway's vetoes and the
            // leases' counts beside them.
            let read = world.view(&name, Role::Single, &ask, false, &mut view);
            let facts = RequestFacts {
                model: &name,
                seed,
                budget_micros: BUDGET_MICROS,
                now_micros: world.micros(ask.now),
                lent_in_flight: read.as_ref().and_then(|read| read.in_flight.as_ref()),
                ..RequestFacts::default()
            };
            let pool = read.map_or_else(
                || PoolFacts {
                    generation: world.spec().generation(),
                    ..PoolFacts::default()
                },
                |read| read.built.pool_facts(),
            );
            let (candidates, stalled) = read
                .map(|read| (read.candidates, read.stalled))
                .unwrap_or_default();
            world.spec().engine().decide(
                &facts,
                &pool,
                candidates,
                &Constraints {
                    stalled,
                    ..Constraints::default()
                },
                &mut scratch,
            )
        });
        let target = decision
            .pick()
            .and_then(|pick| pick.targets.first())
            .expect("a pick over the fleet, in process")
            .id;
        if commit {
            // An id of the host's own minting: the high bit.
            let lease = LeaseId(1 << 63 | seed);
            let filed = store.issue(LeaseRecord {
                lease: Lease {
                    id: lease,
                    ..Lease::default()
                },
                gateway,
                request: seed,
                target,
                picked: Picked::default(),
                decode: None,
                term,
                issued_at: ask.now,
                dispatch_deadline: ask.now + DISPATCH,
                state: LeaseState::Issued,
            });
            assert_eq!(filed, Ok(()), "the store files the lease");
            let ended = store.released(gateway, lease, None, ReleaseReason::ClientGone);
            assert!(ended, "the lease is on the store's file");
        }
        black_box(&decision);
    };
    for seed in 0..2_000 {
        decide(seed);
    }
    counting();
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
        frames: false,
    };
    let (calls, busy) = match mode {
        "in_process" | "commit" => {
            let served = serve(shape);
            let calls = in_process(&served.store, decisions, mode == "commit");
            served.stop();
            (calls, 0)
        }
        "stream" => {
            let row = wire(shape);
            (row.calls, row.busy)
        }
        _ => return writeln!(out, "a count row's mode is in_process, commit or stream"),
    };
    writeln!(
        out,
        "policy_decide_stream/count/{mode}/{members} decisions={decisions} allocator_calls={calls} calls_per_decision={}.{:02} busy={busy}",
        calls / decisions.max(1),
        calls % decisions.max(1) * 100 / decisions.max(1),
    )
}

fn main() -> io::Result<()> {
    // A run on another allocator, or under other settings, is refused
    // before a row starts: its rows would pass for the binary's.
    #[cfg(all(
        not(target_env = "msvc"),
        not(target_env = "musl"),
        not(target_vendor = "apple")
    ))]
    shipped::stands()?;
    // As the binary's own subscriber does, this one keeps the transport's
    // spans off the tracing crate's log path.
    tracing::subscriber::set_global_default(tracing::subscriber::NoSubscriber::default())
        .expect("no subscriber stands before the bench's");
    // As criterion reads a run: `cargo bench` passes `--bench`, and a test
    // of every target passes neither flag.
    let quick = env::args().any(|arg| arg == "--test") || !env::args().any(|arg| arg == "--bench");
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
                frames: false,
            });
            writeln!(
                out,
                "policy_decide_stream/round_trip/{in_flight} members=64 deciders=2 decisions={} p50_us={} p99_us={} p999_us={} max_us={} per_s={} busy={} cpu_s_per_million={} rss_mib={} overtaken={} ties={} leases={}",
                row.nanos.len(),
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                percentile(&row.nanos, 999, 1_000),
                percentile(&row.nanos, 1, 1),
                row.per_second,
                row.busy,
                read(row.cpu_millis_per_million, seconds),
                read(row.rss_mib, |mib| mib.to_string()),
                row.overtaken,
                ties(row.ties),
                row.leases,
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
                frames: false,
            });
            writeln!(
                out,
                "policy_decide_stream/offered/{offered}/{streams} members=64 deciders=2 decisions={} p50_us={} p99_us={} p999_us={} max_us={} per_s={} busy={} local={} late={} rss_mib={} ties={} leases={}",
                row.nanos.len(),
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                percentile(&row.nanos, 999, 1_000),
                percentile(&row.nanos, 1, 1),
                row.per_second,
                row.busy,
                row.local,
                row.late,
                read(row.rss_mib, |mib| mib.to_string()),
                ties(row.ties),
                row.leases,
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
                    frames: false,
                });
                writeln!(
                    out,
                    "policy_decide_stream/rate/{members}/{deciders} models={models} streams=4 in_flight=32 decisions={} per_s={} p50_us={} p99_us={} busy={} cpu_s_per_million={} rss_mib={} overtaken={} ties={} leases={}",
                    row.nanos.len(),
                    row.per_second,
                    percentile(&row.nanos, 1, 2),
                    percentile(&row.nanos, 99, 100),
                    row.busy,
                    read(row.cpu_millis_per_million, seconds),
                    read(row.rss_mib, |mib| mib.to_string()),
                    row.overtaken,
                    ties(row.ties),
                    row.leases,
                )?;
                out.flush()?;
            }
        }
    }
    if all || rows == "gateway" {
        for (deciders, models) in [(1, 1), (8, 64)] {
            let row = wire(Shape {
                members: 64,
                deciders,
                models,
                streams: 4,
                in_flight: 32,
                offered: 0,
                warm,
                measured,
                counted: false,
                frames: true,
            });
            writeln!(
                out,
                "policy_decide_stream/gateway/64/{deciders} models={models} streams=4 in_flight=32 decisions={} per_s={} p50_us={} p99_us={} busy={} cpu_s_per_million={} rss_mib={} overtaken={} ties={} leases={} frames={}",
                row.nanos.len(),
                row.per_second,
                percentile(&row.nanos, 1, 2),
                percentile(&row.nanos, 99, 100),
                row.busy,
                read(row.cpu_millis_per_million, seconds),
                read(row.rss_mib, |mib| mib.to_string()),
                row.overtaken,
                ties(row.ties),
                row.leases,
                row.frames,
            )?;
            out.flush()?;
        }
    }
    Ok(())
}
