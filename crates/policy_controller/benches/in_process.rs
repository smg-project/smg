//! The in-process engine over a host's worker objects, cell by cell.
//!
//! The engine of a gateway's own process ([`PolicyEngine`]) decides over
//! the gateway's lists of worker objects and its load monitor's snapshot.
//! A cell is one such list under one document, with the request its callers
//! make: one entry of [`CELLS`]. A caller asks the one entry a host asks
//! for a request ([`PolicyEngine::decide_at`]) and says what the host then
//! owes the engine of it. A place answered alone owes nothing, and neither
//! does a decision that is not leased. A leased one is held by a guard,
//! which is told the dispatch and the end ([`DecisionGuard`]).
//!
//! The request is made, the clock read and the load snapshot borrowed
//! where the engine asks for them, the snapshot under a document that reads
//! load reports alone: a place answered alone takes none. No other work of
//! a host is done: no route, no metric, no in-flight count.
//!
//! ```text
//! cargo bench -p smg-policy-controller --features in-process --bench in_process -- \
//!     --cell power_of_two --workers 64 --callers 1 --decisions 1000000
//! ```
//!
//! `--workers` is the list's size: every worker is ready and has a load
//! report, published once. `--overloaded 1` puts every worker over its
//! overload threshold, so the list admits no worker, and gives each worker
//! requests in flight: its place plus one, the middle worker none, so the
//! rule for such a list has loads to steer by. A pair's cell hands two
//! lists of that size. `--reports 0` lends no load snapshot, whatever the
//! document reads (`reports=` in the result says whether one was lent).
//! `--callers` threads each make `--decisions` timed decisions behind
//! untimed first ones. The result is one line.
//!
//! A time needs a quiet machine. A count of instructions does not: run a
//! cell twice under callgrind with one caller and two values of
//! `--decisions`, and divide the difference of the counts by the difference
//! of the decisions. The setup and the untimed decisions are in both runs.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "a bench binary: it stops on a broken setup and prints its result"
)]

#[path = "../tests/common/counting_alloc.rs"]
mod counting_alloc;

use std::{
    cell::OnceCell,
    collections::HashMap,
    env,
    hint::black_box,
    ops::Deref,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use counting_alloc::{calls, Counting};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        Constraints, Decided, PairAdmission, PairLists, PartnerClasses, Phase, Plain, PolicyEngine,
        PoolMembers, Request, RequestFacts, RequestMaker, WorkerLists,
    },
    store::GatewayId,
    tracking::{DecisionGuard, Outcome, Side, TransportClass},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    load::state::{one_rank_load, LoadReceiver, LoadSnapshot, TestFeed},
    test_support::{no_health_check, ready_worker},
    worker::{WorkerLoadResponse, WorkerSpec, WorkerStatus, WorkerType},
};

#[global_allocator]
static GLOBAL: Counting = Counting;

const MODEL: &str = "bench-model";

/// Decisions a caller makes untimed first: the buffers a thread keeps.
const CALLER_WARM: u64 = 256;

/// A cell: its name, the preset its document is, and the request of one
/// turn of its callers, by the turn's seed. The request is plain for every
/// seed or for none: the bench asks it once a cell. A cell whose request is
/// a pair's runs under the pair document of its preset on both sides, and
/// its host hands two lists of `--workers` workers each, every pair open.
type Cell = (&'static str, &'static str, fn(u64) -> Request<'static>);

/// The cells. A change that moves a policy's cost adds its entry here.
const CELLS: [Cell; 8] = [
    ("round_robin", "round_robin", plain),
    ("random", "random", plain),
    ("passthrough", "passthrough", plain),
    ("power_of_two", "power_of_two", plain),
    ("least_load", "least_load", plain),
    ("bucket", "bucket", plain),
    ("pair_round_robin", "round_robin", pair),
    ("pair_least_load", "least_load", pair),
];

/// A plain single request of 1,024 prompt tokens: it asks only a worker.
fn plain(seed: u64) -> Request<'static> {
    let facts = RequestFacts {
        model: MODEL,
        phase: Phase::Single,
        prompt_tokens: 1_024,
        seed,
        ..RequestFacts::default()
    };
    Request {
        id: b"bench",
        facts,
        constraints: Constraints::default(),
    }
}

/// A pair's request of the same size, stated for the prefill phase.
fn pair(seed: u64) -> Request<'static> {
    let mut request = plain(seed);
    request.facts.phase = Phase::Prefill;
    request
}

/// A ready worker of `worker_type` at `url`, with no health checker behind it.
fn typed(url: String, worker_type: WorkerType) -> Arc<dyn Worker> {
    let spec = WorkerSpec {
        worker_type,
        ..WorkerSpec::new(url)
    };
    let worker = <BasicWorkerBuilder>::from_spec(spec)
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A pair's host with no limit of its own: every prefill has capacity and
/// every decode a room.
#[derive(Debug)]
struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}

/// A mix of `value`'s bits: a seed a request, a report's numbers a worker.
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// One run, as the switches state it.
struct Run {
    cell: Cell,
    workers: usize,
    callers: usize,
    decisions: u64,
    overloaded: bool,
    reports: bool,
}

fn run() -> Run {
    let mut run = Run {
        cell: CELLS[0],
        workers: 64,
        callers: 1,
        decisions: 1_000_000,
        overloaded: false,
        reports: true,
    };
    let mut switches = env::args().skip(1);
    while let Some(switch) = switches.next() {
        // Cargo's own switch for a bench binary.
        if switch == "--bench" {
            continue;
        }
        let value = switches.next().expect("a value behind each switch");
        let number = || value.parse::<u64>().expect("a number");
        match switch.as_str() {
            "--cell" => {
                let named = CELLS.iter().find(|cell| cell.0 == value);
                run.cell = *named.unwrap_or_else(|| panic!("no cell {value}"));
            }
            "--workers" => run.workers = number() as usize,
            "--callers" => run.callers = number() as usize,
            "--decisions" => run.decisions = number(),
            "--overloaded" => run.overloaded = number() != 0,
            "--reports" => run.reports = number() != 0,
            other => panic!("no switch {other}"),
        }
    }
    run
}

/// One report a worker: queued token-work, running requests, KV pressure
/// and a live rate, as the gateway's decision bench gives its fleet.
fn reports(workers: &[Arc<dyn Worker>]) -> HashMap<String, WorkerLoadResponse> {
    let report = |(at, worker): (usize, &Arc<dyn Worker>)| {
        let draw = |salt: u64| mix(at as u64 * 16 + salt);
        let rate = [0.0, 500.0 + (draw(3) % 3_500) as f64][usize::from(draw(3) % 4 != 0)];
        let queued = (draw(1) % 8_192) as i32;
        let mut report = one_rank_load(queued, (draw(2) % 95) as f64 / 100.0, rate);
        report.loads[0].num_running_reqs = (draw(4) % 12) as i32;
        (worker.url().to_owned(), report)
    };
    workers.iter().enumerate().map(report).collect()
}

/// One turn's request as its host states it: made where the engine makes
/// one, with the load snapshot borrowed where the engine first asks for it.
struct Stated<'a> {
    host: &'a Host,
    seed: u64,
    loads: OnceCell<Arc<LoadSnapshot>>,
}

impl RequestMaker for Stated<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        (self.host.request)(self.seed)
    }

    fn plain(&self) -> Option<Plain> {
        self.host.plain
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(&self.host.members)
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        let feed = self.host.loads.as_ref()?;
        Some(self.loads.get_or_init(|| feed.borrow().clone()))
    }
}

/// A list of worker objects as a host keeps it: the engine with what it
/// keeps of the list, the load monitor's feed where the document reads load
/// reports, and the cell's request.
struct Host {
    engine: PolicyEngine,
    members: PoolMembers,
    loads: Option<LoadReceiver>,
    request: fn(u64) -> Request<'static>,
    plain: Option<Plain>,
}

impl Host {
    /// One request by the engine's one entry: the place in the list it was
    /// sent to, with the dispatch and the end told where a decision is leased.
    fn place(&self, seed: u64) -> Option<u32> {
        let stated = Stated {
            host: self,
            seed,
            loads: OnceCell::new(),
        };
        // The instant the engine dated the decision at, where it dated one.
        let dated = OnceCell::new();
        let clock = || *dated.get_or_init(Instant::now);
        match self.engine.decide_at(&stated, clock).ok()? {
            Decided::Alone(pick) => Some(pick.place),
            Decided::Whole(decided, later) => {
                let target = *decided.decision.pick()?.targets.first()?;
                if decided.lease.is_some() {
                    let now = *dated.get()?;
                    let mut held =
                        DecisionGuard::holding(&self.engine, GatewayId::LOCAL, decided, later, now);
                    held.dispatched(now);
                    let end = Outcome {
                        side: Side::Single,
                        served: target.id,
                        status: 200,
                        class: TransportClass::Ok,
                        ttft_micros: 0,
                        tokens_in: 1_024,
                        tokens_out: 0,
                        duration_micros: 0,
                    };
                    held.completed(&end, now);
                }
                Some(target.position)
            }
        }
    }
}

/// A pair's decodes as its host keeps them beside its prefills (the host's
/// own list), with every worker's pair class.
struct Decodes {
    members: PoolMembers,
    classes: PartnerClasses,
}

/// One turn's pair request as its host states it: the request of
/// [`Stated`], over the host's two lists.
struct StatedPair<'a> {
    stated: Stated<'a>,
    decodes: &'a Decodes,
}

impl RequestMaker for StatedPair<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn make<'w>(&'w self, spec: &'w AppliedSpec) -> Request<'w> {
        self.stated.make(spec)
    }

    fn plain(&self) -> Option<Plain> {
        self.stated.plain()
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Pair(PairLists {
            prefill: &self.stated.host.members,
            decode: &self.decodes.members,
            classes: &self.decodes.classes,
            admission: &Open,
        })
    }

    fn loads(&self) -> Option<&Arc<LoadSnapshot>> {
        self.stated.loads()
    }
}

/// The engine as a pair's guard holds it. The type makes that guard an
/// instance of [`DecisionGuard`] apart from a single request's, whose calls
/// then have the one caller they have with no pair in the binary: a single
/// request's cell counts the same with a pair's cells beside it.
struct PairsEngine<'a>(&'a PolicyEngine);

impl Deref for PairsEngine<'_> {
    type Target = PolicyEngine;

    fn deref(&self) -> &PolicyEngine {
        self.0
    }
}

impl Decodes {
    /// One pair request of `host` by the engine's one entry: the prefill's
    /// place in its list, with both dispatches and each side's end told
    /// where the decision is leased.
    fn place(&self, host: &Host, seed: u64) -> Option<u32> {
        let stated = StatedPair {
            stated: Stated {
                host,
                seed,
                loads: OnceCell::new(),
            },
            decodes: self,
        };
        let dated = OnceCell::new();
        let clock = || *dated.get_or_init(Instant::now);
        let Decided::Whole(decided, later) = host.engine.decide_at(&stated, clock).ok()? else {
            return None;
        };
        let target = *decided.decision.pick()?.targets.first()?;
        let decode = target.pair?.decode_id;
        if decided.lease.is_some() {
            let now = *dated.get()?;
            let engine = PairsEngine(&host.engine);
            let mut held = DecisionGuard::holding(engine, GatewayId::LOCAL, decided, later, now);
            // The decode goes out with its prefill, and each side's end is
            // told under its own lease.
            held.dispatched(now);
            held.dispatched_decode(now);
            let end = |side, served| Outcome {
                side,
                served,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 1_024,
                tokens_out: 0,
                duration_micros: 0,
            };
            held.completed(&end(Side::Prefill, target.id), now);
            held.completed(&end(Side::Decode, decode), now);
        }
        Some(target.position)
    }
}

fn main() {
    let run = run();
    let (_, preset, request) = run.cell;
    let paired = matches!(request(0).facts.phase, Phase::Prefill);
    let side = |side: &str, worker_type| -> Arc<[Arc<dyn Worker>]> {
        let url = |at: usize| format!("grpc://{side}-{at:04}:9000");
        (0..run.workers)
            .map(|at| typed(url(at), worker_type))
            .collect()
    };
    let workers: Arc<[Arc<dyn Worker>]> = if paired {
        side("prefill", WorkerType::Prefill)
    } else {
        (0..run.workers)
            .map(|at| ready_worker(&format!("http://worker-{at:04}:8000"), MODEL))
            .collect()
    };
    let decodes = paired.then(|| side("decode", WorkerType::Decode));
    if run.overloaded {
        for list in std::iter::once(&workers).chain(&decodes) {
            for (at, worker) in list.iter().enumerate() {
                worker.set_overloaded(true);
                if at != list.len() / 2 {
                    for _ in 0..=at {
                        worker.increment_load();
                    }
                }
            }
        }
    }
    let document = if paired {
        PolicySpec::pair_presets(preset, preset)
    } else {
        PolicySpec::preset(preset)
    };
    let document = document.expect("a preset's name");
    let engine = PolicyEngine::new(AppliedSpec::validate(&document).expect("a valid preset"));
    let members = engine.pool_members(&workers);
    // Every prefill pairs with every decode.
    let partners: Vec<u32> = (0..run.workers as u32).collect();
    let decode_side = decodes.as_ref().map(|decodes| {
        let every = (0..workers.len()).map(|_| partners.as_slice());
        Decodes {
            classes: PartnerClasses::number(decodes.len(), every),
            members: engine.pool_members(decodes),
        }
    });
    let mut receiver = None;
    let feed = TestFeed::attached_to(|loads| receiver = loads);
    let mut loads = reports(&workers);
    loads.extend(decodes.iter().flat_map(|decodes| reports(decodes)));
    feed.publish(&loads);
    // `--reports 0`: the host lends no snapshot, whatever the document reads.
    let feed_read = engine.spec().needs().loads && run.reports;
    let host = Host {
        engine,
        members,
        loads: receiver.filter(|_| feed_read),
        request,
        plain: request(0).plain(),
    };
    // Each kind of cell runs its own caller: a pair's lines are in no
    // single request's path.
    match &decode_side {
        None => measure(&run, &host, |seed| host.place(seed)),
        Some(decodes) => measure(&run, &host, |seed| decodes.place(&host, seed)),
    }
}

/// The run's decisions, each made by `decide`, and the run's one line.
fn measure(run: &Run, host: &Host, decide: impl Fn(u64) -> Option<u32> + Sync) {
    // The list's first requests, untimed: what they build once.
    for turn in 0..(run.workers as u64 * 8).max(4_096) {
        assert!(decide(mix(turn)).is_some(), "a worker takes it");
    }

    // Each caller makes its first decisions untimed; all start together.
    let start = Barrier::new(run.callers + 1);
    let (busy_nanos, placed, allocations) =
        (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let wall = thread::scope(|scope| {
        let one_caller = |caller: u64| {
            let seed = |turn: u64| mix(caller << 40 | turn);
            for turn in 0..CALLER_WARM {
                black_box(decide(seed(turn)));
            }
            start.wait();
            let (allocated, started) = (calls(), Instant::now());
            let mut took = 0;
            for turn in CALLER_WARM..CALLER_WARM + run.decisions {
                took += u64::from(black_box(decide(black_box(seed(turn)))).is_some());
            }
            busy_nanos.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
            allocations.fetch_add((calls() - allocated) as u64, Ordering::Relaxed);
            placed.fetch_add(took, Ordering::Relaxed);
        };
        let callers: Vec<_> = (0..run.callers as u64)
            .map(|caller| scope.spawn(move || one_caller(caller)))
            .collect();
        start.wait();
        let started = Instant::now();
        for caller in callers {
            caller.join().expect("a caller that ends");
        }
        started.elapsed()
    });
    let total = run.decisions * run.callers as u64;
    assert_eq!(placed.into_inner(), total, "a worker takes it");
    println!(
        "cell={} workers={} callers={} overloaded={} reports={} allocations_per_decision={:.4} decisions={} wall_s={:.3} decisions_per_s={:.0} ns_per_decision={:.1}",
        run.cell.0,
        run.workers,
        run.callers,
        run.overloaded,
        host.loads.is_some(),
        allocations.into_inner() as f64 / total as f64,
        total,
        wall.as_secs_f64(),
        total as f64 / wall.as_secs_f64(),
        busy_nanos.into_inner() as f64 / total as f64,
    );
}
