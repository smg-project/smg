//! The in-process engine over a host's worker objects, cell by cell.
//!
//! The engine of a gateway's own process ([`PolicyEngine`]) decides over
//! the gateway's lists of worker objects and its load monitor's snapshot.
//! A cell is one such list under one document, with the request its callers
//! make: one entry of [`CELLS`]. A caller asks the one entry a host asks
//! for a request ([`PolicyEngine::decide_at`]) and says what the host then
//! owes the engine of it. A place answered alone owes nothing. A whole
//! decision is held by a guard, which is told the dispatch and the end
//! ([`DecisionGuard`]).
//!
//! The request is made, the clock read and the load snapshot borrowed
//! where the engine asks for them: a place answered alone takes none. No
//! other work of a host is done: no route, no metric, no in-flight count.
//!
//! ```text
//! cargo bench -p smg-policy-controller --features in-process --bench in_process -- \
//!     --cell power_of_two --workers 64 --callers 1 --decisions 1000000
//! ```
//!
//! `--workers` is the list's size: every worker is ready and has a load
//! report. `--publish-ms N` publishes the reports again, as a new snapshot,
//! every `N` milliseconds while the callers run; `0`, the default,
//! publishes once. `--callers` threads each make `--decisions` timed
//! decisions behind untimed first ones. The result is one line.
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
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Barrier,
    },
    thread,
    time::{Duration, Instant},
};

use counting_alloc::{calls, Counting};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{
        Constraints, Decided, Phase, Plain, PolicyEngine, PoolMembers, Request, RequestFacts,
        RequestMaker, WorkerLists,
    },
    store::GatewayId,
    tracking::{DecisionGuard, Outcome, Side, TransportClass},
};
use smg_worker::{
    identity::basic::Worker,
    load::state::{one_rank_load, LoadReceiver, LoadSnapshot, TestFeed},
    test_support::ready_worker,
    worker::WorkerLoadResponse,
};

#[global_allocator]
static GLOBAL: Counting = Counting;

const MODEL: &str = "bench-model";

/// Decisions a caller makes untimed first: the buffers a thread keeps.
const CALLER_WARM: u64 = 256;

/// A cell: its name, the preset its document is, and the request of one
/// turn of its callers, by the turn's seed.
type Cell = (&'static str, &'static str, fn(u64) -> Request<'static>);

/// The cells. A change that moves a policy's cost adds its entry here.
const CELLS: [Cell; 6] = [
    ("round_robin", "round_robin", plain),
    ("random", "random", plain),
    ("passthrough", "passthrough", plain),
    ("power_of_two", "power_of_two", plain),
    ("least_load", "least_load", plain),
    ("bucket", "bucket", plain),
];

/// A plain single request: a first attempt that asks only a worker.
fn plain(seed: u64) -> Request<'static> {
    let facts = RequestFacts {
        model: MODEL,
        phase: Phase::Single,
        prompt_tokens: 64,
        seed,
        ..RequestFacts::default()
    };
    Request {
        id: b"bench",
        facts,
        constraints: Constraints::default(),
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
    publish_ms: u64,
}

fn run() -> Run {
    let mut run = Run {
        cell: CELLS[0],
        workers: 64,
        callers: 1,
        decisions: 1_000_000,
        publish_ms: 0,
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
            "--publish-ms" => run.publish_ms = number(),
            other => panic!("no switch {other}"),
        }
    }
    run
}

/// One report a worker: queued token-work, KV pressure and a live rate.
fn reports(workers: &[Arc<dyn Worker>]) -> HashMap<String, WorkerLoadResponse> {
    let report = |(at, worker): (usize, &Arc<dyn Worker>)| {
        let draw = |salt: u64| mix(at as u64 * 16 + salt);
        let rate = [0.0, 500.0 + (draw(3) % 3_500) as f64][usize::from(draw(3) % 4 != 0)];
        let queued = (draw(1) % 8_192) as i32;
        let report = one_rank_load(queued, (draw(2) % 95) as f64 / 100.0, rate);
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
        Some(self.loads.get_or_init(|| self.host.loads.borrow().clone()))
    }
}

/// A list of worker objects as a host keeps it: the engine with what it
/// keeps of the list, the load monitor's side, and the cell's request.
struct Host {
    engine: PolicyEngine,
    members: PoolMembers,
    loads: LoadReceiver,
    request: fn(u64) -> Request<'static>,
    plain: Option<Plain>,
}

impl Host {
    /// One request by the engine's one entry: the place in the list it was
    /// sent to, with the dispatch and the end told where a decision is held.
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
                let now = *dated.get()?;
                let mut held =
                    DecisionGuard::holding(&self.engine, GatewayId::LOCAL, decided, later, now);
                let target = *held.decision().pick()?.targets.first()?;
                held.dispatched(now);
                let end = Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 64,
                    tokens_out: 0,
                    duration_micros: 0,
                };
                held.completed(&end, now);
                Some(target.position)
            }
        }
    }
}

fn main() {
    let run = run();
    let (name, preset, request) = run.cell;
    let workers: Arc<[Arc<dyn Worker>]> = (0..run.workers)
        .map(|at| ready_worker(&format!("http://worker-{at:04}:8000"), MODEL))
        .collect();
    let mut document = PolicySpec::preset(preset).expect("a preset's name");
    document.generation = 1;
    let engine = PolicyEngine::new(AppliedSpec::validate(&document).expect("a valid preset"));
    let members = engine.pool_members(&workers);
    let mut receiver = None;
    let feed = TestFeed::attached_to(|loads| receiver = loads);
    let loads = reports(&workers);
    feed.publish(&loads);
    let host = Host {
        engine,
        members,
        loads: receiver.expect("the feed's reader"),
        request,
        plain: request(0).plain(),
    };
    let decide = |seed: u64| host.place(seed);
    // The list's first requests, untimed: what they build once.
    for turn in 0..(run.workers as u64 * 8).max(4_096) {
        assert!(decide(mix(turn)).is_some(), "a ready worker takes it");
    }

    // Each caller makes its first decisions untimed; all start together.
    let start = Barrier::new(run.callers + 1);
    let (busy_nanos, placed, allocations) =
        (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let (publications, published) = (AtomicU64::new(0), AtomicBool::new(false));
    let converted_before = host.members.conversions();
    let wall = thread::scope(|scope| {
        // The monitor's side: the reports again, as a new snapshot.
        if run.publish_ms > 0 {
            scope.spawn(|| {
                while !published.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(run.publish_ms));
                    feed.publish(&loads);
                    publications.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
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
        published.store(true, Ordering::Relaxed);
        started.elapsed()
    });
    let total = run.decisions * run.callers as u64;
    assert_eq!(placed.into_inner(), total, "a ready worker takes it");
    println!(
        "cell={} workers={} callers={} publish_ms={} publications={} conversions={} allocations_per_decision={:.4} decisions={} wall_s={:.3} decisions_per_s={:.0} ns_per_decision={:.1}",
        name,
        run.workers,
        run.callers,
        run.publish_ms,
        publications.into_inner(),
        host.members.conversions() - converted_before,
        allocations.into_inner() as f64 / total as f64,
        total,
        wall.as_secs_f64(),
        total as f64 / wall.as_secs_f64(),
        busy_nanos.into_inner() as f64 / total as f64,
    );
}
