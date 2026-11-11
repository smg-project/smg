//! The engine under concurrent decisions: one, then eight threads decide over
//! one built view through one engine, each on its own scratch, and each books
//! the target it picked as a host's commit does, so the stateful stages (the
//! least-load credit tally, the round-robin rotation) are written at the pace
//! they are read. A booking is dated at the decision's clock, after every
//! row's report sample, so its credit stands until a later report releases it
//! (the steady state between two reports under load; a credit released by the
//! next decision would measure the report's moment at full duty). Ids:
//! `policy_engine_threads/<preset>/<threads>/<n>`. A row's time is one
//! decision on one thread (the slowest thread's wall time over its decisions),
//! so the eight-thread row against the one-thread row of the same shape reads
//! what the stages' locks cost under contention; `passthrough` holds no lock
//! and is the control. A bench target of its own, so the gated `decision`
//! bench keeps its single-threaded rows.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::{
    hint::black_box,
    thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    Candidate, Constraints, Engine, Lease, Lifecycle, PoolFacts, RequestFacts, Scratch, Side,
};

/// The presets with a stateful stage (the credit tally, the rotation table)
/// and the stateless control; `round_robin_churn` decides over a survivor
/// set that differs per decision (one tried exclusion rotating through the
/// pool), the rotation table's slot-claim path.
const PRESETS: [&str; 4] = [
    "least_load",
    "round_robin",
    "round_robin_churn",
    "passthrough",
];
const THREADS: [usize; 2] = [1, 8];
const SIZES: [usize; 2] = [64, 512];

/// One thread's `iters` decisions from its own seed range on its own scratch,
/// each pick booked at once; the wall time they took. The first decision
/// fills the scratch outside the timed span.
fn decide_and_book(
    engine: &Engine,
    view: &[Candidate],
    pool: &PoolFacts<'_>,
    cx: &Constraints<'_>,
    thread: u64,
    iters: u64,
    churn: bool,
) -> Duration {
    let mut scratch = Scratch::new();
    // Dated at the decision's clock, after every row's sample: the credit
    // stands, so the next decision prices it and picks elsewhere.
    let lease = Lease {
        issued_micros: 1_000_000,
        tokens: 512,
        ..Lease::default()
    };
    let mut seed = thread << 48;
    let decide = |seed: u64, scratch: &mut Scratch| {
        let tried = [view[(seed % view.len() as u64) as usize].id];
        let facts = RequestFacts {
            model: "m",
            prompt_tokens: 512,
            seed,
            now_micros: 1_000_000,
            tried: if churn { &tried } else { &[] },
            ..RequestFacts::default()
        };
        let decision = engine.decide(&facts, pool, black_box(view), cx, scratch);
        if let Some(target) = decision.pick().and_then(|p| p.targets.first()) {
            engine.on_dispatched(&lease, target, Side::Single, 0);
        }
        black_box(&decision);
    };
    seed += 1;
    decide(seed, &mut scratch);
    let start = Instant::now();
    for _ in 0..iters {
        seed += 1;
        decide(seed, &mut scratch);
    }
    start.elapsed()
}

fn decide_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine_threads");
    for preset in PRESETS {
        let engine = common::engine_of(preset.strip_suffix("_churn").unwrap_or(preset));
        let churn = preset.ends_with("_churn");
        for &n in &SIZES {
            let fleet = common::fleet(n);
            let mut view = Vec::with_capacity(n);
            common::build_view(&fleet, &mut view);
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            for &threads in &THREADS {
                group.bench_with_input(
                    BenchmarkId::new(format!("{preset}/{threads}"), n),
                    &n,
                    |b, _| {
                        b.iter_custom(|iters| {
                            thread::scope(|s| {
                                let deciders: Vec<_> = (0..threads as u64)
                                    .map(|t| {
                                        let (engine, view, pool, cx) = (&engine, &view, &pool, &cx);
                                        s.spawn(move || {
                                            decide_and_book(engine, view, pool, cx, t, iters, churn)
                                        })
                                    })
                                    .collect();
                                deciders
                                    .into_iter()
                                    .map(|h| h.join().expect("a deciding thread"))
                                    .max()
                                    .unwrap_or_default()
                            })
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, decide_threads);
criterion_main!(benches);
