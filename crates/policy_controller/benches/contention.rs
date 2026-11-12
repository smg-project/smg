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
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    Candidate, Constraints, Engine, Lease, Lifecycle, PoolFacts, RequestFacts, Scratch, Side,
};

/// The presets with a stateful stage (the credit tally, the rotation table)
/// and the stateless control. `round_robin_hits` decides over a survivor set
/// that differs per decision (one tried exclusion rotating through the pool):
/// as many sets as rows, each claimed once and hit after that, so the
/// rotation table's hit path spread over many words. `round_robin_churn`
/// excludes three rows drawn from a mixed seed, so most decisions see a set
/// the table has forgotten: the claim path, with its one shared count.
/// `least_load_rising` dates its bookings at one clock every thread advances,
/// so a rank row's lanes interleave by instant and an overflow's batch merges
/// them (the tied rows drain one lane): the price of retiring in order.
const PRESETS: [&str; 6] = [
    "least_load",
    "least_load_rising",
    "round_robin",
    "round_robin_hits",
    "round_robin_churn",
    "passthrough",
];

/// A golden-ratio multiplier: a decision's three drawn exclusions are slices
/// of its seed's one product.
const DRAW_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const THREADS: [usize; 2] = [1, 8];
const SIZES: [usize; 2] = [64, 512];

/// The rising rows' clock: every booking of every thread a later instant.
static CLOCK: AtomicU64 = AtomicU64::new(0);

/// How a preset's decisions run: the tried exclusions per decision (none,
/// one rotating through the pool, or three drawn) and whether the bookings
/// are dated at the rising clock.
#[derive(Clone, Copy)]
struct Shape {
    exclusions: usize,
    rising: bool,
}

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
    shape: Shape,
) -> Duration {
    let mut scratch = Scratch::new();
    // Dated after every row's sample, so the credit stands and the next
    // decision prices it: at one instant, or at the clock every booking of
    // every thread advances, so the lanes interleave.
    let dated = || Lease {
        issued_micros: if shape.rising {
            1_000_000 + CLOCK.fetch_add(1, Ordering::Relaxed)
        } else {
            1_000_000
        },
        tokens: 512,
        ..Lease::default()
    };
    let mut seed = thread << 48;
    let decide = |seed: u64, scratch: &mut Scratch| {
        let n = view.len() as u64;
        let z = seed.wrapping_mul(DRAW_MIX);
        let drawn = [
            view[(seed % n) as usize].id,
            view[((z >> 16) % n) as usize].id,
            view[((z >> 32) % n) as usize].id,
        ];
        let facts = RequestFacts {
            model: "m",
            prompt_tokens: 512,
            seed,
            now_micros: 1_000_000,
            tried: &drawn[..shape.exclusions],
            ..RequestFacts::default()
        };
        let decision = engine.decide(&facts, pool, black_box(view), cx, scratch);
        if let Some(target) = decision.pick().and_then(|p| p.targets.first()) {
            engine.on_dispatched(&dated(), target, Side::Single, 0);
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
        let engine = common::engine_of(
            preset
                .strip_suffix("_churn")
                .or_else(|| preset.strip_suffix("_hits"))
                .or_else(|| preset.strip_suffix("_rising"))
                .unwrap_or(preset),
        );
        let shape = Shape {
            exclusions: match preset {
                "round_robin_churn" => 3,
                "round_robin_hits" => 1,
                _ => 0,
            },
            rising: preset.ends_with("_rising"),
        };
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
                                            decide_and_book(engine, view, pool, cx, t, iters, shape)
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
