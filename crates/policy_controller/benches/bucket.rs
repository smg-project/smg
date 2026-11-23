//! The `bucket` preset's cost per decision: the preset's own decision at 8,
//! 64 and 512 candidates, and a prefill placement with its booking. Ids:
//! `policy_engine/decide/bucket/<n>`,
//! `policy_engine/decide/bucket_prefill/<n>`,
//! `policy_engine/decide/bucket_prefill_stale/<n>`,
//! `policy_engine/decide/bucket_prefill_64_models/8`,
//! `policy_engine/decide/bucket_prefill_64_models_shared/8`,
//! `policy_engine/decide/bucket_prefill_joined/512` and
//! `policy_engine/decide/bucket_prefill_two_rows/256`. The fleet is the one
//! the `decision` bench decides over, and the first ids are the rows that
//! bench times for every other preset, built by the one body the two
//! binaries share (`common/preset.rs`). They stand here because that binary
//! runs at its gate's bound: a preset's rows there are three rows more in
//! every pull request's gate. This file is a pull-request gate's too
//! (`.github/bench-gates/policy-engine-bucket.toml` runs the whole binary
//! three times a side).
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;
#[path = "common/preset.rs"]
mod preset;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use common::{build_view, engine_of, fleet};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use preset::SIZES;
use smg_policy_controller::{
    engine::{Constraints, Phase, PoolFacts, RequestFacts, Scratch},
    stages::Lifecycle,
    tracking::{Lease, Membership},
};

/// How often the stale rows tell their leave again: well inside the ten
/// seconds its fence stands.
const RETOLD: Duration = Duration::from_secs(1);

fn decide(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/decide");
    preset::rows(&mut group, "bucket");
    bucket_prefill(&mut group);
    bucket_prefill_stale(&mut group);
    bucket_prefill_models(&mut group);
    bucket_prefill_parted(&mut group);
    group.finish();
}

/// A prefill placement under the bucket preset, booked as a host books it.
/// Each iteration decides for one request and issues the decision with the
/// clock one millisecond on, so the preset's five-second window holds about
/// five thousand bookings: each decision ages one out, and every
/// five-thousandth reaches an interval's end, where a cut falls due. The
/// first is made from the window, and none after it while each worker's
/// share stays within a factor of two of the last cut's. The sizes step
/// through the span, so the ranges' owners take turns.
fn bucket_prefill(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &n in &SIZES {
        let engine = engine_of("bucket");
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new("bucket_prefill", n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let facts = RequestFacts {
                    model: "m",
                    phase: Phase::Prefill,
                    prompt_tokens: tokens,
                    seed,
                    now_micros,
                    ..RequestFacts::default()
                };
                let decision = engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
}

/// A prefill placement over a view older than a leave. The model's first
/// decision lists every worker, the lifecycle prunes one the filters keep,
/// and each timed decision's view still shows it: the decision finds a
/// survivor its partition does not list, asks the fence of it, lists
/// nothing new and seats the others. The fence stands ten seconds, so the
/// leave is told again once a second, as a feed that repeats it would.
fn bucket_prefill_stale(group: &mut BenchmarkGroup<'_, WallTime>) {
    for &n in &SIZES {
        let engine = engine_of("bucket");
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let listing = RequestFacts {
            model: "m",
            phase: Phase::Prefill,
            now_micros: 1_000_000,
            ..RequestFacts::default()
        };
        black_box(engine.decide(&listing, &pool, &view, &cx, &mut scratch));
        // The first healthy worker from the middle of the view on leaves: a
        // row the filters keep, with half the view behind it.
        let gone = view[n / 2..]
            .iter()
            .find(|row| row.health.healthy)
            .expect("a healthy row in the view's second half")
            .id;
        engine.on_membership(&Membership::Removed(gone));
        let mut told = Instant::now();
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new("bucket_prefill_stale", n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                if seed.is_multiple_of(256) && told.elapsed() >= RETOLD {
                    engine.on_membership(&Membership::Removed(gone));
                    told = Instant::now();
                }
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let facts = RequestFacts {
                    model: "m",
                    phase: Phase::Prefill,
                    prompt_tokens: tokens,
                    seed,
                    now_micros,
                    ..RequestFacts::default()
                };
                let decision = engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
}

/// The same placement where 64 models take turns: each over 8 workers of its
/// own, then every model over the same 8. A decision finds its model's
/// partition by the name and a booking its worker's, so the first cell
/// reads what the count of models adds to a placement. A lease names no
/// model: where the models list the same workers, the second cell books
/// each issued decision in the 64 partitions that list its worker, one
/// guard after the other.
fn bucket_prefill_models(group: &mut BenchmarkGroup<'_, WallTime>) {
    const MODELS: usize = 64;
    let n = 8;
    for (name, shared) in [
        ("bucket_prefill_64_models", false),
        ("bucket_prefill_64_models_shared", true),
    ] {
        let engine = engine_of("bucket");
        let fleet = fleet(MODELS * n);
        let mut rows = Vec::with_capacity(MODELS * n);
        build_view(&fleet, &mut rows);
        // A model's view is 8 rows, each at its place in that view: its own
        // 8, or the fleet's first 8 where the models list the same workers.
        let views: Vec<(String, Vec<_>)> = rows
            .chunks(n)
            .enumerate()
            .map(|(at, own)| {
                let mut view = if shared {
                    rows[..n].to_vec()
                } else {
                    own.to_vec()
                };
                for (position, row) in view.iter_mut().enumerate() {
                    row.position = position as u32;
                }
                (format!("model-{at:02}"), view)
            })
            .collect();
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
            b.iter(|| {
                seed += 1;
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let (model, view) = &views[seed as usize % MODELS];
                let facts = RequestFacts {
                    model: model.as_str(),
                    phase: Phase::Prefill,
                    prompt_tokens: tokens,
                    seed,
                    now_micros,
                    ..RequestFacts::default()
                };
                let decision = engine.decide(&facts, &pool, black_box(view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
}

/// Two views that part from the listing's order, which a placement must not
/// pay for by a search of the listing: at 512 workers one that joined in the
/// middle of the view, and at 256 workers two rows a worker. Every row past
/// the parting is found by its worker's id.
fn bucket_prefill_parted(group: &mut BenchmarkGroup<'_, WallTime>) {
    let fleet = fleet(512);
    let mut whole = Vec::with_capacity(512);
    build_view(&fleet, &mut whole);
    // The view before the join: every row but the middle one.
    let mut before = whole.clone();
    before.remove(256);
    // Two rows a worker: each of the first 256 workers as two ranks.
    let mut twice = Vec::with_capacity(512);
    for row in &whole[..256] {
        for rank in 0..2 {
            let mut row = *row;
            row.dp_rank = rank;
            row.dp_size = 2;
            twice.push(row);
        }
    }
    for view in [&mut before, &mut twice] {
        for (position, row) in view.iter_mut().enumerate() {
            row.position = position as u32;
        }
    }
    for (name, workers, listed, view) in [
        ("bucket_prefill_joined", 512usize, &before, &whole),
        ("bucket_prefill_two_rows", 256usize, &twice, &twice),
    ] {
        let engine = engine_of("bucket");
        let mut scratch = Scratch::new();
        let cx = Constraints::default();
        let facts_of = |tokens: u32, seed: u64, now_micros: u64| RequestFacts {
            model: "m",
            phase: Phase::Prefill,
            prompt_tokens: tokens,
            seed,
            now_micros,
            ..RequestFacts::default()
        };
        let pool_of = |rows: usize| PoolFacts {
            pool_size: rows as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        // One decision lists the model's workers in the order of `listed`.
        black_box(engine.decide(
            &facts_of(0, 0, 1_000_000),
            &pool_of(listed.len()),
            listed,
            &cx,
            &mut scratch,
        ));
        let pool = pool_of(view.len());
        let mut seed = 0u64;
        group.bench_with_input(BenchmarkId::new(name, workers), &workers, |b, _| {
            b.iter(|| {
                seed += 1;
                let now_micros = 1_000_000 + seed * 1_000;
                let tokens = (seed.wrapping_mul(2_654_435_761) % 4_096) as u32;
                let facts = facts_of(tokens, seed, now_micros);
                let decision = engine.decide(&facts, &pool, black_box(view), &cx, &mut scratch);
                let lease = Lease {
                    tokens,
                    issued_micros: now_micros,
                    ..Lease::default()
                };
                engine.on_decision_issued(&lease, &decision);
                decision
            });
        });
    }
}

criterion_group!(benches, decide);
criterion_main!(benches);
