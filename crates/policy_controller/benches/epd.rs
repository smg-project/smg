//! The `epd` plan's encode half and the ring behind it, the groups that grew
//! past the pull-request gate's budget in the `decision` bench (that file is
//! the gate's; `.github/bench-gates/policy-engine.toml` runs it whole, three
//! times a side). Ids: `policy_engine/epd/<preset>/<items>/<n>` (the encode
//! items of one request decided one by one over an encode pool of `n`, each
//! by its own key, under the `epd` plan), `policy_engine/ring/learn/<n>` (a
//! ring learning `n` members in one merge, the cold start of a pool),
//! `policy_engine/ring/learn_one_at_a_time/<n>` (the same pool added one
//! member at a time, the cost the one merge replaces) and
//! `policy_engine/ring/add_remove/<n>` (one member added to a ring of `n` and
//! removed again). The view is built from the synthetic per-worker state of
//! `common` the way a host reads its routing state; the decision runs with a
//! fresh seed per iteration so the drawing pickers draw.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

#[expect(
    dead_code,
    reason = "this bench builds its engine from the epd presets; `common::engine_of` serves the decision and contention benches"
)]
mod common;

use std::{hint::black_box, time::Instant};

use common::{build_view, fleet, WorkerState};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smg_policy_controller::{
    Candidate, Constraints, Identity, Phase, PolicySpec, PoolFacts, RequestFacts, Ring, Role,
    Scratch, WorkerId,
};

const SIZES: [usize; 3] = [8, 64, 512];

/// The encode pool as the host fills it: every encoder compatible with the
/// chosen prefill side and taking images.
fn build_encode_view(fleet: &[WorkerState], out: &mut Vec<Candidate>) {
    build_view(fleet, out);
    for row in out.iter_mut() {
        row.role = Role::Encode;
        row.ec_compatible = true;
        row.vision = true;
    }
}

/// The encode items of one request decided one by one, each by its own
/// content hash as the decision's key: the `epd` plan's encode half under
/// the three encode presets. The ring's members are learned at the first
/// decision, before the measurement.
fn epd(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/epd");
    for preset in ["consistent_hashing", "least_load", "round_robin"] {
        let engine = PolicySpec::epd_presets(preset, "round_robin", "round_robin")
            .expect("an epd spec of this build")
            .validate()
            .expect("the epd spec validates");
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_encode_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            for items in [1u64, 4, 16] {
                group.throughput(Throughput::Elements(items));
                group.bench_with_input(
                    BenchmarkId::new(format!("{preset}/{items}"), n),
                    &n,
                    |b, _| {
                        let mut seed = 0u64;
                        b.iter(|| {
                            seed += 1;
                            let mut picked = 0u32;
                            for item in 0..items {
                                let key = (seed * 16 + item).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                                let facts = RequestFacts {
                                    model: "m",
                                    phase: Phase::Encode,
                                    session_key: Identity::hashed(key | 1),
                                    seed: seed ^ item,
                                    now_micros: 1_000_000,
                                    ..RequestFacts::default()
                                };
                                let decision = engine.decide(
                                    &facts,
                                    &pool,
                                    black_box(&view),
                                    &cx,
                                    &mut scratch,
                                );
                                picked += decision.pick().map_or(0, |p| p.targets.len() as u32);
                            }
                            picked
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

/// A membership change on the ring: one member added and removed again on a
/// ring of `n` members (the rebuild cost a registration or a removal pays).
fn ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/ring");
    for &n in &SIZES {
        let members: Vec<(WorkerId, u32)> = (0..n as u64)
            .map(|i| (WorkerId(1_000 + i), 1_000 + i as u32))
            .collect();
        let now = Instant::now();
        group.bench_with_input(BenchmarkId::new("learn", n), &n, |b, _| {
            b.iter(|| {
                let mut ring = Ring::new();
                ring.learn(black_box(&members).iter().copied(), now);
                ring.len()
            });
        });
        group.bench_with_input(BenchmarkId::new("learn_one_at_a_time", n), &n, |b, _| {
            b.iter(|| {
                let mut ring = Ring::new();
                for &(id, key) in black_box(&members) {
                    ring.add(id, key);
                }
                ring.len()
            });
        });
        let mut ring = Ring::new();
        ring.learn(members.iter().copied(), now);
        let newcomer = WorkerId(1);
        group.bench_with_input(BenchmarkId::new("add_remove", n), &n, |b, _| {
            b.iter(|| {
                ring.add(black_box(newcomer), 1);
                ring.remove(newcomer, now);
                ring.len()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, epd, ring);
criterion_main!(benches);
