//! The decode side decided later, for a prefill a host holds. Ids:
//! `policy_engine/decode_later/<n>`: the decode profile's one pass over the
//! decodes of `n` rows, a quarter of them prefills, and each target named
//! with the held prefill. The held prefill stands in the middle of its
//! pool: the entry finds it by a walk of the rows.
//!
//! The cells stand in a bench of their own. A bench binary is optimized
//! whole, so the entries it links shape every cell in it: beside the pair
//! entries this one would move their cells.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

#[expect(
    dead_code,
    reason = "this bench builds its engine from the epd presets; `common::engine_of` serves the decision and contention benches"
)]
mod common;

use std::hint::black_box;

use common::{build_view, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    document::PolicySpec,
    engine::{Candidate, Constraints, PairPools, PoolFacts, RequestFacts, Role, Scratch, WorkerId},
};

const SIZES: [usize; 3] = [8, 64, 512];

/// One side of the fleet: `n` ready rows of `role` in pair class 1, every
/// admission bit set, each with the synthetic load of its place.
fn side_view(n: usize, role: Role, first: u64) -> Vec<Candidate> {
    let mut out = Vec::with_capacity(n);
    build_view(&fleet(n), &mut out);
    for row in &mut out {
        row.id = WorkerId(first + u64::from(row.position));
        row.key = (first + u64::from(row.position)) as u32;
        row.role = role;
        row.health.healthy = true;
        row.pair_class = 1;
        row.has_capacity = true;
        row.has_room = true;
        row.vision = true;
    }
    out
}

/// The second decision of a pair whose prefill went out first: the decode
/// profile, by its least load, over the decodes of the held prefill.
fn decode_later(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/decode_later");
    let engine = PolicySpec::epd_presets("consistent_hashing", "least_load", "least_load")
        .expect("an epd spec of this build")
        .validate()
        .expect("the epd spec validates");
    for &n in &SIZES {
        let prefills = side_view(n / 4, Role::Prefill, 10_000);
        let decodes = side_view(n - n / 4, Role::Decode, 20_000);
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        // The facts of the decode pool: the entry's profile places there.
        let pool = PoolFacts {
            pool_size: decodes.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let held = prefills[prefills.len() / 2].id;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                engine.decide_decode(
                    &RequestFacts {
                        model: "m",
                        seed,
                        now_micros: 1_000_000,
                        ..RequestFacts::default()
                    },
                    &pool,
                    black_box(pools),
                    held,
                    None,
                    &cx,
                    &mut scratch,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, decode_later);
criterion_main!(benches);
