//! The decode side decided later, for a prefill a host holds. Ids:
//! `policy_engine/decode_later/<n>`: the decode profile's one pass over the
//! decodes of `n` rows, a quarter of them prefills, and each target named
//! with the held prefill. The held prefill stands in the middle of its
//! pool: the entry finds it by a walk of the rows.
//! `policy_engine/decode_later_round_robin/{kept,picked}/<n>`: the same
//! call under round robin, handed the decode the pair's first decision
//! named (it is kept) and handed none (the picker turns again).
//! `policy_engine/decode_later_passthrough/{handed,picked}/<n>`: the same
//! two calls under `passthrough`, which decides again in both.
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

/// The second decision under round robin, a profile that keeps its pick.
/// `kept`: the entry is handed the decode in the middle of the decodes as
/// the one the first decision named, and answers it behind the profile's
/// filters. `picked`: handed none, it asks the picker again.
fn decode_later_round_robin(c: &mut Criterion) {
    decode_later_handed(c, "round_robin", "kept");
}

/// The second decision under `passthrough`, whose pick takes no turn: it
/// asks the picker again, `handed` the first decision's decode or none.
fn decode_later_passthrough(c: &mut Criterion) {
    decode_later_handed(c, "passthrough", "handed");
}

/// The second decision under the `decode` preset, over each size: handed
/// the decode in the middle of the decodes under the cell name `handed`,
/// and handed none under `picked`.
fn decode_later_handed(c: &mut Criterion, decode: &str, handed: &str) {
    let mut group = c.benchmark_group(format!("policy_engine/decode_later_{decode}"));
    let engine = PolicySpec::pair_presets("round_robin", decode)
        .expect("a pair spec of this build")
        .validate()
        .expect("the pair spec validates");
    for &n in &SIZES {
        let prefills = side_view(n / 4, Role::Prefill, 10_000);
        let decodes = side_view(n - n / 4, Role::Decode, 20_000);
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        };
        let pool = PoolFacts {
            pool_size: decodes.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let (cx, mut scratch) = (Constraints::default(), Scratch::new());
        let held = prefills[prefills.len() / 2].id;
        let first = &decodes[decodes.len() / 2];
        let first = Some((first.id, first.dp_rank));
        for (name, first) in [(handed, first), ("picked", None)] {
            let facts = RequestFacts {
                model: "m",
                now_micros: 1_000_000,
                ..RequestFacts::default()
            };
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                b.iter(|| {
                    engine.decide_decode(
                        &facts,
                        &pool,
                        black_box(pools),
                        held,
                        first,
                        &cx,
                        &mut scratch,
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    decode_later,
    decode_later_round_robin,
    decode_later_passthrough
);
criterion_main!(benches);
