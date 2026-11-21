//! The `epd` plan in one decision. Ids:
//! `policy_engine/encode_pair/<preset>/<items>/<n>` (the pair over `n`
//! prefills and decodes, then the encode items over `n / 8` encoders of the
//! chosen prefill's class; with 0 items the request reads no encoder and no
//! membership) and `policy_engine/encode_pair/pair_alone/<n>`
//! (the pair alone over the same pools: the gap is the encode side's cost),
//! `policy_engine/encode_pair/left_out/1/<n>` (rows left out on both
//! sides, so the pair is picked over copies of the admitted rows), and
//! `policy_engine/encode_pair/far/1/<n>` (the same with its encode classes
//! numbered above the rows handed, so the encode side reads its lists).
//!
//! The cells stand in a bench of their own. A bench binary is optimized
//! whole, so the entries it links shape every cell in it: beside the pair
//! entries, the per-item cells of the `epd` bench would no longer hold the
//! profile run inlined in their single decision.
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
    engine::{
        Candidate, Constraints, EncodeItems, EncodePlaced, EncodePool, EncoderFacts, Identity,
        MediaCounts, PairPools, PoolFacts, RequestFacts, Role, Scratch, WorkerId,
    },
};

const SIZES: [usize; 3] = [8, 64, 512];

/// One side of the one decision's fleet: `n` ready rows of `role` in pair
/// class 1, every admission bit set, each with the synthetic load of its
/// place.
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

/// A request of the `epd` plan in one decision: its pair over `n / 4`
/// prefills and the rest of `n` decodes, each side by its least load, then
/// its encode items over `n / 8` encoders of one pushing class, each item by
/// its own content hash. Beside it the pair alone over the same two pools:
/// what the encode side costs is the gap. The ring's members are learned at
/// the first decision, before the measurement.
fn encode_pair(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/encode_pair");
    for preset in ["consistent_hashing", "least_load"] {
        let engine = PolicySpec::epd_presets(preset, "least_load", "least_load")
            .expect("an epd spec of this build")
            .validate()
            .expect("the epd spec validates");
        for &n in &SIZES {
            let prefills = side_view(n / 4, Role::Prefill, 10_000);
            let decodes = side_view(n - n / 4, Role::Decode, 20_000);
            let encoders = side_view((n / 8).max(1), Role::Encode, 30_000);
            let of_encoders = vec![
                EncoderFacts {
                    class: 1,
                    pushes: true,
                    references: false,
                };
                encoders.len()
            ];
            let prefill_classes = vec![1u32; prefills.len()];
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let encode = EncodePool {
                encoders: &encoders,
                facts: &of_encoders,
                prefill_classes: &prefill_classes,
                affinity: &[],
                ..EncodePool::default()
            };
            let pool = PoolFacts {
                pool_size: prefills.len() as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            let mut scratch = Scratch::new();
            let mut placed = EncodePlaced::new();
            let facts_of = |seed: u64| RequestFacts {
                model: "m",
                media: MediaCounts {
                    images: 1,
                    ..MediaCounts::default()
                },
                seed,
                now_micros: 1_000_000,
                ..RequestFacts::default()
            };
            if preset == "consistent_hashing" {
                group.bench_with_input(BenchmarkId::new("pair_alone", n), &n, |b, _| {
                    let mut seed = 0u64;
                    b.iter(|| {
                        seed += 1;
                        engine.decide_pair(
                            &facts_of(seed),
                            &pool,
                            black_box(pools),
                            &cx,
                            &cx,
                            &mut scratch,
                        )
                    });
                });
            }
            for items in [0usize, 1, 4] {
                group.bench_with_input(
                    BenchmarkId::new(format!("{preset}/{items}"), n),
                    &n,
                    |b, _| {
                        let mut seed = 0u64;
                        let mut keys = vec![Identity::NONE; items];
                        b.iter(|| {
                            seed += 1;
                            for (item, key) in (0u64..).zip(keys.iter_mut()) {
                                let hash = (seed * 16 + item).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                                *key = Identity::hashed(hash | 1);
                            }
                            let asked = EncodeItems {
                                keys: &keys,
                                one: None,
                                references: false,
                                decode_images: false,
                            };
                            let decision = engine.decide_encode_pair(
                                &facts_of(seed),
                                &pool,
                                black_box(pools),
                                black_box(encode),
                                asked,
                                &cx,
                                &cx,
                                &cx,
                                &mut scratch,
                                &mut placed,
                            );
                            black_box(placed.encoders().len());
                            decision
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

/// The one decision with rows left out on both sides: the prefills name two
/// encode classes in turn and the second class has no ready encoder; every
/// other decode takes no image, and the request asks one of its decode. The
/// pair is picked over copies of the admitted rows, half of each side. The
/// cell runs again as `far`, its two classes numbered above the rows handed
/// as a gateway's view of host-local pairs holds them.
fn left_out(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/encode_pair");
    let engine = PolicySpec::epd_presets("consistent_hashing", "least_load", "least_load")
        .expect("an epd spec of this build")
        .validate()
        .expect("the epd spec validates");
    for &n in &SIZES {
        let prefills = side_view(n / 4, Role::Prefill, 10_000);
        let mut decodes = side_view(n - n / 4, Role::Decode, 20_000);
        let mut encoders = side_view((n / 8).max(2), Role::Encode, 30_000);
        for row in decodes.iter_mut().skip(1).step_by(2) {
            row.vision = false;
        }
        for row in encoders.iter_mut().skip(1).step_by(2) {
            row.health.healthy = false;
        }
        for (name, first) in [("left_out", 1u32), ("far", 1 << 20)] {
            let of_encoders: Vec<EncoderFacts> = (0..encoders.len() as u32)
                .map(|at| EncoderFacts {
                    class: first + at % 2,
                    pushes: true,
                    references: false,
                })
                .collect();
            let prefill_classes: Vec<u32> = (0..prefills.len() as u32)
                .map(|at| first + at % 2)
                .collect();
            let pools = PairPools {
                prefills: &prefills,
                decodes: &decodes,
                ..PairPools::default()
            };
            let encode = EncodePool {
                encoders: &encoders,
                facts: &of_encoders,
                prefill_classes: &prefill_classes,
                affinity: &[],
                ..EncodePool::default()
            };
            let pool = PoolFacts {
                pool_size: prefills.len() as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            let mut scratch = Scratch::new();
            let mut placed = EncodePlaced::new();
            group.bench_with_input(BenchmarkId::new(format!("{name}/1"), n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let facts = RequestFacts {
                        model: "m",
                        media: MediaCounts {
                            images: 1,
                            ..MediaCounts::default()
                        },
                        seed,
                        now_micros: 1_000_000,
                        ..RequestFacts::default()
                    };
                    let keys = [Identity::hashed(
                        (seed * 16).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1,
                    )];
                    let asked = EncodeItems {
                        keys: &keys,
                        one: None,
                        references: false,
                        decode_images: true,
                    };
                    let decision = engine.decide_encode_pair(
                        &facts,
                        &pool,
                        black_box(pools),
                        black_box(encode),
                        asked,
                        &cx,
                        &cx,
                        &cx,
                        &mut scratch,
                        &mut placed,
                    );
                    black_box(placed.encoders().len());
                    decision
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, encode_pair, left_out);
criterion_main!(benches);
