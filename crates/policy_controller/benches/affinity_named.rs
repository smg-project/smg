//! The cost of one decision over the rows a host names in the affinity
//! column's place: the `cache_aware` preset over `common`'s settled fleet,
//! every holder of its column named and no column fed, at 8, 64 and 512
//! candidates. Ids: `policy_affinity_named/decide/cache_aware/<n>`; the same
//! decision over the column is `policy_affinity/decide/cache_aware/<n>` of
//! the `affinity` target. The settled fleet warms nobody, so no warming row
//! is named. A bench target of its own: a row added to the gated `affinity`
//! binary would move that binary's rows through its inlining.

#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

#[path = "common/column.rs"]
mod column;
mod common;

use std::hint::black_box;

use column::build_column;
use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    Constraints, NamedAffinity, NamedHolder, PoolFacts, RequestFacts, Scratch,
};

const SIZES: [usize; 3] = [8, 64, 512];

fn facts(seed: u64, holders: &[NamedHolder]) -> RequestFacts<'_> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        seed,
        now_micros: 1_000_000,
        named: Some(NamedAffinity {
            holders,
            warming: &[],
            thin: false,
        }),
        ..RequestFacts::default()
    }
}

fn decide(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_affinity_named/decide");
    let engine = engine_of("cache_aware");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut rows = Vec::with_capacity(n);
        build_view(&fleet, &mut rows);
        let mut held = Vec::with_capacity(n);
        build_column(&fleet, &mut held);
        // Every holder of the fleet's column, named in its place.
        let holders: Vec<NamedHolder> = (0u32..)
            .zip(&held)
            .filter(|(_, entry)| entry.overlap_blocks > 0 || entry.predicted_blocks > 0)
            .map(|(row, entry)| NamedHolder {
                row,
                overlap_blocks: entry.overlap_blocks,
                predicted_blocks: entry.predicted_blocks,
            })
            .collect();
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let first = engine.decide(&facts(0, &holders), &pool, &rows, &cx, &mut scratch);
        assert!(first.pick().is_some(), "{n}: the decision picks a worker");
        group.bench_with_input(BenchmarkId::new("cache_aware", n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                engine.decide(
                    &facts(seed, black_box(&holders)),
                    &pool,
                    black_box(&rows),
                    &cx,
                    &mut scratch,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, decide);
criterion_main!(benches);
