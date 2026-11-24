//! The cost of one decision over the rows a host names in the affinity
//! column's place: the `cache_aware` preset over `common`'s settled fleet,
//! the holders of its column named as a host names them (the deepest first,
//! at most the picker's sample) and no column fed, at 8, 64, 128, 512 and
//! 1,024 candidates. Ids: `policy_affinity_named/decide/cache_aware/<n>`; the same
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
use smg_policy_controller::engine::{
    AffinityHolder, AffinityRows, Constraints, Phase, PoolFacts, RequestFacts, Scratch,
};

const SIZES: [usize; 5] = [8, 64, 128, 512, 1024];

fn facts(seed: u64, holders: &[AffinityHolder]) -> RequestFacts<'_> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        seed,
        now_micros: 1_000_000,
        named: Some(AffinityRows {
            holders,
            warming: &[],
            thin: false,
            mean_in_flight: None,
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
        // The holders of the fleet's column as a host names them: the deepest
        // first, at most the sample of the profile that decides.
        let mut holders: Vec<AffinityHolder> = (0u32..)
            .zip(&held)
            .filter(|(_, entry)| entry.overlap_blocks > 0 || entry.predicted_blocks > 0)
            .map(|(row, entry)| AffinityHolder {
                row,
                overlap_blocks: entry.overlap_blocks,
                predicted_blocks: entry.predicted_blocks,
            })
            .collect();
        let cap = engine.sample_for("m", Phase::Single) as usize;
        if cap > 0 && holders.len() > cap {
            holders.sort_by_key(|holder| std::cmp::Reverse(holder.overlap_blocks));
            let edge = holders[cap].overlap_blocks;
            let fit = holders
                .iter()
                .take_while(|holder| holder.overlap_blocks > edge)
                .count();
            let group: Vec<AffinityHolder> = holders[fit..]
                .iter()
                .copied()
                .take_while(|holder| holder.overlap_blocks == edge)
                .collect();
            // The group at the cap's edge: an even stride through it stands
            // in for the host's draw; which of its rows are named moves no
            // cost.
            let left = cap - fit;
            holders.truncate(fit);
            holders.extend((0..left).map(|k| group[k * group.len() / left]));
            holders.sort_by_key(|holder| holder.row);
        }
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
