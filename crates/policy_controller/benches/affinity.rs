//! The cost of the affinity column a host feeds beside the candidate rows:
//! the view build that writes each row and its entry of the column, and the
//! decision of the preset whose stages read the column, at 8, 64 and 512
//! candidates. Ids: `policy_affinity/view_build/<n>` and
//! `policy_affinity/decide/cache_aware/<n>`. The fleet is `common`'s with its
//! settled column: the warm-up stage reads the fleet's index level and
//! slices nothing, the affinity stage ranks the holders. A bench target of
//! its own: the pull-request gate's rows are the `decision` binary's, and a
//! row added there could move them through that binary's inlining.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

#[path = "common/column.rs"]
mod column;
mod common;
#[path = "common/placed.rs"]
mod placed;

use std::hint::black_box;

use column::build_column;
use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use placed::page_start_buffer;
use smg_policy_controller::{AffinityFacts, Constraints, PoolFacts, RequestFacts, Scratch};

const SIZES: [usize; 3] = [8, 64, 512];

fn view_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_affinity/view_build");
    for &n in &SIZES {
        let fleet = fleet(n);
        // Both timed buffers stand on a page's start (`placed`), as the
        // decision bench's view build does.
        let mut rows = page_start_buffer(n);
        let mut held = page_start_buffer(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                build_view(black_box(&fleet), &mut rows);
                build_column(black_box(&fleet), &mut held);
                black_box(&rows);
                black_box(&held);
                rows.len()
            });
        });
    }
    group.finish();
}

fn facts(seed: u64, held: &[AffinityFacts]) -> RequestFacts<'_> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        seed,
        now_micros: 1_000_000,
        affinity: held,
        ..RequestFacts::default()
    }
}

fn decide(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_affinity/decide");
    let engine = engine_of("cache_aware");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut rows = Vec::with_capacity(n);
        build_view(&fleet, &mut rows);
        let mut held = Vec::with_capacity(n);
        build_column(&fleet, &mut held);
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        // A column of another length is refused before any stage runs: the
        // cell would time the refusal.
        let first = engine.decide(&facts(0, &held), &pool, &rows, &cx, &mut scratch);
        assert!(first.pick().is_some(), "{n}: the decision picks a worker");
        group.bench_with_input(BenchmarkId::new("cache_aware", n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                engine.decide(
                    &facts(seed, black_box(&held)),
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

criterion_group!(benches, view_build, decide);
criterion_main!(benches);
