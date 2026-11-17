//! The cost of one decision under the `prefix_hash` preset for a request
//! keyed by its prompt's prefix, at 8, 64 and 512 candidates: its key's
//! owner takes it (`owner`), and its key's owner is past the bounded-load
//! rule, so the fewest requests in flight take it (`passed_over`). Ids:
//! `policy_engine/prefix_hash/<path>/<n>`. The fleet is the one the
//! `decision` bench decides over, whose `policy_engine/decide/prefix_hash/<n>`
//! ids time the preset for a request with nothing to hash. The request
//! carries the 64 block hashes of a prompt of 1,024 tokens, and the decision
//! runs with a fresh seed per iteration.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a preset that fails to validate is a bug the unit tests catch"
)]

mod common;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use smg_policy_controller::{Candidate, Constraints, Engine, PoolFacts, RequestFacts, Scratch};

const SIZES: [usize; 3] = [8, 64, 512];

/// The block hashes of a prompt of 1,024 tokens in blocks of sixteen. The
/// key is the sixteenth: the last whole block at or under 256 tokens.
fn prompt() -> Vec<u64> {
    (1..=64u64)
        .map(|block| block.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect()
}

fn facts(hashes: &[u64], seed: u64) -> RequestFacts<'_> {
    RequestFacts {
        model: "m",
        prompt_tokens: 1_024,
        block_hashes: hashes,
        seed,
        now_micros: 1_000_000,
        ..RequestFacts::default()
    }
}

fn pool_of(view: &[Candidate]) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: view.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    }
}

/// The row the prompt is placed on over `view`.
fn placed(engine: &Engine, view: &[Candidate], hashes: &[u64]) -> usize {
    let decision = engine.decide(
        &facts(hashes, 1),
        &pool_of(view),
        view,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.pick().expect("a pick");
    pick.targets.first().expect("a target").position as usize
}

fn decide_over(
    group: &mut BenchmarkGroup<'_, WallTime>,
    path: &str,
    engine: &Engine,
    view: &[Candidate],
    hashes: &[u64],
) {
    let n = view.len();
    let mut scratch = Scratch::new();
    let pool = pool_of(view);
    let cx = Constraints::default();
    group.bench_with_input(BenchmarkId::new(path, n), &n, |b, _| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            engine.decide(
                &facts(hashes, seed),
                &pool,
                black_box(view),
                &cx,
                &mut scratch,
            )
        });
    });
}

fn paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/prefix_hash");
    let hashes = prompt();
    for &n in &SIZES {
        // An engine a size: the ring learns its members from the view.
        let engine = engine_of("prefix_hash");
        let mut view = Vec::with_capacity(n);
        build_view(&fleet(n), &mut view);
        // No row of the fleet holds sixteen requests, so the owner keeps its
        // key; then the same owner holds a thousand and is passed over.
        let owner = placed(&engine, &view, &hashes);
        decide_over(&mut group, "owner", &engine, &view, &hashes);
        view[owner].in_flight = 1_000;
        assert_ne!(
            placed(&engine, &view, &hashes),
            owner,
            "the owner is passed over"
        );
        decide_over(&mut group, "passed_over", &engine, &view, &hashes);
    }
    group.finish();
}

criterion_group!(benches, paths);
criterion_main!(benches);
