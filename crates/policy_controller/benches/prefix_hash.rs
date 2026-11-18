//! The cost of one decision under the `prefix_hash` preset for a request
//! keyed by its prompt's prefix, at 8, 64 and 512 candidates: its key's
//! owner takes it (`owner`), and its key's owner is past the bounded-load
//! rule, so the fewest requests in flight take it (`passed_over`). The two
//! `levels_` paths are the same two under the boundaries 64 and 256 tokens:
//! the prompt is keyed at 256, its deepest, and a passed owner hands it to
//! the owner at 64 before the fewest requests take it. Ids:
//! `policy_engine/prefix_hash/<path>/<n>`. The fleet is the one the
//! `decision` bench decides over, whose `policy_engine/decide/prefix_hash/<n>`
//! ids time the preset for a request with nothing to hash. The request
//! carries the 64 block hashes of a prompt of 1,024 tokens, and the decision
//! runs with a fresh seed per iteration. The pull-request gate builds its
//! base side with this file: a build that refuses the boundaries times
//! neither `levels_` path.
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
use smg_policy_controller::{
    Candidate, Constraints, Engine, PolicySpec, PoolFacts, RequestFacts, Scratch,
};

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

/// The row that owns the prompt's key: the one the prompt is placed on over
/// an idle copy of `view`. The view's own loads leave that row under its
/// bound, so it takes the prompt over the view as it stands too.
fn owner_of(engine: &Engine, view: &[Candidate], hashes: &[u64]) -> usize {
    let mut idle = view.to_vec();
    for row in &mut idle {
        row.in_flight = 0;
    }
    let owner = placed(engine, &idle, hashes);
    assert_eq!(
        placed(engine, view, hashes),
        owner,
        "the owner keeps its key over the fleet's loads"
    );
    owner
}

/// The preset under the boundaries 64 and 256 tokens; none where the build
/// refuses them, as the gate's base side does before the boundary levels.
fn levelled() -> Option<Engine> {
    let mut spec = PolicySpec::preset("prefix_hash").expect("a preset of this build");
    spec.view.boundaries = vec![64, 256];
    spec.validate().ok()
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
        // An engine a size and a form: the ring learns its members from
        // the view. The owner keeps its key over the fleet's loads; then the
        // same owner holds a thousand requests and is passed over.
        for (engine, kept, passed) in [
            (Some(engine_of("prefix_hash")), "owner", "passed_over"),
            (levelled(), "levels_owner", "levels_passed_over"),
        ] {
            let Some(engine) = engine else {
                continue;
            };
            let mut view = Vec::with_capacity(n);
            build_view(&fleet(n), &mut view);
            let owner = owner_of(&engine, &view, &hashes);
            decide_over(&mut group, kept, &engine, &view, &hashes);
            view[owner].in_flight = 1_000;
            assert_ne!(
                placed(&engine, &view, &hashes),
                owner,
                "the owner is passed over"
            );
            decide_over(&mut group, passed, &engine, &view, &hashes);
        }
    }
    group.finish();
}

criterion_group!(benches, paths);
criterion_main!(benches);
