//! The cost of one decision for a request that carries a session key under
//! the `consistent_hashing` preset, at 8, 64 and 512 candidates. The ring
//! places the key, and the key keeps its worker over the overload veto: the
//! all-vetoed rule's home decides such a request with the veto lifted. Two
//! paths. `kept_key` is the fleet the `decision` bench decides over.
//! `kept_key_over` is that fleet with every row over its thresholds: the
//! veto as it stands keeps nobody there, and the kept pick says the lifted
//! veto. Ids: `policy_engine/vetoed/<path>/<n>`. The request is one of 64
//! session keys in turn, and the decision runs with a fresh seed per
//! iteration. The `decision` bench times the preset for a request with no
//! key (`policy_engine/decide/consistent_hashing/<n>`) and the rule's steer
//! (`policy_engine/decide/all_vetoed/<n>`). A bench target of its own: the
//! `decision` binary takes no new id. A build that sheds a keyed request
//! over such a pool lists no `kept_key_over` id.
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
use smg_policy_controller::engine::{
    Candidate, Constraints, Decision, Engine, Identity, LiftedVeto, PoolFacts, RequestFacts,
    Scratch,
};

const SIZES: [usize; 3] = [8, 64, 512];

/// The session keys a cell asks with, in turn.
const KEYS: u64 = 64;

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        seed,
        session_key: Identity::hashed(1 + seed % KEYS),
        now_micros: 1_000_000,
        ..RequestFacts::default()
    }
}

fn pool_of(view: &[Candidate]) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: view.len() as u32,
        generation: 1,
        // Every row of the bench's fleet reports a rate.
        reported: true,
        ..PoolFacts::default()
    }
}

fn decide_over(
    group: &mut BenchmarkGroup<'_, WallTime>,
    path: &str,
    engine: &Engine,
    view: &[Candidate],
) {
    let n = view.len();
    let mut scratch = Scratch::new();
    let pool = pool_of(view);
    let cx = Constraints::default();
    group.bench_with_input(BenchmarkId::new(path, n), &n, |b, _| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            engine.decide(&facts(seed), &pool, black_box(view), &cx, &mut scratch)
        });
    });
}

fn paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_engine/vetoed");
    for &n in &SIZES {
        // An engine a size: the ring learns its members from the view.
        let engine = engine_of("consistent_hashing");
        let mut view = Vec::with_capacity(n);
        build_view(&fleet(n), &mut view);
        decide_over(&mut group, "kept_key", &engine, &view);
        for row in &mut view {
            row.health.overloaded = true;
        }
        // The ids stand only on an engine that keeps a key over the veto:
        // a shed is no base to time a kept pick against.
        let kept = matches!(
            engine.decide(
                &facts(1),
                &pool_of(&view),
                &view,
                &Constraints::default(),
                &mut Scratch::new(),
            ),
            Decision::Pick(pick) if pick.trace.lifted == LiftedVeto::Overload
        );
        if kept {
            decide_over(&mut group, "kept_key_over", &engine, &view);
        }
    }
    group.finish();
}

criterion_group!(benches, paths);
criterion_main!(benches);
