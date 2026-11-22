//! One live decision followed by its shadow over the same rows. Both
//! profiles use one preset's stages: the least-load preset's over a pool
//! read whole, and the hash ring's for a request placed by its client's key
//! over a pool that lists its engines. The rows are built before timing,
//! and each iteration gives both decisions the same fresh seed.
#![expect(clippy::expect_used, reason = "the bench requires a valid binding")]

mod common;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    document::{PolicySpec, ShadowSpec},
    engine::{Constraints, Engines, PoolFacts, RequestFacts, Scratch, ShadowKind},
};

/// An iteration's client key is its seed times this odd number, so no
/// iteration's key is the zero of a request with none.
const CLIENT_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// A live decision of `preset` and its shadow, a second profile of the same
/// stages, for `facts`. Where `listed`, the pool lists its engines, each
/// worker an engine of one rank. A request with a client key carries one
/// of its own each iteration.
fn shadowed(c: &mut Criterion, preset: &str, group: &str, facts: RequestFacts<'_>, listed: bool) {
    let plain = engine_of(preset);
    let mut document = PolicySpec::preset(preset).expect("a preset");
    let next = "next";
    document
        .profiles
        .insert(next.into(), document.profiles[preset].clone());
    document.bindings[0].shadow = Some(ShadowSpec::Profile(next.into()));
    let engine = document.validate().expect("a binding with a shadow");
    let mut group = c.benchmark_group(format!("policy_shadow/{group}"));
    for n in [8, 64, 512] {
        let mut rows = Vec::with_capacity(n);
        build_view(&fleet(n), &mut rows);
        let places: Vec<u32> = (0..n as u32).collect();
        let ends: Vec<u32> = (1..=n as u32).collect();
        let mut scratch = Scratch::new();
        // A list goes under a membership epoch, each pool under a number
        // of its own.
        let (membership_epoch, engines) = if listed {
            (1, Engines::new(n as u32, &places, &ends))
        } else {
            (0, Engines::default())
        };
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: engine.generation(),
            membership_epoch,
            engines,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(live, plain.decide(&facts, &pool, &rows, &cx, &mut scratch));
        assert_eq!(ShadowKind::of(&live, &shadow), ShadowKind::Agreed);
        // Both decisions walk a listed pool: a walk's trace counts no
        // survivors. The group would time the whole pass otherwise.
        if listed {
            let walked = [&live, &shadow].iter().all(|decision| {
                let pick = decision.pick().expect("a pick");
                pick.trace.survivors.get().is_none()
            });
            assert!(walked, "both decisions walk the ring over {n} rows");
        }

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                let facts = RequestFacts {
                    seed,
                    client_key: facts.client_key.wrapping_mul(seed),
                    ..facts
                };
                let rows = black_box(rows.as_slice());
                let live = engine.decide(&facts, &pool, rows, &cx, &mut scratch);
                let shadow = engine.shadow(&facts, &pool, rows, &cx, &mut scratch);
                (live, shadow)
            });
        });
    }
    group.finish();
}

fn decide_with_shadow(c: &mut Criterion) {
    let now_micros = 1_000_000;
    let sized = RequestFacts {
        prompt_tokens: 512,
        now_micros,
        ..RequestFacts::default()
    };
    shadowed(c, "least_load", "decide_with_shadow", sized, false);
    let by_client = RequestFacts {
        client_key: CLIENT_MIX,
        now_micros,
        ..RequestFacts::default()
    };
    shadowed(
        c,
        "consistent_hashing",
        "ring_by_client_key",
        by_client,
        true,
    );
}

criterion_group!(benches, decide_with_shadow);
criterion_main!(benches);
