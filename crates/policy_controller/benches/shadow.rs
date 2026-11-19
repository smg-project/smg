//! One live decision followed by its shadow over the same rows. Both
//! profiles use the least-load preset's stages. The rows are built before
//! timing, and each iteration gives both decisions the same fresh seed.
#![expect(clippy::expect_used, reason = "the bench requires a valid binding")]

mod common;

use std::hint::black_box;

use common::{build_view, engine_of, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smg_policy_controller::{
    decision::ShadowKind, spec::ShadowSpec, Constraints, PolicySpec, PoolFacts, RequestFacts,
    Scratch,
};

fn decide_with_shadow(c: &mut Criterion) {
    let preset = "least_load";
    let plain = engine_of(preset);
    let mut document = PolicySpec::preset(preset).expect("a preset");
    let next = "next";
    document
        .profiles
        .insert(next.into(), document.profiles[preset].clone());
    document.bindings[0].shadow = Some(ShadowSpec::Profile(next.into()));
    let engine = document.validate().expect("a binding with a shadow");
    let mut group = c.benchmark_group("policy_shadow/decide_with_shadow");
    for n in [8, 64, 512] {
        let mut rows = Vec::with_capacity(n);
        build_view(&fleet(n), &mut rows);
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: engine.generation(),
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let facts = RequestFacts {
            prompt_tokens: 512,
            now_micros: 1_000_000,
            ..RequestFacts::default()
        };
        let live = engine.decide(&facts, &pool, &rows, &cx, &mut scratch);
        let shadow = engine
            .shadow(&facts, &pool, &rows, &cx, &mut scratch)
            .expect("a shadow decision");
        assert_eq!(live, plain.decide(&facts, &pool, &rows, &cx, &mut scratch));
        assert_eq!(ShadowKind::of(&live, &shadow), ShadowKind::Agreed);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                let facts = RequestFacts { seed, ..facts };
                let rows = black_box(rows.as_slice());
                let live = engine.decide(&facts, &pool, rows, &cx, &mut scratch);
                let shadow = engine.shadow(&facts, &pool, rows, &cx, &mut scratch);
                (live, shadow)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, decide_with_shadow);
criterion_main!(benches);
