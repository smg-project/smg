//! The WebAssembly plugin host's cost per decision: a profile whose filter,
//! scorer and picker are one plugin, over the record and the packed world,
//! at 8, 64 and 512 candidates, on a kept instance and without one (every
//! decision instantiates); the fuel-exhaustion latency of a guest that
//! spins; one lifecycle event; and the same profile out of built-in stages
//! as the floor. Ids: `policy_wasm/decide/<world>/<n>`,
//! `policy_wasm/decide_fresh/<world>/<n>`, `policy_wasm/fuel_exhausted/<n>`,
//! `policy_wasm/observe/<world>` and `policy_wasm/native/<n>`. The guests
//! are the test guests, assembled at bench start.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a guest that fails to load is a bug the tests catch"
)]

#[expect(
    dead_code,
    reason = "the native floor is the guest's own shape out of built-ins, not a preset"
)]
mod common;
#[expect(
    dead_code,
    reason = "the bench assembles the honest guests; the misbehaving modes are the tests'"
)]
#[path = "../tests/support/wasm_guest.rs"]
mod guest;

use std::{hint::black_box, sync::Arc};

use common::{build_view, fleet};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use smg_policy_controller::{
    builtins::{Best, Healthy, LeastRequests},
    wasm::{Plugin, PluginSpec, Runtime, WasmFilter, WasmPicker, WasmScorer},
    Binding, Constraints, Engine, Lease, Lifecycle, Needs, Plan, PoolFacts, Profile, RequestFacts,
    Scratch, Side, Target, Weighted,
};

const SIZES: [usize; 3] = [8, 64, 512];

/// A frame of 10 ms per stage call: wide enough that no honest call meets it.
const FRAME_MICROS: u32 = 10_000;

fn plugin(runtime: &Arc<Runtime>, name: &str, component: Vec<u8>, config: Vec<u8>) -> Arc<Plugin> {
    let mut spec = PluginSpec::new(name, component);
    spec.config = config;
    spec.budget_micros = FRAME_MICROS;
    spec.metrics = vec![
        "observed".into(),
        "now".into(),
        "random".into(),
        "observe-now".into(),
        "observe-session".into(),
        "pick-random".into(),
    ];
    Plugin::load(runtime, spec).expect("the guest loads")
}

/// One profile whose three stages are the plugin's.
fn plugin_engine(plugin: &Arc<Plugin>) -> Engine {
    let bound = plugin.bind();
    let profile = Profile {
        name: "plugin".into(),
        filters: vec![Box::new(WasmFilter::new(Arc::clone(&bound), None))],
        scorers: vec![Weighted {
            scorer: Box::new(WasmScorer::new(Arc::clone(&bound), None)),
            weight: 1.0,
        }],
        picker: Box::new(WasmPicker::new(bound, None)),
    };
    engine_of(profile)
}

/// The same shape out of built-in stages.
fn native_engine() -> Engine {
    engine_of(Profile {
        name: "native".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![Weighted {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    })
}

fn engine_of(profile: Profile) -> Engine {
    Engine::new(
        vec![profile],
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::Single,
            profile: 0,
            decode_profile: None,
            sessions: None,
        }],
        1,
        Needs::default(),
    )
}

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        budget_micros: FRAME_MICROS,
        seed,
        now_micros: 1_000_000,
        ..RequestFacts::default()
    }
}

fn worlds() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ]
}

fn decide(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new().expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/decide");
    for (world, component) in worlds() {
        let plugin = plugin(
            &runtime,
            world,
            component,
            vec![guest::mode::LEAST_IN_FLIGHT],
        );
        let engine = plugin_engine(&plugin);
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(world, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
        let stats = plugin.stats();
        assert_eq!(
            (
                stats.traps,
                stats.fuel_exhausted,
                stats.invalid,
                stats.declined
            ),
            (0, 0, 0, 0),
            "{world}: the honest guest answered every call: {stats:?}"
        );
    }
    group.finish();
}

fn decide_fresh(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new().expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/decide_fresh");
    for (world, component) in worlds() {
        let plugin = plugin(
            &runtime,
            world,
            component,
            vec![guest::mode::LEAST_IN_FLIGHT],
        );
        let engine = plugin_engine(&plugin);
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(world, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter_batched(
                    || plugin.forget_instance(),
                    |()| {
                        seed += 1;
                        engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }
    group.finish();
}

fn fuel_exhausted(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new().expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/fuel_exhausted");
    // A 100 microsecond frame, so the spin ends at 100 microseconds of fuel;
    // a cool-down of zero keeps the breaker from bypassing the guest after
    // its threshold (every iteration is a failure by design). Each exhaustion
    // drops the instance, so an iteration pays one instantiation as well:
    // `decide_fresh` prices that part.
    let mut spec = PluginSpec::new("spin", guest::record_guest());
    spec.config = vec![guest::mode::SPIN_IN_SCORE];
    spec.budget_micros = 100;
    spec.breaker.cooldown_micros = 0;
    spec.metrics = vec![
        "observed".into(),
        "now".into(),
        "random".into(),
        "observe-now".into(),
        "observe-session".into(),
        "pick-random".into(),
    ];
    let plugin = Plugin::load(&runtime, spec).expect("the guest loads");
    let engine = plugin_engine(&plugin);
    let n = 8usize;
    let fleet = fleet(n);
    let mut view = Vec::with_capacity(n);
    build_view(&fleet, &mut view);
    let mut scratch = Scratch::new();
    let pool = PoolFacts {
        pool_size: n as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
        });
    });
    group.finish();
}

fn observe(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new().expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/observe");
    for (world, component) in worlds() {
        let plugin = plugin(
            &runtime,
            world,
            component,
            vec![guest::mode::LEAST_IN_FLIGHT],
        );
        let engine = plugin_engine(&plugin);
        let lease = Lease {
            issued_micros: 1_000_000,
            tokens: 512,
            ..Lease::default()
        };
        let target = Target::default();
        group.bench_function(world, |b| {
            b.iter(|| engine.on_dispatched(black_box(&lease), &target, Side::Single, 0));
        });
    }
    group.finish();
}

fn native(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_wasm/native");
    let engine = native_engine();
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        let mut scratch = Scratch::new();
        let pool = PoolFacts {
            pool_size: n as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    decide,
    decide_fresh,
    fuel_exhausted,
    observe,
    native
);
criterion_main!(benches);
