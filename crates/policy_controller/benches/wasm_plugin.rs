//! The WebAssembly plugin host's cost per decision: a profile whose filter,
//! scorer and picker are one plugin, over the record and the packed world,
//! at 8, 64 and 512 candidates, on a kept instance, on a kept instance with
//! the affinity column fed (both worlds' fills then read each row's entry
//! of it) and without an instance (every decision instantiates); a decision
//! at 64 candidates whose request carries a prompt's block hashes, for a row
//! that declares them and for one that does not; the
//! fuel-exhaustion latency of a guest that spins; one lifecycle event; and
//! the same profile out of built-in stages as the floor. Ids:
//! `policy_wasm/decide/<world>/<n>`,
//! `policy_wasm/prompt/<declared|undeclared>/<world>/<hashes>`,
//! `policy_wasm/decide_held/<world>/<n>`,
//! `policy_wasm/decide_fresh/<world>/<n>`, `policy_wasm/fuel_exhausted/<n>`,
//! `policy_wasm/observe/<world>`, `policy_wasm/unheard/<world>` (an event of a
//! kind the row does not list), `policy_wasm/decide_named/<world>/<n>` (the
//! decision made as a named decider), `policy_wasm/membership/told/<world>`
//! (the host's telling of one change over a list of 64 members),
//! `policy_wasm/membership/heard/<world>`
//! (a decision whose instance hears one change first) and
//! `policy_wasm/native/<n>`. The guests are the test guests, assembled at
//! bench start.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a guest that fails to load is a bug the tests catch"
)]

#[path = "common/column.rs"]
mod column;
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

use column::build_column;
use common::{build_view, fleet};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use smg_policy_controller::{
    document::{Needs, MAX_BLOCK_HASHES},
    engine::{
        Binding, Constraints, DecisionSlot, Engine, Plan, PoolFacts, Profile, RequestFacts,
        Scratch, Target, WeightedScorer, WorkerId,
    },
    plugins::{
        shared::OpenDecisions,
        wasm::{Plugin, PluginSpec, Runtime, WasmFilter, WasmPicker, WasmScorer},
    },
    stages::{Best, Healthy, LeastRequests, Lifecycle},
    tracking::{EventSet, Lease, Membership, Side},
};

const SIZES: [usize; 3] = [8, 64, 512];

/// A frame of 10 ms per stage call: wide enough that no honest call meets it.
const FRAME_MICROS: u32 = 10_000;

fn plugin(runtime: &Arc<Runtime>, name: &str, component: Vec<u8>, config: Vec<u8>) -> Arc<Plugin> {
    Plugin::load(runtime, row(name, component, config)).expect("the guest loads")
}

/// The row of a bench's plugin.
fn row(name: &str, component: Vec<u8>, config: Vec<u8>) -> PluginSpec {
    let mut spec = PluginSpec::new(name, component);
    spec.config = config;
    spec.budget_micros = FRAME_MICROS;
    // A cool-down of zero: an opening then ends at its own instant, so the
    // breaker bypasses no measured decision, whatever opened it.
    spec.breaker.cooldown_micros = 0;
    // Every kind of event, as the `observe` cells tell one.
    spec.observes = EventSet::ALL;
    spec.metrics = vec![
        "observed".into(),
        "now".into(),
        "random".into(),
        "observe-now".into(),
        "observe-session".into(),
        "pick-random".into(),
    ];
    spec
}

/// One profile whose three stages are the plugin's.
fn plugin_engine(plugin: &Arc<Plugin>) -> Engine {
    let bound = plugin.bind();
    let profile = Profile {
        name: "plugin".into(),
        filters: vec![Box::new(WasmFilter::new(Arc::clone(&bound), None))],
        scorers: vec![WeightedScorer {
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
        scorers: vec![WeightedScorer {
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
            shadow: None,
            decode_shadow: None,
            sessions: None,
            overload: None,
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

/// What a decide group checks once its cells are measured: the guest
/// answered what it was asked. It asserts what no event of the host can make
/// false: no trap, no invalid answer, no decline. A call past its frame and a
/// stage that finds the decision's budget spent fall back, and a thread the
/// host held leaves either behind, so they are bounded and said, not refused:
/// one, and one in a thousand of the answered calls. More is a bench whose
/// cells timed the fallback.
#[expect(clippy::print_stderr, reason = "a bench says what its cells left out")]
fn answered(world: &str, plugin: &Plugin) {
    let stats = plugin.stats();
    assert_eq!(
        (stats.traps, stats.invalid, stats.declined),
        (0, 0, 0),
        "{world}: the honest guest answered every call it was asked: {stats:?}"
    );
    let fell_back = stats.fuel_exhausted + stats.budget_spent;
    assert!(
        fell_back <= 1 + stats.answered / 1_000,
        "{world}: the cells timed the fallback: {stats:?}"
    );
    if fell_back != 0 {
        eprintln!(
            "{world}: {fell_back} calls fell back among {} answered: {stats:?}",
            stats.answered
        );
    }
}

fn decide(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
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
        answered(world, &plugin);
    }
    group.finish();
}

/// The kept-instance decision with the affinity column fed: each world's
/// fill reads a row and the row's entry of the column.
fn decide_held(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/decide_held");
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
            let mut held = Vec::with_capacity(n);
            build_column(&fleet, &mut held);
            let mut scratch = Scratch::new();
            let pool = PoolFacts {
                pool_size: n as u32,
                generation: 1,
                ..PoolFacts::default()
            };
            let cx = Constraints::default();
            // A column of another length is refused before the plugin is
            // called: the cell would time the refusal.
            let first = RequestFacts {
                affinity: &held,
                ..facts(0)
            };
            let first = engine.decide(&first, &pool, &view, &cx, &mut scratch);
            assert!(
                first.pick().is_some(),
                "{world}/{n}: the decision picks a worker"
            );
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(world, n), &n, |b, _| {
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let facts = RequestFacts {
                        affinity: black_box(&held),
                        ..facts(seed)
                    };
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
        let stats = plugin.stats();
        assert_eq!(
            (
                stats.traps,
                stats.fuel_exhausted,
                stats.invalid,
                stats.declined,
                stats.bypassed
            ),
            (0, 0, 0, 0, 0),
            "{world}: the honest guest answered every call of every decision: {stats:?}"
        );
    }
    group.finish();
}

fn decide_fresh(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
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
        answered(world, &plugin);
    }
    group.finish();
}

fn fuel_exhausted(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
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

/// One decision at 64 candidates whose request carries a prompt's block
/// hashes, 256 and the range's 8,190, for a row that declares them and for
/// one that does not: the view of the second holds none of them.
fn prompt(c: &mut Criterion) {
    const CANDIDATES: usize = 64;
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/prompt");
    let hashes: Vec<u64> = (1..=u64::from(MAX_BLOCK_HASHES)).collect();
    let fleet = fleet(CANDIDATES);
    let mut view = Vec::with_capacity(CANDIDATES);
    build_view(&fleet, &mut view);
    let pool = PoolFacts {
        pool_size: CANDIDATES as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    for (world, component) in worlds() {
        for (declares, declared) in [("undeclared", false), ("declared", true)] {
            let mut row = row(world, component.clone(), vec![guest::mode::LEAST_IN_FLIGHT]);
            row.needs.token_hashes = declared;
            let plugin = Plugin::load(&runtime, row).expect("the guest loads");
            let engine = plugin_engine(&plugin);
            let mut scratch = Scratch::new();
            for carried in [256, hashes.len()] {
                let id = BenchmarkId::new(format!("{declares}/{world}"), carried);
                group.bench_function(id, |b| {
                    let mut seed = 0u64;
                    b.iter(|| {
                        seed += 1;
                        let facts = RequestFacts {
                            block_hashes: &hashes[..carried],
                            ..facts(seed)
                        };
                        engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                    });
                });
            }
            answered(world, &plugin);
        }
    }
    group.finish();
}

fn observe(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
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

/// One lifecycle event of a kind the plugin's row does not list: no call is
/// made.
fn unheard(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/unheard");
    for (world, component) in worlds() {
        let mut row = row(world, component, vec![guest::mode::LEAST_IN_FLIGHT]);
        row.observes = EventSet::NONE;
        let plugin = Plugin::load(&runtime, row).expect("the guest loads");
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
        assert_eq!(plugin.stats().observed, 0, "{world}: nothing was told");
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

/// The kept-instance decision at 64 candidates made as a named decider: the
/// decider's instance is brought to the thread at the decision's first open
/// and given back at its last close.
fn decide_named(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/decide_named");
    let n = 64;
    for (world, component) in worlds() {
        let plugin = plugin(
            &runtime,
            world,
            component,
            vec![guest::mode::LEAST_IN_FLIGHT],
        );
        let engine = plugin_engine(&plugin);
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
        let turn = DecisionSlot::new(0).enter();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(world, n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
            });
        });
        drop(turn);
        answered(world, &plugin);
        assert_eq!(
            plugin.stats().instantiations,
            1,
            "{world}: the named decider decided on the apply's own instance"
        );
    }
    group.finish();
}

/// The membership feed of a plugin whose row lists the kind, over the pool
/// of 64 the decisions read, told first as one snapshot. `told`: the host
/// tells one change, a worker that comes or goes, which is folded into the
/// feed's list of 64 and kept; no guest call is made. `heard`: one change is
/// told and a decision at 64 candidates is made, whose instance hears the
/// change first, in one `observe` call.
fn membership(c: &mut Criterion) {
    let runtime = Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"));
    let mut group = c.benchmark_group("policy_wasm/membership");
    let n = 64;
    // A worker the pool does not hold: its coming and its going each read
    // the feed's whole list.
    let other = WorkerId(u64::MAX);
    let (comes, goes) = (Membership::Added(other), Membership::Removed(other));
    for (world, component) in worlds() {
        let plugin = plugin(
            &runtime,
            world,
            component,
            vec![guest::mode::LEAST_IN_FLIGHT],
        );
        let engine = plugin_engine(&plugin);
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
        let members: Vec<WorkerId> = view.iter().map(|row| row.id).collect();
        engine.on_membership(&Membership::Snapshot(&members));
        let mut came = false;
        let mut tell = || {
            came = !came;
            engine.on_membership(black_box(if came { &comes } else { &goes }));
        };
        group.bench_function(BenchmarkId::new("told", world), |b| b.iter(&mut tell));
        assert_eq!(
            plugin.stats().observed,
            0,
            "{world}: a telling calls no guest"
        );
        group.bench_function(BenchmarkId::new("heard", world), |b| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                tell();
                engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
            });
        });
        answered(world, &plugin);
    }
    group.finish();
}

criterion_group!(
    benches,
    decide,
    decide_named,
    prompt,
    decide_held,
    decide_fresh,
    fuel_exhausted,
    observe,
    unheard,
    membership,
    native
);
criterion_main!(benches);
