//! The gRPC plugin's cost per decision: a profile bound to one plugin served
//! in the bench's own process over a Unix socket, so a cell is the host's
//! side of a round trip plus a plugin that answers at once: the frame built
//! and sent, the wait, the answer read. Bound as the three stages (asked at
//! the open) and as the picker alone (asked at the pick with the combined
//! scores), at 8, 64 and 512 candidates; a decision the open breaker
//! bypasses; one lifecycle event the plugin hears and one it does not; the
//! same profile out of built-in stages as the floor; and a request that
//! carries a prompt's block hashes, to a plugin whose row declares them and
//! to one whose row does not. Ids: `policy_grpc/decide/<binding>/<n>`,
//! `policy_grpc/bypassed/<n>`, `policy_grpc/observe`, `policy_grpc/unheard`,
//! `policy_grpc/native/<n>` and `policy_grpc/prompt/<row>/<hashes>`. A
//! closing check is asked of the cells that ran: a filter times a cell alone.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer; a plugin that fails to load is a bug the tests catch"
)]

#[expect(
    dead_code,
    reason = "the native floor is the plugin's own shape out of built-ins, not a preset"
)]
mod common;
#[expect(
    dead_code,
    reason = "the bench serves the honest plugin; the misbehaving scripts are the tests'"
)]
#[path = "../tests/support/grpc_plugin.rs"]
mod served;

use std::{hint::black_box, sync::Arc};

use common::{build_view, fleet};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use served::{Behaviour, TestPlugin, ALL, FILTER, PICK, SCORE};
use smg_policy_controller::{
    document::Needs,
    engine::{
        Binding, Constraints, Engine, Filter, Picker, Plan, PoolFacts, Profile, RequestFacts,
        Scorer, Scratch, Target, WeightedScorer,
    },
    plugins::{
        grpc::{GrpcFilter, GrpcPicker, GrpcPluginSpec, GrpcScorer, Plugin, Runtime, Transport},
        shared::{BreakerSpec, OpenDecisions},
    },
    stages::{Best, Healthy, LeastRequests, Lifecycle},
    tracking::{EventSet, Lease, Side},
};

const SIZES: [usize; 3] = [8, 64, 512];

/// A prompt's block hashes: the view's default cap and the contract's most.
const HASHES: [usize; 2] = [256, 8_190];

/// A budget of one second per decision, which the plugin states as its own
/// frame too: no honest round trip meets it, so every cell is an answered
/// decision.
const BUDGET_MICROS: u32 = 1_000_000;

/// A served plugin that keeps none of the frames and events it is sent.
fn serve() -> TestPlugin {
    let served = TestPlugin::serve();
    served.script(|s| s.forget = true);
    served
}

fn spec(served: &TestPlugin, name: &str) -> GrpcPluginSpec {
    let mut spec = GrpcPluginSpec::new(
        name,
        Transport::Unix {
            path: served.socket.clone(),
            peer_uid: None,
        },
    );
    spec.connect_micros = 5_000_000;
    // A plugin that states no budget is cut at the default frame, a tenth
    // of a decision's budget here.
    spec.budget_micros = BUDGET_MICROS;
    spec.max_candidates = 512;
    // Every kind of event, as the `observe` cell tells one.
    spec.observes = EventSet::ALL;
    spec
}

/// One profile that binds the plugin as `stages` and runs the built-in
/// stage elsewhere.
fn plugin_engine(plugin: &Arc<Plugin>, stages: u32) -> Engine {
    let bound = plugin.bind();
    let filter: Box<dyn Filter> = if stages & FILTER != 0 {
        Box::new(GrpcFilter::new(Arc::clone(&bound), None))
    } else {
        Box::new(Healthy)
    };
    let scorer: Box<dyn Scorer> = if stages & SCORE != 0 {
        Box::new(GrpcScorer::new(Arc::clone(&bound), None))
    } else {
        Box::new(LeastRequests)
    };
    let picker: Box<dyn Picker> = if stages & PICK != 0 {
        Box::new(GrpcPicker::new(bound, None))
    } else {
        Box::new(Best)
    };
    engine_of(Profile {
        name: "plugin".into(),
        filters: vec![filter],
        scorers: vec![WeightedScorer {
            scorer,
            weight: 1.0,
        }],
        picker,
    })
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
        budget_micros: BUDGET_MICROS,
        seed,
        now_micros: 1_000_000,
        ..RequestFacts::default()
    }
}

fn pool_of(n: usize) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: n as u32,
        generation: 1,
        ..PoolFacts::default()
    }
}

fn decide(c: &mut Criterion) {
    let runtime = Runtime::new(&OpenDecisions::new()).expect("a runtime");
    let mut group = c.benchmark_group("policy_grpc/decide");
    for (binding, stages) in [("all", ALL), ("pick", PICK)] {
        let served = serve();
        // A stall of the host reads as a slow plugin and opens the breaker;
        // without a cool-down no measured decision takes the fallback for
        // it, and the check below fails the run if one did.
        let mut spec = spec(&served, binding);
        spec.breaker.cooldown_micros = 0;
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        let engine = plugin_engine(&plugin, stages);
        let mut ran = false;
        for &n in &SIZES {
            let fleet = fleet(n);
            let mut view = Vec::with_capacity(n);
            build_view(&fleet, &mut view);
            let mut scratch = Scratch::new();
            let pool = pool_of(n);
            let cx = Constraints::default();
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(binding, n), &n, |b, _| {
                ran = true;
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
        assert_answered(binding, &plugin, ran);
    }
    group.finish();
}

/// The plugin answered every frame of its cells and no decision took a
/// fallback: a run that timed anything else fails here. `ran` is whether a
/// cell of the plugin ran: under a filter or a listing none may, and the
/// plugin was then asked nothing.
fn assert_answered(row: &str, plugin: &Plugin, ran: bool) {
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed,
            stats.invalid,
            stats.declined,
            stats.cold,
            stats.refused,
            stats.bypassed
        ),
        (0, 0, 0, 0, 0, 0),
        "{row}: the plugin answered every frame and no decision was bypassed: {stats:?}"
    );
    assert!(!ran || stats.answered > 0, "{row}: {stats:?}");
}

/// A request that carries a prompt's block hashes, at 64 candidates, bound
/// as the three stages. A plugin whose row declares the hashes is sent
/// them; one whose row declares none is sent none, so its cell is the
/// decision's without a prompt, whatever the prompt's length.
fn prompt(c: &mut Criterion) {
    let runtime = Runtime::new(&OpenDecisions::new()).expect("a runtime");
    let n = 64;
    let fleet = fleet(n);
    let mut view = Vec::with_capacity(n);
    build_view(&fleet, &mut view);
    let pool = pool_of(n);
    let cx = Constraints::default();
    let mut group = c.benchmark_group("policy_grpc/prompt");
    for (row, declared) in [("declared", true), ("undeclared", false)] {
        let served = serve();
        let mut spec = spec(&served, row);
        spec.breaker.cooldown_micros = 0;
        spec.needs.token_hashes = declared;
        let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
        let engine = plugin_engine(&plugin, ALL);
        let mut ran = false;
        for &hashes in &HASHES {
            let held: Vec<u64> = (1..).take(hashes).collect();
            let mut scratch = Scratch::new();
            group.bench_with_input(BenchmarkId::new(row, hashes), &hashes, |b, _| {
                ran = true;
                let mut seed = 0u64;
                b.iter(|| {
                    seed += 1;
                    let facts = RequestFacts {
                        block_hashes: black_box(&held),
                        ..facts(seed)
                    };
                    engine.decide(&facts, &pool, black_box(&view), &cx, &mut scratch)
                });
            });
        }
        assert_answered(row, &plugin, ran);
    }
    group.finish();
}

/// What a decision costs while the plugin's breaker is open: the open's
/// look, no frame, the fallbacks.
fn bypassed(c: &mut Criterion) {
    let runtime = Runtime::new(&OpenDecisions::new()).expect("a runtime");
    let served = serve();
    served.behave(Behaviour::ScoreNan);
    let mut spec = spec(&served, "bypassed");
    spec.breaker = BreakerSpec {
        failures: 1,
        window_micros: 3_600_000_000,
        cooldown_micros: 3_600_000_000,
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let bound = plugin.bind();
    let engine = engine_of(Profile {
        name: "plugin".into(),
        filters: vec![Box::new(GrpcFilter::new(
            Arc::clone(&bound),
            Some(Box::new(Healthy)),
        ))],
        scorers: vec![WeightedScorer {
            scorer: Box::new(GrpcScorer::new(
                Arc::clone(&bound),
                Some(Box::new(LeastRequests)),
            )),
            weight: 1.0,
        }],
        picker: Box::new(GrpcPicker::new(bound, Some(Box::new(Best)))),
    });
    let mut group = c.benchmark_group("policy_grpc/bypassed");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        let mut scratch = Scratch::new();
        let pool = pool_of(n);
        let cx = Constraints::default();
        // One invalid answer opens the breaker for the bench's length.
        engine.decide(&facts(0), &pool, &view, &cx, &mut scratch);
        assert_eq!(plugin.stats().breaker_opens, 1);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut seed = 0u64;
            b.iter(|| {
                seed += 1;
                engine.decide(&facts(seed), &pool, black_box(&view), &cx, &mut scratch)
            });
        });
    }
    let stats = plugin.stats();
    assert_eq!(stats.invalid, 1, "one frame in all: {stats:?}");
    group.finish();
}

/// One lifecycle event handed to the `Observe` queue of a plugin whose row
/// lists its kind, and one told where the row lists no kind: nothing is
/// handed, and the plugin has no such stream.
fn observe(c: &mut Criterion) {
    let runtime = Runtime::new(&OpenDecisions::new()).expect("a runtime");
    let served = serve();
    let lease = Lease::default();
    let target = Target::default();

    let plugin = Plugin::load(&runtime, spec(&served, "observer")).expect("the plugin loads");
    let engine = plugin_engine(&plugin, ALL);
    let mut ran = false;
    c.bench_function("policy_grpc/observe", |b| {
        ran = true;
        b.iter(|| engine.on_dispatched(black_box(&lease), &target, Side::Single, 3));
    });
    let stats = plugin.stats();
    assert!(
        !ran || stats.observed + stats.observe_dropped > 0,
        "{stats:?}"
    );

    let mut unlisted = spec(&served, "unheard");
    unlisted.observes = EventSet::NONE;
    let plugin = Plugin::load(&runtime, unlisted).expect("the plugin loads");
    let engine = plugin_engine(&plugin, ALL);
    c.bench_function("policy_grpc/unheard", |b| {
        b.iter(|| engine.on_dispatched(black_box(&lease), &target, Side::Single, 3));
    });
    let stats = plugin.stats();
    assert_eq!((stats.observed, stats.observe_dropped), (0, 0), "{stats:?}");
}

fn native(c: &mut Criterion) {
    let engine = native_engine();
    let mut group = c.benchmark_group("policy_grpc/native");
    for &n in &SIZES {
        let fleet = fleet(n);
        let mut view = Vec::with_capacity(n);
        build_view(&fleet, &mut view);
        let mut scratch = Scratch::new();
        let pool = pool_of(n);
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

criterion_group!(benches, decide, bypassed, observe, native, prompt);
criterion_main!(benches);
