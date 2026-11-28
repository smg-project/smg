//! The gRPC user plugin against a `PolicyPlugin` served in the test's own
//! process (`support::grpc_plugin`): what a load needs, what each binding
//! asks, the answer's validity rules, the budget and the breaker, a lost
//! stream, the lifecycle and both transports.
//!
//! The answered cases wait under a budget no loaded host misses, and the
//! cases that miss do so against a plugin that never answers (or answers
//! many frames later), so no case reads a race between an answer and its
//! deadline.
//!
//! One module a subject holds the cases, each a file under `grpc_plugin_cases/`:
//! `connection.rs`, `stages.rs`, `budgets.rs`, `stream.rs`, `lifecycle.rs`,
//! `exposure.rs` and `measurement.rs`. This file holds what they share: the
//! loads, the profiles and the decisions over the served plugin.
#![expect(
    clippy::expect_used,
    reason = "the served plugin, the certificates and the loads run inside the tests; a failure there is the test's failure"
)]

#[path = "../support"]
mod support {
    pub mod grpc_plugin;
    pub mod said;
    #[cfg(feature = "wasm")]
    #[expect(
        dead_code,
        reason = "the one case over both kinds loads one guest; the others are the WebAssembly binary's"
    )]
    pub mod wasm_guest;
}

use std::{sync::Arc, time::Duration};

#[cfg(feature = "wasm")]
use smg_grand_central::plugins::wasm;
use smg_grand_central::{
    document::Needs,
    engine::{
        Binding, Candidate, Constraints, Decision, Engine, Filter, PairPools, Picker, Plan,
        PoolFacts, Profile, RequestFacts, Scorer, Scratch, WeightedScorer, WorkerId,
    },
    plugins::{
        grpc::{
            GrpcFilter, GrpcPicker, GrpcPluginSpec, GrpcScorer, Plugin, PluginBinding, Runtime,
            Transport,
        },
        shared::OpenDecisions,
    },
    stages::{Best, Healthy, LeastRequests},
    tracking::EventSet,
};
use support::grpc_plugin::{TestPlugin, ALL, FILTER, PICK, SCORE};

#[path = "grpc_plugin_cases/budgets.rs"]
mod budgets;
#[path = "grpc_plugin_cases/connection.rs"]
mod connection;
#[path = "grpc_plugin_cases/exposure.rs"]
mod exposure;
#[path = "grpc_plugin_cases/lifecycle.rs"]
mod lifecycle;
#[path = "grpc_plugin_cases/measurement.rs"]
mod measurement;
#[path = "grpc_plugin_cases/stages.rs"]
mod stages;
#[path = "grpc_plugin_cases/stream.rs"]
mod stream;

/// The decision budget of the answered cases, and the connect frame of
/// every load: far past what a loaded host takes for one round trip.
const BUDGET_MICROS: u32 = 5_000_000;
/// The budget of the cases that miss on purpose.
const SHORT_MICROS: u32 = 20_000;

fn runtime() -> Arc<Runtime> {
    Runtime::new(&OpenDecisions::new()).expect("a runtime")
}

fn unix(served: &TestPlugin) -> Transport {
    Transport::Unix {
        path: served.socket.clone(),
        peer_uid: None,
    }
}

fn spec(served: &TestPlugin, name: &str) -> GrpcPluginSpec {
    let mut spec = GrpcPluginSpec::new(name, unix(served));
    spec.connect_micros = BUDGET_MICROS;
    // The plugin states its frame, as wide as an answered case's budget. A
    // plugin that states none takes the default frame, which has a case of
    // its own.
    spec.budget_micros = BUDGET_MICROS;
    // Every kind of event: the lifecycle's cases tell them, and a decision
    // reads none of it.
    spec.observes = EventSet::ALL;
    spec
}

fn load(runtime: &Arc<Runtime>, served: &TestPlugin, name: &str) -> Arc<Plugin> {
    Plugin::load(runtime, spec(served, name)).expect("the plugin loads")
}

/// A try or a round a hold of the test's runner can have ended says so: the
/// one line of a try that asserts nothing and is made again (read under
/// `--nocapture`), so a row of runs can say in how many a hold reached a
/// case.
#[expect(
    clippy::print_stdout,
    reason = "a try that is made again reports it on stdout under --nocapture"
)]
fn made_again(case: &str, why: &str) {
    println!("{case}: made again in this run: {why}");
}

/// The loads of the one WebAssembly guest a case makes at most.
#[cfg(feature = "wasm")]
const GUEST_LOADS: u32 = 10;

/// Loads the WebAssembly guest of the case over both kinds. Nothing watches
/// an `init`: the host ends it at its frame on the wall clock, so a runner
/// that holds the loading thread ends an `init` that did nothing. The load
/// claims nothing of that time, so it is made again whenever the host ended
/// its `init` at the frame, ten loads at most: the last load asserts what it
/// says. The WebAssembly plugin binary's loads that must succeed do the
/// same.
#[cfg(feature = "wasm")]
fn guest_loaded(runtime: &Arc<wasm::Runtime>, spec: &wasm::PluginSpec) -> Arc<wasm::Plugin> {
    use smg_grand_central::plugins::wasm;

    let mut loads = 0;
    loop {
        let load = wasm::Plugin::load(runtime, spec.clone());
        loads += 1;
        let ended = matches!(
            &load,
            Err(wasm::LoadError::Trap { reason, .. }) if reason.ends_with("wasm trap: interrupt")
        );
        if !ended || loads == GUEST_LOADS {
            return load.expect("the guest loads");
        }
        made_again(
            "a guest's load",
            "the host ended its init at the frame's deadline",
        );
    }
}

/// A profile that binds the plugin as `stages` and runs the built-in stage
/// elsewhere; with `fallbacks`, each bound stage falls back to the built-in
/// one. The one profile of its build that binds the plugin.
fn profile_of(plugin: &Arc<Plugin>, stages: u32, fallbacks: bool) -> Profile {
    bound_profile(plugin.bind(), stages, fallbacks)
}

/// `profile_of` over a bound set the caller took.
fn bound_profile(bound: Arc<PluginBinding>, stages: u32, fallbacks: bool) -> Profile {
    let filter: Box<dyn Filter> = if stages & FILTER != 0 {
        let fallback: Option<Box<dyn Filter>> = fallbacks.then(|| Box::new(Healthy) as _);
        Box::new(GrpcFilter::new(Arc::clone(&bound), fallback))
    } else {
        Box::new(Healthy)
    };
    let scorer: Box<dyn Scorer> = if stages & SCORE != 0 {
        let fallback: Option<Box<dyn Scorer>> = fallbacks.then(|| Box::new(LeastRequests) as _);
        Box::new(GrpcScorer::new(Arc::clone(&bound), fallback))
    } else {
        Box::new(LeastRequests)
    };
    let picker: Box<dyn Picker> = if stages & PICK != 0 {
        let fallback: Option<Box<dyn Picker>> = fallbacks.then(|| Box::new(Best) as _);
        Box::new(GrpcPicker::new(bound, fallback))
    } else {
        Box::new(Best)
    };
    let mut profile = Profile::of("plugin".into(), picker);
    profile.filters = vec![filter];
    profile.scorers = vec![WeightedScorer::of(scorer, 1.0)];
    profile
}

fn profile(plugin: &Arc<Plugin>, stages: u32) -> Profile {
    profile_of(plugin, stages, true)
}

/// A profile whose filter is one plugin and whose picker another, each
/// with its built-in fallback. The picker's frame leaves at the pick and
/// names what the decision has left by then.
fn filter_then_picker(filter: &Arc<Plugin>, picker: &Arc<Plugin>) -> Profile {
    let healthy: Option<Box<dyn Filter>> = Some(Box::new(Healthy));
    let best: Option<Box<dyn Picker>> = Some(Box::new(Best));
    let mut profile = Profile::of(
        "two-plugins".into(),
        Box::new(GrpcPicker::new(picker.bind(), best)),
    );
    profile.filters = vec![Box::new(GrpcFilter::new(filter.bind(), healthy))];
    profile.scorers = vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
    profile
}

fn binding(plan: Plan, decode_profile: Option<u16>) -> Binding {
    let mut binding = Binding::of("*".into(), 0);
    binding.phases = vec![];
    binding.plan = plan;
    binding.decode_profile = decode_profile;
    binding.shadow = None;
    binding.decode_shadow = None;
    binding.sessions = None;
    binding
}

fn single_engine(profile: Profile) -> Engine {
    Engine::new(
        vec![profile],
        vec![binding(Plan::Single, None)],
        1,
        Needs::default(),
    )
}

/// `n` healthy candidates (ids from 100) whose in-flight counts descend:
/// row 0 is the most loaded, the last row the least.
fn pool(n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|i| {
            let mut row = Candidate::default();
            row.position = i;
            row.id = WorkerId(u64::from(i) + 100);
            row.health.healthy = true;
            row.in_flight = n - i;
            row
        })
        .collect()
}

fn facts(seed: u64) -> RequestFacts<'static> {
    let mut facts = RequestFacts::default();
    facts.model = "m";
    facts.budget_micros = BUDGET_MICROS;
    facts.seed = seed;
    facts.now_micros = 777_000;
    facts
}

fn pool_facts(candidates: &[Candidate]) -> PoolFacts<'static> {
    let mut pool = PoolFacts::default();
    pool.pool_size = candidates.len() as u32;
    pool.generation = 1;
    pool
}

fn decide_with(engine: &Engine, facts: &RequestFacts<'_>, candidates: &[Candidate]) -> Decision {
    engine.decide(
        facts,
        &pool_facts(candidates),
        candidates,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

fn decide(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    decide_with(engine, &facts(seed), candidates)
}

/// A decision under the short budget: the frame's deadline is 20 ms.
fn decide_short(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    let mut short = facts(seed);
    short.budget_micros = SHORT_MICROS;
    decide_with(engine, &short, candidates)
}

/// Decides under `facts` until `judged` of the plugin's frames were answered
/// or missed with the I/O thread seen running, whatever the host's load. A
/// frame the stream's window refuses, a cold stream, or a frame the I/O
/// thread stalled under (held off its core) is no sample, so a decision
/// that judged nothing gives the I/O thread a moment and is tried again.
fn decide_until_judged(
    engine: &Engine,
    plugin: &Plugin,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    judged: u64,
) {
    let count = || {
        let stats = plugin.stats();
        stats.answered + stats.missed - stats.stalls
    };
    let mut tries = 0_u32;
    while count() < judged {
        tries += 1;
        assert!(
            tries <= 10_000,
            "the plugin's frames do not leave: {:?}",
            plugin.stats()
        );
        let before = count();
        assert!(matches!(
            decide_with(engine, facts, candidates),
            Decision::Pick(_)
        ));
        if count() == before {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// The lead target's id; a shed or a queue has none.
fn lead(decision: &Decision) -> Option<u64> {
    decision
        .pick()
        .and_then(|pick| pick.targets.first().map(|t| t.id.0))
}

/// One pair decision with `plugin` bound on both sides: three prefills,
/// four decodes of one pair class.
fn decide_pair(plugin: &Arc<Plugin>, facts: &RequestFacts<'_>) -> Decision {
    let engine = Engine::new(
        vec![profile(plugin, ALL)],
        vec![binding(Plan::PdPair, Some(0))],
        1,
        Needs::default(),
    );
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 7;
    pool_facts.generation = 1;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    engine.decide_pair(
        facts,
        &pool_facts,
        pools,
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
}
