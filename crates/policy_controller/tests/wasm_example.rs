//! The example policy plugin (`examples/wasm/wasm-guest-policy`) as its own
//! build line leaves it: the host loads the component, the plugin decides,
//! sheds an unusable pool and counts, and a configuration it cannot read
//! fails the load.
//! `examples/wasm/wasm-guest-policy/build.sh` builds it; without the
//! artifact the cases skip, and say so, unless `SMG_REQUIRE_POLICY_GUEST` is
//! set: CI builds the example where it runs this crate's tests, and a
//! missing artifact there is a failure.
#![cfg(feature = "wasm")]
#![expect(
    clippy::expect_used,
    reason = "the load and the reads run inside the tests; a failure there is the test's failure"
)]

use std::sync::Arc;

use smg_policy_controller::{
    wasm::{
        Failure, LoadError, Plugin, PluginSpec, Runtime, WasmFilter, WasmPicker, WasmScorer, World,
    },
    Binding, Candidate, Constraints, Decision, Engine, Health, Lease, Lifecycle, Needs, Outcome,
    Plan, PoolFacts, Profile, ReleaseReason, RequestFacts, Scratch, ShedReason, Side,
    TransportClass, Weighted, WorkerId,
};

/// What the example's build line writes.
const ARTIFACT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/wasm/wasm-guest-policy/target/wasm32-unknown-unknown/release/wasm_guest_policy.component.wasm"
);

/// Set to a non-empty value where the example is built before the tests (CI
/// sets it on the job that builds it): the cases then fail, instead of
/// skipping, when the artifact is not there.
const REQUIRE_POLICY_GUEST: &str = "SMG_REQUIRE_POLICY_GUEST";

/// The example as built, for a test that needs it: where it is not built, a
/// failure where [`REQUIRE_POLICY_GUEST`] is set, else a skip said on stderr.
#[expect(
    clippy::print_stderr,
    reason = "a skip says so; the harness shows it under --nocapture"
)]
fn example_or_skip(test: &str) -> Option<Vec<u8>> {
    match std::fs::read(ARTIFACT) {
        Ok(component) => Some(component),
        Err(error) => {
            let required =
                std::env::var_os(REQUIRE_POLICY_GUEST).is_some_and(|value| !value.is_empty());
            assert!(
                !required,
                "the example plugin is not built ({ARTIFACT}: {error}), and {REQUIRE_POLICY_GUEST} is set: {test}"
            );
            eprintln!("the example plugin is not built: skipping {test}");
            None
        }
    }
}

/// The example's spec with `config` as its configuration: empty for the
/// example's default, else one byte, the session bonus in percent.
fn spec_with(component: Vec<u8>, config: &[u8]) -> PluginSpec {
    let mut spec = PluginSpec::new("example", component);
    spec.config = config.to_vec();
    spec.budget_micros = 10_000;
    spec.metrics = vec!["completed".to_string()];
    spec
}

/// The example loaded with `config` under the default failure mode and
/// bound as the three stages of one profile, with no fallback stage.
fn engine_with(component: Vec<u8>, config: &[u8]) -> (Arc<Plugin>, Engine) {
    engine_under(spec_with(component, config))
}

/// The example loaded as `spec` says and bound as the three stages of one
/// profile, with no fallback stage.
fn engine_under(spec: PluginSpec) -> (Arc<Plugin>, Engine) {
    let runtime = Arc::new(Runtime::new().expect("a runtime"));
    let plugin = Plugin::load(&runtime, spec).expect("the example loads");
    let bound = plugin.bind();
    let profile = Profile {
        name: "example".into(),
        filters: vec![Box::new(WasmFilter::new(Arc::clone(&bound), None))],
        scorers: vec![Weighted {
            scorer: Box::new(WasmScorer::new(Arc::clone(&bound), None)),
            weight: 1.0,
        }],
        picker: Box::new(WasmPicker::new(bound, None)),
    };
    let engine = Engine::new(
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
    );
    (plugin, engine)
}

/// Four workers with capacity whose requests in flight descend from four to
/// one; `healthy` says which of them are healthy.
fn pool(healthy: [bool; 4]) -> Vec<Candidate> {
    (0..4u32)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(100 + u64::from(i)),
            health: Health {
                healthy: healthy[i as usize],
                ..Health::default()
            },
            in_flight: 4 - i,
            has_capacity: true,
            ..Candidate::default()
        })
        .collect()
}

/// The decision over `candidates`.
fn decide(engine: &Engine, candidates: &[Candidate]) -> Decision {
    let facts = RequestFacts {
        model: "m",
        budget_micros: 10_000,
        seed: 1,
        ..RequestFacts::default()
    };
    let pool_facts = PoolFacts {
        pool_size: candidates.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    engine.decide(
        &facts,
        &pool_facts,
        candidates,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// The first target of the decision over `candidates`.
fn first_pick(engine: &Engine, candidates: &[Candidate]) -> Option<u64> {
    decide(engine, candidates)
        .pick()
        .and_then(|pick| pick.targets.iter().next().map(|target| target.id.0))
}

/// The component the build line wrote is a record-world plugin the host
/// loads (it imports nothing but the package's own interfaces), and it
/// picks the healthy worker with the fewest requests in flight.
#[test]
fn the_example_loads_and_picks_the_least_loaded_healthy_worker() {
    let Some(component) =
        example_or_skip("the_example_loads_and_picks_the_least_loaded_healthy_worker")
    else {
        return;
    };
    let (plugin, engine) = engine_with(component, &[20]);
    assert_eq!(plugin.world(), World::Record);
    // The idlest worker, the fourth, is unhealthy: the third is picked.
    assert_eq!(
        first_pick(&engine, &pool([true, true, true, false])),
        Some(102)
    );
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.answered,
            stats.declined,
            stats.traps,
            stats.fuel_exhausted,
            stats.invalid
        ),
        (4, 0, 0, 0, 0),
        "the constructor and the three stages answered: {stats:?}"
    );

    // A healthy worker with no capacity left is no candidate either.
    let mut full = pool([true, true, true, false]);
    full[2].has_capacity = false;
    assert_eq!(first_pick(&engine, &full), Some(101));

    // Two workers as idle as each other: the lower position is picked.
    let mut tied = pool([true, true, true, false]);
    tied[1].in_flight = tied[2].in_flight;
    assert_eq!(first_pick(&engine, &tied), Some(101));
}

/// The configuration byte reaches `init`: under a bonus of twenty percent
/// the busiest worker wins when it holds the session, and under none it
/// does not. An empty configuration is the example's default, a fifth.
#[test]
fn the_examples_session_bonus_is_its_configuration() {
    let Some(component) = example_or_skip("the_examples_session_bonus_is_its_configuration") else {
        return;
    };
    let mut candidates = pool([true, true, true, false]);
    candidates[0].session_hit = true;
    let (_, with_bonus) = engine_with(component.clone(), &[20]);
    assert_eq!(
        first_pick(&with_bonus, &candidates),
        Some(100),
        "a fifth for the session's worker and a fifth of bonus pass a third"
    );
    // The bonus holds at the top of the range, where the host clamps a
    // score: of two idle workers the session's is picked, not the first.
    let mut idle = pool([true, true, true, false]);
    idle[0].in_flight = 0;
    idle[1].in_flight = 0;
    idle[1].session_hit = true;
    assert_eq!(first_pick(&with_bonus, &idle), Some(101));
    let (_, without) = engine_with(component.clone(), &[0]);
    assert_eq!(first_pick(&without, &candidates), Some(102));
    let (_, by_default) = engine_with(component, &[]);
    assert_eq!(first_pick(&by_default, &candidates), Some(100));
}

/// A configuration the example cannot read fails the load with the
/// plugin's own message, so an apply that carries it is refused: a bonus
/// past a hundred percent, and more bytes than the one it reads.
#[test]
fn the_example_refuses_a_configuration_it_cannot_read() {
    let Some(component) = example_or_skip("the_example_refuses_a_configuration_it_cannot_read")
    else {
        return;
    };
    let runtime = Arc::new(Runtime::new().expect("a runtime"));
    for (config, said) in [
        (&[101][..], "0 to 100 percent, got 101"),
        (&[20, 20][..], "one byte, got 2"),
    ] {
        let refused = Plugin::load(&runtime, spec_with(component.clone(), config))
            .map(|_| ())
            .expect_err("the load is refused");
        assert!(
            matches!(
                &refused,
                LoadError::Init { name, reason } if name == "example" && reason.contains(said)
            ),
            "{config:?}: {refused:?}"
        );
    }
}

/// With no usable worker the example's filter empties the pool and the
/// request is shed, naming the plugin: no worker is picked, with no
/// fallback bound, under either failure mode. A completed dispatch reaches
/// `observe` and the counter the spec declared.
#[test]
fn the_example_sheds_an_unusable_pool_and_counts_completions() {
    let Some(component) =
        example_or_skip("the_example_sheds_an_unusable_pool_and_counts_completions")
    else {
        return;
    };
    // No healthy worker; then healthy workers with no capacity left.
    let mut full = pool([true; 4]);
    for candidate in &mut full {
        candidate.has_capacity = false;
    }
    let unusable = [pool([false; 4]), full];
    for failure in [Failure::Closed, Failure::Open] {
        let mut spec = spec_with(component.clone(), &[20]);
        spec.failure = failure;
        let (plugin, engine) = engine_under(spec);
        for pool in &unusable {
            let decision = decide(&engine, pool);
            assert!(
                matches!(
                    &decision,
                    Decision::Shed(shed)
                        if shed.reason == ShedReason::NoEndpoints && shed.emptied_by == "example"
                ),
                "failure {failure:?}: {decision:?}"
            );
        }
        let stats = plugin.stats();
        assert_eq!(
            (stats.declined, stats.traps, stats.invalid),
            (0, 0, 0),
            "failure {failure:?}: the filter answered and nothing failed: {stats:?}"
        );
        assert!(plugin.is_warm(), "an emptied pool keeps the instance");
    }

    let (plugin, engine) = engine_with(component, &[20]);
    let completed = || {
        plugin
            .counters()
            .read()
            .next()
            .map(|(_, value)| value)
            .expect("the declared counter")
    };
    // A lifecycle event of another kind reaches `observe` and is not
    // counted: the counter is of completions.
    engine.on_released(&Lease::default(), ReleaseReason::ClientGone);
    assert_eq!(plugin.stats().observed, 1, "the release was observed");
    assert_eq!(completed(), 0);
    engine.on_completed(
        &Lease::default(),
        &Outcome {
            side: Side::Single,
            served: WorkerId(102),
            status: 200,
            class: TransportClass::Ok,
            ttft_micros: 0,
            tokens_in: 0,
            tokens_out: 0,
            duration_micros: 0,
        },
    );
    assert_eq!(completed(), 1);
}
