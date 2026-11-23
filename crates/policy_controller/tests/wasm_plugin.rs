//! The WebAssembly plugin host against the test guests: what loads and what
//! is refused, the honest pick through both worlds, the four host imports,
//! and the failure rules (a decline keeps the instance, a trap, an exhausted
//! budget or an invalid answer drops it and counts on the breaker, the
//! breaker bypasses, the fallbacks and the closed policy), one resource per
//! decision and per side of a pair, the lifecycle once per engine build, and
//! the instances kept per decider.
#![cfg(feature = "wasm")]
#![expect(
    clippy::expect_used,
    reason = "the guests' assembly and the loads run inside the tests; a failure there is the test's failure"
)]

mod support {
    pub mod wasm_guest;
}

use std::{
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering},
        mpsc, Arc,
    },
};

use smg_policy_controller::{
    document::Needs,
    engine::{
        Binding, Candidate, Constraints, Decision, DecisionSlot, Engine, Health, Identity,
        PairPools, Pick, Plan, PoolFacts, Profile, RequestFacts, Scorer as _, Scratch, ShedReason,
        Target, WeightedScorer, WorkerId,
    },
    plugins::{
        shared::{BreakerSpec, Failure, FrameBudget, OpenDecisions, Reveal},
        wasm::{
            Abi, LoadError, Plugin, PluginBinding, PluginSpec, Runtime, WasmFilter, WasmPicker,
            WasmScorer, World,
        },
    },
    policies::power_of_two::PowerOfTwo,
    stages::{Best, Healthy, LeastRequests, Lifecycle, SessionAffinity},
    tracking::{EventKind, EventSet, Lease, Side},
    Draw,
};
use support::wasm_guest::{self as guest, canonical, counter, mode};
use wit_component::{ComponentEncoder, StringEncoding};
use wit_parser::Resolve;

const FRAME_MICROS: u32 = 10_000;
/// The decision budget of a case that tests neither the frame nor the
/// decision's budget: a minute, which no hold of a runner spends, so such a
/// case is not decided by the wall clock. A call draws the decision's budget
/// by its elapsed time: under a budget of the frame's ten milliseconds, one
/// hold of that length inside one call sent every later stage to its
/// fallback.
const UNTIMED_BUDGET_MICROS: u32 = 60_000_000;
/// A thousand times below this host's speed: a charge at this calibration is
/// its fuel unless one call is held off the core for longer than a burn
/// (near a tenth of a second), so a case's budget arithmetic in burns holds
/// with sibling test threads on a loaded runner. A fifty-fold margin lost a
/// neighbour's score to a stolen timeslice at eight threads on four cores.
const SLOW_CALIBRATION: u64 = 1;
/// The per-call frame of a slow-calibrated plugin: a second of fuel-time,
/// near ten burns, so the frame never ends a burn.
const SLOW_FRAME_MICROS: u32 = 1_000_000;
/// The pair cases' decision budget at the slow calibration: 1.3 seconds of
/// fuel-time, a burn nine percent of it, so six burns pass half of it with
/// near six hundred milliseconds of slack for a stolen timeslice (the decode
/// case runs on half of it, three burns against three hundred).
const PAIR_BUDGET_MICROS: u32 = 1_300_000;
const COUNTERS: [&str; 6] = [
    "observed",
    "now",
    "random",
    "observe-now",
    "observe-session",
    "pick-random",
];

fn runtime() -> Arc<Runtime> {
    Arc::new(Runtime::new(&OpenDecisions::new()).expect("a runtime"))
}

fn spec(name: &str, component: Vec<u8>, mode: u8) -> PluginSpec {
    let mut spec = PluginSpec::new(name, component);
    spec.config = vec![mode];
    spec.budget_micros = FRAME_MICROS;
    spec.metrics = COUNTERS.iter().map(|s| s.to_string()).collect();
    // Every kind of event: the lifecycle's cases tell them, and a decision
    // reads none of it.
    spec.observes = EventSet::ALL;
    spec
}

/// The host's frame of an `init`. Nothing watches an `init`: it ends at
/// this deadline on the wall clock, whatever its thread had of the CPU.
const INIT_FRAME: std::time::Duration = std::time::Duration::from_millis(100);

/// How many times a case makes what a hold of the runner can have moved: a
/// load, a close's rebuild, a lazy `init`, a destructor's time. A try whose
/// witness says a hold reached it says so and is made again. No witness
/// passes a run: the last try asserts whatever its witness says, so a host
/// that fails by itself fails each try, and the case. A window of short
/// holds has a count of its own (`WINDOWS`).
const TRIES: u32 = 3;

/// Loads `spec` and returns what the load said. Its `init` ends at a
/// deadline on the wall clock, so a runner that holds the loading thread for
/// the init frame has the host end an `init` that did nothing. Such a load
/// is made again, [`TRIES`] loads at most: its thread was off the CPU half
/// the frame. A guest that spends the frame itself has the CPU, and its load
/// ends here as it did.
fn load_unheld(runtime: &Arc<Runtime>, spec: PluginSpec) -> Result<Arc<Plugin>, LoadError> {
    let mut loads = 0;
    loop {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let load = Plugin::load(runtime, spec.clone());
        loads += 1;
        let held = matches!(
            &load,
            Err(LoadError::Trap { reason, .. }) if reason.ends_with("wasm trap: interrupt")
        ) && off_cpu_since(wall, cpu) >= INIT_FRAME / 2;
        if !held || loads == TRIES {
            return load;
        }
        made_again("a load", "its thread was off the CPU half the init frame");
    }
}

/// How many times a load that must succeed is made.
const LOADS: u32 = 10;

/// A load that must succeed: its case claims nothing of the `init`'s time.
/// An `init` the host ended at the frame's deadline was ended by the
/// runner, off the CPU or slowed on its core, where the thread's clock reads
/// it on the CPU and no witness of time off the CPU sees it. Such a load is
/// made again with no witness asked, [`LOADS`] loads at most. The last
/// asserts what it says, so a host that ends every `init` fails the case.
fn loaded(runtime: &Arc<Runtime>, spec: PluginSpec) -> Arc<Plugin> {
    loaded_by(|| Plugin::load(runtime, spec.clone()))
}

/// [`loaded`] over the load it makes: the rule apart from the host.
fn loaded_by(mut load: impl FnMut() -> Result<Arc<Plugin>, LoadError>) -> Arc<Plugin> {
    // The host's warnings are kept from a case's first load on: a case reads
    // there what the host said of its plugin's rebuild.
    said();
    let mut loads = 0;
    loop {
        let made = load();
        loads += 1;
        let ended = matches!(
            &made,
            Err(LoadError::Trap { reason, .. }) if reason.ends_with("wasm trap: interrupt")
        );
        if !ended || loads == LOADS {
            return made.expect("the guest loads");
        }
        made_again("a load", "the host ended its init at the frame's deadline");
    }
}

thread_local! {
    /// The warm-ups this thread made again ([`warmed`]): an instance each,
    /// for the case that counts its instances behind one.
    static WARMED_AGAIN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// A warm-up that must succeed: its case claims nothing of the `init`'s
/// time. As a load ([`loaded`]): made again where the host ended the `init`
/// at the frame's deadline, [`LOADS`] warm-ups at most, and the last
/// asserts what it says. One made again is an instance more, counted for
/// its thread in [`WARMED_AGAIN`].
fn warmed(plugin: &Plugin) {
    let mut warms = 0;
    loop {
        let made = plugin.warm();
        warms += 1;
        let ended = matches!(
            &made,
            Err(LoadError::Trap { reason, .. }) if reason.ends_with("wasm trap: interrupt")
        );
        if !ended || warms == LOADS {
            return made.expect("it warms");
        }
        WARMED_AGAIN.set(WARMED_AGAIN.get() + 1);
        made_again(
            "a warm-up",
            "the host ended its init at the frame's deadline",
        );
    }
}

/// An `init` the host ended at the frame's deadline, as a load says it.
fn ended_at_the_frame() -> LoadError {
    LoadError::Trap {
        name: "held".to_owned(),
        reason: "wasm trap: interrupt".to_owned(),
    }
}

/// A load that must succeed is made again after an `init` the host ended at
/// the frame, with no witness asked. Here every load but the last one
/// allowed is handed back as such an end, and the last hands back the
/// guest's own load. That load is made ahead, by the rule: made here as the
/// last one allowed, an `init` the host ended would have no load behind it.
#[test]
fn a_load_the_host_ended_is_made_again() {
    let runtime = runtime();
    let own = loaded(
        &runtime,
        spec("held", guest::record_guest(), mode::LEAST_IN_FLIGHT),
    );
    let mut made = 0;
    let plugin = loaded_by(|| {
        made += 1;
        if made < LOADS {
            return Err(ended_at_the_frame());
        }
        Ok(Arc::clone(&own))
    });
    assert_eq!(made, LOADS);
    assert!(Arc::ptr_eq(&plugin, &own), "the last load's own answer");
    assert_eq!(plugin.stats().traps, 0);
}

/// The last load asserts what it says: a host that ends every `init` fails
/// the case, at the last load allowed.
#[test]
fn a_host_that_ends_every_init_fails_the_load() {
    let mut made = 0;
    let load = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loaded_by(|| {
            made += 1;
            Err(ended_at_the_frame())
        })
    }));
    assert!(load.is_err(), "a load the host ended every time stood");
    assert_eq!(made, LOADS);
}

/// A load refused for any other reason is the case's failure at once: it is
/// made one time.
#[test]
fn a_load_refused_for_another_reason_is_made_once() {
    let mut made = 0;
    let load = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loaded_by(|| {
            made += 1;
            Err(LoadError::Runtime("refused".to_owned()))
        })
    }));
    assert!(load.is_err(), "a refused load stood");
    assert_eq!(made, 1);
}

fn load(runtime: &Arc<Runtime>, name: &str, component: Vec<u8>, mode: u8) -> Arc<Plugin> {
    loaded(runtime, spec(name, component, mode))
}

/// `load` at the slow calibration, with its frame.
fn load_slow(runtime: &Arc<Runtime>, name: &str, component: Vec<u8>, mode: u8) -> Arc<Plugin> {
    let mut spec = spec(name, component, mode);
    spec.fuel_per_micro = SLOW_CALIBRATION;
    spec.budget_micros = SLOW_FRAME_MICROS;
    loaded(runtime, spec)
}

/// A profile whose three stages are the plugin's, each with the given
/// fallback: the one profile of its build that binds the plugin.
fn plugin_profile(
    plugin: &Arc<Plugin>,
    filter_fallback: Option<Box<dyn smg_policy_controller::engine::Filter>>,
    scorer_fallback: Option<Box<dyn smg_policy_controller::engine::Scorer>>,
    picker_fallback: Option<Box<dyn smg_policy_controller::engine::Picker>>,
) -> Profile {
    bound_profile(
        plugin.bind(),
        filter_fallback,
        scorer_fallback,
        picker_fallback,
    )
}

/// A profile whose three stages share `bound`, each with the given fallback.
fn bound_profile(
    bound: Arc<PluginBinding>,
    filter_fallback: Option<Box<dyn smg_policy_controller::engine::Filter>>,
    scorer_fallback: Option<Box<dyn smg_policy_controller::engine::Scorer>>,
    picker_fallback: Option<Box<dyn smg_policy_controller::engine::Picker>>,
) -> Profile {
    Profile {
        name: "plugin".into(),
        filters: vec![Box::new(WasmFilter::new(
            Arc::clone(&bound),
            filter_fallback,
        ))],
        scorers: vec![WeightedScorer {
            scorer: Box::new(WasmScorer::new(Arc::clone(&bound), scorer_fallback)),
            weight: 1.0,
        }],
        picker: Box::new(WasmPicker::new(bound, picker_fallback)),
    }
}

fn single_engine(profile: Profile) -> Engine {
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

fn plugin_engine(plugin: &Arc<Plugin>) -> Engine {
    single_engine(plugin_profile(plugin, None, None, None))
}

/// `n` healthy candidates whose in-flight counts descend, so the least
/// loaded is the last one.
fn pool(n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(u64::from(i) + 100),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight: n - i,
            ..Candidate::default()
        })
        .collect()
}

/// A request under the untimed budget. A case that tests the decision's
/// budget names its own (`timed`, or the field).
fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        budget_micros: UNTIMED_BUDGET_MICROS,
        seed,
        now_micros: 777_000,
        ..RequestFacts::default()
    }
}

/// A request whose decision budget is one frame: for the cases that assert
/// what a spent budget does.
fn timed(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        budget_micros: FRAME_MICROS,
        ..facts(seed)
    }
}

fn decide(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    decide_with(engine, candidates, &facts(seed))
}

fn decide_with(engine: &Engine, candidates: &[Candidate], facts: &RequestFacts<'_>) -> Decision {
    let pool_facts = PoolFacts {
        pool_size: candidates.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    engine.decide(
        facts,
        &pool_facts,
        candidates,
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

/// The targets' ids; a shed or a queue has none, so an assert reads `[]`.
fn pick_ids(decision: &Decision) -> Vec<u64> {
    decision
        .pick()
        .map(|pick| pick.targets.iter().map(|t| t.id.0).collect())
        .unwrap_or_default()
}

fn counter_value(plugin: &Plugin, id: u32) -> u64 {
    plugin
        .counters()
        .read()
        .nth(id as usize)
        .map(|(_, v)| v)
        .expect("a declared counter")
}

#[test]
fn the_guests_offsets_match_the_canonical_layout() {
    guest::check_canonical_layout();
    assert_eq!(canonical::CANDIDATE_SIZE % 8, 0);
}

#[test]
fn both_worlds_load_and_the_load_instantiates_on_the_applying_thread() {
    let runtime = runtime();
    let record = load(
        &runtime,
        "record",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let packed = load(
        &runtime,
        "packed",
        guest::packed_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    assert_eq!(record.world(), World::Record);
    assert_eq!(packed.world(), World::Packed);
    assert_eq!(record.abi(), Abi::V1_2);
    assert_eq!(packed.abi(), Abi::V1_2);
    assert!(record.is_warm() && packed.is_warm());
    assert_eq!(record.stats().instantiations, 1);
    assert_eq!(record.name(), "record");
}

#[test]
fn a_wasi_import_is_refused_at_load() {
    let runtime = runtime();
    let err = Plugin::load(
        &runtime,
        spec("wasi", guest::wasi_importing_guest(), mode::LEAST_IN_FLIGHT),
    )
    .expect_err("refused");
    assert!(
        matches!(&err, LoadError::Wasi { import, .. } if import.starts_with("wasi:clocks/monotonic-clock")),
        "{err}"
    );
}

#[test]
fn a_component_of_another_world_has_no_policy_world() {
    let mut resolve = Resolve::new();
    let package = resolve
        .push_str("empty.wit", "package test:empty;\nworld nothing {}")
        .expect("the world parses");
    let world = resolve
        .select_world(&[package], Some("nothing"))
        .expect("the world exists");
    let mut module = wat::parse_str("(module)").expect("assembles");
    wit_component::embed_component_metadata(&mut module, &resolve, world, StringEncoding::UTF8)
        .expect("embeds");
    let component = ComponentEncoder::default()
        .module(&module)
        .expect("matches")
        .validate(true)
        .encode()
        .expect("encodes");
    let err = Plugin::load(&runtime(), spec("empty", component, 0)).expect_err("refused");
    assert_eq!(
        err,
        LoadError::NoWorld {
            name: "empty".into()
        }
    );
}

#[test]
fn a_component_with_an_async_export_is_refused_at_its_compile() {
    // The runtime's own engine keeps no state for concurrent calls, so a
    // component that needs the async component support does not compile. An
    // engine that kept it would compile this one, and the loader would
    // refuse it later, for having no policy world.
    let needs_it = wat::parse_str(
        r#"(component
            (core module $m
                (func (export "run") (result i32) i32.const 0)
                (func (export "cb") (param i32 i32 i32) (result i32) i32.const 0))
            (core instance $i (instantiate $m))
            (type $run (func async))
            (func (export "run") (type $run)
                (canon lift (core func $i "run") async (callback (core func $i "cb")))))"#,
    )
    .expect("the component assembles");
    let err = Plugin::load(&runtime(), spec("async-export", needs_it, 0)).expect_err("refused");
    assert!(matches!(err, LoadError::Compile { .. }), "{err}");
}

#[test]
fn the_abi_pin_the_memory_floor_the_init_failure_and_bad_bytes_are_refused() {
    let runtime = runtime();
    let mut pinned = spec("pinned", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    pinned.abi = Some(Abi { major: 1, minor: 3 });
    let err = Plugin::load(&runtime, pinned).expect_err("refused");
    assert!(
        matches!(
            err,
            LoadError::AbiPin {
                found: Abi::V1_2,
                ..
            }
        ),
        "{err}"
    );

    let mut no_memory = spec("tiny", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    no_memory.memory_mib = 0;
    assert!(matches!(
        Plugin::load(&runtime, no_memory).expect_err("refused"),
        LoadError::Memory { .. }
    ));

    let mut unfuelled = spec("unfuelled", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    unfuelled.fuel_per_micro = 0;
    assert!(matches!(
        Plugin::load(&runtime, unfuelled).expect_err("refused"),
        LoadError::Fuel { .. }
    ));

    let err = load_unheld(
        &runtime,
        spec("refusing", guest::record_guest(), mode::REFUSE_INIT),
    )
    .expect_err("init failed");
    assert_eq!(
        err,
        LoadError::Init {
            name: "refusing".into(),
            reason: "refused".into()
        }
    );

    let err = Plugin::load(&runtime, spec("bytes", b"not wasm".to_vec(), 0)).expect_err("refused");
    assert!(matches!(err, LoadError::Compile { .. }), "{err}");

    let specs = [
        spec("a", Vec::new(), 0),
        spec("b", Vec::new(), 0),
        spec("c", Vec::new(), 0),
    ];
    assert_eq!(
        Plugin::check_memory_budget(&specs, 47),
        Err(LoadError::MemoryBudget {
            asked: 48,
            budget: 47
        })
    );
    assert_eq!(Plugin::check_memory_budget(&specs, 48), Ok(()));
}

#[test]
fn the_honest_guest_picks_the_least_in_flight_through_either_world() {
    let runtime = runtime();
    for (name, component) in [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ] {
        let plugin = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        let engine = plugin_engine(&plugin);
        let candidates = pool(8);
        let decision = decide(&engine, &candidates, 5);
        let ids = pick_ids(&decision);
        assert_eq!(
            ids[0], 107,
            "{name}: the last row has one request in flight"
        );
        assert_eq!(
            ids,
            vec![107, 106, 105, 104],
            "{name}: the alternates follow by score"
        );
        let Some(pick) = decision.pick() else {
            panic!("a pick, not {decision:?}")
        };
        assert_eq!(pick.trace.survivors.get(), Some(8));
        let stats = plugin.stats();
        assert_eq!(
            stats.answered, 4,
            "{name}: the constructor, filter, score and pick"
        );
        assert_eq!(
            (
                stats.declined,
                stats.traps,
                stats.fuel_exhausted,
                stats.invalid
            ),
            (0, 0, 0, 0)
        );
        assert!(stats.fuel_used > 0);
        // The host imports: the decision's clock and its seed's draw.
        assert_eq!(counter_value(&plugin, counter::NOW), 777_000, "{name}");
        assert_eq!(
            counter_value(&plugin, counter::RANDOM),
            Draw::new(5).next_u64(),
            "{name}"
        );
        assert_eq!(plugin.counters().undeclared(), 0);
    }
}

const BAND_SEEDS: Range<u64> = 129..193;

/// Each score differs from the next. Only the two highest fit within the
/// guest's band of the top score.
fn band_pool() -> Vec<Candidate> {
    let mut candidates = pool(4);
    for (candidate, in_flight) in candidates.iter_mut().zip([87, 63, 29, 11]) {
        candidate.in_flight = in_flight;
    }
    candidates
}

fn band_guests() -> [(&'static str, Vec<u8>); 2] {
    [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ]
}

fn band_engine(
    plugin: &Arc<Plugin>,
    fallback: Option<Box<dyn smg_policy_controller::engine::Scorer>>,
) -> Engine {
    single_engine(Profile {
        name: "band".into(),
        filters: vec![],
        scorers: vec![WeightedScorer {
            scorer: Box::new(WasmScorer::new(plugin.bind(), fallback)),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    })
}

fn lead(decision: &Decision) -> Option<WorkerId> {
    decision.pick()?.targets.first().map(|target| target.id)
}

/// A guest's band reaches the built-in picker. Over 64 seeds the lead is
/// one of the two rows within the band of the best score, and each of the
/// two leads at least once, in either world.
#[test]
fn a_guests_band_widens_the_tie_the_built_in_picker_draws_from() {
    use std::collections::BTreeSet;

    let runtime = runtime();
    let candidates = band_pool();
    let leads = band_guests().map(|(world, component)| {
        let plugin = load(&runtime, world, component, mode::SCORE_BAND);
        let engine = band_engine(&plugin, None);
        let leads = BAND_SEEDS
            .map(|seed| lead(&decide(&engine, &candidates, seed)))
            .collect::<BTreeSet<_>>();
        let stats = plugin.stats();
        assert_eq!(stats.answered, 2 * BAND_SEEDS.count() as u64, "{world}");
        assert_eq!(stats.invalid, 0, "{world}");
        leads
    });
    let tied = BTreeSet::from([Some(candidates[2].id), Some(candidates[3].id)]);
    assert_eq!(leads, [tied.clone(), tied], "record and packed leads");
}

/// A guest that answers no band draws within the default one: the best row
/// leads at every seed.
#[test]
fn a_guest_that_answers_no_band_decides_as_before() {
    let runtime = runtime();
    let candidates = band_pool();
    for (world, component) in band_guests() {
        let plugin = load(&runtime, world, component, mode::LEAST_IN_FLIGHT);
        let engine = band_engine(&plugin, None);
        for seed in BAND_SEEDS {
            assert_eq!(
                lead(&decide(&engine, &candidates, seed)),
                Some(candidates[3].id),
                "{world}, seed {seed}"
            );
        }
        let stats = plugin.stats();
        assert_eq!(stats.answered, 2 * BAND_SEEDS.count() as u64, "{world}");
        assert_eq!(stats.invalid, 0, "{world}");
    }
}

/// A band that is no number, below zero or above one is an invalid answer,
/// as a score that is no number is: the stage's fallback decides and the
/// fault is counted once.
#[test]
fn a_band_that_is_no_value_from_0_to_1_is_an_invalid_answer() {
    let runtime = runtime();
    let mut candidates = band_pool();
    candidates[1].session_hit = true;
    for ((world, component), (_, negative)) in band_guests()
        .into_iter()
        .zip(guest::guests_with_band(-0.375))
    {
        for (label, component, mode) in [
            ("bad score", &component, mode::NAN_SCORE),
            ("NaN band", &component, mode::NAN_BAND),
            ("band above one", &component, mode::ABOVE_ONE_BAND),
            ("band below zero", &negative, mode::SCORE_BAND),
        ] {
            let plugin = load(&runtime, world, component.clone(), mode);
            let engine = band_engine(&plugin, Some(Box::new(SessionAffinity)));
            let picked = lead(&decide(&engine, &candidates, BAND_SEEDS.start));
            assert_eq!(
                (picked, plugin.stats().invalid),
                (Some(candidates[1].id), 1),
                "{world}: {label}"
            );
        }
    }
}

/// A band belongs to its decision. On one instance each decision with a
/// band is followed by one without, and the second draws within the default
/// band.
#[test]
fn a_band_is_not_carried_to_the_next_decision() {
    let runtime = runtime();
    let candidates = band_pool();
    for (world, component) in band_guests() {
        let plugin = load(&runtime, world, component, mode::SCORE_BAND);
        let engine = band_engine(&plugin, None);
        let mut second_row_led = false;
        for seed in BAND_SEEDS {
            let first = lead(&decide(&engine, &candidates, seed));
            second_row_led |= first == Some(candidates[2].id);
            let without_band = RequestFacts {
                now_micros: 0,
                ..facts(seed)
            };
            assert_eq!(
                lead(&decide_with(&engine, &candidates, &without_band)),
                Some(candidates[3].id),
                "{world}, seed {seed}: the next decision has no band"
            );
        }
        assert!(second_row_led, "{world}: the band put the second row first");
        let stats = plugin.stats();
        assert_eq!(stats.answered, 4 * BAND_SEEDS.count() as u64, "{world}");
        assert_eq!(stats.invalid, 0, "{world}");
        assert_eq!(stats.instantiations, 1, "{world}");
    }
}

/// The band stays with the scores of its decision: a second read of the
/// kept answer says the same band, at both ends of the range and at the
/// default, and asks the guest no second time.
#[test]
fn a_score_answer_keeps_its_band_with_its_scores() {
    let runtime = runtime();
    let candidates = band_pool();
    let facts = facts(BAND_SEEDS.start);
    let pool = PoolFacts {
        pool_size: candidates.len() as u32,
        ..PoolFacts::default()
    };
    for band in [Some(0.0), Some(1.0), None] {
        let guest_mode = if band.is_some() {
            mode::SCORE_BAND
        } else {
            mode::LEAST_IN_FLIGHT
        };
        let expected = band.unwrap_or(smg_policy_controller::engine::TieBand::NORMALIZED.0);
        for (world, component) in guest::guests_with_band(band.unwrap_or_default()) {
            let plugin = load(&runtime, world, component, guest_mode);
            let scorer = WasmScorer::new(plugin.bind(), None);
            let scope = scorer.scope().expect("the scorer opens a decision");
            scope.open(&facts, &candidates, &pool);
            let mut scores = [0.0; 2];
            let first = scorer.score(&facts, &candidates, &[1, 3], &mut scores);
            let [lower, higher] = scores;
            let cached = scorer.score(&facts, &candidates, &[3, 1], &mut scores);
            scope.close();
            assert_eq!(first.0, expected, "{world}: the first read");
            assert_eq!(cached.0, expected, "{world}: the cached read");
            assert!(higher > lower, "{world}: scoring answered");
            assert_eq!(scores, [higher, lower], "{world}: cached scores");
            assert_eq!(plugin.stats().answered, 2, "{world}: one guest score");
        }
    }
}

#[test]
fn a_decision_on_a_kept_instance_opens_one_resource_and_leaves_no_instance_behind() {
    let runtime = runtime();
    let plugin = load(&runtime, "p", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    for seed in 0..50 {
        decide(&engine, &candidates, seed);
    }
    let stats = plugin.stats();
    assert_eq!(
        stats.instantiations, 1,
        "one kept instance for fifty decisions"
    );
    assert_eq!(stats.answered, 200);
    assert!(plugin.is_warm());
}

#[test]
fn the_filters_answer_narrows_the_pool_once_per_decision_and_row_by_row() {
    let runtime = runtime();
    let plugin = load(&runtime, "even", guest::record_guest(), mode::KEEP_EVEN);
    let engine = plugin_engine(&plugin);
    let candidates = pool(8);
    let decision = decide(&engine, &candidates, 1);
    assert_eq!(pick_ids(&decision), vec![106, 104, 102, 100]);
    let Some(pick) = decision.pick() else {
        panic!("a pick, not {decision:?}")
    };
    assert_eq!(pick.trace.survivors.get(), Some(4));
    assert_eq!(plugin.stats().answered, 4, "filter asked once");
}

/// A plugin scorer under a picker that narrows is handed the set compared, as
/// any scorer is: the guest scores its candidates once, the host reads the
/// rows compared in the picker's order, and no line of the interface moves.
/// Under the two-draw picker the better of the two drawn by the guest's score
/// leads, the other follows, and the trace's top rows are the two.
#[test]
fn a_plugin_scorer_under_a_two_draw_picker_scores_the_two_drawn() {
    let runtime = runtime();
    for (name, component) in [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ] {
        let plugin = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        let engine = single_engine(Profile {
            name: "two-draw".into(),
            filters: vec![],
            scorers: vec![WeightedScorer {
                scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
                weight: 1.0,
            }],
            picker: Box::new(PowerOfTwo),
        });
        // The in-flight counts descend, so of two rows the later scores
        // higher: one less a hundredth a request in flight.
        let candidates = pool(8);
        let score = |id: u64| 1.0 - (108 - id) as f32 / 100.0;
        let mut leads = std::collections::BTreeSet::new();
        for seed in 1..=64 {
            let decision = decide(&engine, &candidates, seed);
            let ids = pick_ids(&decision);
            assert_eq!(ids.len(), 2, "{name}, seed {seed}: the two drawn");
            assert!(
                ids[0] > ids[1],
                "{name}, seed {seed}: the fewer in flight leads, {ids:?}"
            );
            let Some(pick) = decision.pick() else {
                panic!("a pick, not {decision:?}")
            };
            assert_eq!(
                pick.trace.survivors.get(),
                Some(8),
                "{name}: the filters' count"
            );
            let top = &pick.trace.top[..usize::from(pick.trace.top_len)];
            assert_eq!(
                top.iter().map(|(id, _)| id.0).collect::<Vec<_>>(),
                ids,
                "{name}, seed {seed}: the trace's top rows are the two"
            );
            assert!(
                top.iter().all(|(id, got)| (got - score(id.0)).abs() < 1e-6),
                "{name}, seed {seed}: each row's own score, {top:?}"
            );
            leads.insert(ids[0]);
        }
        assert!(
            leads.len() > 2,
            "{name}: the draw moves the pair, {leads:?}"
        );
        let stats = plugin.stats();
        assert_eq!(
            stats.answered, 128,
            "{name}: the constructor and one score a decision"
        );
        assert_eq!(
            (
                stats.declined,
                stats.traps,
                stats.fuel_exhausted,
                stats.invalid
            ),
            (0, 0, 0, 0),
            "{name}"
        );
    }
}

#[test]
fn a_declined_filter_keeps_the_instance_and_takes_its_fallback() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "declining",
        guest::record_guest(),
        mode::DECLINE_FILTER,
    );
    let engine = single_engine(plugin_profile(&plugin, Some(Box::new(Healthy)), None, None));
    let mut candidates = pool(4);
    candidates[3].health.healthy = false;
    let decision = decide(&engine, &candidates, 1);
    assert_eq!(
        pick_ids(&decision),
        vec![102, 101, 100],
        "the fallback dropped the unhealthy best row"
    );
    let stats = plugin.stats();
    assert_eq!(stats.declined, 1);
    assert_eq!(stats.instantiations, 1, "a decline keeps the instance");
    assert!(plugin.is_warm());

    // Without a fallback: open keeps every row, closed empties the pool.
    let open = load(
        &runtime,
        "open",
        guest::record_guest(),
        mode::DECLINE_FILTER,
    );
    assert_eq!(
        pick_ids(&decide(&plugin_engine(&open), &candidates, 1)),
        vec![103, 102, 101, 100]
    );
    let mut closed = spec("closed", guest::record_guest(), mode::DECLINE_FILTER);
    closed.failure = Failure::Closed;
    let closed = loaded(&runtime, closed);
    match decide(&plugin_engine(&closed), &candidates, 1) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
            assert_eq!(shed.emptied_by, "closed");
        }
        other => panic!("a shed, not {other:?}"),
    }
}

#[test]
fn a_trap_drops_the_instance_and_the_close_rebuilds_it() {
    tried(a_trap_drops_the_instance_and_the_close_rebuilds_it_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_trap_drops_the_instance_and_the_close_rebuilds_it_in_a_try(made: u32) -> bool {
    const CASE: &str = "a trap";
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "trapping",
        guest::record_guest(),
        mode::TRAP_IN_FILTER,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = decide(&engine, &candidates, 1);
    let held = off_cpu_since(wall, cpu);
    let mut ids = pick_ids(&decision);
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![100, 101, 102, 103],
        "the filter fell back to keep-all; the flat scores tie and the seed orders them"
    );
    let stats = plugin.stats();
    assert_eq!(stats.traps, 1);
    if !the_close_rebuilt(CASE, &plugin, held, made) {
        return false;
    }
    assert_eq!((stats.instantiations, stats.lazy_instantiations), (2, 1));
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    decide(&engine, &candidates, 2);
    let held = off_cpu_since(wall, cpu);
    if !the_close_rebuilt(CASE, &plugin, held, made) {
        return false;
    }
    let stats = plugin.stats();
    assert_eq!(
        stats.instantiations, 3,
        "rebuilt after each fault, never reused"
    );
    assert_eq!(stats.lazy_instantiations, 2);
    assert_eq!(stats.traps, 2);
    true
}

#[test]
fn an_exhausted_budget_is_a_fuel_fault_and_the_scorer_falls_back() {
    tried(an_exhausted_budget_is_a_fuel_fault_and_the_scorer_falls_back_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn an_exhausted_budget_is_a_fuel_fault_and_the_scorer_falls_back_in_a_try(made: u32) -> bool {
    const CASE: &str = "an exhausted budget";
    let runtime = runtime();
    // The slow calibration: the frame's fuel is gone a thousand times sooner
    // than its bound on the clock, so the fuel ends the spin on any host. At
    // the default calibration the two race, and a spin its bound ended first
    // is the same fault with less than the frame's fuel used.
    let plugin = load_slow(
        &runtime,
        "spinning",
        guest::record_guest(),
        mode::SPIN_IN_SCORE,
    );
    let engine = single_engine(plugin_profile(
        &plugin,
        None,
        Some(Box::new(LeastRequests)),
        None,
    ));
    let candidates = pool(4);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = decide(&engine, &candidates, 1);
    let held = off_cpu_since(wall, cpu);
    assert_eq!(
        pick_ids(&decision).first().copied(),
        Some(103),
        "the fallback scorer decided"
    );
    let stats = plugin.stats();
    assert_eq!(stats.fuel_exhausted, 1);
    assert_eq!(stats.traps, 0);
    assert!(
        stats.fuel_used >= u64::from(SLOW_FRAME_MICROS) * SLOW_CALIBRATION,
        "the spin used its frame's fuel: {stats:?}"
    );
    the_close_rebuilt(CASE, &plugin, held, made)
}

/// A guest that states no budget takes the default frame, under a request
/// that states none too: a stage is shown the widest budget there, and the
/// guest's spin is cut at the default frame's fuel, not at the request's.
/// The fallback scorer decides, and the cut is the guest's own failure.
#[test]
fn a_guest_with_no_budget_is_cut_at_the_default_frame_under_a_request_with_none() {
    tried(a_guest_with_no_budget_is_cut_at_the_default_frame_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_guest_with_no_budget_is_cut_at_the_default_frame_in_a_try(made: u32) -> bool {
    const CASE: &str = "a guest with no budget";
    let runtime = runtime();
    // The slow calibration, as for an exhausted budget: the fuel ends the
    // spin long before the frame's bound on the clock.
    let mut unset = spec("unbudgeted", guest::record_guest(), mode::SPIN_IN_SCORE);
    unset.fuel_per_micro = SLOW_CALIBRATION;
    unset.budget_micros = 0;
    let plugin = loaded(&runtime, unset);
    let engine = single_engine(plugin_profile(
        &plugin,
        None,
        Some(Box::new(LeastRequests)),
        None,
    ));
    let candidates = pool(4);
    // What a stage is shown for a request that states no budget.
    let widest = RequestFacts {
        budget_micros: u32::MAX,
        ..facts(1)
    };
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = decide_with(&engine, &candidates, &widest);
    let held = off_cpu_since(wall, cpu);
    assert_eq!(
        pick_ids(&decision).first().copied(),
        Some(103),
        "the fallback scorer decided"
    );
    let stats = plugin.stats();
    // The cut is the guest's own failure: it ran out under its own frame,
    // not under what a request's budget had left.
    assert_eq!(
        (stats.fuel_exhausted, stats.budget_spent, stats.traps),
        (1, 0, 0),
        "{stats:?}"
    );
    let frame = u64::from(FrameBudget::DEFAULT_MICROS) * SLOW_CALIBRATION;
    assert!(
        (frame..2 * frame).contains(&stats.fuel_used),
        "the spin used the default frame's fuel and no more: {stats:?}"
    );
    the_close_rebuilt(CASE, &plugin, held, made)
}

#[test]
fn an_invalid_answer_is_a_fault_of_the_plugin() {
    let runtime = runtime();
    let cases = [
        (mode::FILTER_OUT_OF_RANGE, "out of range"),
        (mode::FILTER_DUPLICATE, "duplicate"),
        (mode::NAN_SCORE, "nan"),
        (mode::PICK_FILTERED_OUT, "filtered out"),
    ];
    for (mode, name) in cases {
        let case = format!("an invalid answer, {name}");
        tried(|made| {
            let plugin = load(&runtime, name, guest::record_guest(), mode);
            let engine = plugin_engine(&plugin);
            let (wall, cpu) = (std::time::Instant::now(), bound_clock());
            let decision = decide(&engine, &pool(4), 1);
            let held = off_cpu_since(wall, cpu);
            assert!(
                matches!(decision, Decision::Pick(_)),
                "{name}: the fallbacks decide"
            );
            let stats = plugin.stats();
            assert_eq!(stats.invalid, 1, "{name}");
            if !the_close_rebuilt(&case, &plugin, held, made) {
                return false;
            }
            assert_eq!(plugin.stats().lazy_instantiations, 1, "{name}");
            true
        });
    }
}

#[test]
fn a_declined_pick_takes_the_fallback_picker_or_sheds_when_closed() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "no-pick",
        guest::record_guest(),
        mode::DECLINE_PICK,
    );
    let engine = single_engine(plugin_profile(&plugin, None, None, Some(Box::new(Best))));
    assert_eq!(
        pick_ids(&decide(&engine, &pool(4), 1)).first().copied(),
        Some(103)
    );
    assert_eq!(plugin.stats().declined, 1);

    // Without a fallback: open ranks by the combined scores, closed sheds.
    let open = load(
        &runtime,
        "open-pick",
        guest::record_guest(),
        mode::DECLINE_PICK,
    );
    assert_eq!(
        pick_ids(&decide(&plugin_engine(&open), &pool(4), 1)),
        vec![103, 102, 101, 100],
        "the plugin's own scores rank the rows"
    );
    assert_eq!(open.stats().declined, 1);
    let mut closed = spec("closed-pick", guest::record_guest(), mode::DECLINE_PICK);
    closed.failure = Failure::Closed;
    let plugin = loaded(&runtime, closed);
    match decide(&plugin_engine(&plugin), &pool(4), 1) {
        Decision::Shed(shed) => {
            assert_eq!(shed.reason, ShedReason::PolicyDeclined);
            assert_eq!(shed.emptied_by, "closed-pick");
        }
        other => panic!("a shed, not {other:?}"),
    }
}

/// A score is a preference and has no veto: declined with no fallback it is
/// a flat score under either failure mode, the plugin weighs nothing and
/// nothing is shed (what must hold or shed is a filter). The plugin's own
/// pick, which with its scores names the least loaded row, names the first.
#[test]
fn a_declined_score_has_no_veto_under_either_failure_mode() {
    let runtime = runtime();
    let scoring = load(
        &runtime,
        "scoring",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    assert_eq!(
        pick_ids(&decide(&plugin_engine(&scoring), &pool(4), 1))
            .first()
            .copied(),
        Some(103),
        "with its scores the plugin picks the least loaded row"
    );
    for failure in [Failure::Open, Failure::Closed] {
        let mut declining = spec("no-score", guest::record_guest(), mode::DECLINE_SCORE);
        declining.failure = failure;
        let plugin = loaded(&runtime, declining);
        let decision = decide(&plugin_engine(&plugin), &pool(4), 1);
        let mut ids = pick_ids(&decision);
        assert_eq!(
            ids.first().copied(),
            Some(100),
            "{failure:?}: over flat scores the plugin's pick is the first row: {decision:?}"
        );
        ids.sort_unstable();
        assert_eq!(ids, vec![100, 101, 102, 103], "{failure:?}: no row is shed");
        let stats = plugin.stats();
        assert_eq!(
            (stats.declined, stats.answered),
            (1, 3),
            "{failure:?}: the score declined; the constructor, the filter and the pick answered"
        );
    }
}

#[test]
fn the_breaker_opens_on_the_failure_threshold_and_bypasses_the_guest() {
    let runtime = runtime();
    let mut spec = spec("breaking", guest::record_guest(), mode::TRAP_IN_FILTER);
    spec.breaker = BreakerSpec {
        failures: 2,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = loaded(&runtime, spec);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    decide(&engine, &candidates, 1);
    decide(&engine, &candidates, 2);
    let stats = plugin.stats();
    assert_eq!(
        (stats.traps, stats.breaker_opens, stats.bypassed),
        (2, 1, 0)
    );
    assert!(matches!(decide(&engine, &candidates, 3), Decision::Pick(_)));
    let stats = plugin.stats();
    assert_eq!(
        stats.bypassed, 1,
        "the third decision never reached the guest"
    );
    assert_eq!(
        stats.instantiations, 2,
        "no instance made for a bypassed decision"
    );
    assert_eq!(stats.traps, 2);
}

/// What the host says: a subscriber that keeps every warning of this crate
/// as its level and its fields. It also takes a guest's own log lines (the
/// plugin host writes them at their level), keeps none of them, and holds
/// the calling thread in one where a case asked ([`hold`]), or panics in
/// one ([`panic_in_a_line_of`]).
struct Said(Arc<std::sync::Mutex<Vec<String>>>);

/// The holds the cases asked: a plugin's name, how long, how many lines.
static HOLDS: std::sync::Mutex<Vec<(String, std::time::Duration, u32)>> =
    std::sync::Mutex::new(Vec::new());

/// Holds the thread that writes `plugin`'s next `lines` log lines, `nap`
/// each: the guest's log line is a host call, so the thread is off the CPU
/// inside a guest call for as long as the case says, whatever the load. The
/// cases that hold run where the host reads a thread's clock: elsewhere
/// nothing is watched, and a held call ends at its deadline on the wall
/// clock.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn hold(plugin: &str, nap: std::time::Duration, lines: u32) {
    said();
    HOLDS
        .lock()
        .expect("the holds")
        .push((format!("plugin=\"{plugin}\""), nap, lines));
}

/// The panics the cases asked: a plugin's name, as the host's line says it.
static PANICS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Panics inside `plugin`'s next log line, once. The guest's log line is a
/// host call, so the panic unwinds through a guest call and leaves its
/// instance where it stood: a clock that panics behind a failed call finds
/// the instance dropped.
fn panic_in_a_line_of(plugin: &str) {
    said();
    PANICS
        .lock()
        .expect("the panics")
        .push(format!("plugin=\"{plugin}\""));
}

impl tracing::Subscriber for Said {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        (*metadata.level() <= tracing::Level::WARN
            && metadata.target().starts_with("smg_policy_controller"))
            || (*metadata.level() == tracing::Level::INFO
                && metadata.target() == "smg_policy_controller::plugins::wasm::bindings")
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                let _ = write!(self.0, " {}={value:?}", field.name());
            }
        }
        let mut fields = Fields(event.metadata().level().to_string());
        event.record(&mut fields);
        if *event.metadata().level() <= tracing::Level::WARN {
            self.0.lock().expect("the kept lines").push(fields.0);
            return;
        }
        // A guest's own line: kept nowhere. It panics where a case asked,
        // the list's lock given up first, and is held where a case asked.
        let panics = {
            let mut asked = PANICS.lock().expect("the panics");
            asked
                .iter()
                .position(|named| fields.0.contains(named.as_str()))
                .map(|at| asked.swap_remove(at))
        };
        if panics.is_some() {
            // A panic with a payload of the case's own; the hook stays quiet.
            std::panic::resume_unwind(Box::new("the host panics inside a guest's log line"));
        }
        let nap = HOLDS
            .lock()
            .expect("the holds")
            .iter_mut()
            .find(|(named, _, lines)| *lines > 0 && fields.0.contains(named.as_str()))
            .map(|(_, nap, lines)| {
                *lines -= 1;
                *nap
            });
        if let Some(nap) = nap {
            std::thread::sleep(nap);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The warnings this test process logs, for the cases that read a line of
/// the host. The subscriber is the process's, installed on the first call:
/// a thread's own would miss a line whose callsite another case's thread
/// met first, the callsite cache being the process's. Every warning from
/// then on lands in one buffer, and a case reads the lines that name its
/// plugin.
fn said() -> Arc<std::sync::Mutex<Vec<String>>> {
    static SAID: std::sync::OnceLock<Arc<std::sync::Mutex<Vec<String>>>> =
        std::sync::OnceLock::new();
    Arc::clone(SAID.get_or_init(|| {
        let lines = Arc::new(std::sync::Mutex::new(Vec::new()));
        // Set once per process; nothing else in this binary sets one.
        tracing::subscriber::set_global_default(Said(Arc::clone(&lines)))
            .expect("no other subscriber is the process's");
        lines
    }))
}

/// The lines of an opening that name `plugin`.
fn openings(plugin: &str) -> Vec<String> {
    let named = format!("plugin=\"{plugin}\"");
    said()
        .lock()
        .expect("the kept lines")
        .iter()
        .filter(|line| line.contains("breaker opened") && line.contains(&named))
        .cloned()
        .collect()
}

/// An opening is otherwise silent (the stages take their fallback): under
/// the failure rule it writes one line with the plugin, the rule and the
/// cool-down, and the decisions it bypasses write none.
#[test]
fn an_opening_under_the_failure_rule_writes_one_line() {
    said();
    let runtime = runtime();
    let mut spec = spec("trapper", guest::record_guest(), mode::TRAP_IN_FILTER);
    spec.breaker = BreakerSpec {
        failures: 2,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = loaded(&runtime, spec);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    decide(&engine, &candidates, 1);
    assert!(
        openings("trapper").is_empty(),
        "one failure under a threshold of two opens nothing"
    );
    decide(&engine, &candidates, 2);
    decide(&engine, &candidates, 3);
    assert_eq!(plugin.stats().bypassed, 1);
    let lines = openings("trapper");
    assert_eq!(lines.len(), 1, "one opening, one line: {lines:?}");
    assert!(
        lines[0].contains("rule=\"failures\"") && lines[0].contains("cooldown_micros=60000000"),
        "{}",
        lines[0]
    );
}

#[test]
fn a_gone_budget_is_never_forwarded_and_counts_on_the_breaker() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "budgetless",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    let pool_facts = PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    };
    let decide_gone = |seed: u64| {
        let gone = RequestFacts {
            budget_micros: 0,
            ..facts(seed)
        };
        engine.decide(
            &gone,
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    assert!(
        matches!(decide_gone(1), Decision::Pick(_)),
        "the fallbacks decide"
    );
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.declined), (0, 0));
    assert_eq!(
        stats.budget_gone, 1,
        "one budget hit per decision: the rest is bypassed"
    );
    assert!(plugin.is_warm(), "no churn for a request past its budget");
    plugin.forget_instance();
    decide_gone(2);
    let stats = plugin.stats();
    assert!(
        !plugin.is_warm() && stats.instantiations == 1,
        "a gone budget instantiates nothing: a slot the decision never touched stays cold for the next open: {stats:?}"
    );
    // Three more reach the default threshold of five: the breaker opens.
    for seed in 3..6 {
        decide_gone(seed);
    }
    let stats = plugin.stats();
    assert_eq!((stats.budget_gone, stats.breaker_opens), (5, 1));
    assert!(matches!(decide(&engine, &candidates, 7), Decision::Pick(_)));
    assert_eq!(
        plugin.stats().bypassed,
        1,
        "a budgeted decision inside the cool-down is bypassed"
    );
}

#[test]
fn the_packed_form_skips_a_request_it_cannot_carry() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "packed",
        guest::packed_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    let long = "m".repeat(70_000);
    let pool_facts = PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    };
    let long_model = RequestFacts {
        model: &long,
        ..facts(1)
    };
    let decision = engine.decide(
        &long_model,
        &pool_facts,
        &candidates,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let stats = plugin.stats();
    assert_eq!(stats.too_long, 1);
    assert_eq!(stats.answered, 0);
    assert!(plugin.is_warm());
}

#[test]
fn the_memory_cap_is_the_specs_and_a_growth_past_it_traps() {
    let runtime = runtime();
    let capped = load(&runtime, "capped", guest::record_guest(), mode::GROW_32_MIB);
    decide(&plugin_engine(&capped), &pool(4), 1);
    assert_eq!(
        capped.stats().traps,
        1,
        "16 MiB by default: 32 MiB more traps"
    );

    let mut roomy = spec("roomy", guest::record_guest(), mode::GROW_32_MIB);
    roomy.memory_mib = 64;
    let roomy = loaded(&runtime, roomy);
    decide(&plugin_engine(&roomy), &pool(4), 1);
    let stats = roomy.stats();
    assert_eq!(
        (stats.traps, stats.answered),
        (0, 4),
        "under 64 MiB the growth fits"
    );
}

/// A module's view carries the request's block hashes where its row
/// declares them, in the token form or the text form, and none where it
/// declares neither: a guest of each world counts the hashes its decisions
/// handed it.
#[test]
fn a_view_carries_block_hashes_to_a_row_that_declares_them_alone() {
    let hashes: Vec<u64> = (1..=300).collect();
    let token = Needs::NONE.with_token_hashes();
    let text = Needs::NONE.with_text_hashes();
    let candidates = pool(4);
    for (world, component) in guest::hash_counting_guests() {
        for (needs, handed) in [(Needs::default(), 0), (token, 300), (text, 300)] {
            let runtime = runtime();
            let mut row = spec("counting", component.clone(), mode::LEAST_IN_FLIGHT);
            row.metrics.push("hashes".to_string());
            row.needs = needs;
            let plugin = loaded(&runtime, row);
            let engine = plugin_engine(&plugin);
            let decision = engine.decide(
                &RequestFacts {
                    block_hashes: &hashes,
                    ..facts(1)
                },
                &PoolFacts {
                    pool_size: 4,
                    generation: 1,
                    ..PoolFacts::default()
                },
                &candidates,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            assert!(decision.pick().is_some(), "{world} {needs:?}: {decision:?}");
            // The guest ran: a plugin the host passed by would count nothing too.
            assert_eq!(
                plugin.stats().answered,
                4,
                "the {world} world, {needs:?}: the constructor, the filter, the score and the pick"
            );
            assert_eq!(
                counter_value(&plugin, counter::HASHES),
                handed,
                "the {world} world, {needs:?}"
            );
        }
    }
}

#[test]
fn the_lifecycle_reaches_the_plugin_once_per_profile_on_the_hosts_clock() {
    let runtime =
        Arc::new(Runtime::with_clock(&OpenDecisions::new(), || 424_242).expect("a runtime"));
    let plugin = load(
        &runtime,
        "observer",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = plugin_engine(&plugin);
    let lease = Lease::default();
    engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
    assert_eq!(
        counter_value(&plugin, counter::OBSERVED),
        1,
        "three adapters of one profile, one observe"
    );
    assert_eq!(counter_value(&plugin, counter::OBSERVE_NOW), 424_242);
    engine.on_released(
        &lease,
        smg_policy_controller::tracking::ReleaseReason::ClientGone,
    );
    engine.on_membership(&smg_policy_controller::tracking::Membership::Snapshot(&[
        WorkerId(1),
        WorkerId(2),
    ]));
    engine.on_signal(&smg_policy_controller::tracking::Signal::gateway_veto(
        WorkerId(1),
        500_000,
    ));
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 4);
    assert_eq!(plugin.stats().observed, 4);
}

#[test]
fn a_pair_decision_opens_one_resource_per_side() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "pair",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let prefill = plugin_profile(&plugin, None, None, None);
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    let decision = pair_decide(&engine);
    let Some(pick) = decision.pick() else {
        panic!("a pair pick, not {decision:?}")
    };
    assert_pair_pick(pick);
    let stats = plugin.stats();
    assert_eq!(
        stats.answered, 8,
        "per side: the constructor, one filter (the partner pass reads it), one score, one pick"
    );
    assert_eq!(stats.instantiations, 1);
}

/// One profile bound to both sides: the decode side opens its own resource
/// (the open decision is keyed by phase), so neither side reads the other's.
#[test]
fn one_profile_on_both_pair_sides_opens_a_resource_per_side() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "both-sides",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let shared = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![shared], 0);
    let decision = pair_decide(&engine);
    let Some(pick) = decision.pick() else {
        panic!("a pair pick, not {decision:?}")
    };
    assert_pair_pick(pick);
    let stats = plugin.stats();
    assert_eq!(
        stats.answered, 8,
        "the decode side's four calls are its own resource's, not the prefill's"
    );
    assert_eq!(
        (stats.declined, stats.invalid, stats.instantiations),
        (0, 0, 1)
    );
}

fn pair_engine(profiles: Vec<Profile>, decode_profile: u16) -> Engine {
    Engine::new(
        profiles,
        vec![Binding {
            model: "*".into(),
            phases: vec![],
            plan: Plan::PdPair,
            profile: 0,
            decode_profile: Some(decode_profile),
            shadow: None,
            decode_shadow: None,
            sessions: None,
            overload: None,
        }],
        1,
        Needs::default(),
    )
}

/// Three prefills (100..102) and four decodes (200..203) of one pair class,
/// each pool's last row the least loaded.
fn pair_decide(engine: &Engine) -> Decision {
    pair_decide_under(engine, UNTIMED_BUDGET_MICROS)
}

/// `pair_decide` under a decision budget of the case's own.
fn pair_decide_under(engine: &Engine, budget_micros: u32) -> Decision {
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let pool_facts = PoolFacts {
        pool_size: 7,
        generation: 1,
        ..PoolFacts::default()
    };
    engine.decide_pair(
        &RequestFacts {
            budget_micros,
            ..facts(9)
        },
        &pool_facts,
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
}

fn assert_pair_pick(pick: &Pick) {
    let lead = pick.targets.first().expect("a target");
    assert_eq!(lead.id, WorkerId(102), "the least-loaded prefill");
    assert_eq!(
        lead.pair.map(|p| p.decode_id),
        Some(WorkerId(203)),
        "the least-loaded decode of its class"
    );
}

/// The alternates follow the lead by position, so a second data-parallel
/// rank of the lead's worker (same id, another row) stays in the answer.
#[test]
fn the_alternates_keep_the_lead_workers_other_ranks() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "ranks",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = plugin_engine(&plugin);
    let mut candidates = pool(4);
    candidates[0].id = WorkerId(103);
    candidates[0].dp_rank = 1;
    let decision = decide(&engine, &candidates, 3);
    assert_eq!(pick_ids(&decision), vec![103, 102, 101, 103]);
    let Some(pick) = decision.pick() else {
        panic!("a pick, not {decision:?}")
    };
    let positions: Vec<u32> = pick.targets.iter().map(|t| t.position).collect();
    assert_eq!(
        positions,
        vec![3, 2, 1, 0],
        "the lead, then the rest by score"
    );
}

#[test]
fn warm_creates_the_threads_instance_and_a_cold_thread_counts_its_first_decision() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "warm",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    // Both threads live at once: the slot crate recycles the id of a thread
    // that exited, and a later thread would adopt its instance.
    let gate = Arc::new(std::sync::Barrier::new(3));
    let warmer = {
        let (plugin, gate) = (Arc::clone(&plugin), Arc::clone(&gate));
        std::thread::spawn(move || {
            let before = plugin.is_warm();
            // The verdict leaves through the gate: a warm that panicked
            // here would leave the other two waiting at it.
            let warmed = plugin.warm();
            let after = plugin.is_warm();
            gate.wait();
            warmed.map(|()| (before, after))
        })
    };
    let cold = {
        let (plugin, gate) = (Arc::clone(&plugin), Arc::clone(&gate));
        std::thread::spawn(move || {
            let cold = plugin.is_warm();
            gate.wait();
            cold
        })
    };
    gate.wait();
    assert_eq!(warmer.join().expect("joins").expect("warms"), (false, true));
    assert!(
        !cold.join().expect("joins"),
        "a thread that neither warmed nor decided holds nothing"
    );
    let engine = plugin_engine(&plugin);
    plugin.forget_instance();
    decide(&engine, &pool(4), 1);
    let stats = plugin.stats();
    assert_eq!(stats.instantiations, 3, "the load, the warm, the lazy one");
    assert_eq!(stats.lazy_instantiations, 1);
}

#[test]
fn a_profile_of_built_ins_holds_no_scope_and_a_plugin_profile_does() {
    let native = Profile {
        name: "native".into(),
        filters: vec![Box::new(Healthy)],
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    };
    assert!(!native.scoped());
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "scoped",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    assert!(plugin_profile(&plugin, None, None, None).scoped());
}

/// The exposure gates the lifecycle as it gates the request: a withheld
/// session key reads zero in every lease the plugin observes.
#[test]
fn a_withheld_session_key_is_withheld_from_the_lifecycle_too() {
    let runtime = runtime();
    for (name, component) in [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ] {
        let mut withheld = spec(name, component.clone(), mode::LEAST_IN_FLIGHT);
        withheld.view.session_key = Reveal::None;
        let withheld = loaded(&runtime, withheld);
        let hashed = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        // A lease carries a request's session, which the request's facts
        // alone make.
        let session = RequestFacts {
            model: "m",
            session_key: Identity::hashed(0xBEEF),
            ..RequestFacts::default()
        }
        .session();
        let lease = Lease {
            session,
            ..Lease::default()
        };
        for plugin in [&withheld, &hashed] {
            plugin_engine(plugin).on_dispatched(&lease, &Target::default(), Side::Single, 3);
        }
        assert_eq!(
            counter_value(&withheld, counter::OBSERVE_SESSION),
            0,
            "{name}: withheld"
        );
        assert_eq!(
            counter_value(&hashed, counter::OBSERVE_SESSION),
            session.get(),
            "{name}: the session's value, as in the request"
        );
    }
}

/// The sandbox is one linear memory under the cap and tables of a bounded
/// size: a second memory is refused at instantiation, a table grown past the
/// cap traps.
#[test]
fn a_second_memory_and_a_table_past_the_cap_are_refused() {
    let runtime = runtime();
    let err = Plugin::load(
        &runtime,
        spec(
            "two-memories",
            guest::two_memory_guest(),
            mode::LEAST_IN_FLIGHT,
        ),
    )
    .expect_err("a second memory is refused");
    assert!(matches!(err, LoadError::Trap { .. }), "{err}");

    // The close's rebuild is an `init`, on the wall clock.
    tried(|made| {
        let plugin = load(
            &runtime,
            "table-grower",
            guest::record_guest(),
            mode::GROW_TABLE,
        );
        let engine = plugin_engine(&plugin);
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = decide(&engine, &pool(4), 1);
        let held = off_cpu_since(wall, cpu);
        assert!(matches!(decision, Decision::Pick(_)));
        let stats = plugin.stats();
        assert_eq!(
            (stats.traps, stats.answered),
            (1, 1),
            "the constructor answered, the filter trapped"
        );
        the_close_rebuilt("a table past the cap", &plugin, held, made)
    });

    let err = Plugin::load(
        &runtime,
        spec(
            "four-tables",
            guest::four_table_guest(),
            mode::LEAST_IN_FLIGHT,
        ),
    )
    .expect_err("a fifth table is refused");
    assert!(matches!(err, LoadError::Trap { .. }), "{err}");
}

/// The request's budget bounds the decision, not each call: a call finds the
/// remainder, and a call that runs past it ran out of its frame.
#[test]
fn the_decisions_budget_bounds_the_sum_of_its_calls() {
    const CASE: &str = "the decision's budget and the sum of its calls";
    /// Half the time the constructor's fuel prices at the slow calibration
    /// (over two hundred microseconds, for a few of work): the least fuel of
    /// the decision's calls, so no shorter hold has a call charged its time.
    const A_HOLD: std::time::Duration = std::time::Duration::from_micros(100);
    tried(|made| {
        let runtime = runtime();
        let mut burner = spec("burner", guest::record_guest(), mode::BURN_IN_STAGES);
        // A frame wider than any budget below: each call may take all of
        // what is left of the request's.
        burner.budget_micros = u32::MAX;
        // The budgets below are burns of fuel: at the slow calibration a call
        // held off its core is still charged its fuel.
        burner.fuel_per_micro = SLOW_CALIBRATION;
        let fuel_per_micro = burner.fuel_per_micro;
        let plugin = loaded(&runtime, burner);
        let engine = plugin_engine(&plugin);
        let candidates = pool(4);
        let pool_facts = PoolFacts {
            pool_size: 4,
            generation: 1,
            ..PoolFacts::default()
        };
        let decide_under = |budget_micros: u32, seed: u64| {
            let facts = RequestFacts {
                budget_micros,
                ..facts(seed)
            };
            engine.decide(
                &facts,
                &pool_facts,
                &candidates,
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        let before = plugin.stats().fuel_used;
        assert!(matches!(
            decide_under(SLOW_FRAME_MICROS, 1),
            Decision::Pick(_)
        ));
        let one = plugin.stats();
        let whole = one.fuel_used - before;
        assert_eq!(
            (one.answered, one.budget_gone, one.fuel_exhausted),
            (4, 0, 0)
        );
        // Two and a half calls' worth: the third stage call finds the remainder
        // short of its work and stops there.
        let per_call = whole / 3;
        let budget_micros =
            u32::try_from((per_call * 5 / 2).div_ceil(fuel_per_micro)).expect("fits");
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = decide_under(budget_micros, 2);
        let held = off_cpu_since(wall, cpu);
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
        let two = plugin.stats();
        // The run-out is the plugin's own only where no call was charged past
        // its fuel, and a call is charged its time once that passes its fuel at
        // the calibration: one hold inside the constructor makes the picker's
        // run-out the decision's budget spent. A decision whose thread was off
        // the CPU that long says so and is made again.
        let ran_out = (
            two.answered - one.answered,
            two.budget_gone,
            two.fuel_exhausted,
            two.traps,
        );
        if ran_out != (3, 0, 1, 0) && held >= A_HOLD && made < TRIES {
            made_again(
                CASE,
                &format!("the second decision's thread was off the CPU {held:?}: a call was charged its time, past its fuel"),
            );
            return false;
        }
        assert_eq!(
            ran_out,
            (3, 0, 1, 0),
            "the constructor, the filter and the scorer answered, and the picker's grant was the remainder and ran out: the calls together exceeded the budget (try {made}, the thread off the CPU {held:?}): {two:?}"
        );
        // The run-out emptied the slot and the close rebuilt it: one `init`, the
        // load's cost, burnt outside the decision's budget.
        let init = before;
        assert!(
            two.fuel_used - one.fuel_used <= u64::from(budget_micros) * fuel_per_micro + init,
            "no decision burns past its budget: {two:?}"
        );
        the_close_rebuilt(CASE, &plugin, held, made)
    });
}

/// A declined filter declines once per open decision: the pair plan's
/// partner scan asks row by row and reads the cached decline.
#[test]
fn a_declined_filter_declines_once_per_side_of_a_pair() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "decliner",
        guest::record_guest(),
        mode::DECLINE_FILTER,
    );
    let shared = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![shared], 0);
    let decision = pair_decide(&engine);
    let Some(pick) = decision.pick() else {
        panic!("a pair pick, not {decision:?}")
    };
    assert_pair_pick(pick);
    let stats = plugin.stats();
    assert_eq!(
        stats.declined, 2,
        "one decline per side, however many rows the scan asks"
    );
    assert_eq!(
        stats.answered, 6,
        "per side: the constructor, the scorer and the picker"
    );
}

/// Each open decision of a pair draws its own sequence from its own seed:
/// the prefill side's stages never read the decode side's draws.
#[test]
fn each_open_decision_of_a_pair_draws_its_own_sequence() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "drawer",
        guest::record_guest(),
        mode::DRAW_IN_PICK,
    );
    let shared = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![shared], 0);
    let decision = pair_decide(&engine);
    let Some(pick) = decision.pick() else {
        panic!("a pair pick, not {decision:?}")
    };
    assert_pair_pick(pick);
    // The prefill side draws from the request's seed (9), the decode side
    // from the first draw of it (the engine's derivation); the constructor
    // takes the first value of each sequence, the pick the second.
    let mut prefill = Draw::new(9);
    let (p1, p2) = (prefill.next_u64(), prefill.next_u64());
    let mut decode = Draw::new(p1);
    let (d1, d2) = (decode.next_u64(), decode.next_u64());
    assert_eq!(
        counter_value(&plugin, counter::RANDOM),
        p1.saturating_add(d1)
    );
    assert_eq!(
        counter_value(&plugin, counter::PICK_RANDOM),
        p2.saturating_add(d2),
        "each pick drew the second value of its own side's sequence"
    );
}

/// A plugin bound to two profiles of one build observes each lifecycle
/// event once: the engine broadcasts to every profile, and the bound sets
/// the build took from its one binder share one observer, the first adapter
/// made.
#[test]
fn a_plugin_bound_to_two_profiles_observes_each_event_once() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "twice-bound",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let binder = plugin.binder();
    let prefill = bound_profile(binder.bind(), None, None, None);
    let decode = bound_profile(binder.bind(), None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    assert_eq!(plugin.stats().observed, 1);
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 1);
}

/// A loaded plugin outlives the engine that bound it first. The engine
/// built to replace it, while the first still stands, forwards the
/// lifecycle for itself from its first event: each build has its own
/// observer, so an event reaches the plugin once per engine that hears it,
/// while the first engine stands and once it is gone.
#[test]
fn a_plugin_bound_into_a_later_engine_observes_through_it() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "rebound",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let dispatch = |engine: &Engine| {
        engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        plugin.stats().observed
    };
    let first = plugin_engine(&plugin);
    assert_eq!(dispatch(&first), 1);
    let second = plugin_engine(&plugin);
    assert_eq!(
        dispatch(&second),
        2,
        "the later engine forwards beside the first"
    );
    assert_eq!(dispatch(&first), 3, "and the first still does");
    drop(first);
    assert_eq!(dispatch(&second), 4, "and once the first is gone");
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 4);
}

/// A row lists the kinds of event its module hears, none by default. A
/// module whose row lists none is never told: no call is made, and a thread
/// that only tells makes no instance. One whose row lists some is told
/// those kinds alone. A loaded module says the kinds its row lists.
#[test]
fn a_module_hears_the_kinds_of_event_its_row_lists_and_no_other() {
    let tell = |engine: &Engine| {
        let lease = Lease::default();
        engine.on_dispatched(&lease, &Target::default(), Side::Single, 3);
        engine.on_released(
            &lease,
            smg_policy_controller::tracking::ReleaseReason::ClientGone,
        );
        engine.on_membership(&smg_policy_controller::tracking::Membership::Snapshot(&[
            WorkerId(1),
            WorkerId(2),
        ]));
        engine.on_signal(&smg_policy_controller::tracking::Signal::gateway_veto(
            WorkerId(1),
            500_000,
        ));
    };
    let runtime = runtime();
    // The default row lists nothing.
    let mut unlisted = spec("unlisted", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    unlisted.observes = PluginSpec::new("default", Vec::new()).observes;
    assert_eq!(unlisted.observes, EventSet::NONE);
    let plugin = loaded(&runtime, unlisted);
    assert_eq!(plugin.observes(), EventSet::NONE);
    let engine = plugin_engine(&plugin);
    assert!(!engine.needs_lifecycle());
    // Told on a thread that never decided, then on the loading thread.
    std::thread::scope(|scope| {
        scope
            .spawn(|| tell(&engine))
            .join()
            .expect("the telling thread");
    });
    tell(&engine);
    let stats = plugin.stats();
    assert_eq!(
        (stats.observed, stats.lazy_instantiations),
        (0, 0),
        "nothing told, and no instance made by a thread that only tells: {stats:?}"
    );
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 0);

    let mut listed = spec("listed", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    listed.observes = EventSet::of([EventKind::Released, EventKind::Signal]);
    let plugin = loaded(&runtime, listed);
    assert_eq!(
        plugin.observes(),
        EventSet::of([EventKind::Released, EventKind::Signal])
    );
    let engine = plugin_engine(&plugin);
    assert!(engine.needs_lifecycle());
    tell(&engine);
    assert_eq!(plugin.stats().observed, 2);
    assert_eq!(counter_value(&plugin, counter::OBSERVED), 2);
}

/// A loop of whole-memory fills under a calibration that underprices it:
/// fuel prices a fill by its size, and at this calibration the frame's fuel
/// lasts seconds of them. The frame's bound on the clock ends the call, on
/// the time its thread had the CPU, counted as the frame running out.
#[test]
fn a_bulk_memory_loop_meets_the_frames_bound_on_the_clock() {
    tried(a_bulk_memory_loop_meets_the_frames_bound_on_the_clock_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_bulk_memory_loop_meets_the_frames_bound_on_the_clock_in_a_try(made: u32) -> bool {
    const CASE: &str = "a bulk-memory loop";
    let runtime = runtime();
    // A calibration no host has: at it the frame's fuel lasts seconds of
    // fills (a fill is charged by its length), so only the clock can end the
    // loop inside the bound below; with the deadline late or the ticker
    // asleep the case reads seconds and fails.
    let mut filler = spec("filler", guest::record_guest(), mode::FILL_IN_SCORE);
    filler.fuel_per_micro = 20_000_000;
    let plugin = loaded(&runtime, filler);
    let engine = plugin_engine(&plugin);
    let (wall, started) = (std::time::Instant::now(), bound_clock());
    assert!(
        matches!(decide(&engine, &pool(4), 1), Decision::Pick(_)),
        "the fallback scorer decided"
    );
    let had = bound_clock().saturating_sub(started);
    let off_the_cpu = wall.elapsed().saturating_sub(had);
    let stats = plugin.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.traps),
        (1, 0),
        "the frame ran out on the clock, not on fuel: {stats:?}"
    );
    // The bound ended the call, so the call is the plugin's and never
    // unread. A call of the decision the runner itself held is unread by
    // right, so the count is asserted where the decision kept its thread on
    // the CPU.
    if off_the_cpu < std::time::Duration::from_micros(u64::from(FRAME_MICROS) / 4) {
        assert_eq!(
            stats.unread, 0,
            "a call its bound ended is never unread: {stats:?}"
        );
    } else {
        excused(
            "a bulk-memory loop",
            "the decision's thread was off the CPU a quarter of the frame",
        );
    }
    // A fill costs its sixteen million bytes under a frame of two hundred
    // thousand million units, so the loop would run for seconds on fuel
    // alone; the bound ends it within a few ticks of the frame (a trapped
    // call's fuel reads as spent: the store's counter is not synced on an
    // interrupt). The bound counts the time this thread had the CPU, so the
    // case reads the clock the bound counts: on a starved runner the wall
    // clock reads more, by the share of the CPU the thread was given.
    let bound = std::time::Duration::from_micros(u64::from(FRAME_MICROS) * 20);
    assert!(
        had < bound,
        "the bound held: {had:?} on the clock it counts, for a {FRAME_MICROS} us frame"
    );
    the_close_rebuilt(CASE, &plugin, off_the_cpu, made)
}

/// One plugin on both sides of a pair draws on one decision budget: the
/// sides' calls together never exceed the request's.
#[test]
fn a_pair_shares_one_decision_budget() {
    let runtime = runtime();
    let mut burner = spec("pair-burner", guest::record_guest(), mode::BURN_IN_STAGES);
    // A frame wider than any budget below: the request's alone bounds a
    // call.
    burner.budget_micros = u32::MAX;
    // Budgets in burns of fuel, at the slow calibration.
    burner.fuel_per_micro = SLOW_CALIBRATION;
    let fuel_per_micro = burner.fuel_per_micro;
    let plugin = loaded(&runtime, burner);
    // One burning decision on a single profile calibrates a call's cost.
    let single = plugin_engine(&plugin);
    let before = plugin.stats().fuel_used;
    let facts_of = |budget_micros: u32| RequestFacts {
        budget_micros,
        ..facts(1)
    };
    let pool_facts = PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    };
    single.decide(
        &facts_of(SLOW_FRAME_MICROS),
        &pool_facts,
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let per_call = (plugin.stats().fuel_used - before) / 3;
    // The load's one `init`: what the close's rebuild of an emptied slot
    // burns, outside the decision's budget.
    let init = before;
    // Four calls' worth for a pair that would take six: the budget is one.
    let budget_micros = u32::try_from((per_call * 4).div_ceil(fuel_per_micro)).expect("fits");
    let shared = plugin_profile(&plugin, None, None, None);
    let pair = pair_engine(vec![shared], 0);
    let before = plugin.stats().fuel_used;
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let decision = pair.decide_pair(
        &facts_of(budget_micros),
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let stats = plugin.stats();
    assert!(
        stats.fuel_used - before <= u64::from(budget_micros) * fuel_per_micro + init,
        "both sides drew on one budget, the close's rebuild aside: {stats:?}"
    );
    assert!(
        stats.fuel_exhausted + stats.budget_spent >= 1,
        "the later side met the remainder the earlier side spent: {stats:?}"
    );
    assert_eq!(
        stats.budget_gone, 0,
        "nothing was gone on arrival: {stats:?}"
    );
}

/// One plugin bound by a prefill profile and a decode profile of their own:
/// a fault on the decode side drops the one instance both sides' resources
/// live on. The prefill side's close finds its resource's store gone and
/// drops nothing through the rebuilt instance: one trap, one rebuild.
#[test]
fn a_fault_on_one_side_of_a_two_profile_pair_rebuilds_once() {
    tried(a_fault_on_one_side_of_a_two_profile_pair_rebuilds_once_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_fault_on_one_side_of_a_two_profile_pair_rebuilds_once_in_a_try(made: u32) -> bool {
    const CASE: &str = "a fault on one side of a pair";
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "four-trap",
        guest::record_guest(),
        mode::TRAP_ON_FOUR_CANDIDATES,
    );
    let prefill = plugin_profile(&plugin, None, None, None);
    let decode = plugin_profile(&plugin, None, None, None);
    let pair = pair_engine(vec![prefill, decode], 1);
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = pair.decide_pair(
        &facts(1),
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let held = off_cpu_since(wall, cpu);
    let stats = plugin.stats();
    assert_eq!(
        stats.traps, 1,
        "the decode filter's trap and no second one from a stale handle: {stats:?}"
    );
    if !the_close_rebuilt(CASE, &plugin, held, made) {
        return false;
    }
    assert_eq!(
        (stats.instantiations, stats.lazy_instantiations),
        (2, 1),
        "rebuilt once: {stats:?}"
    );
    true
}

/// A budget the decision's own calls spend to the last microsecond drops no
/// healthy instance: the later stage falls back without a call, the
/// destructor runs under its own small frame, nothing is rebuilt.
#[test]
fn a_spent_budget_drops_no_healthy_instance() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "thrifty",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = plugin_engine(&plugin);
    // Three microseconds: every call is charged at least one, so the
    // constructor, the filter and the scorer spend it and the pick finds
    // nothing left.
    let facts = RequestFacts {
        budget_micros: 3,
        ..facts(1)
    };
    let pool_facts = PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    };
    let decision = engine.decide(
        &facts,
        &pool_facts,
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let stats = plugin.stats();
    assert!(
        stats.budget_spent >= 1,
        "a later call met the spent remainder: {stats:?}"
    );
    assert_eq!(
        (stats.fuel_exhausted, stats.traps, stats.instantiations),
        (0, 0, 1),
        "no fault, no rebuild: {stats:?}"
    );
    assert!(
        plugin.is_warm(),
        "the instance stands through the destructor's own frame"
    );
}

/// A long call that answers, then an endless one, end at the budget
/// together, because the second call's bound is what the first one left, not
/// its whole frame. Fuel cannot end either call inside the bound at this
/// calibration; the clock does, counting the time the thread had the CPU.
/// The decision's own deadline is the wall clock's: where the runner holds
/// the thread past the budget inside the first call, the remainder is spent
/// and the second is never made. The case names both worlds, and in both
/// the calls end at the budget on the clock the bound counts. In both the
/// decision's budget is spent and the breaker is told nothing: the long
/// call answered with no reading, so by its fuel the run-out under what it
/// left is the budget's and by the time it took the plugin's, and nothing
/// says which. It is the budget's, counted with the calls nothing read, and
/// it arms the thread; what holds a guest like this one is its decisions'
/// samples. These are the counts of a host that reads a thread's clock: where
/// none is, a call spends its charge, the run-out under what the long call
/// left is the plugin's own, and nothing is counted unread.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn the_decisions_cpu_is_bounded_by_its_budget_whatever_the_fuel() {
    const CASE: &str = "the decision's CPU is bounded by its budget";
    /// The decision's budget, the second the plugin and the request name.
    const BUDGET: std::time::Duration = std::time::Duration::from_secs(1);
    tried(|made| {
        let runtime = runtime();
        let mut spec = spec(
            "fill-then-fill",
            guest::record_guest(),
            mode::FILL_IN_FILTER_THEN_FILL,
        );
        spec.fuel_per_micro = 20_000_000;
        spec.budget_micros = 1_000_000;
        let plugin = loaded(&runtime, spec);
        // A filter of the host's own behind the plugin's: it is first asked
        // when the plugin's filter, the long fill, has ended. A scorer of the
        // host's own behind the plugin's: it is asked when the plugin's score,
        // the endless fill, has ended, and reads this thread's clock there.
        let asked = Arc::new(std::sync::OnceLock::new());
        let scored = Arc::new(std::sync::OnceLock::new());
        let mut profile = plugin_profile(&plugin, None, None, None);
        profile
            .filters
            .push(Box::new(SaysWhenAsked(Arc::clone(&asked))));
        profile.scorers.push(WeightedScorer {
            scorer: Box::new(ReadsTheClockWhenAsked(Arc::clone(&scored))),
            weight: 1.0,
        });
        let engine = single_engine(profile);
        let cpu_before = bound_clock();
        let started = std::time::Instant::now();
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: 1_000_000,
                ..facts(1)
            },
            &PoolFacts {
                pool_size: 4,
                generation: 1,
                ..PoolFacts::default()
            },
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let elapsed = started.elapsed();
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
        let cpu = bound_clock().saturating_sub(cpu_before);
        let off_the_cpu = elapsed.saturating_sub(cpu);
        let stats = plugin.stats();
        assert_eq!(stats.traps, 0, "{stats:?}");
        assert_eq!(
            (
                stats.fuel_exhausted,
                stats.budget_spent,
                stats.breaker_opens
            ),
            (0, 1, 0),
            "the decision's budget was spent, by a run-out or with no call: {stats:?}"
        );
        // The close's rebuild is an `init`, on the wall clock.
        if !the_close_rebuilt(CASE, &plugin, off_the_cpu, made) {
            return false;
        }
        // The long fill answers inside the decision's second, and the endless
        // one is called and runs out on the clock, mid-call. A runner can slow
        // the long fill past the whole second, with its thread on the CPU or
        // off it, and the endless fill then gets no call. The host's own
        // filter says so: the long fill had ended at or past the budget. The
        // fuel cannot: a call the clock ends reads as no fuel burnt. Such a
        // decision is made again, and so is one whose thread was off the CPU
        // half its budget.
        let ran_out = (stats.unread, stats.stalls, stats.lazy_instantiations);
        let long_fill = asked
            .get()
            .expect("the host's filter is asked behind the plugin's")
            .duration_since(started);
        let uncalled = long_fill >= BUDGET;
        let held = off_the_cpu >= BUDGET / 2;
        if ran_out != (1, 0, 1) && (uncalled || held) && made < TRIES {
            made_again(
                CASE,
                &format!(
                    "the long fill ended {long_fill:?} into the decision, past its budget ({uncalled}), or the thread was off the CPU half the budget ({held}): {elapsed:?} on the wall, {cpu:?} on the thread's clock"
                ),
            );
            return false;
        }
        assert_eq!(
            ran_out,
            (1, 0, 1),
            "the run-out was counted with the calls nothing read, and its instance went (try {made}, {elapsed:?} on the wall, {cpu:?} on the thread's clock): {stats:?}"
        );
        // Eight hundred fills take two to four hundred milliseconds, by the
        // host, and answer; the endless fill then gets what the second has
        // left, plus a tick and the fill in flight, so the two calls end within
        // milliseconds of the budget. A whole-frame bound would run the endless
        // fill the frame's whole second on top: a fifth of a second past the
        // budget at the least. The line is the budget and a tenth. The bound
        // counts the time this thread had the CPU, so the case reads the clock
        // the bound counts: on a starved runner the wall clock reads more, by
        // the share of the CPU the thread was given. It reads that clock where
        // the endless fill ended, in the host's own scorer, and not at the
        // decision's end: the close behind the calls drops the instance and
        // rebuilds the slot, memory work no budget of the decision bounds, and
        // a loaded runner slows it on this thread's own clock.
        let calls = scored
            .get()
            .expect("the host's scorer is asked behind the plugin's")
            .saturating_sub(cpu_before);
        assert!(
            calls < std::time::Duration::from_millis(1_100),
            "the two calls ended at the budget on the clock its bound counts: {calls:?}; the decision whole {cpu:?}, {elapsed:?} on the wall"
        );
        true
    });
}

/// A filter of the host's own that keeps every candidate and says when it
/// was first asked: behind a plugin's filter, when that filter's call ended.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug)]
struct SaysWhenAsked(Arc<std::sync::OnceLock<std::time::Instant>>);

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl smg_policy_controller::engine::Filter for SaysWhenAsked {
    fn name(&self) -> &'static str {
        "says-when-asked"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        self.0.get_or_init(std::time::Instant::now);
        true
    }
}

/// A scorer of the host's own that scores every candidate alike and reads
/// its thread's clock when it is first asked: behind a plugin's scorer, what
/// the thread had when that scorer's call ended.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug)]
struct ReadsTheClockWhenAsked(Arc<std::sync::OnceLock<std::time::Duration>>);

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Lifecycle for ReadsTheClockWhenAsked {}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl smg_policy_controller::engine::Scorer for ReadsTheClockWhenAsked {
    fn name(&self) -> &'static str {
        "reads-the-clock-when-asked"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> smg_policy_controller::engine::TieBand {
        self.0.get_or_init(bound_clock);
        out.fill(0.0);
        smg_policy_controller::engine::TieBand::NORMALIZED
    }
}

/// A plugin that alone spends its decisions' budget is charged for it on an
/// armed thread. A lone plugin whose score spins, under a frame of the whole
/// budget at the default calibration, where a call takes longer than its
/// fuel (the host's marshalling is time and no fuel). On a quiet thread its
/// first run-out is the decision's budget spent, counted with the calls
/// nothing read, and it arms the thread. From then on a reading gives each
/// earlier call its one time, the plugin's own spend and the grant reach
/// the budget, and the run-out is the plugin's own: the failure rule opens
/// the breaker at the fifth, long before the ring's first read at the
/// sixty-fourth decision. A reading can run a microsecond short of its
/// call, and that run-out is the budget's: the case asserts the opening and
/// its floor, never every decision. Both are asserted where no hold moved
/// the run: a call the host held is short by its reading, and its
/// decision's run-out is the budget's.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_lone_spinner_on_an_armed_thread_is_charged_and_the_failure_rule_opens_it() {
    let runtime = runtime();
    let spinner = load(
        &runtime,
        "lone-spinner",
        guest::record_guest(),
        mode::SPIN_IN_SCORE,
    );
    let engine = plugin_engine(&spinner);
    let candidates = pool(4);
    let decide = |seed: u64| {
        engine.decide(
            &timed(seed),
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    let (decisions, on_the_cpu) = decide_until_open(&spinner, decide);
    let s = spinner.stats();
    assert_eq!(
        (s.traps, s.breaker_opens),
        (0, 1),
        "the spinner's breaker opened once, on no trap: {s:?}"
    );
    assert!(
        s.unread >= 1,
        "the first run-out of a quiet thread was counted with the calls nothing read: {s:?}"
    );
    if no_hold("a lone spinner", on_the_cpu, &[&spinner]) {
        assert!(
            s.fuel_exhausted >= 5,
            "on the armed thread the run-outs were the plugin's own: {s:?}"
        );
        assert!(
            decisions < 64,
            "the failure rule opened it before the ring's first read: {decisions} decisions, {s:?}"
        );
    }
}

/// Two plugins in one profile: the scorer spends most of the budget, the
/// picker runs out under what is left. The decision's budget is spent, the
/// picker's instance goes, and the picker's breaker is untouched. A call is
/// late by the wall clock: a try whose thread was held off the CPU reads a
/// late call that nothing read, and is made again.
#[test]
fn an_exhaustion_under_another_plugins_clamp_is_the_decisions_budget_spent() {
    tried(an_exhaustion_under_another_plugins_clamp_is_the_decisions_budget_spent_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn an_exhaustion_under_another_plugins_clamp_is_the_decisions_budget_spent_in_a_try(
    made: u32,
) -> bool {
    const CASE: &str = "an exhaustion under another plugin's clamp";
    let runtime = runtime();
    let scorer = load_slow(
        &runtime,
        "heavy-scorer",
        guest::record_guest(),
        mode::BURN_IN_STAGES,
    );
    let mut picker_spec = spec("picker", guest::record_guest(), mode::BURN_IN_STAGES);
    picker_spec.fuel_per_micro = SLOW_CALIBRATION;
    picker_spec.budget_micros = SLOW_FRAME_MICROS;
    let fuel_per_micro = picker_spec.fuel_per_micro;
    let picker = loaded(&runtime, picker_spec);
    // One burn's fuel, from the scorer alone at a large budget.
    let before = scorer.stats().fuel_used;
    plugin_engine(&scorer).decide(
        &RequestFacts {
            budget_micros: 1_000_000,
            ..facts(1)
        },
        &PoolFacts {
            pool_size: 4,
            generation: 1,
            ..PoolFacts::default()
        },
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let burn = (scorer.stats().fuel_used - before) / 3;
    let profile = Profile {
        name: "two".into(),
        filters: vec![],
        scorers: vec![WeightedScorer {
            scorer: Box::new(WasmScorer::new(scorer.bind(), None)),
            weight: 1.0,
        }],
        picker: Box::new(WasmPicker::new(picker.bind(), None)),
    };
    let engine = single_engine(profile);
    let gauged = scorer.stats().answered;
    // A burn and a half, plus a little for the constructors: the scorer's
    // burn fits, the picker's does not.
    let budget_micros = u32::try_from((burn * 3 / 2).div_ceil(fuel_per_micro) + 20).expect("fits");
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = engine.decide(
        &RequestFacts {
            budget_micros,
            ..facts(1)
        },
        &PoolFacts {
            pool_size: 4,
            generation: 1,
            ..PoolFacts::default()
        },
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let held = off_cpu_since(wall, cpu);
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallback picker decides"
    );
    let (s, p) = (scorer.stats(), picker.stats());
    let stood = (s.fuel_exhausted, s.traps, s.answered - gauged) == (0, 0, 2)
        && (p.fuel_exhausted, p.traps, p.budget_spent) == (0, 0, 1)
        && (p.unread, p.stalls) == (0, 0)
        && (p.instantiations, p.lazy_instantiations) == (2, 1)
        && p.breaker_opens == 0;
    if held_counts_are_made_again(CASE, stood, held, made) {
        return false;
    }
    assert_eq!(
        (s.fuel_exhausted, s.traps, s.answered - gauged),
        (0, 0, 2),
        "the scorer's constructor and score answered: {s:?}"
    );
    assert_eq!(
        (p.fuel_exhausted, p.traps, p.budget_spent),
        (0, 0, 1),
        "the picker ran out under the scorer's clamp: {p:?}"
    );
    assert_eq!(
        (p.unread, p.stalls),
        (0, 0),
        "by its fuel and by its time the run-out is the budget's: no reading was owed: {p:?}"
    );
    assert_eq!(
        (p.instantiations, p.lazy_instantiations),
        (2, 1),
        "the picker's interrupted instance went and the close rebuilt it: {p:?}"
    );
    assert_eq!(p.breaker_opens, 0);
    true
}

/// The decision's first open seeds its budget and every later open draws on
/// it: one plugin on both sides of a pair, a constructor that burns, a budget
/// of a burn and a half. The second constructor gets what the first left and
/// runs out; a budget re-seeded per open would let it through.
#[test]
fn the_first_open_seeds_the_budget_and_later_opens_draw_on_it() {
    tried(the_first_open_seeds_the_budget_and_later_opens_draw_on_it_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn the_first_open_seeds_the_budget_and_later_opens_draw_on_it_in_a_try(made: u32) -> bool {
    const CASE: &str = "the first open's seed";
    let runtime = runtime();
    let mut spec = spec(
        "ctor-burner",
        guest::record_guest(),
        mode::BURN_IN_CONSTRUCTOR,
    );
    // A frame wider than any budget below: the request's alone bounds a
    // call.
    spec.budget_micros = u32::MAX;
    // Budgets in burns of fuel, at the slow calibration.
    spec.fuel_per_micro = SLOW_CALIBRATION;
    let fuel_per_micro = spec.fuel_per_micro;
    let plugin = loaded(&runtime, spec);
    // One decision's fuel on a single profile: the burning constructor and
    // three cheap stages.
    let before = plugin.stats().fuel_used;
    plugin_engine(&plugin).decide(
        &RequestFacts {
            budget_micros: 1_000_000,
            ..facts(1)
        },
        &PoolFacts {
            pool_size: 4,
            generation: 1,
            ..PoolFacts::default()
        },
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let one = plugin.stats().fuel_used - before;
    let budget_micros = u32::try_from((one * 3 / 2).div_ceil(fuel_per_micro)).expect("fits");
    let shared = plugin_profile(&plugin, None, None, None);
    let pair = pair_engine(vec![shared], 0);
    let mut prefills = pool(3);
    let mut decodes = pool(4);
    for c in prefills.iter_mut().chain(decodes.iter_mut()) {
        c.pair_class = 7;
    }
    for (i, c) in decodes.iter_mut().enumerate() {
        c.id = WorkerId(200 + i as u64);
    }
    let before = plugin.stats();
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = pair.decide_pair(
        &RequestFacts {
            budget_micros,
            ..facts(1)
        },
        &PoolFacts {
            pool_size: 7,
            generation: 1,
            ..PoolFacts::default()
        },
        PairPools {
            prefills: &prefills,
            decodes: &decodes,
            ..PairPools::default()
        },
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let held = off_cpu_since(wall, cpu);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted - before.fuel_exhausted,
            stats.answered - before.answered
        ),
        (1, 1),
        "the first constructor answered and the second ran out on what it left: {stats:?}"
    );
    the_close_rebuilt(CASE, &plugin, held, made)
}

/// Two runtimes built over the same open decisions: a decision that opens a
/// plugin of each is one decision to both, with one remainder. The scorer,
/// of the first runtime, spends most of the budget; the picker, of the
/// second, runs out under what is left, and the run-out is the decision's
/// budget spent. Built apart, each over open decisions of its own, the
/// picker's runtime seeds a remainder of its own and the same picker
/// answers: whoever hosts the engine builds its runtimes over one.
#[test]
fn runtimes_built_over_the_same_open_decisions_draw_one_remainder_and_apart_draw_two() {
    let under = |shared: bool| {
        let open = OpenDecisions::new();
        let other = if shared {
            Arc::clone(&open)
        } else {
            OpenDecisions::new()
        };
        let first = Arc::new(Runtime::new(&open).expect("a runtime"));
        let second = Arc::new(Runtime::new(&other).expect("a runtime"));
        let scorer = load_slow(
            &first,
            "heavy-scorer",
            guest::record_guest(),
            mode::BURN_IN_STAGES,
        );
        let picker = load_slow(
            &second,
            "picker",
            guest::record_guest(),
            mode::BURN_IN_STAGES,
        );
        // One burn's fuel, from the scorer alone at a large budget.
        let before = scorer.stats().fuel_used;
        plugin_engine(&scorer).decide(
            &RequestFacts {
                budget_micros: 1_000_000,
                ..facts(1)
            },
            &PoolFacts {
                pool_size: 4,
                generation: 1,
                ..PoolFacts::default()
            },
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let burn = (scorer.stats().fuel_used - before) / 3;
        let engine = single_engine(Profile {
            name: "two".into(),
            filters: vec![],
            scorers: vec![WeightedScorer {
                scorer: Box::new(WasmScorer::new(scorer.bind(), None)),
                weight: 1.0,
            }],
            picker: Box::new(WasmPicker::new(picker.bind(), None)),
        });
        let gauged = scorer.stats().answered;
        // A burn and a half, plus a little for the constructors: the
        // scorer's burn fits, and the picker's fits only in a remainder the
        // scorer did not draw.
        let budget_micros =
            u32::try_from((burn * 3 / 2).div_ceil(SLOW_CALIBRATION) + 20).expect("fits");
        let decision = engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(1)
            },
            &PoolFacts {
                pool_size: 4,
                generation: 1,
                ..PoolFacts::default()
            },
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the picker or its fallback decides"
        );
        let s = scorer.stats();
        assert_eq!(
            (s.fuel_exhausted, s.traps, s.answered - gauged),
            (0, 0, 2),
            "the scorer's constructor and score answered: {s:?}"
        );
        picker.stats()
    };
    let p = under(true);
    assert_eq!(
        (p.answered, p.budget_spent, p.fuel_exhausted, p.traps),
        (1, 1, 0, 0),
        "built over the same: the picker's constructor answered and its pick ran out under what the scorer left: {p:?}"
    );
    let p = under(false);
    assert_eq!(
        (p.answered, p.budget_spent, p.fuel_exhausted, p.traps),
        (2, 0, 0, 0),
        "built apart: the picker drew a remainder of its own: {p:?}"
    );
}

/// One loaded plugin in two engine builds, as a spec applied again leaves
/// it: a thread decides under the old build and the plugin fails that
/// decision, then the thread's next decision is the new build's first. It is
/// another decision to the plugin: its stages are asked, not bypassed for
/// the old build's failure. A plugin keys its part in a decision by the
/// decision's number alone, so the numbers come from the open decisions its
/// runtime was built over, as wide as the plugin's sharing: counted per
/// build, the new build's first decision on the thread would carry the
/// number of the old build's.
#[test]
fn a_new_builds_first_decision_on_a_thread_is_not_the_old_builds_last() {
    const CASE: &str = "a new build's first decision on a thread";
    tried(|made| {
        let runtime = runtime();
        let plugin = load(
            &runtime,
            "four-trapper",
            guest::record_guest(),
            mode::TRAP_ON_FOUR_CANDIDATES,
        );
        let old = plugin_engine(&plugin);
        let new = plugin_engine(&plugin);
        // A thread of its own: its one decision under the old build, a lazy
        // `init` among what that decision paid, then its first under the new.
        let (failed, asked, took, held) = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                    decide(&old, &pool(4), 1);
                    let (took, held) = (wall.elapsed(), off_cpu_since(wall, cpu));
                    let failed = plugin.stats();
                    decide(&new, &pool(3), 2);
                    (failed, plugin.stats(), took, held)
                })
                .join()
                .expect("the deciding thread")
        });
        // Both instances the old build's decision makes come of an `init`, on
        // the wall clock: the cold slot's, and the close's rebuild. An `init`
        // the host ended took its frame on that clock, and the decision with
        // it. The thread's own clock can count that time the thread's
        // ([`loaded`] says where), so the witness is read on the clock of the
        // bound.
        let made_both = (failed.traps, failed.lazy_instantiations);
        if made_both != (1, 2) && took >= INIT_FRAME && made < TRIES {
            made_again(
                CASE,
                &format!(
                    "the old build's decision took {took:?} on the wall clock, the frame of an init or more, its thread off the CPU {held:?} of it"
                ),
            );
            return false;
        }
        assert_eq!(
            made_both,
            (1, 2),
            "the old build's decision made its instance, trapped in the filter and was rebuilt (try {made}, {took:?} on the wall clock, the thread off the CPU {held:?}): {failed:?}"
        );
        assert_eq!(
            (asked.traps, asked.answered - failed.answered),
            (1, 4),
            "the new build's first decision asked the constructor and the three stages: {asked:?}"
        );
        true
    });
}

/// A stage of the host's own whose scope panics at its open, once.
#[derive(Debug)]
struct PanicsAtItsOpen(AtomicBool);

impl smg_policy_controller::engine::Filter for PanicsAtItsOpen {
    fn name(&self) -> &'static str {
        "panics-at-its-open"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for PanicsAtItsOpen {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        assert!(
            !self.0.swap(false, Ordering::Relaxed),
            "this scope's open panics, once"
        );
    }

    fn close(&self) {}
}

/// The engine closes a decision on every path, the unwinding one included:
/// a scope that panics at its open leaves the scopes opened before it
/// closed. A host that catches the panic keeps its thread, and the thread's
/// next decision is a decision of its own, numbered and seeded anew. Left
/// open, the first decision would hand the next its number, and the plugin
/// that failed it would read every later decision as failed.
#[test]
fn a_scope_that_panics_at_its_open_leaves_the_threads_next_decision_its_own() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "beside-a-panic",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    // The plugin's scope opens first, then the one that panics.
    let mut profile = plugin_profile(&plugin, None, None, None);
    profile
        .filters
        .push(Box::new(PanicsAtItsOpen(AtomicBool::new(true))));
    let engine = single_engine(profile);
    let candidates = pool(3);
    let pool_facts = PoolFacts {
        pool_size: 3,
        generation: 1,
        ..PoolFacts::default()
    };
    // A budget gone on arrival: the plugin fails this decision at its open.
    let gone = RequestFacts {
        budget_micros: 0,
        ..facts(1)
    };
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &gone,
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }));
    assert!(caught.is_err(), "the second scope's open panicked");
    let failed = plugin.stats();
    assert_eq!(failed.budget_gone, 1, "{failed:?}");
    let next = decide(&engine, &candidates, 2);
    let asked = plugin.stats();
    assert_eq!(
        (asked.budget_gone, asked.answered - failed.answered),
        (1, 4),
        "the next decision asked the constructor and the three stages: {asked:?}"
    );
    assert_eq!(
        pick_ids(&next).first(),
        Some(&102),
        "the plugin's pick leads: the least in flight"
    );
}

/// What a scope of the host's own saw: its opens and its closes.
#[derive(Debug, Default)]
struct ScopeCounts {
    opens: AtomicU32,
    closes: AtomicU32,
}

impl ScopeCounts {
    fn read(&self) -> (u32, u32) {
        (
            self.opens.load(Ordering::Relaxed),
            self.closes.load(Ordering::Relaxed),
        )
    }
}

/// A stage of the host's own whose scope stores at its open and takes what
/// it stored at its close: a close with nothing stored panics. Told to, its
/// open panics before its store, once. It counts its opens and its closes.
#[derive(Debug)]
struct StoresAtItsOpen {
    counts: Arc<ScopeCounts>,
    stored: AtomicBool,
    panics_once: AtomicBool,
}

impl StoresAtItsOpen {
    fn stage(panics_once: bool) -> (Box<Self>, Arc<ScopeCounts>) {
        let counts = Arc::new(ScopeCounts::default());
        let stage = Box::new(Self {
            counts: Arc::clone(&counts),
            stored: AtomicBool::new(false),
            panics_once: AtomicBool::new(panics_once),
        });
        (stage, counts)
    }
}

impl smg_policy_controller::engine::Filter for StoresAtItsOpen {
    fn name(&self) -> &'static str {
        "stores-at-its-open"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for StoresAtItsOpen {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.counts.opens.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.panics_once.swap(false, Ordering::Relaxed),
            "this scope's open panics before its store, once"
        );
        self.stored.store(true, Ordering::Relaxed);
    }

    fn close(&self) {
        self.counts.closes.fetch_add(1, Ordering::Relaxed);
        assert!(
            self.stored.swap(false, Ordering::Relaxed),
            "a close with nothing stored"
        );
    }
}

/// The engine closes a scope only after its open returned. When a scope
/// panics at its open, the scopes whose open returned are closed, and
/// neither that scope nor any after it: an open that stops half-way undoes
/// itself. So a host's scope whose close takes what its open stored is never
/// closed with nothing stored. Such a close would panic inside the first
/// panic's unwinding and end the process; here the first panic reaches the
/// host's catch.
#[test]
fn a_scope_is_closed_only_after_its_open_returned() {
    let (before, opened_before) = StoresAtItsOpen::stage(false);
    let (panics, panicked) = StoresAtItsOpen::stage(true);
    let (after, never_reached) = StoresAtItsOpen::stage(false);
    let filters: Vec<Box<dyn smg_policy_controller::engine::Filter>> = vec![before, panics, after];
    let engine = single_engine(Profile {
        name: "host-scopes".into(),
        filters,
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    });
    let candidates = pool(3);
    let pool_facts = PoolFacts {
        pool_size: 3,
        generation: 1,
        ..PoolFacts::default()
    };
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &facts(1),
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }));
    assert!(
        caught.is_err(),
        "the second scope's panic reached the catch"
    );
    assert_eq!(opened_before.read(), (1, 1), "its open returned: closed");
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    assert_eq!(never_reached.read(), (0, 0), "never opened: not closed");
    // The next decision opens and closes all three.
    let next = decide(&engine, &candidates, 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(opened_before.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
    assert_eq!(never_reached.read(), (1, 1));
}

/// A stage of the host's own whose scope's close panics, once. It counts its
/// opens and its closes.
#[derive(Debug)]
struct PanicsAtItsClose {
    counts: Arc<ScopeCounts>,
    panics_once: AtomicBool,
}

impl smg_policy_controller::engine::Filter for PanicsAtItsClose {
    fn name(&self) -> &'static str {
        "panics-at-its-close"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for PanicsAtItsClose {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        self.counts.opens.fetch_add(1, Ordering::Relaxed);
    }

    fn close(&self) {
        self.counts.closes.fetch_add(1, Ordering::Relaxed);
        assert!(
            !self.panics_once.swap(false, Ordering::Relaxed),
            "this scope's close panics, once"
        );
    }
}

/// The engine closes the scopes whose open returned inside the unwinding of
/// a later scope's open, and a close that panics there stops at the engine's
/// own catch: the scopes after it are still closed, and the open's panic is
/// the one the host catches. A second panic let out inside that unwinding
/// would end the process.
#[test]
fn a_close_that_panics_inside_an_opens_unwinding_stops_at_the_engine() {
    let close_counts = Arc::new(ScopeCounts::default());
    let close_panics = Box::new(PanicsAtItsClose {
        counts: Arc::clone(&close_counts),
        panics_once: AtomicBool::new(true),
    });
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (panics, panicked) = StoresAtItsOpen::stage(true);
    let filters: Vec<Box<dyn smg_policy_controller::engine::Filter>> =
        vec![close_panics, after_it, panics];
    let engine = single_engine(Profile {
        name: "host-scopes".into(),
        filters,
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    });
    let candidates = pool(3);
    let pool_facts = PoolFacts {
        pool_size: 3,
        generation: 1,
        ..PoolFacts::default()
    };
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &facts(1),
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }));
    let payload = caught.expect_err("the third scope's panic reached the catch");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "closed once, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the scope after it is still closed"
    );
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    // The next decision opens and closes all three.
    let next = decide(&engine, &candidates, 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// A profile of the host's own stages, in their order.
fn host_profile(filters: Vec<Box<dyn smg_policy_controller::engine::Filter>>) -> Profile {
    Profile {
        name: "host-scopes".into(),
        filters,
        scorers: vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    }
}

/// A scope whose close panics, once, with its counts.
fn close_panics_once() -> (Box<PanicsAtItsClose>, Arc<ScopeCounts>) {
    let counts = Arc::new(ScopeCounts::default());
    let stage = Box::new(PanicsAtItsClose {
        counts: Arc::clone(&counts),
        panics_once: AtomicBool::new(true),
    });
    (stage, counts)
}

/// A stage of the host's own whose scope's close panics once with its
/// number as the payload, so a case can tell two such panics apart.
#[derive(Debug)]
struct ClosePanicsWith(u32, AtomicBool);

impl smg_policy_controller::engine::Filter for ClosePanicsWith {
    fn name(&self) -> &'static str {
        "close-panics-with"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for ClosePanicsWith {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {
        if self.1.swap(false, Ordering::Relaxed) {
            // A panic with a payload of the case's own type; the hook stays quiet.
            std::panic::resume_unwind(Box::new(self.0));
        }
    }
}

/// A stage of the host's own, with no scope, whose `keep` panics once.
#[derive(Debug)]
struct PanicsAtItsKeep(AtomicBool);

impl smg_policy_controller::engine::Filter for PanicsAtItsKeep {
    fn name(&self) -> &'static str {
        "panics-at-its-keep"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        assert!(
            !self.0.swap(false, Ordering::Relaxed),
            "this stage's keep panics, once"
        );
        true
    }
}

/// One decision of three candidates, its panic caught.
fn decide_caught(engine: &Engine) -> Box<dyn std::any::Any + Send> {
    let candidates = pool(3);
    let pool_facts = PoolFacts {
        pool_size: 3,
        generation: 1,
        ..PoolFacts::default()
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.decide(
            &facts(1),
            &pool_facts,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        )
    }))
    .expect_err("the decision panicked")
}

/// A close that panics on a decision's own way out keeps no scope after it
/// open: the engine runs every close it owes, and the first close's panic is
/// the one the host sees, once the rest are closed.
#[test]
fn a_close_that_panics_alone_keeps_no_scope_after_it_open() {
    let first = Box::new(ClosePanicsWith(1, AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let second = Box::new(ClosePanicsWith(2, AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![first, after_it, second]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<u32>().copied(),
        Some(1),
        "the first close's panic is the one the host sees"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the scope after the first panic is closed"
    );
    // Both closes that panic did so in that decision: the next one closes
    // clean, so the third scope's close ran too.
    let next = decide(&engine, &pool(3), 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(closed_after_it.read(), (2, 2));
}

/// A close that panics while a stage's panic unwinds stops at the engine:
/// the scopes after it are still closed, and the stage's panic is the one
/// the host sees. Let out inside that unwinding, it would end the process.
#[test]
fn a_close_that_panics_while_a_stages_panic_unwinds_stops_at_the_engine() {
    let (close_panics, close_counts) = close_panics_once();
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let keep_panics = Box::new(PanicsAtItsKeep(AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![close_panics, after_it, keep_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this stage's keep panics, once"),
        "the stage's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "closed, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the scope after it is still closed"
    );
    let next = decide(&engine, &pool(3), 2);
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
}

/// The payload of a pair decision that panics.
fn pair_decide_caught(engine: &Engine) -> Box<dyn std::any::Any + Send> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pair_decide(engine)))
        .expect_err("the pair decision panicked")
}

/// A pair opens its prefill side whole before its decode side. When a decode
/// scope panics at its open, the prefill side's closes run inside that
/// unwinding, by the decision's own guard: one that panics there stops at
/// the engine, the scopes after it are still closed, and the open's panic is
/// the one the host sees.
#[test]
fn a_pairs_prefill_close_that_panics_while_its_decode_side_opens_stops_at_the_engine() {
    let (close_panics, close_counts) = close_panics_once();
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (decode_panics, panicked) = StoresAtItsOpen::stage(true);
    let engine = pair_engine(
        vec![
            host_profile(vec![close_panics, after_it]),
            host_profile(vec![decode_panics]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "the prefill scope is closed, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the prefill scope after it is still closed"
    );
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    // The next pair decision opens and closes all three.
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// The same with a decode scope already open when the panic starts: the
/// opening guard closes that one, the decision's guard then runs the prefill
/// side's closes, and the one that panics there stops at the engine.
#[test]
fn a_pairs_prefill_close_that_panics_after_a_decode_scope_opened_stops_at_the_engine() {
    let (close_panics, close_counts) = close_panics_once();
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (decode_opened, decode_counts) = StoresAtItsOpen::stage(false);
    let (decode_panics, panicked) = StoresAtItsOpen::stage(true);
    let engine = pair_engine(
        vec![
            host_profile(vec![close_panics, after_it]),
            host_profile(vec![decode_opened, decode_panics]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        decode_counts.read(),
        (1, 1),
        "the decode scope that opened is closed, once"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "the prefill scope is closed, and its close panicked"
    );
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "the prefill scope after it is still closed"
    );
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(decode_counts.read(), (2, 2));
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(closed_after_it.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// The close that panics on the decode side itself, inside the unwinding of
/// its own profile's open: the opening guard stops it, and the prefill side,
/// open whole, is closed by the decision's guard.
#[test]
fn a_pairs_decode_close_that_panics_inside_its_own_opens_unwinding_stops_at_the_engine() {
    let (prefill, prefill_counts) = StoresAtItsOpen::stage(false);
    let (close_panics, close_counts) = close_panics_once();
    let (decode_panics, panicked) = StoresAtItsOpen::stage(true);
    let engine = pair_engine(
        vec![
            host_profile(vec![prefill]),
            host_profile(vec![close_panics, decode_panics]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(
        close_counts.read(),
        (1, 1),
        "the decode scope is closed, and its close panicked"
    );
    assert_eq!(prefill_counts.read(), (1, 1), "the prefill side is closed");
    assert_eq!(
        panicked.read(),
        (1, 0),
        "its open did not return: not closed"
    );
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(close_counts.read(), (2, 2));
    assert_eq!(prefill_counts.read(), (2, 2));
    assert_eq!(panicked.read(), (2, 1));
}

/// Both sides' closes panic on a pair decision's own way out. The decode
/// side closes first, so the panic of its close goes on once that side is
/// closed: the prefill side's closes run inside it, the one that panics
/// there stops at the engine, and the decode close's panic is the one the
/// host sees.
#[test]
fn a_pairs_prefill_close_that_panics_while_its_decode_sides_close_unwinds_stops_at_the_engine() {
    let (prefill_panics, prefill_counts) = close_panics_once();
    let (after_prefill, closed_after_prefill) = StoresAtItsOpen::stage(false);
    let decode_panics = Box::new(ClosePanicsWith(2, AtomicBool::new(true)));
    let (after_decode, closed_after_decode) = StoresAtItsOpen::stage(false);
    let engine = pair_engine(
        vec![
            host_profile(vec![prefill_panics, after_prefill]),
            host_profile(vec![decode_panics, after_decode]),
        ],
        1,
    );
    let payload = pair_decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<u32>().copied(),
        Some(2),
        "the decode close's panic is the one the host sees"
    );
    assert_eq!(
        closed_after_decode.read(),
        (1, 1),
        "the decode scope after the close that panicked is closed"
    );
    assert_eq!(
        prefill_counts.read(),
        (1, 1),
        "the prefill scope is closed, and its close panicked"
    );
    assert_eq!(
        closed_after_prefill.read(),
        (1, 1),
        "the prefill scope after it is still closed"
    );
    // The next pair decision opens and closes all four, and no close panics.
    let next = pair_decide(&engine);
    assert!(next.pick().is_some(), "the next decision picks");
    assert_eq!(closed_after_decode.read(), (2, 2));
    assert_eq!(prefill_counts.read(), (2, 2));
    assert_eq!(closed_after_prefill.read(), (2, 2));
}

/// A panic payload whose own destructor panics, with a payload whose
/// destructor panics in turn for as many drops as it counts.
struct DropPanics(u8);

impl Drop for DropPanics {
    fn drop(&mut self) {
        match self.0 {
            0 => std::panic::resume_unwind(Box::new("this payload's drop panics")),
            more => std::panic::resume_unwind(Box::new(DropPanics(more - 1))),
        }
    }
}

/// A stage of the host's own whose scope's close panics, once, with a
/// payload whose own drop panics, as does the drop of that panic's payload.
#[derive(Debug)]
struct ClosePanicsWithADrop(AtomicBool);

impl smg_policy_controller::engine::Filter for ClosePanicsWithADrop {
    fn name(&self) -> &'static str {
        "close-panics-with-a-drop"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for ClosePanicsWithADrop {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {
        if self.0.swap(false, Ordering::Relaxed) {
            std::panic::resume_unwind(Box::new(DropPanics(1)));
        }
    }
}

/// The engine drops a payload it caught and does not resume under a catch
/// of its own: the payload's destructor can panic, and so can the
/// destructor of that panic's payload, which is forgotten. Let out inside
/// an unwinding, either panic would end the process; on a decision's own
/// way out it would take the first panic's place and leave the scopes
/// after it open.
#[test]
fn a_caught_payload_whose_drop_panics_stops_at_the_engine() {
    // Inside an open's unwinding: the closes of the scopes that had opened.
    let with_a_drop = Box::new(ClosePanicsWithADrop(AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let (open_panics, _) = StoresAtItsOpen::stage(true);
    let engine = single_engine(host_profile(vec![with_a_drop, after_it, open_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once")
    );
    assert_eq!(closed_after_it.read(), (1, 1), "inside an open's unwinding");

    // Inside a stage's unwinding: the decision's own closes.
    let with_a_drop = Box::new(ClosePanicsWithADrop(AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let keep_panics = Box::new(PanicsAtItsKeep(AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![with_a_drop, after_it, keep_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this stage's keep panics, once")
    );
    assert_eq!(closed_after_it.read(), (1, 1), "inside a stage's unwinding");

    // On a decision's own way out: the first close's panic goes on, and the
    // second close's payload is released.
    let first = Box::new(ClosePanicsWith(1, AtomicBool::new(true)));
    let with_a_drop = Box::new(ClosePanicsWithADrop(AtomicBool::new(true)));
    let (after_it, closed_after_it) = StoresAtItsOpen::stage(false);
    let engine = single_engine(host_profile(vec![first, with_a_drop, after_it]));
    let payload = decide_caught(&engine);
    assert_eq!(payload.downcast_ref::<u32>().copied(), Some(1));
    assert_eq!(
        closed_after_it.read(),
        (1, 1),
        "on a decision's own way out"
    );
}

/// A panic payload that counts its drops, so a case can read whether the
/// engine freed it.
struct CountsItsDrop(Arc<AtomicU32>);

impl Drop for CountsItsDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// A stage of the host's own whose scope's close panics, once, with a
/// payload that counts its drops.
#[derive(Debug)]
struct ClosePanicsWithACount(AtomicBool, Arc<AtomicU32>);

impl ClosePanicsWithACount {
    fn stage() -> (Box<Self>, Arc<AtomicU32>) {
        let drops = Arc::new(AtomicU32::new(0));
        let stage = Box::new(Self(AtomicBool::new(true), Arc::clone(&drops)));
        (stage, drops)
    }
}

impl smg_policy_controller::engine::Filter for ClosePanicsWithACount {
    fn name(&self) -> &'static str {
        "close-panics-with-a-count"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for ClosePanicsWithACount {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}

    fn close(&self) {
        if self.0.swap(false, Ordering::Relaxed) {
            std::panic::resume_unwind(Box::new(CountsItsDrop(Arc::clone(&self.1))));
        }
    }
}

/// The engine frees a payload it caught and does not resume. Forgotten, a
/// fault that repeats would leave one payload allocated a decision, with
/// nothing to bound it.
#[test]
fn a_caught_payload_the_engine_does_not_resume_is_freed() {
    // Inside an open's unwinding: the closes of the scopes that had opened.
    let (with_a_count, drops) = ClosePanicsWithACount::stage();
    let (open_panics, _) = StoresAtItsOpen::stage(true);
    let engine = single_engine(host_profile(vec![with_a_count, open_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this scope's open panics before its store, once")
    );
    assert_eq!(
        drops.load(Ordering::Relaxed),
        1,
        "inside an open's unwinding"
    );

    // Inside a stage's unwinding: the decision's own closes.
    let (with_a_count, drops) = ClosePanicsWithACount::stage();
    let keep_panics = Box::new(PanicsAtItsKeep(AtomicBool::new(true)));
    let engine = single_engine(host_profile(vec![with_a_count, keep_panics]));
    let payload = decide_caught(&engine);
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("this stage's keep panics, once")
    );
    assert_eq!(
        drops.load(Ordering::Relaxed),
        1,
        "inside a stage's unwinding"
    );

    // On a decision's own way out: the first close's panic goes on, and the
    // second close's payload is freed.
    let first = Box::new(ClosePanicsWith(1, AtomicBool::new(true)));
    let (with_a_count, drops) = ClosePanicsWithACount::stage();
    let engine = single_engine(host_profile(vec![first, with_a_count]));
    let payload = decide_caught(&engine);
    assert_eq!(payload.downcast_ref::<u32>().copied(), Some(1));
    assert_eq!(
        drops.load(Ordering::Relaxed),
        1,
        "on a decision's own way out"
    );
}

/// The plugin's own open marks its phase open and counts the decision
/// before its instance's steps, and the engine does not close a scope whose
/// open did not return. So an open that stops half-way undoes itself: here
/// the host's clock panics inside it, and the thread's next decision is a
/// decision of its own that asks the plugin. Left half-open, the phase would
/// read as open to every later decision, and none would ask the plugin
/// again.
#[test]
fn a_plugins_open_that_stops_half_way_undoes_itself() {
    let clock_panics = Arc::new(AtomicBool::new(false));
    let told = Arc::clone(&clock_panics);
    let runtime = Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), move || {
            assert!(
                !told.swap(false, Ordering::Relaxed),
                "the host's clock panics, once"
            );
            1_000_000
        })
        .expect("a runtime"),
    );
    let plugin = load(
        &runtime,
        "half-open",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let engine = single_engine(plugin_profile(&plugin, None, None, None));
    let candidates = pool(3);
    // A whole decision first: the slot is warm, and what stops the next open
    // is the clock alone.
    let first = decide(&engine, &candidates, 1);
    assert_eq!(pick_ids(&first).first(), Some(&102));
    let before = plugin.stats();
    clock_panics.store(true, Ordering::Relaxed);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decide(&engine, &candidates, 2)
    }));
    assert!(caught.is_err(), "the plugin's open panicked at the clock");
    assert_eq!(
        plugin.stats().answered,
        before.answered,
        "the open stopped before any call"
    );
    let next = decide(&engine, &candidates, 3);
    let asked = plugin.stats();
    assert_eq!(
        asked.answered - before.answered,
        4,
        "the next decision asked the constructor and the three stages: {asked:?}"
    );
    assert_eq!(
        pick_ids(&next).first(),
        Some(&102),
        "the plugin's pick leads: the least in flight"
    );
}

/// What the case below shares between its picker and the host's clock.
#[derive(Debug, Default)]
struct PastThePick {
    /// The case wants a close's read of the clock to panic.
    armed: AtomicBool,
    /// The decision's pick ran: its thread's next read of the clock is a
    /// close's.
    picked: AtomicBool,
}

/// The built-in best picker, telling the clock when it ran.
#[derive(Debug)]
struct TellsThePick(Arc<PastThePick>);

impl Lifecycle for TellsThePick {}

impl smg_policy_controller::engine::Picker for TellsThePick {
    fn name(&self) -> &'static str {
        "tells-the-pick"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: smg_policy_controller::engine::TieBand,
        out: &mut smg_policy_controller::engine::Targets,
    ) {
        if self.0.armed.load(Ordering::Relaxed) {
            self.0.picked.store(true, Ordering::Relaxed);
        }
        smg_policy_controller::engine::Picker::pick(
            &Best, facts, candidates, kept, scores, band, out,
        );
    }
}

/// The plugin's own close runs each phase's destructor, the guest's code
/// under the host's, and a destructor that runs out reads the host's clock
/// for the breaker. So a close that stops half-way leaves its scope closed:
/// every phase stops reading as open, and its place in the thread's decision
/// is given back, before the first destructor runs, and a walk that does not
/// return forgets the slot it marked for the rebuild. Here the clock panics
/// at that read, and every later decision of the thread is a decision of its
/// own that asks the plugin. Left open, the phase would read as open to the
/// next decision: it would make no resource and ask the plugin nothing. With
/// its place not given back, the later decisions would be one decision to
/// the thread, under one number and one budget. The destructor runs out in
/// each, and a plugin that failed a decision is not asked again in it: the
/// second later decision would ask nothing.
///
/// The plugin is bound at one stage, so a decision closes its scope once.
/// Bound at two, the engine closes the same bound set twice, and the second
/// close forgets what the first left open.
///
/// A decision makes its instance under the init frame on the wall clock
/// when the close before it could not. A try in which a decision asked
/// nothing while its thread, or the thread of the decision before it, was
/// off the CPU half that frame is made again, [`TRIES`] tries at most: the
/// last try asserts whatever it read.
#[test]
fn a_plugins_close_that_stops_half_way_leaves_its_scope_closed() {
    const LATER: u64 = 8;
    let mut made = 0;
    loop {
        made += 1;
        let shared = Arc::new(PastThePick::default());
        let told = Arc::clone(&shared);
        let deciding = std::thread::current().id();
        let runtime = Arc::new(
            Runtime::with_clock(&OpenDecisions::new(), move || {
                // Past an armed pick the deciding thread reads the clock in
                // a close alone: that read panics, once.
                assert!(
                    std::thread::current().id() != deciding
                        || !told.picked.swap(false, Ordering::Relaxed),
                    "the host's clock panics inside a close"
                );
                1_000_000
            })
            .expect("a runtime"),
        );
        // A destructor that fills without end runs out of its frame, and
        // the fault's read of the clock is inside the close. At the slow
        // calibration that frame is a hundred units of fuel: the fill runs
        // out of fuel, not of time.
        let mut filler = spec(
            &format!("half-closed-{made}"),
            guest::record_guest(),
            mode::FILL_IN_DESTRUCTOR,
        );
        filler.fuel_per_micro = SLOW_CALIBRATION;
        filler.budget_micros = SLOW_FRAME_MICROS;
        // Every decision's destructor runs out: the failure rule is set out
        // of reach, so the breaker stays shut through the later decisions.
        filler.breaker = BreakerSpec {
            failures: u32::MAX,
            ..BreakerSpec::default()
        };
        let plugin = loaded(&runtime, filler);
        let engine = single_engine(Profile {
            name: "one-stage".into(),
            filters: vec![],
            scorers: vec![WeightedScorer {
                scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
                weight: 1.0,
            }],
            picker: Box::new(TellsThePick(Arc::clone(&shared))),
        });
        let candidates = pool(3);
        shared.armed.store(true, Ordering::Relaxed);
        let stopped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decide(&engine, &candidates, 1)
        }));
        shared.armed.store(false, Ordering::Relaxed);
        shared.picked.store(false, Ordering::Relaxed);
        assert!(
            stopped.is_err(),
            "the plugin's close panicked at the clock: {:?}",
            plugin.stats()
        );
        let mut silent = None;
        let mut before_it = std::time::Duration::ZERO;
        for later in 0..LATER {
            let answered = plugin.stats().answered;
            let (wall, cpu) = (std::time::Instant::now(), bound_clock());
            decide(&engine, &candidates, 2 + later);
            let off_the_cpu = off_cpu_since(wall, cpu);
            if plugin.stats().answered == answered {
                silent = Some((later, off_the_cpu.max(before_it)));
                break;
            }
            before_it = off_the_cpu;
        }
        let Some((later, off_the_cpu)) = silent else {
            break;
        };
        assert!(
            off_the_cpu >= INIT_FRAME / 2 && made < TRIES,
            "every later decision asks the plugin as a decision of its own: decision {later} after the stop asked nothing (try {made}, a thread off the CPU {off_the_cpu:?}): {:?}",
            plugin.stats()
        );
        made_again(
            "a close that stops half-way",
            &format!(
                "decision {later} after the stop asked nothing with a thread off the CPU {off_the_cpu:?}"
            ),
        );
    }
}

/// How many pairs of threads a case of a taken place makes for one pair to
/// share a place.
const PLACES: u32 = 64;

/// A plugin one of whose working places a thread left lent to decider 1 at
/// its exit, with the engine that decides on it. The thread made a decision
/// as decider 1 whose close the host stopped half-way, by a panic inside
/// the destructor's log line: the close did not reach the give-back, and
/// the thread exited with the decider's instance in its place.
///
/// `thread_local` hands a new thread the lowest id that is free, and the
/// first thread's is free at its join, which returns behind the thread's
/// destructors, that crate's among them. The suite's other threads take and
/// free ids all the while, and that crate keys every map of its kind by the
/// one id, so an entry of the case's own says whether a later thread stands
/// in the first's place. A pair that shared none is made again, [`PLACES`]
/// pairs at most, and a case asserts that one pair did. The calling thread
/// takes its id ahead of the first thread: its place is its own.
struct LeftLent {
    plugin: Arc<Plugin>,
    engine: Engine,
    candidates: Vec<Candidate>,
    place: thread_local::ThreadLocal<std::cell::Cell<bool>>,
}

impl LeftLent {
    fn by_a_thread_that_exited(name: &str) -> Self {
        let row = spec(name, guest::record_guest(), mode::LOG_IN_DESTRUCTOR);
        Self::of(&runtime(), row)
    }

    /// As [`Self::by_a_thread_that_exited`], the plugin loaded from `row` on
    /// `runtime`: a guest of the row's mode writes a log line in its
    /// destructor.
    fn of(runtime: &Arc<Runtime>, row: PluginSpec) -> Self {
        let name = row.name.clone();
        let plugin = loaded(runtime, row);
        let engine = single_engine(Profile {
            name: "one-stage".into(),
            filters: vec![],
            scorers: vec![WeightedScorer {
                scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
                weight: 1.0,
            }],
            picker: Box::new(Best),
        });
        let left = Self {
            plugin,
            engine,
            candidates: pool(3),
            place: thread_local::ThreadLocal::new(),
        };
        left.place.get_or_default().set(false);
        panic_in_a_line_of(&name);
        let ended = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    left.place.get_or_default().set(true);
                    let _turn = DecisionSlot::new(1).enter();
                    decide(&left.engine, &left.candidates, 1)
                })
                .join()
        });
        assert!(ended.is_err(), "the first thread's close panicked");
        left
    }

    /// Whether a new thread stood in the first thread's place, and what
    /// `act` said there.
    fn taken<T: Send>(&self, act: impl FnOnce(&Self) -> T + Send) -> (bool, T) {
        std::thread::scope(|scope| {
            scope
                .spawn(|| (self.place.get_or_default().get(), act(self)))
                .join()
                .expect("the second thread")
        })
    }

    /// Decider 1's decision on the calling thread, and the `observe` calls
    /// its catch-up made.
    fn heard_as_decider_1(&self) -> u64 {
        let before = self.plugin.stats().observed;
        let _turn = DecisionSlot::new(1).enter();
        assert!(
            matches!(decide(&self.engine, &self.candidates, 1), Decision::Pick(_)),
            "decider 1 decides"
        );
        self.plugin.stats().observed - before
    }

    /// Decider 1's two turns on the calling thread, with what each heard:
    /// the first makes an instance, and the second hears the list, one
    /// member told between them. Two more members are told behind them.
    fn two_turns_then_two_members(&self) -> [u64; 2] {
        use smg_policy_controller::tracking::Membership;
        let anew = self.heard_as_decider_1();
        self.engine.on_membership(&Membership::Added(WorkerId(102)));
        let listed = self.heard_as_decider_1();
        self.engine.on_membership(&Membership::Added(WorkerId(103)));
        self.engine.on_membership(&Membership::Added(WorkerId(104)));
        [anew, listed]
    }
}

/// A thread under no name holds no instance but its own, where it takes the
/// working place of a thread that exited with a named decider's state lent
/// ([`LeftLent`]). The thread that takes the place reads the mark the first
/// left there: under no name it is not warm, and decider 1's state is at
/// its rest again for the thread that next acts as decider 1.
#[test]
fn a_thread_under_no_name_gives_back_the_state_its_place_was_left_with() {
    // One pair: whether the second thread stood in the first's place, and
    // whether it was warm there under no name, then as decider 1.
    let pair = || {
        LeftLent::by_a_thread_that_exited("left-lent").taken(|left| {
            let own = left.plugin.is_warm();
            let _turn = DecisionSlot::new(1).enter();
            (own, left.plugin.is_warm())
        })
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, _)| *shared);
    assert_eq!(
        shared,
        Some((true, (false, true))),
        "a second thread stood in the first's place: under no name it held no instance there, and decider 1's state was at its rest behind it"
    );
}

/// A state that comes back late does not replace the one that rests. The
/// first thread exits with decider 1's instance in its place ([`LeftLent`]).
/// Decider 1 then takes its turns on the case's own thread: its rest is
/// empty, so its first decision makes an instance. One member is then told,
/// and its second decision hears the list as it stands. The thread that
/// takes the first thread's place gives back what it finds there, late. The
/// rest keeps the instance of the later turns: it is level with the feed,
/// and decider 1's third decision hears nothing. The instance the first
/// thread left never heard the feed: put over the newer one, it would hear
/// the list there.
#[test]
fn a_state_given_back_late_does_not_replace_the_one_at_rest() {
    // One pair: whether the second thread stood in the first's place and
    // was warm there under no name, the `observe` calls of decider 1's three
    // decisions on this thread, and the instances made.
    let pair = || {
        let left = LeftLent::by_a_thread_that_exited("came-late");
        let anew = left.heard_as_decider_1();
        left.engine
            .on_membership(&smg_policy_controller::tracking::Membership::Added(
                WorkerId(102),
            ));
        let behind = left.heard_as_decider_1();
        let (shared, own) = left.taken(|left| left.plugin.is_warm());
        (
            shared,
            own,
            [anew, behind, left.heard_as_decider_1()],
            left.plugin.stats().instantiations,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, false, [0, 1, 0], 2)),
        "decider 1's later instance heard the list once and nothing behind the late give-back: it rests"
    );
}

/// A turn under the decider's own name in a place left lent runs on the
/// state at rest. The first thread exits with decider 1's instance in its
/// place ([`LeftLent`]). Decider 1 takes two turns on the case's own
/// thread, and two more members are told. The thread that takes the first
/// thread's place then acts as decider 1 itself. The hold
/// it finds there is stale, since the rest names the other place: it drops
/// it and decides on the state at rest, which hears the two changes. That
/// state rests behind the turn: decider 1's next decision on the case's
/// thread hears nothing. On the instance the first thread left, the turn
/// would hear the list as one snapshot and its close would come late: the
/// next decision would hear the two changes again.
#[test]
fn a_turn_under_the_same_name_in_a_place_left_lent_runs_on_the_state_at_rest() {
    // One pair: whether the second thread stood in the first's place, the
    // `observe` calls of decider 1's decisions (two on this thread, the
    // second thread's, one more on this thread), and the instances made.
    let pair = || {
        let left = LeftLent::by_a_thread_that_exited("same-name");
        let before = left.two_turns_then_two_members();
        // Behind its turn the second thread reads, under no name, whether
        // an instance is its own: the stale one is dropped, not set aside.
        let (shared, turn) = left.taken(|left| (left.heard_as_decider_1(), left.plugin.is_warm()));
        (
            shared,
            before,
            turn,
            left.heard_as_decider_1(),
            left.plugin.stats().instantiations,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, [0, 1], (2, false), 0, 2)),
        "the second thread's turn as decider 1 heard the two changes on the state at rest, kept no instance of the stale hold, and left that state at rest"
    );
}

/// A hold of the same decider is checked by the open that reaches the bring,
/// also where the open breaker turned the decision's first open away. The
/// first thread exits with decider 1's instance in its place ([`LeftLent`]),
/// and decider 1 takes a turn on the case's own thread: the rest names that
/// place. A warm-up then faults on a snapshot of five and opens the breaker,
/// and four members go. The thread that takes the first thread's place
/// decides a pair as decider 1. Its prefill side meets the open breaker, the
/// cool-down ends inside that side's scope, and the decode side's open is
/// the first to reach the bring. The hold it finds is stale: it drops it and
/// decides on the state at rest, which hears the list as it stands and rests
/// behind the decision, so decider 1's next decision on the case's thread
/// hears nothing. On the instance the first thread left, the close would
/// come late, and that next decision would hear the list again.
#[test]
fn a_hold_is_checked_by_the_open_that_reaches_the_bring_behind_an_open_breaker() {
    use smg_policy_controller::tracking::Membership;
    const START: u64 = 1_000_000;
    const COOL_DOWN: u64 = 1_000_000;
    // The host's clock is the case's own: the cool-down ends when the
    // prefill side's scope says so.
    static CLOCK: AtomicU64 = AtomicU64::new(START);
    // One pair: whether the second thread stood in the first's place, what
    // decider 1's first turn here heard, the warm-up's fault, what the pair
    // decision heard and bypassed, and what decider 1's next turn here heard.
    let pair = || {
        CLOCK.store(START, Ordering::Relaxed);
        let runtime = Arc::new(
            Runtime::with_clock(&OpenDecisions::new(), || CLOCK.load(Ordering::Relaxed))
                .expect("a runtime"),
        );
        let mode = mode::TRAP_ON_FIVE_AND_LOG_IN_DESTRUCTOR;
        let mut row = spec("corner", guest::record_guest(), mode);
        row.breaker = BreakerSpec {
            failures: 1,
            window_micros: 60_000_000,
            cooldown_micros: COOL_DOWN,
        };
        let left = LeftLent::of(&runtime, row);
        let anew = left.heard_as_decider_1();
        let five: Vec<WorkerId> = (100..105).map(WorkerId).collect();
        left.engine.on_membership(&Membership::Snapshot(&five));
        left.plugin.warm().expect("the warm-up's own work is done");
        let warmed = left.plugin.stats();
        for id in [104, 103, 102, 101] {
            left.engine
                .on_membership(&Membership::Removed(WorkerId(id)));
        }
        let plain = Arc::new(plugin_engine(&left.plugin));
        let ends = Inside::MovesTheClock(&CLOCK, START + COOL_DOWN);
        let prefill = profile_that(&left.plugin, &plain, ends);
        let decode = plugin_profile(&left.plugin, None, None, None);
        let engine = pair_engine(vec![prefill, decode], 1);
        let (shared, paired) = left.taken(|left| {
            let before = left.plugin.stats();
            let _turn = DecisionSlot::new(1).enter();
            let _ = pair_decide(&engine);
            let after = left.plugin.stats();
            (
                after.observed - before.observed,
                after.bypassed - before.bypassed,
            )
        });
        let next = left.heard_as_decider_1();
        (
            shared,
            anew,
            (warmed.traps, warmed.breaker_opens),
            paired,
            next,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, 0, (1, 1), (1, 1), 0)),
        "the decode side's open checked the hold: the state at rest heard the list and rests behind the pair"
    );
}

/// An event told under the decider's own name in a place left lent reaches
/// the state at rest: the start of a told event checks the hold as a
/// decision's first open does. The staging is the case's above. The thread
/// that takes the first thread's place tells one dispatch as decider 1: the
/// state at rest comes, hears the two changes and then the event, and rests
/// behind it, so decider 1's next decision hears nothing. On the instance
/// the first thread left, the event would be heard behind one snapshot and
/// stay in the place: the next decision would hear the two changes.
#[test]
fn an_event_told_under_the_same_name_in_a_place_left_lent_reaches_the_state_at_rest() {
    // One pair, as above: the second thread's `observe` calls are its told
    // event's.
    let pair = || {
        let left = LeftLent::by_a_thread_that_exited("same-name-told");
        let before = left.two_turns_then_two_members();
        let (shared, told) = left.taken(|left| {
            let before = left.plugin.stats().observed;
            let _turn = DecisionSlot::new(1).enter();
            left.engine
                .on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            left.plugin.stats().observed - before
        });
        (
            shared,
            before,
            [told, left.heard_as_decider_1()],
            left.plugin.stats().instantiations,
        )
    };
    let shared = (0..PLACES).map(|_| pair()).find(|(shared, ..)| *shared);
    assert_eq!(
        shared,
        Some((true, [0, 1], [3, 0], 2)),
        "the second thread's event as decider 1 was heard behind the two changes on the state at rest, and that state rests behind it"
    );
}

/// A warm-up under a name holds the decider's state for its own span: at
/// its end the state is back at the decider's rest, for whichever thread
/// acts as the decider next. The first thread lives until the second has
/// read, so the second stands in a place of its own. The warm-up takes the
/// apply's own instance: no thread here makes one.
#[test]
fn a_named_warm_up_leaves_the_state_at_its_deciders_rest() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "warmed-as-one",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let (warmed, first_warmed) = mpsc::channel();
    let (read, second_read) = mpsc::channel::<()>();
    let warm_behind_it = std::thread::scope(|scope| {
        let plugin = &plugin;
        scope.spawn(move || {
            {
                let _turn = DecisionSlot::new(1).enter();
                plugin.warm().expect("decider 1 warms");
            }
            warmed.send(()).expect("the case waits for the warm-up");
            // Alive, under no name, until the second thread has read.
            let _ = second_read.recv();
        });
        first_warmed.recv().expect("the first thread warmed");
        let warm = scope
            .spawn(move || {
                let _turn = DecisionSlot::new(1).enter();
                plugin.is_warm()
            })
            .join();
        drop(read);
        warm.expect("the second thread")
    });
    assert!(
        warm_behind_it,
        "decider 1 is warm on the second thread: the first left its state at the rest"
    );
    assert_eq!(
        plugin.stats().instantiations,
        1,
        "the warm-up took the load's own instance"
    );
}

/// The plugin's own undo runs inside its open's unwinding too, by the walk
/// its close does. If that walk panics, it stops at the plugin's own catch
/// and the open's panic is the one the host catches: a second panic let out
/// inside that unwinding would end the process. Here the constructor traps,
/// which empties the slot, and the host's clock panics from then on: at the
/// fault's read inside the open, and again at the undo's rebuild of the slot.
#[test]
fn an_undo_that_panics_inside_its_opens_unwinding_stops_at_the_plugin() {
    tried(an_undo_that_panics_inside_its_opens_unwinding_stops_at_the_plugin_in_a_try);
}

/// One try of the case above, told its number: whether it stood. The
/// stopped undo leaves the slot empty, so the next decision makes its
/// instance inside its own frame, and a hold of the runner can end it there.
fn an_undo_that_panics_inside_its_opens_unwinding_stops_at_the_plugin_in_a_try(made: u32) -> bool {
    const CASE: &str = "an undo that panics inside its open's unwinding";
    let clock_panics = Arc::new(AtomicBool::new(true));
    let told = Arc::clone(&clock_panics);
    let loaded: Arc<std::sync::OnceLock<std::sync::Weak<Plugin>>> = Arc::default();
    let seen = Arc::clone(&loaded);
    let runtime = Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), move || {
            let trapped = seen
                .get()
                .and_then(std::sync::Weak::upgrade)
                .is_some_and(|plugin| plugin.stats().traps > 0);
            assert!(
                !(trapped && told.load(Ordering::Relaxed)),
                "the host's clock panics"
            );
            1_000_000
        })
        .expect("a runtime"),
    );
    let plugin = load(
        &runtime,
        "undo-panics",
        guest::record_guest(),
        mode::TRAP_IN_CONSTRUCTOR,
    );
    loaded
        .set(Arc::downgrade(&plugin))
        .expect("the plugin is set once");
    let engine = single_engine(plugin_profile(&plugin, None, None, None));
    let candidates = pool(3);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decide(&engine, &candidates, 1)
    }));
    let payload = caught.expect_err("the clock's panic inside the open reached the catch");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("the host's clock panics"),
        "the open's panic is the one the host sees"
    );
    assert_eq!(plugin.stats().traps, 1, "the constructor trapped once");
    // The clock mended, the thread's next decision is its own: it asks the
    // plugin, whose constructor traps again, and the fallbacks decide.
    clock_panics.store(false, Ordering::Relaxed);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let next = decide(&engine, &candidates, 2);
    let held = off_cpu_since(wall, cpu);
    if plugin.stats().traps != 2 && held >= INIT_FRAME / 2 && made < TRIES {
        made_again(
            CASE,
            "the next decision's thread was off the CPU half the init frame, the bound of its cold slot's init",
        );
        return false;
    }
    assert!(!pick_ids(&next).is_empty(), "the next decision picks");
    assert_eq!(
        plugin.stats().traps,
        2,
        "the next decision asked the plugin"
    );
    true
}

/// A destructor long on the clock but cheap in fuel runs under its own
/// bound of two milliseconds and is no latency sample: sixty-five decisions
/// of it open no breaker. Its fifty host calls cost a few microseconds each,
/// a small fraction of the bound; one held off the core past it on a loaded
/// runner is not ended, the bound counting the time its thread had the CPU.
#[test]
fn a_slow_destructor_within_its_own_bound_opens_nothing() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "slow-dtor",
        guest::record_guest(),
        mode::SLOW_DESTRUCTOR,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    for seed in 0..65 {
        assert!(
            matches!(decide(&engine, &candidates, seed), Decision::Pick(_)),
            "decision {seed}"
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.breaker_opens, stats.traps),
        (0, 0),
        "no sample, no failure: {stats:?}"
    );
    assert!(plugin.is_warm());
}

/// A destructor the host holds past its bound is not ended: the bound counts
/// the time its thread had the CPU. The guest's log line is a host call, and
/// the case holds the thread in it for four bounds, off the CPU whatever the
/// load, in two decisions. The promise is exact in both: the destructor
/// finishes, the instance stays, the pick is the guest's, and nothing counts
/// on the breaker (one failure would open it). The two tallies are floors,
/// for a hold of the runner's own is a hold too: the first hold has no
/// reading of its start, so it is unread and arms the thread; the second is
/// read, and is the host's.
///
/// A runner that stops the whole process makes another lawful world: the
/// kernel's stop path runs on each stopped thread's own clock (about this
/// destructor's whole bound in a process of two hundred threads), so the
/// held destructor is a call whose thread had its bound. The case reads its
/// thread's clock across each decision: where the clock says the thread had
/// the bound, that world's two outcomes are asserted by their counts and the
/// run says which in one line; a count off anywhere else fails.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_destructor_the_host_held_past_its_bound_is_not_ended() {
    const CASE: &str = "a held destructor";
    // The promise's counts: calls past a frame, traps, instances made,
    // openings, bypasses. Kept: nothing was ended. Ended: a tick found the
    // destructor's thread had its bound, and the one-failure breaker opened.
    const KEPT: [u64; 5] = [0, 0, 1, 0, 0];
    const ENDED: [u64; 5] = [1, 0, 1, 1, 0];
    let runtime = runtime();
    let mut held = spec("held-dtor", guest::record_guest(), mode::LOG_IN_DESTRUCTOR);
    held.breaker = BreakerSpec {
        failures: 1,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = loaded(&runtime, held);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    // The destructor's bound (the host's own constant, private to it, so by
    // copy) and the grain of the host's readings of the thread's clock (whole
    // microseconds).
    let bound = std::time::Duration::from_millis(2);
    let grain = std::time::Duration::from_micros(1);
    // One decision: the guest decided; the promise's counts after it; and
    // the case's witness, the thread's own clock across the decision. That
    // bounds what the destructor's thread had inside its call from above:
    // short of the bound by the grain, no reading of the host's can say the
    // thread had its bound.
    let decide_and_read = |seed: u64, when: &str| {
        let cpu = bound_clock();
        let decision = decide(&engine, &candidates, seed);
        let on_the_cpu = bound_clock().saturating_sub(cpu);
        assert_eq!(
            pick_ids(&decision),
            vec![103, 102, 101, 100],
            "{when}: the guest's pick"
        );
        let stats = plugin.stats();
        let counts = [
            stats.fuel_exhausted,
            stats.traps,
            stats.instantiations,
            stats.breaker_opens,
            stats.bypassed,
        ];
        (stats, counts, on_the_cpu + grain > bound)
    };
    // The promise in the world the case makes: the destructor was not ended.
    let kept = |counts: [u64; 5], stats: &dyn std::fmt::Debug, when: &str| {
        assert_eq!(
            counts, KEPT,
            "{when}: it was not ended: the instance stays and nothing counts: {stats:?}"
        );
        assert!(plugin.is_warm(), "{when}");
    };
    // The one line of a run in the other world.
    let stopped = |when: &str, outcome: &str| {
        other_world(
            CASE,
            &format!("{when}: the thread's clock says it had the destructor's bound: {outcome}"),
        );
    };
    let nap = 4 * bound;
    hold("held-dtor", nap, 1);
    let (stats, counts, had_the_bound) = decide_and_read(1, "the first hold");
    if had_the_bound && counts == ENDED {
        stopped("the first hold", "ended at a tick");
        return;
    }
    kept(counts, &stats, "the first hold");
    assert!(
        stats.unread >= 1,
        "no reading covered the first hold: {stats:?}"
    );
    let unread = stats.unread;
    hold("held-dtor", nap, 1);
    let (stats, counts, had_the_bound) = decide_and_read(2, "the second hold");
    if had_the_bound && counts == ENDED {
        stopped("the second hold", "ended at a tick");
        return;
    }
    if had_the_bound && counts == KEPT && (stats.stalls, stats.unread) == (0, unread) {
        stopped("the second hold", "judged the plugin's own at its end");
        return;
    }
    kept(counts, &stats, "the second hold");
    assert!(
        stats.stalls >= 1,
        "the thread was armed: the second hold is read, and the host's: {stats:?}"
    );
    // The next decision, not held, runs on the same instance.
    let (stats, counts, had_the_bound) = decide_and_read(3, "no hold");
    if had_the_bound && counts == ENDED {
        stopped("no hold", "ended at a tick");
        return;
    }
    kept(counts, &stats, "no hold");
}

/// A stage call the host holds past its frame is not ended, so the instance
/// stays and nothing fails. Three scores are held, three budgets each. The
/// first has no reading of its start: it is unread, its decision's sample is
/// the time it took, and it arms the thread. The other two are read: each is
/// the host's, and its decision's sample is the plugin's own time, never the
/// hold. So the ring of sixty-four holds one slow sample and opens nothing;
/// sampled by the wall clock the three are its slowest, and the second
/// slowest opens the breaker. The decision's deadline stays real: the pick
/// after a held score finds the budget spent and falls back, nobody's
/// failure. The promise is exact; the tallies are floors, for a hold of the
/// runner's own is a hold too.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_stage_the_host_held_past_its_frame_fails_nothing_and_opens_nothing() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "held-score",
        guest::record_guest(),
        mode::LOG_IN_SCORE,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    hold(
        "held-score",
        std::time::Duration::from_micros(u64::from(FRAME_MICROS) * 3),
        3,
    );
    for seed in 0..64 {
        let decision = engine.decide(
            &timed(seed),
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    }
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted,
            stats.traps,
            stats.instantiations,
            stats.breaker_opens
        ),
        (0, 0, 1, 0),
        "none was ended, and one slow sample opens nothing: {stats:?}"
    );
    assert!(
        stats.unread >= 1,
        "no reading covered the first held score: {stats:?}"
    );
    assert!(
        stats.stalls >= 2,
        "the thread was armed: the next two are read, and the host's: {stats:?}"
    );
    assert!(
        stats.budget_spent >= 3,
        "the pick after each held score found the decision's budget spent: {stats:?}"
    );
}

/// A hold excuses lateness, never a fault. A score the host holds past its
/// frame that then traps by itself is the plugin's failure: the trap is
/// counted, the instance goes and the breaker is told, both times, and the
/// breaker opens at its threshold of two. The first held score has no
/// reading of its start: it is unread and arms the thread, as any late call
/// no reading covers. The second is read, the host held it, and its trap is
/// still its own: no stall is counted for it. That count is exact where the
/// case's own witness says no hold but the case's two fell in its decisions:
/// a hold of the runner's own on a call that answered is a stall too. The
/// two decisions run under the helper's budget, which no hold spends: the
/// score's bound is still the plugin's frame, so the case's hold makes it
/// late.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_score_the_host_held_that_then_traps_is_the_plugins_failure_and_no_stall() {
    let runtime = runtime();
    let mut held = spec("held-trap", guest::record_guest(), mode::LOG_IN_SCORE);
    held.breaker = BreakerSpec {
        failures: 2,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = loaded(&runtime, held);
    let engine = plugin_engine(&plugin);
    // On three candidates the guest's score traps after its log line.
    let candidates = pool(3);
    let three = PoolFacts {
        pool_size: 3,
        generation: 1,
        ..PoolFacts::default()
    };
    let nap = std::time::Duration::from_micros(u64::from(FRAME_MICROS) * 3);
    let slack = std::time::Duration::from_micros(u64::from(FRAME_MICROS) / 4);
    hold("held-trap", nap, 2);
    // No hold but the case's own two fell in its decisions.
    let mut quiet = true;
    for (seed, traps) in [(1, 1), (2, 2)] {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &facts(seed),
            &three,
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let off_the_cpu = off_cpu_since(wall, cpu);
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: the fallbacks decide"
        );
        let stats = plugin.stats();
        quiet &= off_the_cpu < nap + slack;
        assert_eq!(
            (stats.traps, stats.fuel_exhausted, stats.instantiations),
            (traps, 0, 2),
            "decision {seed}: the held score's trap is the plugin's and its instance went: {stats:?}"
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.traps, stats.breaker_opens, stats.bypassed),
        (2, 1, 0),
        "the breaker was told of both traps and opened at its threshold: {stats:?}"
    );
    assert!(
        stats.unread >= 1,
        "no reading covered the first held score: {stats:?}"
    );
    if quiet {
        assert_eq!(
            stats.stalls, 0,
            "a held call that failed by itself is no stall: {stats:?}"
        );
    } else {
        excused(
            "a held score that then traps",
            "a decision's thread was off the CPU past the case's own hold",
        );
    }
}

/// One decision of a plugin at the slow calibration whose score the host
/// holds for `held` inside its log line and whose pick then burns. A burn is
/// 117 ms of fuel-time there, more than `held` leaves of the decision's
/// `budget_micros`, so the pick runs out under the decision's remainder. One
/// failure opens the plugin's breaker: an opening is a failure the breaker
/// was told. Returns the plugin, its engine and the case's own witness: the
/// decision took no longer than the hold and half of what the hold leaves,
/// so a remainder reached the pick and the route is the case's to assert.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn burn_after_a_hold(
    name: &str,
    frame_micros: u32,
    budget_micros: u32,
    held: std::time::Duration,
) -> (Arc<Plugin>, Engine, bool) {
    let runtime = runtime();
    let mut slow = spec(name, guest::record_guest(), mode::LOG_THEN_BURN);
    slow.fuel_per_micro = SLOW_CALIBRATION;
    slow.budget_micros = frame_micros;
    slow.breaker = BreakerSpec {
        failures: 1,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = loaded(&runtime, slow);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    hold(name, held, 1);
    let wall = std::time::Instant::now();
    let decision = engine.decide(
        &RequestFacts {
            budget_micros,
            ..facts(1)
        },
        &four(),
        &candidates,
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let took = wall.elapsed();
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let left = std::time::Duration::from_micros(u64::from(budget_micros)).saturating_sub(held);
    (plugin, engine, took < held + left / 2)
}

/// Whether the close rebuilt the instance a run-out dropped. Two other
/// worlds are lawful, and a run in either asserts that world's counts here,
/// says so, and asserts no more. A host that holds the decision's thread
/// past the case's own hold can end the budget inside the score: the pick
/// has no grant and is not called, nothing runs out and the instance stays.
/// Only a run whose witness is silent can be that one (`quiet`: a remainder
/// reached the pick). And the rebuild is an `init`, judged under its own
/// frame on the wall clock: one the host holds for that frame fails, and
/// that failure is the breaker's, where the run-out itself told it nothing.
/// That run says so with the host's own line.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn rebuilt(case: &str, plugin: &Plugin, quiet: bool) -> bool {
    let stats = plugin.stats();
    if stats.lazy_instantiations != 0 {
        return true;
    }
    if !quiet && stats.instantiations == 1 {
        assert_eq!(
            stats.breaker_opens, 0,
            "a score held past the whole budget fails nothing: {stats:?}"
        );
        other_world(
            case,
            "the hold spent the whole budget and the pick was not called",
        );
        return false;
    }
    assert_eq!(
        (stats.instantiations, stats.breaker_opens),
        (2, 1),
        "a rebuild that failed is one failure, and the one-failure breaker opened on it: {stats:?}"
    );
    let named = format!("plugin=\"{}\"", plugin.name());
    let said = said()
        .lock()
        .expect("the kept lines")
        .iter()
        .filter(|line| line.contains("rebuild") && line.contains(&named))
        .cloned()
        .collect::<Vec<_>>();
    other_world(case, &format!("the close's rebuild failed: {said:?}"));
    false
}

/// Time nothing read draws the decision's remainder, never the plugin's own
/// spend. A score the host holds for nearly half of the decision's budget is
/// short of late, so nothing reads the thread's clock for it and its own
/// spend is its fuel, not the time it took. The pick's burn then needs more
/// than the hold left and runs out: the decision's budget is spent, the
/// interrupted instance goes and the breaker is told nothing, for the plugin
/// spent a burn's fuel of a budget the host's hold ended. Charged the hold,
/// the plugin's spend with the grant reached the budget and the run-out was
/// its own failure. Nothing said whether the time before the run-out was a
/// hold or the plugin's own past its fuel, so the run-out is counted with
/// the calls nothing read and arms the thread: a second score, held past its
/// bound this time, has a reading of its start and is the host's stall,
/// where a thread that was not armed counts it unread too. The promise is
/// exact in every run where the pick ran out and the close rebuilt the
/// dropped instance (a hold past the whole budget leaves the pick no call,
/// and a rebuild that fails is a failure of `init`, the breaker's: either
/// run says so); the route is asserted where the case's witness says a
/// remainder reached the pick.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_burn_that_runs_out_under_what_a_hold_short_of_late_left_is_the_decisions_budget_spent() {
    // The hold leaves 110 ms of fuel-time, under a burn's 117: the pick runs
    // out. It leaves more than half the budget, so the pick's grant counted
    // twice into the plugin's spend would reach the budget.
    let (plugin, engine, quiet) = burn_after_a_hold(
        "starved-short",
        SLOW_FRAME_MICROS,
        200_000,
        std::time::Duration::from_millis(90),
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.budget_spent),
        (0, 1),
        "a run-out under what a hold left is the decision's budget spent: {stats:?}"
    );
    if !rebuilt("a burn after a hold short of late", &plugin, quiet) {
        return;
    }
    assert_eq!(
        stats.breaker_opens, 0,
        "the breaker was told nothing of the run-out: {stats:?}"
    );
    if quiet {
        assert_eq!(
            (stats.instantiations, stats.unread, stats.stalls),
            (2, 1, 0),
            "the pick ran out mid-call behind a score short of late, and nothing read which: {stats:?}"
        );
    } else {
        excused(
            "a burn after a hold short of late",
            "the decision's thread was held past the case's own hold",
        );
    }
    // The second decision: its budget (20 ms) is short of this hold (30 ms),
    // so the held score is late past its bound and spends the remainder.
    hold("starved-short", std::time::Duration::from_millis(30), 1);
    let decision = engine.decide(
        &RequestFacts {
            budget_micros: 20_000,
            ..facts(2)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted,
            stats.breaker_opens,
            stats.budget_spent
        ),
        (0, 0, 2),
        "the held score spent the second decision's budget, and the pick was not called: {stats:?}"
    );
    assert!(
        stats.stalls >= 1,
        "the thread was armed: the late score was read, and the host's: {stats:?}"
    );
    if quiet {
        assert_eq!(
            stats.unread, 1,
            "the run-out armed the thread: the late score after it did not go unread: {stats:?}"
        );
    }
}

/// The same end by the other path: a frame under the decision's budget. The
/// held score is late by its frame and, on a thread that read no clock at
/// its start, unread; an unread call's own spend is its fuel too. The hold
/// leaves an eighth of the budget, the pick's burn runs out under it, and it
/// is the decision's budget spent as above, counted with the calls nothing
/// read beside the late score.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_burn_that_runs_out_under_what_an_unread_late_hold_left_is_the_decisions_budget_spent() {
    let (plugin, _engine, quiet) = burn_after_a_hold(
        "starved-late",
        200_000,
        400_000,
        std::time::Duration::from_millis(350),
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.budget_spent),
        (0, 1),
        "a run-out under what a hold left is the decision's budget spent: {stats:?}"
    );
    if !rebuilt("a burn after an unread late hold", &plugin, quiet) {
        return;
    }
    assert_eq!(
        stats.breaker_opens, 0,
        "the breaker was told nothing of the run-out: {stats:?}"
    );
    if quiet {
        assert_eq!(
            (stats.instantiations, stats.unread, stats.stalls),
            (2, 2, 0),
            "no reading covered the held score, nor said whose the pick's run-out was: {stats:?}"
        );
    } else {
        excused(
            "a burn after an unread late hold",
            "the decision's thread was held past the case's own hold",
        );
    }
}

/// A hold excuses a decline as it does an answer: the guest said the stage
/// does not apply, late because the host held it. Two scores the host holds
/// past the frame, each declining after its log line: both are declines,
/// nothing fails, the instance is kept and the breaker is told nothing. The
/// first has no reading of its start and is unread; the second is read, and
/// the host's stall. The decline count is exact; the two late tallies are
/// floors, for a hold of the runner's own is a hold too.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_score_the_host_held_that_then_declines_is_a_decline_and_the_hosts_stall() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "held-decline",
        guest::record_guest(),
        mode::LOG_THEN_DECLINE,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    hold(
        "held-decline",
        std::time::Duration::from_micros(u64::from(FRAME_MICROS) * 3),
        2,
    );
    for seed in [1, 2] {
        let decision = decide(&engine, &candidates, seed);
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: a declined score has no veto"
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.declined,
            stats.traps,
            stats.fuel_exhausted,
            stats.instantiations,
            stats.breaker_opens
        ),
        (2, 0, 0, 1, 0),
        "both held scores declined, and a decline fails nothing: {stats:?}"
    );
    assert!(
        stats.unread >= 1,
        "no reading covered the first held score: {stats:?}"
    );
    assert!(
        stats.stalls >= 1,
        "the second was read: the host held it, and it declined: {stats:?}"
    );
}

/// Sixty-four decisions of a plugin whose score the host holds inside its
/// log line for six tenths of the decision's budget, short of late. With
/// `arm`, the first is held past the frame instead: unread, it arms the
/// thread. Returns the plugin and the case's two witnesses. The thread's own
/// clock across each decision says none had a quarter of the budget on the
/// CPU. The same clock says whether the window's faults are the runner's:
/// each fell in a decision held short whose thread was off the CPU for its
/// hold and half of what the hold leaves of the budget, where the calls
/// after the score run out under what is left. The case's hold is off the
/// CPU and a decision's own work is not: a call that work ends is the
/// host's fault, however long its decision took. A budget spent with no
/// call failed is the runner's wherever the time went: the hold leaves
/// four tenths of the budget, and the clock alone took the rest.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn one_window_of_short_holds(name: &str, arm: bool) -> (Arc<Plugin>, bool, bool) {
    let runtime = runtime();
    let mut held = spec(name, guest::record_guest(), mode::LOG_IN_SCORE);
    // Every score's log line reaches the host's subscriber, where the case
    // holds it.
    held.log.per_second = 1_000;
    let plugin = loaded(&runtime, held);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    let frame = std::time::Duration::from_micros(u64::from(FRAME_MICROS));
    let short = frame * 6 / 10;
    if arm {
        hold(name, 3 * frame, 1);
        hold(name, short, 63);
    } else {
        hold(name, short, 64);
    }
    // What a short hold leaves of the budget bounds the calls after the
    // score. A decision whose thread was off the CPU for its hold and half
    // of that again was held by the runner too, and a call that runs out in
    // it ran out under what the runner left.
    let late = short + (frame - short) / 2;
    // Such a call is the decision's budget spent and no fault of the
    // plugin's, so the witness reads that count beside the two faults. The
    // arming decision's long hold spends its own budget by design.
    let ran_out = || {
        let stats = plugin.stats();
        (stats.fuel_exhausted, stats.traps, stats.budget_spent)
    };
    let (mut cheap, mut by_the_runner, mut by_the_host) = (true, 0, 0);
    for seed in 0..64 {
        let before = ran_out();
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &timed(seed),
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let off_the_cpu = off_cpu_since(wall, cpu);
        cheap &= bound_clock().saturating_sub(cpu) < frame / 4;
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
        let arming = arm && seed == 0;
        let (fuel, traps, spent) = ran_out();
        let faulted = (fuel, traps) != (before.0, before.1);
        let budget_gone = spent != before.2 && !arming;
        if faulted {
            if arming || off_the_cpu < late {
                by_the_host += 1;
            } else {
                by_the_runner += 1;
            }
        } else if budget_gone {
            // No call failed and the decision's calls still took its whole
            // budget: on the CPU or off it, that time was the clock's.
            by_the_runner += 1;
        }
    }
    (plugin, cheap, by_the_runner > 0 && by_the_host == 0)
}

/// How many windows of short holds a case makes at most. A window is
/// sixty-four decisions, each with its budget on the wall clock: on a host
/// loaded past its cores the runner spends a budget in one window of a few,
/// and three such windows in a row are not rare there. Twelve are, and a
/// host that spends a budget in every window still fails the case.
#[cfg(any(target_os = "linux", target_os = "macos"))]
const WINDOWS: u32 = 12;

/// What a window of short holds says when its faults are the runner's: it
/// is made again.
#[cfg(any(target_os = "linux", target_os = "macos"))]
const A_HOLD_REACHED_A_PICK: &str =
    "a call ran out in a decision whose thread was off the CPU for its short hold and half of what the hold leaves of the budget, or a decision's budget ran out with no call failed";

/// [`one_window_of_short_holds`], made again while its faults are the
/// runner's, [`WINDOWS`] windows at most, each under a plugin name of its own
/// (a hold is kept by name). Returns the last window's plugin and its
/// witness on the CPU: a case asserts that window's counts, whatever its
/// faults were.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn a_window_of_short_holds(case: &str, name: &str, arm: bool) -> (Arc<Plugin>, bool) {
    let mut made = 0;
    loop {
        made += 1;
        let (plugin, cheap, by_the_runner) =
            one_window_of_short_holds(&format!("{name}-{made}"), arm);
        if !by_the_runner || made == WINDOWS {
            return (plugin, cheap);
        }
        made_again(case, A_HOLD_REACHED_A_PICK);
    }
}

/// A reading gives a call one time, late or not. On an armed thread a score
/// the host holds for six tenths of the decision's budget is short of late,
/// and its time is what its thread had, not the hold: sixty-three such
/// decisions are no slow samples. The ring of sixty-four holds one slow
/// sample, the unread call that armed the thread, and opens nothing. The
/// unread tally is one: the event that arms a thread is counted once, at
/// the arming, and on an armed thread nothing is counted again, a hold of
/// the runner's own included (it is read).
/// That the breaker stays shut is asserted where the case's witness says no
/// decision had a quarter of the budget on the CPU: a stop of the whole
/// process is off the CPU by the thread's own clock, but one that lands
/// inside the case's own hold makes that sleep end on the CPU: the clock
/// moves by about the hold, and that time is the call's by any reading.
/// A call that runs out is the host's fault unless its decision's thread
/// was off the CPU for its short hold and half of what the hold leaves of
/// the budget: a hold of the runner's own past that reaches the remainder
/// the later calls run under. A decision's budget spent with no call failed
/// is the runner's too. A window whose faults are all the runner's says so
/// and is made again, [`WINDOWS`] windows at most.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_hold_short_of_late_on_an_armed_thread_is_no_slow_sample() {
    let (plugin, cheap) =
        a_window_of_short_holds("a window of holds short of late", "held-short", true);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted,
            stats.traps,
            stats.budget_spent,
            stats.instantiations
        ),
        (0, 0, 1, 1),
        "a hold fails nothing, and the arming hold alone spends its decision's budget: {stats:?}"
    );
    assert!(
        stats.unread >= 1,
        "no reading covered the first held score: {stats:?}"
    );
    if cheap {
        assert_eq!(
            stats.unread, 1,
            "one event armed the thread, and nothing on an armed thread is counted again: {stats:?}"
        );
        assert_eq!(
            stats.breaker_opens, 0,
            "a window of holds short of late on an armed thread reads no slow median: {stats:?}"
        );
    } else {
        excused(
            "a window of holds short of late",
            "a decision's thread had a quarter of the budget on the CPU",
        );
    }
}

/// The other end of the same interval. On a thread nothing armed no reading
/// covers a call, and nothing tells a hold of the host from a guest slow in
/// what its fuel does not price that answers in time: the ring takes the
/// time the call took. So the first of the same sixty-four scores, held for
/// six tenths of the budget and not late, is a sample slow by time nothing
/// read alone: past half the budget by the time the score took, nowhere
/// near it by its fuel. It is counted with the unread calls and arms the
/// thread, and the sixty-three after it are read as the armed window's are:
/// the ring holds one slow sample and opens nothing. Before that rule all
/// sixty-four read slow and the window opened at its first read. A guest
/// slow by itself is slow by those readings and opens all the same (the
/// case of the guest slow in what fuel does not price). That the breaker
/// stays shut is asserted under the armed case's witness. A window whose
/// faults are the runner's by the same reading is made again.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_window_of_holds_short_of_late_on_a_thread_nothing_armed_is_one_slow_sample() {
    let (plugin, cheap) = a_window_of_short_holds(
        "a window of holds short of late on a thread nothing armed",
        "held-quiet",
        false,
    );
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted,
            stats.traps,
            stats.budget_spent,
            stats.instantiations
        ),
        (0, 0, 0, 1),
        "a hold fails nothing: {stats:?}"
    );
    assert!(
        stats.unread >= 1,
        "the first held score's sample was slow by time nothing read: {stats:?}"
    );
    if cheap {
        assert_eq!(
            stats.breaker_opens, 0,
            "one slow sample, the first: the sixty-three after it were read: {stats:?}"
        );
    } else {
        excused(
            "a window of holds short of late on a thread nothing armed",
            "a decision's thread had a quarter of the budget on the CPU",
        );
    }
}

/// An arming at a decision's close lasts the ring's own length and no
/// longer: the sample that armed the thread and the 127 after it, so the
/// next sample no reading covers overwrites the first. A score the host
/// holds for six tenths of the budget is short of late. At the first
/// decision its sample is slow by time nothing read: counted, it arms the
/// thread. At the 128th the thread is still armed: the hold is read and
/// counts nothing. At the 129th the arm has fallen: the same hold is
/// counted again, and arms again.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_arming_at_a_close_lasts_the_rings_length_and_no_longer() {
    let name = "held-a-ring-apart";
    let runtime = runtime();
    let mut held = spec(name, guest::record_guest(), mode::LOG_IN_SCORE);
    // Every score's log line reaches the host's subscriber, where the case
    // holds three of them.
    held.log.per_second = 1_000;
    let plugin = loaded(&runtime, held);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    let short = std::time::Duration::from_micros(u64::from(FRAME_MICROS)) * 6 / 10;
    let unread_after = |seed: u64, held: bool| {
        if held {
            hold(name, short, 1);
        }
        let decision = engine.decide(
            &timed(seed),
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
        plugin.stats().unread
    };
    assert_eq!(
        unread_after(1, true),
        1,
        "the first hold is slow by time nothing read"
    );
    for seed in 2..=127 {
        unread_after(seed, false);
    }
    assert_eq!(
        unread_after(128, true),
        1,
        "the 128th decision is the last of the arming: its hold is read"
    );
    assert_eq!(
        unread_after(129, true),
        2,
        "the 129th decision finds the arm fallen: its hold is counted again"
    );
}

/// Where the host reads a thread's clock, no hold opens a neighbour's
/// breaker either. A scorer the host holds inside its log line for seven
/// tenths of the budget, short of late, in thirty-three of its first
/// sixty-four decisions, in front of a scorer that spins. The first held
/// score is a sample slow by time nothing read alone: it is counted with
/// the unread calls and arms the thread, and the thirty-two after it are
/// read, so the held scorer's window holds one slow sample and reads no
/// slow median. Before that rule thirty-three of the sixty-four read slow
/// and its own window opened it. A held decision leaves the spinner three
/// tenths of the budget, a sample under half, so the spinner's first window
/// reads no slow median either and its second opens it: the case decides
/// until that opening. That the held scorer stays shut, and that the unread
/// tally is the one arming, is asserted where no held decision's thread had
/// the hold's own length on the CPU: the spinner behind a hold has what the
/// hold left and two ticks, and a stop of the whole process that lands
/// inside the case's own hold makes that sleep end on the CPU, where the
/// time is the call's by any reading.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_scorer_held_short_of_late_through_half_a_window_before_a_spinner_stays_shut() {
    const HELD: u32 = 33;
    let name = "held-neighbour";
    let runtime = runtime();
    let mut held = spec(name, guest::record_guest(), mode::LOG_IN_SCORE);
    // Every score's log line reaches the host's subscriber, where the case
    // holds the first thirty-three.
    held.log.per_second = 1_000;
    let held = loaded(&runtime, held);
    let spinner = load(
        &runtime,
        "spinner-behind",
        guest::record_guest(),
        mode::SPIN_IN_SCORE,
    );
    let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
        scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
        weight: 1.0,
    };
    let engine = single_engine(Profile {
        name: "held-first".into(),
        filters: vec![],
        scorers: vec![scorer(&held), scorer(&spinner)],
        picker: Box::new(Best),
    });
    let candidates = pool(4);
    let frame = std::time::Duration::from_micros(u64::from(FRAME_MICROS));
    let short = frame * 7 / 10;
    hold(name, short, HELD);
    let (mut decisions, mut cheap) = (0, true);
    while spinner.stats().breaker_opens == 0 {
        decisions += 1;
        assert!(
            decisions <= 256,
            "no opening in 256 decisions: {:?}",
            spinner.stats()
        );
        let cpu = bound_clock();
        let decision = engine.decide(
            &timed(decisions),
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        if decisions <= u64::from(HELD) {
            cheap &= bound_clock().saturating_sub(cpu) < short;
        }
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
    }
    let (s, h) = (spinner.stats(), held.stats());
    assert_eq!(
        (s.traps, s.breaker_opens),
        (0, 1),
        "the spinner's breaker opened once, on no trap: {s:?}"
    );
    assert_eq!(h.traps, 0, "a hold traps nothing: {h:?}");
    if cheap {
        assert_eq!(
            h.breaker_opens, 0,
            "one slow sample, the first: the thirty-two after it were read, and no hold opens the held scorer: {h:?}"
        );
        // The arming holds for the ring's length, a hundred and twenty-eight
        // decisions: within it nothing is counted again.
        if decisions <= 128 {
            assert_eq!(h.unread, 1, "one event, one count, at the arming: {h:?}");
        }
    } else {
        excused(
            "a scorer held through half a window before a spinner",
            "a held decision's thread had the hold's own length on the CPU",
        );
    }
    assert!(
        h.unread >= 1,
        "the first held score's sample was slow by time nothing read: {h:?}"
    );
}

/// The ring's tail is its second-slowest sample, so two samples past the
/// budget open a window. Under a budget that is no whole number of ticks a
/// score the host holds past the budget and short of late (the budget in
/// whole ticks) is no late call, and on a thread nothing armed its decision
/// is sampled at the time the score took. The first such sample is slow by
/// time nothing read alone: it is counted with the unread calls and arms
/// the thread. The second held score is read, and its time is what its
/// thread had. So a window of sixty-four with two held scores holds one
/// slow sample and opens nothing; before that rule it held two, and opened.
/// A hold the runner stretches to late arms the thread by the late call's
/// own rule, and the counts are the same. That nothing ran out and the
/// breaker stays shut is asserted where no decision's thread had half the
/// budget on the CPU: a stop of the whole process that lands inside the
/// case's own hold makes that sleep end on the CPU, the thread's clock
/// moves by about the hold, and a held score that had its bound by that
/// reading is ended as the plugin's own.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn two_scores_held_past_a_small_budget_and_short_of_late_are_one_slow_sample() {
    // No whole number of ticks: late is two ticks, and the hold is between
    // the budget and late.
    const BUDGET_MICROS: u32 = 1_500;
    const HOLD_MICROS: u64 = 1_700;
    let name = "held-twice";
    let runtime = runtime();
    let mut held = spec(name, guest::record_guest(), mode::LOG_IN_SCORE);
    // Every score's log line reaches the host's subscriber, where the case
    // holds the first two.
    held.log.per_second = 1_000;
    let plugin = loaded(&runtime, held);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    hold(name, std::time::Duration::from_micros(HOLD_MICROS), 2);
    let half = std::time::Duration::from_micros(u64::from(BUDGET_MICROS) / 2);
    let mut cheap = true;
    for seed in 1..=64 {
        let cpu = bound_clock();
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: BUDGET_MICROS,
                ..facts(seed)
            },
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        cheap &= bound_clock().saturating_sub(cpu) < half;
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    }
    let stats = plugin.stats();
    assert_eq!(stats.traps, 0, "a hold traps nothing: {stats:?}");
    assert!(
        stats.unread >= 1,
        "the first held score's sample was slow by time nothing read: {stats:?}"
    );
    if cheap {
        assert_eq!(
            (stats.fuel_exhausted, stats.instantiations),
            (0, 1),
            "a hold fails nothing: {stats:?}"
        );
        assert_eq!(
            stats.breaker_opens, 0,
            "two held scores are one slow sample: the second was read: {stats:?}"
        );
    } else {
        excused(
            "two scores held past a small budget",
            "a decision's thread had half the budget on the CPU",
        );
    }
}

/// What holds a guest slow in what its fuel does not price is its
/// decisions' samples. A filter long on the CPU for milliseconds, cheap in
/// fuel, answers; the score then fills without end and its bound ends it
/// under what the filter left, every decision. Behind a call that long no
/// run-out need be the plugin's own by readings (a thread's clock reads
/// short of its call by what the kernel does not bill it), so the failure
/// rule, put out of reach here, may never count one. Each decision's sample
/// is the whole budget all the same: the ring is read at the sixty-fourth
/// decision, not before, and opens the breaker there. That opening is
/// asserted where the case's witness says no decision's thread was off the
/// CPU for half the budget: the time a filter is held leaves the score less,
/// and a decision held that long samples under half.
#[test]
fn a_guest_slow_in_what_fuel_does_not_price_is_opened_by_its_samples_at_the_first_window() {
    const BUDGET_MICROS: u32 = 20_000;
    // The opening's line is read below: the subscriber is installed before
    // the first decision.
    said();
    let runtime = runtime();
    let mut slow = spec(
        "brief-fill",
        guest::record_guest(),
        mode::FILL_BRIEFLY_THEN_FILL,
    );
    slow.fuel_per_micro = 20_000_000;
    slow.budget_micros = BUDGET_MICROS;
    slow.breaker = BreakerSpec {
        failures: 1_000,
        window_micros: 60_000_000,
        cooldown_micros: 60_000_000,
    };
    let plugin = loaded(&runtime, slow);
    let engine = plugin_engine(&plugin);
    let hold = std::time::Duration::from_micros(u64::from(BUDGET_MICROS) / 2);
    let mut on_the_cpu = true;
    for seed in 1..=64 {
        assert_eq!(
            plugin.stats().breaker_opens,
            0,
            "decision {seed}: the ring is not read before the sixty-fourth"
        );
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: BUDGET_MICROS,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let off_the_cpu = off_cpu_since(wall, cpu);
        on_the_cpu &= off_the_cpu < hold;
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: the fallbacks decide"
        );
    }
    let stats = plugin.stats();
    assert_eq!(stats.traps, 0, "{stats:?}");
    assert_eq!(
        stats.budget_spent + stats.fuel_exhausted,
        64,
        "every decision's budget was spent, the budget's or the plugin's by what its thread's clock read: {stats:?}"
    );
    if on_the_cpu {
        assert_eq!(
            stats.breaker_opens, 1,
            "no hold moved the run: the window opened at its first read: {stats:?}"
        );
        let lines = openings("brief-fill");
        assert_eq!(lines.len(), 1, "one opening, one line: {lines:?}");
        assert!(lines[0].contains("rule=\"latency\""), "{}", lines[0]);
    } else {
        excused(
            "a guest slow in what fuel does not price",
            "a decision's thread was off the CPU half the budget",
        );
    }
}

/// What one stage's burn costs at the slow calibration, in microseconds: a
/// decision of a gauge plugin bound as the three stages, its fuel in three
/// parts. A call's charge there is its fuel: a part is one stage's burn and
/// a third of the little the rest of the decision costs.
fn one_burn_micros(runtime: &Arc<Runtime>) -> u32 {
    let mut gauge = spec("burn-gauge", guest::record_guest(), mode::BURN_IN_STAGES);
    gauge.fuel_per_micro = SLOW_CALIBRATION;
    gauge.budget_micros = SLOW_FRAME_MICROS;
    let gauge = loaded(runtime, gauge);
    let before = gauge.stats().fuel_used;
    let decision = plugin_engine(&gauge).decide(
        &RequestFacts {
            budget_micros: SLOW_FRAME_MICROS,
            ..facts(1)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    let fuel = gauge.stats().fuel_used - before;
    u32::try_from((fuel / 3).div_ceil(SLOW_CALIBRATION)).expect("fits")
}

/// Sixty-four decisions of `plugin`, bound as the three stages, under a
/// request that states no budget: a stage is shown the widest one there.
fn decide_64_with_no_budget(plugin: &Arc<Plugin>) {
    let engine = plugin_engine(plugin);
    for seed in 1..=64 {
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: u32::MAX,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: {decision:?}"
        );
    }
}

/// Under a request that states no budget a guest's time is set against the
/// frames of its own calls, not against the widest budget. A guest whose
/// three stages each burn nine tenths of their frame answers every call and
/// fails none, so the failure rule counts nothing. Its decisions sample past
/// half of their frames all the same, and the ring's first read opens the
/// breaker.
#[test]
fn a_guest_near_its_frame_in_every_call_is_opened_by_its_samples_under_a_request_with_no_budget() {
    // The opening's line is read below: the subscriber is installed before
    // the first decision.
    said();
    let runtime = runtime();
    let burn_micros = one_burn_micros(&runtime);
    let mut near = spec(
        "near-its-frame",
        guest::record_guest(),
        mode::BURN_IN_STAGES,
    );
    near.fuel_per_micro = SLOW_CALIBRATION;
    near.budget_micros = burn_micros * 10 / 9;
    let plugin = loaded(&runtime, near);
    decide_64_with_no_budget(&plugin);
    let stats = plugin.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.traps, stats.budget_spent),
        (0, 0, 0),
        "every call fit its frame: {stats:?}"
    );
    assert_eq!(
        stats.breaker_opens, 1,
        "the window opened at its first read: {stats:?}"
    );
    let lines = openings("near-its-frame");
    assert_eq!(lines.len(), 1, "one opening, one line: {lines:?}");
    assert!(lines[0].contains("rule=\"latency\""), "{}", lines[0]);
}

/// The frames of a decision's calls add up. A guest whose three stages each
/// burn four tenths of their frame spends more than one frame a decision
/// and under half of the frames of its calls: under a request that states
/// no budget its sixty-four decisions open nothing.
#[test]
fn three_healthy_calls_of_a_guest_are_read_against_their_three_frames() {
    let runtime = runtime();
    let burn_micros = one_burn_micros(&runtime);
    let mut healthy = spec("three-healthy", guest::record_guest(), mode::BURN_IN_STAGES);
    healthy.fuel_per_micro = SLOW_CALIBRATION;
    healthy.budget_micros = burn_micros * 5 / 2;
    let plugin = loaded(&runtime, healthy);
    decide_64_with_no_budget(&plugin);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted,
            stats.traps,
            stats.budget_spent,
            stats.breaker_opens
        ),
        (0, 0, 0, 0),
        "three calls at four tenths of their frames are no slow sample: {stats:?}"
    );
}

/// A guest's constructor is a call under a frame of its own, so a guest bound
/// as one stage is read over two frames a decision. One whose only stage
/// burns nine tenths of its frame takes under half of its two frames: under
/// a request that states no budget its sixty-four decisions open nothing.
/// Bound as three stages, the same burn opens the breaker at the ring's
/// first read.
#[test]
fn a_guest_bound_as_one_stage_is_read_against_its_two_frames() {
    let runtime = runtime();
    let burn_micros = one_burn_micros(&runtime);
    let mut near = spec(
        "one-stage-near",
        guest::record_guest(),
        mode::BURN_IN_STAGES,
    );
    near.fuel_per_micro = SLOW_CALIBRATION;
    near.budget_micros = burn_micros * 10 / 9;
    let plugin = loaded(&runtime, near);
    let engine = single_engine(Profile {
        name: "one-stage".into(),
        filters: vec![],
        scorers: vec![WeightedScorer {
            scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    });
    for seed in 1..=64 {
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: u32::MAX,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: {decision:?}"
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.fuel_exhausted,
            stats.traps,
            stats.budget_spent,
            stats.breaker_opens
        ),
        (0, 0, 0, 0),
        "nine tenths of one frame of two is no slow sample: {stats:?}"
    );
}

/// A call the decision's remainder clamped is not its plugin's own, though
/// its bound ended it: the plugin failed nothing there, so its decision's
/// sample is still asked whether it is unread. A scorer whose constructor
/// burns draws most of the budget in front of a scorer the host holds inside
/// its log line for as long as the whole budget. Released, that score spins
/// until the bound of what the decision had left ends it: the decision's
/// budget spent, and no failure of the plugin. Its sample holds the hold,
/// past the budget, and its fuel is far short of half: slow by time nothing
/// read alone, it is counted with the unread calls and arms the thread, so a
/// second hold there is read.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_held_call_its_bound_ended_under_the_remainder_is_still_an_unread_sample() {
    // What the decision leaves the held scorer.
    const LEFT_MICROS: u32 = 30_000;
    let name = "held-then-spins";
    let runtime = runtime();
    let front = load_slow(
        &runtime,
        "draws-first",
        guest::record_guest(),
        mode::BURN_IN_CONSTRUCTOR,
    );
    let mut held = spec(name, guest::record_guest(), mode::LOG_THEN_SPIN);
    // The score's log line reaches the host's subscriber, where the case
    // holds it.
    held.log.per_second = 1_000;
    // A calibration no host has: the spin's fuel lasts far past its bound,
    // so only the clock ends it. And a frame the remainder is well under.
    held.fuel_per_micro = 20_000_000;
    held.budget_micros = LEFT_MICROS * 4;
    let held = loaded(&runtime, held);
    let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
        scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
        weight: 1.0,
    };
    let decide_with_budget = |engine: &Engine, budget_micros: u32, seed: u64| {
        let decision = engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: {decision:?}"
        );
    };
    // What a decision costs the front plugin, read off one of its own under
    // its frame: a charge at the slow calibration is its fuel.
    let alone = single_engine(Profile {
        name: "front-alone".into(),
        filters: vec![],
        scorers: vec![scorer(&front)],
        picker: Box::new(Best),
    });
    let before = front.stats().fuel_used;
    decide_with_budget(&alone, SLOW_FRAME_MICROS, 0);
    let front_micros =
        u32::try_from((front.stats().fuel_used - before).div_ceil(SLOW_CALIBRATION)).expect("fits");
    let budget_micros = front_micros + LEFT_MICROS;
    let engine = single_engine(Profile {
        name: "front-then-held".into(),
        filters: vec![],
        scorers: vec![scorer(&front), scorer(&held)],
        picker: Box::new(Best),
    });
    hold(
        name,
        std::time::Duration::from_micros(u64::from(budget_micros)),
        2,
    );
    decide_with_budget(&engine, budget_micros, 1);
    let stats = held.stats();
    assert_eq!(
        (
            stats.budget_spent,
            stats.fuel_exhausted,
            stats.traps,
            stats.breaker_opens
        ),
        (1, 0, 0, 0),
        "the remainder clamped the call its bound ended: the decision's budget spent, no failure of the plugin: {stats:?}"
    );
    assert_eq!(
        stats.unread, 1,
        "its sample is slow by the hold alone: counted unread, which arms the thread: {stats:?}"
    );
    // Armed, the thread reads its clock around the next held score: its time
    // is what the thread had, and nothing is counted again.
    decide_with_budget(&engine, budget_micros, 2);
    let stats = held.stats();
    assert_eq!(
        (stats.budget_spent, stats.unread, stats.breaker_opens),
        (2, 1, 0),
        "the second hold is read: {stats:?}"
    );
}

/// At its bound a call is the plugin's, though the host held it: a score
/// the host holds inside its log line for two frames, which then spins
/// until its own whole frame ends it, under a request that states no
/// budget. The run-out is the plugin's failure. Its decision's sample holds
/// the hold, past its two frames, and its fuel is no more than half of
/// them: slow by time nothing read alone, and never counted unread, for the
/// bound ended the call as the plugin's own.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_held_call_its_own_bound_ended_is_the_plugins_and_never_unread() {
    const OWN_FRAME_MICROS: u32 = 30_000;
    let name = "held-then-spins-alone";
    let runtime = runtime();
    let mut held = spec(name, guest::record_guest(), mode::LOG_THEN_SPIN);
    // The score's log line reaches the host's subscriber, where the case
    // holds it.
    held.log.per_second = 1_000;
    // A calibration no host has: the spin's fuel lasts far past its bound,
    // so only the clock ends it.
    held.fuel_per_micro = 20_000_000;
    held.budget_micros = OWN_FRAME_MICROS;
    let held = loaded(&runtime, held);
    let engine = single_engine(Profile {
        name: "held-alone".into(),
        filters: vec![],
        scorers: vec![WeightedScorer {
            scorer: Box::new(WasmScorer::new(held.bind(), None)),
            weight: 1.0,
        }],
        picker: Box::new(Best),
    });
    hold(
        name,
        std::time::Duration::from_micros(u64::from(OWN_FRAME_MICROS) * 2),
        1,
    );
    let decision = engine.decide(
        &RequestFacts {
            budget_micros: u32::MAX,
            ..facts(1)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    let stats = held.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.budget_spent, stats.traps),
        (1, 0, 0),
        "the plugin's own whole frame ran out on the clock: {stats:?}"
    );
    assert_eq!(
        stats.unread, 0,
        "a call its bound ended is never unread, by its verdict or through its decision's sample: {stats:?}"
    );
}

/// A call's frame in its decision's sample is its plugin's own, whatever
/// another plugin's draw left of its grant. A scorer whose constructor burns
/// opens in front of a picker whose constructor burns as much, under a
/// budget that leaves the picker three tenths more than it uses: its
/// decisions take well under half the budget, and sixty-four of them open
/// nothing. Read against what the remainder left of its two grants, each
/// would be past half of them, and the window's first read would open the
/// breaker.
#[test]
fn a_guest_another_plugins_draw_left_little_is_read_against_its_own_frames() {
    let runtime = runtime();
    let scorer = load_slow(
        &runtime,
        "heavy-scorer",
        guest::record_guest(),
        mode::BURN_IN_CONSTRUCTOR,
    );
    let picker = load_slow(
        &runtime,
        "left-little",
        guest::record_guest(),
        mode::BURN_IN_CONSTRUCTOR,
    );
    let engine = single_engine(Profile {
        name: "two".into(),
        filters: vec![],
        scorers: vec![WeightedScorer {
            scorer: Box::new(WasmScorer::new(scorer.bind(), None)),
            weight: 1.0,
        }],
        picker: Box::new(WasmPicker::new(picker.bind(), None)),
    });
    let decide_under = |budget_micros: u32, seed: u64| {
        let decision = engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: {decision:?}"
        );
    };
    // What a decision costs each plugin, read off one under the plugins'
    // own frame: a charge at the slow calibration is its fuel.
    let before = (scorer.stats().fuel_used, picker.stats().fuel_used);
    decide_under(SLOW_FRAME_MICROS, 0);
    let scorer_micros = (scorer.stats().fuel_used - before.0).div_ceil(SLOW_CALIBRATION);
    let picker_micros = (picker.stats().fuel_used - before.1).div_ceil(SLOW_CALIBRATION);
    // The scorer's draw, and thirteen tenths of the picker's.
    let budget_micros = u32::try_from(scorer_micros + picker_micros * 13 / 10).expect("fits");
    let spent_before = picker.stats().budget_spent;
    for seed in 1..=64 {
        decide_under(budget_micros, seed);
    }
    let stats = picker.stats();
    // The picker's own calls ran in each of them: a stage the remainder
    // left nothing would fall back, and its window would read no sample.
    assert_eq!(
        (
            stats.traps,
            stats.fuel_exhausted,
            stats.budget_spent - spent_before,
            stats.breaker_opens
        ),
        (0, 0, 0, 0),
        "under half the budget a decision, whatever the scorer left: {stats:?}"
    );
}

/// Reads `VmRSS` of this process, in KiB.
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

/// This thread's user and system CPU time, in clock ticks.
fn cpu_ticks() -> u64 {
    std::fs::read_to_string("/proc/thread-self/stat")
        .ok()
        .and_then(|s| {
            let after = s.rsplit(") ").next()?;
            let fields: Vec<&str> = after.split_whitespace().collect();
            let utime: u64 = fields.get(11)?.parse().ok()?;
            let stime: u64 = fields.get(12)?.parse().ok()?;
            Some(utime + stime)
        })
        .unwrap_or(0)
}

/// A measurement for the Numbers table, not a check: run in release with
/// `--ignored measure --nocapture`. RSS per kept instance over 64 warmed
/// threads (their stacks included), and the CPU of one deciding thread over
/// 200,000 decisions at 64 candidates, read from the thread's own clock
/// ticks.
#[test]
#[ignore = "a measurement for the Numbers table; run in release with --nocapture"]
#[expect(
    clippy::print_stdout,
    reason = "the measurement reports its rows on stdout under --nocapture"
)]
fn measure_rss_per_kept_instance_and_cpu_per_decision() {
    let runtime = runtime();
    for (name, component) in [
        ("record", guest::record_guest()),
        ("packed", guest::packed_guest()),
    ] {
        let plugin = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        let threads = 64u64;
        let barrier = Arc::new(std::sync::Barrier::new(threads as usize + 1));
        let before = rss_kib();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let plugin = Arc::clone(&plugin);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let warmed = plugin.warm();
                    barrier.wait();
                    barrier.wait();
                    warmed
                })
            })
            .collect();
        barrier.wait();
        let after = rss_kib();
        println!(
            "{name}: RSS per kept instance {} KiB ({threads} instances: {before} -> {after} KiB)",
            after.saturating_sub(before) / threads
        );
        barrier.wait();
        for h in handles {
            h.join().expect("joins").expect("warms");
        }
        let engine = plugin_engine(&plugin);
        let candidates = pool(64);
        let pool_facts = PoolFacts {
            pool_size: 64,
            generation: 1,
            ..PoolFacts::default()
        };
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let decisions = 200_000u64;
        let ticks0 = cpu_ticks();
        let started = std::time::Instant::now();
        for seed in 0..decisions {
            engine.decide(&facts(seed), &pool_facts, &candidates, &cx, &mut scratch);
        }
        let wall = started.elapsed();
        let cpu_ms = (cpu_ticks() - ticks0) * 10;
        let per_decision_us = wall.as_secs_f64() * 1e6 / decisions as f64;
        // CPU per decision times 10,000 per second, as a share of one core.
        let core_share = cpu_ms as f64 / decisions as f64 * 10_000.0 / 1_000.0 * 100.0;
        println!(
            "{name}: {decisions} decisions at 64 candidates: wall {per_decision_us:.2} us each, CPU {cpu_ms} ms in all = {core_share:.2}% of a core at 10,000 decisions/s"
        );
        let stats = plugin.stats();
        println!(
            "{name}: fuel per decision {} (four calls), fuel per microsecond of wall {:.0}",
            stats.fuel_used / decisions,
            stats.fuel_used as f64 / (wall.as_secs_f64() * 1e6)
        );
    }
    // The epoch ticker with no call in flight: its CPU over two idle seconds.
    let idle = std::time::Duration::from_secs(2);
    let ticks0 = ticker_cpu_ticks();
    std::thread::sleep(idle);
    let ticker_ms = (ticker_cpu_ticks() - ticks0) * 10;
    println!(
        "epoch ticker idle: {ticker_ms} ms CPU over {} ms = {:.3}% of a core",
        idle.as_millis(),
        ticker_ms as f64 / idle.as_millis() as f64 * 100.0
    );
}

/// The ids of the process's threads, from `/proc`. Empty where the host
/// does not say.
fn threads() -> Vec<std::ffi::OsString> {
    std::fs::read_dir("/proc/self/task")
        .map(|tasks| tasks.flatten().map(|task| task.file_name()).collect())
        .unwrap_or_default()
}

/// A runtime, and the threads born while it was made: its epoch ticker is
/// one of them. A case beside this one can make a runtime in the same
/// moment, so this one is made again, eight times at most, until no second
/// ticker was born with it.
fn runtime_and_the_threads_born_with_it() -> (Arc<Runtime>, Vec<std::ffi::OsString>) {
    let mut made = 0;
    loop {
        made += 1;
        let before = threads();
        let runtime = runtime();
        // A thread names itself as it starts.
        std::thread::sleep(std::time::Duration::from_millis(1));
        let born: Vec<_> = threads()
            .into_iter()
            .filter(|thread| !before.contains(thread))
            .collect();
        if ticker_waits(&born).len() <= 1 || made == 8 {
            return (runtime, born);
        }
    }
}

/// The `/proc` directory of `thread` where it is an epoch ticker.
fn ticker_task(thread: &std::ffi::OsString) -> Option<std::path::PathBuf> {
    let task = std::path::Path::new("/proc/self/task").join(thread);
    let comm = std::fs::read_to_string(task.join("comm")).ok()?;
    (comm.trim() == "wasm-epoch").then_some(task)
}

/// How long each epoch ticker among `threads` has waited on a runqueue, in
/// nanoseconds, by thread from `/proc`: runnable, and kept from the CPU. A
/// ticker that sleeps or is parked waits on none.
fn ticker_waits(threads: &[std::ffi::OsString]) -> Vec<(std::ffi::OsString, u64)> {
    threads
        .iter()
        .filter_map(|thread| {
            let schedstat = std::fs::read_to_string(ticker_task(thread)?.join("schedstat")).ok()?;
            let waited = schedstat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
            Some((thread.clone(), waited))
        })
        .collect()
}

/// Waits, a second at most, until no epoch ticker among `threads` is
/// runnable. The kernel adds what a thread waited on a runqueue when the
/// thread next has the CPU, so a ticker still waiting reads short. With no
/// call in flight a ticker parks as soon as it has the CPU.
fn tickers_at_rest(threads: &[std::ffi::OsString]) {
    for _ in 0..1_000 {
        let runnable = threads.iter().filter_map(ticker_task).any(|task| {
            std::fs::read_to_string(task.join("stat")).is_ok_and(|stat| {
                stat.rsplit(") ")
                    .next()
                    .is_some_and(|rest| rest.starts_with('R'))
            })
        });
        if !runnable {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// The least any epoch ticker among `threads` waited on a runqueue since
/// `before` was read, read once the tickers are at rest. The case's own
/// ticker is one of them: it waited no less. `None` where the host does not
/// say.
fn least_ticker_wait_since(
    before: &[(std::ffi::OsString, u64)],
    threads: &[std::ffi::OsString],
) -> Option<std::time::Duration> {
    tickers_at_rest(threads);
    ticker_waits(threads)
        .into_iter()
        .map(|(thread, waited)| {
            let earlier = before
                .iter()
                .find(|(known, _)| *known == thread)
                .map_or(0, |(_, earlier)| *earlier);
            std::time::Duration::from_nanos(waited.saturating_sub(earlier))
        })
        .min()
}

/// The CPU ticks (user and system, at 100 per second) of the runtime's epoch
/// ticker thread, from `/proc`.
fn ticker_cpu_ticks() -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|task| {
            let comm = std::fs::read_to_string(task.path().join("comm")).ok()?;
            (comm.trim() == "wasm-epoch").then_some(task.path().join("stat"))
        })
        .filter_map(|stat| {
            let s = std::fs::read_to_string(stat).ok()?;
            let after = s.rsplit(") ").next()?;
            let fields: Vec<&str> = after.split_whitespace().collect();
            let utime: u64 = fields.get(11)?.parse().ok()?;
            let stime: u64 = fields.get(12)?.parse().ok()?;
            Some(utime + stime)
        })
        .sum()
}

/// The pool of four every shared-budget case decides over.
fn four() -> PoolFacts<'static> {
    PoolFacts {
        pool_size: 4,
        generation: 1,
        ..PoolFacts::default()
    }
}

/// How long the calling thread was off the CPU since `wall` and `cpu` were
/// read together: the wall clock's time less the thread's own.
fn off_cpu_since(wall: std::time::Instant, cpu: std::time::Duration) -> std::time::Duration {
    wall.elapsed()
        .saturating_sub(bound_clock().saturating_sub(cpu))
}

/// What a hold of the runner must reach to move a charge at the slow
/// calibration. A call there is charged its fuel, near 118 ms a burn, until
/// its time on the wall passes that; a cheap call's charge moves sooner. The
/// cases that count charges over many decisions read the longest any one
/// decision's thread was off the CPU against a quarter of a burn, inside
/// each one's slack.
const A_HOLD_THAT_MOVES_A_CHARGE: std::time::Duration = std::time::Duration::from_millis(25);

/// Whether a try that read other counts than the made ones is made again:
/// under such a hold, before the last try. It says so in one line.
fn held_counts_are_made_again(
    case: &str,
    stood: bool,
    worst: std::time::Duration,
    made: u32,
) -> bool {
    if stood || worst < A_HOLD_THAT_MOVES_A_CHARGE || made >= TRIES {
        return false;
    }
    made_again(
        case,
        "a decision's thread was off the CPU a quarter of a burn, which moves a charge at the slow calibration",
    );
    true
}

/// Makes a case its tries: `once` is one try, told its number, and says
/// whether it stood. A try a hold of the runner reached says so (see
/// [`made_again`]) and did not stand. The last try asserts whatever its
/// witness says: one try stands, or the case fails. A try reads the host's
/// lines from where it began ([`warnings`]).
fn tried(once: impl Fn(u32) -> bool) {
    assert!(
        (1..=TRIES).any(|made| {
            TRY_BEGAN.set(said().lock().expect("the kept lines").len());
            once(made)
        }),
        "the last try asserts, so one try stands"
    );
}

thread_local! {
    /// How many lines were kept when this thread's try began.
    static TRY_BEGAN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether the close rebuilt the instance a fault dropped: asserted, unless
/// the decision is to be made again. The rebuild is an `init`, on the wall
/// clock, and the `init` the load made inside the same frame. One the host
/// ended at that frame's deadline was ended by the runner, with the thread
/// off the CPU or slowed on its core, and the host says so in a warning
/// ([`the_frame_ended_the_rebuild`]). Such a decision says so and is made
/// again, [`TRIES`] at most (`made` counts them), whatever the thread's
/// clock read: a thread slowed on its core reads there as on the CPU, and
/// `held` is what it read, for the message. A rebuild that fails another
/// way is the host's, and so is one that fails in the last try.
fn the_close_rebuilt(case: &str, plugin: &Plugin, held: std::time::Duration, made: u32) -> bool {
    if !plugin.is_warm() && the_frame_ended_the_rebuild(plugin.name()) && made < TRIES {
        made_again(
            case,
            "the host ended the rebuild's init at the init frame's deadline, on the wall clock",
        );
        return false;
    }
    assert!(
        plugin.is_warm(),
        "{case}: the instance is dropped and the close rebuilds the slot (try {made}, the thread off the CPU {held:?}); \
         the plugin read {:?}; the host said {:?}",
        plugin.stats(),
        warnings(plugin.name())
    );
    true
}

/// Whether the host said that it ended a rebuild of `plugin`'s dropped
/// instance at the init frame's deadline, and of no rebuild of that plugin
/// that it failed another way. The lines are those kept since the try
/// began: a case's tries load one name, and an earlier try's ended rebuild
/// is not this one's.
fn the_frame_ended_the_rebuild(plugin: &str) -> bool {
    let rebuilds: Vec<String> = warnings(plugin)
        .into_iter()
        .filter(|line| line.contains("the rebuild of a dropped instance failed"))
        .collect();
    !rebuilds.is_empty()
        && rebuilds
            .iter()
            .all(|line| line.contains("wasm trap: interrupt"))
}

/// The warnings the host wrote that name `plugin` since the thread's try
/// began (all of them, on a thread that makes no try), where a case
/// installed the process's subscriber before them: what a failed rebuild
/// said.
fn warnings(plugin: &str) -> Vec<String> {
    let named = format!("plugin=\"{plugin}\"");
    said()
        .lock()
        .expect("the kept lines")
        .iter()
        .skip(TRY_BEGAN.get())
        .filter(|line| line.contains(&named))
        .cloned()
        .collect()
}

/// A try reads the lines the host wrote since it began: an ended rebuild
/// that an earlier try's line tells of is not this try's, so the second
/// try here reads none.
#[test]
fn a_try_reads_the_hosts_lines_since_it_began() {
    const NAME: &str = "a-name-no-case-loads";
    let ended =
        format!("plugin=\"{NAME}\" the rebuild of a dropped instance failed: wasm trap: interrupt");
    tried(|made| {
        if made == 1 {
            said().lock().expect("the kept lines").push(ended.clone());
            assert!(
                the_frame_ended_the_rebuild(NAME),
                "the first try's own line"
            );
            return false;
        }
        assert!(
            !the_frame_ended_the_rebuild(NAME),
            "a line of the first try, read in the second"
        );
        true
    });
}

/// Decides once, and says with the decision whether its thread kept the
/// CPU: its time on the wall clock passed the thread's CPU time over it by
/// less than a quarter of the budget.
fn decide_on_the_cpu(decide: impl FnOnce() -> Decision) -> (Decision, bool) {
    let hold = std::time::Duration::from_micros(u64::from(FRAME_MICROS) / 4);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = decide();
    (decision, off_cpu_since(wall, cpu) < hold)
}

/// Decides until `plugin`'s breaker has opened; says how many decisions that
/// took, and whether each of them kept its thread on the CPU (its time on
/// the wall clock passed the thread's CPU time over it by less than a
/// quarter of the budget). The latency window is sixty-four SAMPLES, and a
/// decision that charged the plugin nothing writes none: a hold of the
/// runner inside an earlier call can spend a ten-millisecond budget before
/// the plugin's first call. So a case that waits for the window waits for
/// what opens it, never for sixty-four decisions, and asserts its counts only
/// where no hold can have moved them. A hold under a quarter of the budget
/// leaves the plugin its call and a sample past half the budget. The
/// plugins' counters do not say that much: a hold that leaves a call short
/// of late is in neither where nothing read the call, and can spend what a
/// later call needed. On a host with no thread clock the two clocks are one
/// and the word is always yes. Red past 256 decisions.
fn decide_until_open(plugin: &Plugin, decide: impl Fn(u64) -> Decision) -> (u64, bool) {
    let (mut decisions, mut on_the_cpu) = (0, true);
    while plugin.stats().breaker_opens == 0 {
        decisions += 1;
        assert!(
            decisions <= 256,
            "no opening in 256 decisions: {:?}",
            plugin.stats()
        );
        let (decision, kept) = decide_on_the_cpu(|| decide(decisions));
        on_the_cpu &= kept;
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
    }
    (decisions, on_the_cpu)
}

/// A count a host can excuse says when it was excused: the one line of a
/// run that does not assert it, with the word that failed (read under
/// `--nocapture`), so a row of runs can say in how many the count was held.
#[expect(
    clippy::print_stdout,
    reason = "a run that excuses a count reports it on stdout under --nocapture"
)]
fn excused(case: &str, why: &str) {
    println!("{case}: a count is not asserted in this run: {why}");
}

/// A try a hold of the runner can have ended says so: the one line of a try
/// that asserts nothing and is made again (read under `--nocapture`), so a
/// row of runs can say in how many a hold reached the case.
#[expect(
    clippy::print_stdout,
    reason = "a try that is made again reports it on stdout under --nocapture"
)]
fn made_again(case: &str, why: &str) {
    println!("{case}: made again in this run: {why}");
}

/// A run its host put in another lawful world says so: the one line of a
/// run that asserted that world's counts in place of the made one's, with
/// what the case's witness read (under `--nocapture`), so a row of runs can
/// say in how many the host made which world.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[expect(
    clippy::print_stdout,
    reason = "a run in another lawful world reports it on stdout under --nocapture"
)]
fn other_world(case: &str, what: &str) {
    println!("{case}: this run asserted another lawful world: {what}");
}

/// What a run on a host with no thread clock says when a hold opened the
/// honest neighbour's breaker. No thread arms there: a call's two ends are
/// one, a held call is its plugin's time, and a window of holds short of
/// late inside the neighbour's own calls opens it.
const A_HOLD_OPENED_THE_NEIGHBOUR: &str =
    "on a host with no thread clock a hold opened the honest neighbour's breaker";

/// The honest neighbour's breaker after a window case. Where the host reads
/// a thread's clock no hold opens it: a hold inside the neighbour's own
/// call is a sample slow by time nothing read, the first arms its thread
/// and the rest are read, so it is asserted shut in every run, hold or no
/// hold. On a host with no thread clock it is asserted where the case's
/// witness saw no hold (`quiet`), and a run a hold moved says when it
/// opened ([`excused`]).
fn the_neighbour_stays(case: &str, quiet: bool, neighbour: &Plugin) {
    let stats = neighbour.stats();
    if quiet || cfg!(any(target_os = "linux", target_os = "macos")) {
        assert_eq!(
            stats.breaker_opens, 0,
            "{case}: the honest neighbour stays: {stats:?}"
        );
    } else if stats.breaker_opens != 0 {
        excused(case, A_HOLD_OPENED_THE_NEIGHBOUR);
    }
}

/// Whether a run asserts a count a hold of the runner's own can move: every
/// decision kept its thread on the CPU (`on_the_cpu`, the case's own
/// witness) and no call was counted held on these plugins (`stalls`). Then
/// no hold moved a route, and the route is the case's to assert; otherwise
/// the run says so ([`excused`]). The unread count is not read here: it
/// takes a plugin's own first slow sample of a window too, a spinner's in
/// every run, and that arms its thread as a late call would. A hold that
/// only makes a call at its bound late does no more, and a hold long enough
/// to move a count is off the CPU a quarter of the budget.
fn no_hold(case: &str, on_the_cpu: bool, plugins: &[&Plugin]) -> bool {
    let held: u64 = plugins.iter().map(|plugin| plugin.stats().stalls).sum();
    if !on_the_cpu {
        excused(
            case,
            "a decision's thread was off the CPU a quarter of the budget",
        );
    } else if held != 0 {
        excused(case, &format!("{held} held call(s) counted"));
    }
    on_the_cpu && held == 0
}

/// The clock a call's bound counts on this host, from an origin of its own:
/// the thread's CPU time where the host has a thread clock, the wall clock
/// elsewhere.
fn bound_clock() -> std::time::Duration {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let now = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        std::time::Duration::new(
            u64::try_from(now.tv_sec).unwrap_or(0),
            u32::try_from(now.tv_nsec).unwrap_or(0),
        )
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        ORIGIN.get_or_init(std::time::Instant::now).elapsed()
    }
}

/// Case (a) of the shared budget, in both scope orders: an honest scorer and
/// a scorer that spins, both with a frame of the whole budget. Neither
/// order charges the spinner on the breaker's failure count: it never consumed
/// the budget alone (the other's constructor spent first), so each decision
/// is the decision's budget spent. Its stage samples score the remainder it
/// took of the budget, the ring survives its rebuilds, and the window (64
/// samples) opens its breaker; the honest scorer's never opens. The case
/// decides until that opening: a decision whose budget a hold of the runner
/// spent before the spinner's constructor writes no sample.
#[test]
fn a_spinner_behind_an_honest_scorer_opens_its_breaker_at_the_window_in_either_order() {
    for spinner_first in [false, true] {
        let runtime = runtime();
        let honest = load(
            &runtime,
            "honest",
            guest::record_guest(),
            mode::LEAST_IN_FLIGHT,
        );
        let spinner = load(
            &runtime,
            "spinner",
            guest::record_guest(),
            mode::SPIN_IN_SCORE,
        );
        let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
            scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
            weight: 1.0,
        };
        let scorers = if spinner_first {
            vec![scorer(&spinner), scorer(&honest)]
        } else {
            vec![scorer(&honest), scorer(&spinner)]
        };
        let engine = single_engine(Profile {
            name: "shared".into(),
            filters: vec![],
            scorers,
            picker: Box::new(Best),
        });
        let decide = |seed: u64| {
            engine.decide(
                &timed(seed),
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        let (decisions, on_the_cpu) = decide_until_open(&spinner, decide);
        let s = spinner.stats();
        assert_eq!(
            (s.traps, s.breaker_opens),
            (0, 1),
            "spinner first {spinner_first}: the spinner's breaker opened once, on no trap: {s:?}"
        );
        let quiet = no_hold(
            "a spinner behind an honest scorer",
            on_the_cpu,
            &[&spinner, &honest],
        );
        if quiet {
            assert_eq!(
                s.answered, 64,
                "spinner first {spinner_first}: the window opened at the spinner's 64th sample, one a decision that reached its constructor: {s:?}"
            );
            assert_eq!(
                decisions, 64,
                "spinner first {spinner_first}: no hold moved the run, so every decision reached the spinner"
            );
            assert_eq!(
                s.fuel_exhausted, 0,
                "spinner first {spinner_first}: the window opened the spinner's breaker, no charge: {s:?}"
            );
            assert!(
                s.budget_spent >= 64,
                "spinner first {spinner_first}: every decision was the budget spent: {s:?}"
            );
        }
        the_neighbour_stays("a spinner behind an honest scorer", quiet, &honest);
        decide(decisions + 1);
        assert!(
            spinner.stats().bypassed >= 1,
            "spinner first {spinner_first}: open, the spinner is bypassed"
        );
    }
}

/// Cases (b) and (d): three honest scorers at a quarter of the budget each,
/// then a scorer that spins. The spinner takes the last quarter and a half,
/// more than any one of them and less than its own frame: the decision's
/// budget is spent, visibly in its counters, and nobody is charged (no share
/// rule: under a shared budget the design promises a bounded decision and
/// the per-plugin counters, not a culprit); its samples, a third of the
/// budget, open nothing over a window.
#[test]
fn three_honest_quarters_and_a_spinner_charge_nobody() {
    tried(three_honest_quarters_and_a_spinner_charge_nobody_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn three_honest_quarters_and_a_spinner_charge_nobody_in_a_try(made: u32) -> bool {
    const CASE: &str = "three honest quarters and a spinner";
    let runtime = runtime();
    let burners: Vec<Arc<Plugin>> = (0..3)
        .map(|i| {
            load_slow(
                &runtime,
                &format!("quarter-{i}"),
                guest::record_guest(),
                mode::BURN_IN_STAGES,
            )
        })
        .collect();
    let mut spinner_spec = spec("spinner", guest::record_guest(), mode::SPIN_IN_SCORE);
    spinner_spec.fuel_per_micro = SLOW_CALIBRATION;
    spinner_spec.budget_micros = SLOW_FRAME_MICROS;
    let fuel_per_micro = spinner_spec.fuel_per_micro;
    let spinner = loaded(&runtime, spinner_spec);
    // One burn's fuel, from a burner alone at a large budget.
    let before = burners[0].stats().fuel_used;
    plugin_engine(&burners[0]).decide(
        &RequestFacts {
            budget_micros: 1_000_000,
            ..facts(1)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let burn = (burners[0].stats().fuel_used - before) / 3;
    let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
        scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
        weight: 1.0,
    };
    let engine = single_engine(Profile {
        name: "quarters".into(),
        filters: vec![],
        scorers: vec![
            scorer(&burners[0]),
            scorer(&burners[1]),
            scorer(&burners[2]),
            scorer(&spinner),
        ],
        picker: Box::new(Best),
    });
    // Four burns and a half, plus a little for the constructors.
    let budget_micros = u32::try_from((burn * 9 / 2).div_ceil(fuel_per_micro) + 40).expect("fits");
    let mut worst = std::time::Duration::ZERO;
    for seed in 1..=65 {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        worst = worst.max(off_cpu_since(wall, cpu));
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
    }
    let s = spinner.stats();
    let stood = (
        s.fuel_exhausted,
        s.traps,
        s.breaker_opens,
        s.bypassed,
        s.budget_spent,
    ) == (0, 0, 0, 0, 65)
        && burners.iter().all(|burner| {
            let b = burner.stats();
            (b.fuel_exhausted, b.budget_spent, b.breaker_opens) == (0, 0, 0)
        });
    if held_counts_are_made_again(CASE, stood, worst, made) {
        return false;
    }
    assert_eq!(
        (s.fuel_exhausted, s.traps, s.breaker_opens, s.bypassed),
        (0, 0, 0, 0),
        "the spinner is charged nothing and stays: {s:?}"
    );
    assert_eq!(
        s.budget_spent, 65,
        "every decision was the budget spent: {s:?}"
    );
    for burner in &burners {
        let b = burner.stats();
        assert_eq!(
            (b.fuel_exhausted, b.budget_spent, b.breaker_opens),
            (0, 0, 0),
            "an honest quarter is untouched: {b:?}"
        );
    }
    true
}

/// Case (e): a plugin whose `init` outlasts a small decision budget. The
/// remainder clamps the cold slot's `init` below its cost: the decision's
/// budget is spent (the clamp drawn whole, so a neighbour opened after it
/// finds the remainder gone), the breaker (which judges `init` under its own
/// frame) is untouched, and the close rebuilds the slot under the full frame,
/// so the next decision finds it warm. Five such decisions open nothing.
#[test]
fn an_init_the_remainder_clamps_is_the_decisions_budget_spent_and_the_close_rebuilds() {
    tried(
        an_init_the_remainder_clamps_is_the_decisions_budget_spent_and_the_close_rebuilds_in_a_try,
    );
}

/// One try of the case above, told its number: whether it stood.
fn an_init_the_remainder_clamps_is_the_decisions_budget_spent_and_the_close_rebuilds_in_a_try(
    made: u32,
) -> bool {
    const CASE: &str = "an init the remainder clamps";
    // The host's warnings are kept from here on: a rebuild that fails says why.
    said();
    let runtime = runtime();
    let heavy = load(
        &runtime,
        "heavy-init",
        guest::record_guest(),
        mode::BURN_IN_INIT,
    );
    let neighbour = load(
        &runtime,
        "after-the-init",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let bound = heavy.bind();
    let engine = single_engine(Profile {
        name: "clamped-init".into(),
        filters: vec![Box::new(WasmFilter::new(bound.clone(), None))],
        scorers: vec![
            WeightedScorer {
                scorer: Box::new(WasmScorer::new(bound.clone(), None)),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(WasmScorer::new(neighbour.bind(), None)),
                weight: 1.0,
            },
        ],
        picker: Box::new(WasmPicker::new(bound, None)),
    });
    for seed in 1..=5 {
        heavy.forget_instance();
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: 200,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let held = off_cpu_since(wall, cpu);
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
        if !the_close_rebuilt(CASE, &heavy, held, made) {
            return false;
        }
    }
    let s = heavy.stats();
    assert_eq!(
        (s.budget_spent, s.breaker_opens, s.fuel_exhausted, s.traps),
        (5, 0, 0, 0),
        "a clamped init is the decision's budget spent, never a failure: {s:?}"
    );
    assert_eq!(
        s.lazy_instantiations, 5,
        "one rebuild per decision, at its close: {s:?}"
    );
    let n = neighbour.stats();
    assert_eq!(
        (n.budget_spent, n.answered, n.breaker_opens),
        (5, 0, 0),
        "the clamp drew the whole remainder: the neighbour's open finds it spent: {n:?}"
    );
    true
}

/// A clock the test steers: at `0` it answers zero at once; at any other
/// value it naps, then answers one less than the value. The nap is a host
/// call's wall time, which costs the guest that waits for it no fuel.
fn steered_runtime(steer: &Arc<AtomicU8>, nap: std::time::Duration) -> Arc<Runtime> {
    steered_runtime_counting(steer, nap).0
}

/// [`steered_runtime`] with the count of its naps: a steered decision's
/// thread is off the CPU for each of them, and what it was off the CPU past
/// them is what a runner held it.
fn steered_runtime_counting(
    steer: &Arc<AtomicU8>,
    nap: std::time::Duration,
) -> (Arc<Runtime>, Arc<AtomicU32>) {
    let steer = Arc::clone(steer);
    let naps = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&naps);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        match steer.load(Ordering::Relaxed) {
            0 => 0,
            value => {
                std::thread::sleep(nap);
                counted.fetch_add(1, Ordering::Relaxed);
                u64::from(value - 1)
            }
        }
    })
    .expect("a runtime");
    (Arc::new(runtime), naps)
}

/// A plugin whose `init` asks the steered clock once, loaded while the clock
/// answers at once; `tweak` sets the rest of its row.
fn plugin_whose_init_asks(
    nap_millis: u64,
    tweak: impl FnOnce(&mut PluginSpec),
) -> (Arc<AtomicU8>, Arc<Runtime>, Arc<Plugin>) {
    let clock = Arc::new(AtomicU8::new(0));
    let runtime = steered_runtime(&clock, std::time::Duration::from_millis(nap_millis));
    let mut asks = spec("init-asks", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    asks.config.push(mode::INIT_ASKS_THE_CLOCK);
    tweak(&mut asks);
    let plugin = loaded(&runtime, asks);
    (clock, runtime, plugin)
}

/// One decision under `budget_micros` on a cold slot of `cold` while the
/// clock is at `steer`: the plugin's three stages with a neighbour's score
/// behind them, no fallback named. Says how long the decision's thread was
/// off the CPU in it: the steered clock's naps, and what a runner held it.
fn decide_on_a_cold_slot(
    clock: &AtomicU8,
    steer: u8,
    cold: &Arc<Plugin>,
    neighbour: &Arc<Plugin>,
    budget_micros: u32,
) -> std::time::Duration {
    let bound = cold.bind();
    let engine = single_engine(Profile {
        name: "cold-init".into(),
        filters: vec![Box::new(WasmFilter::new(bound.clone(), None))],
        scorers: vec![
            WeightedScorer {
                scorer: Box::new(WasmScorer::new(bound.clone(), None)),
                weight: 1.0,
            },
            WeightedScorer {
                scorer: Box::new(WasmScorer::new(neighbour.bind(), None)),
                weight: 1.0,
            },
        ],
        picker: Box::new(WasmPicker::new(bound, None)),
    });
    cold.forget_instance();
    clock.store(steer, Ordering::Relaxed);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let decision = engine.decide(
        &RequestFacts {
            budget_micros,
            ..facts(1)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let off_the_cpu = off_cpu_since(wall, cpu);
    clock.store(0, Ordering::Relaxed);
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
    );
    off_the_cpu
}

/// An `init` its fuel does not end is ended at its bound on the wall clock.
/// Nothing watches an `init` (a cold slot's is charged the time it took), so
/// its deadline ends it: the arm every call takes on a host with no thread
/// clock. The guest's `init` spins without end under a calibration that gives
/// its frame of a hundred milliseconds the fuel of seconds: the load fails on
/// the clock, no sooner than the frame and long before the fuel.
#[test]
fn an_init_its_fuel_does_not_end_is_ended_at_its_bound_on_the_wall_clock() {
    // The clock's answer of two makes the guest's `init` spin.
    let clock = Arc::new(AtomicU8::new(3));
    let runtime = steered_runtime(&clock, std::time::Duration::ZERO);
    let mut spins = spec("init-spins", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    spins.config.push(mode::INIT_ASKS_THE_CLOCK);
    spins.fuel_per_micro = 100_000;
    let (started, cpu) = (std::time::Instant::now(), bound_clock());
    let err = Plugin::load(&runtime, spins).expect_err("the init was ended");
    let (took, spun) = (started.elapsed(), bound_clock().saturating_sub(cpu));
    assert!(
        matches!(&err, LoadError::Trap { reason, .. } if reason.contains("interrupt")),
        "ended on the clock, not on its fuel: {err}"
    );
    // No sooner than its frame on the wall clock, and long before its fuel:
    // the fuel is seconds of spinning, read on the thread's own clock, which
    // a runner's hold of the thread does not move.
    assert!(
        took >= std::time::Duration::from_millis(100) && spun < std::time::Duration::from_secs(2),
        "at its frame of a hundred milliseconds on the wall clock: {took:?} on the wall, {spun:?} on the thread's clock"
    );
}

/// A cold slot's `init` that is long on the clock and cheap in fuel: one host
/// call the host takes 20 ms to answer, under a 10 ms decision. The budget is
/// wall time for `init` as for a call, so the decision's budget is spent: the
/// plugin's own constructor and the neighbour opened behind it find no
/// remainder. The instance is kept and nothing counts on the breaker.
#[test]
fn a_lazy_init_long_on_the_clock_is_the_decisions_budget_spent() {
    let (clock, runtime, cold) = plugin_whose_init_asks(20, |_| {});
    let neighbour = load(
        &runtime,
        "after-the-init",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    decide_on_a_cold_slot(&clock, 1, &cold, &neighbour, FRAME_MICROS);
    let s = cold.stats();
    assert_eq!(
        (
            s.lazy_instantiations,
            s.answered,
            s.budget_spent,
            s.breaker_opens
        ),
        (1, 0, 1, 0),
        "the init's 20 ms are the 10 ms decision's budget spent: {s:?}"
    );
    assert!(cold.is_warm(), "the instance the init made is kept");
    let n = neighbour.stats();
    assert_eq!(
        (n.answered, n.budget_spent),
        (0, 1),
        "the neighbour's open finds the remainder gone: {n:?}"
    );
}

/// The same `init` refusing after its 20 ms: the failure is the breaker's,
/// and the time it took is the decision's all the same.
#[test]
fn a_lazy_init_that_fails_long_on_the_clock_is_the_decisions_budget_spent_too() {
    let (clock, runtime, cold) = plugin_whose_init_asks(20, |_| {});
    let neighbour = load(
        &runtime,
        "after-the-init",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    decide_on_a_cold_slot(&clock, 2, &cold, &neighbour, FRAME_MICROS);
    let s = cold.stats();
    assert_eq!(
        (s.lazy_instantiations, s.answered, s.breaker_opens),
        (0, 0, 0),
        "the init refused: one failure, short of the breaker's five: {s:?}"
    );
    assert!(!cold.is_warm(), "a refused init leaves no instance");
    let n = neighbour.stats();
    assert_eq!(
        (n.answered, n.budget_spent),
        (0, 1),
        "the neighbour's open finds the remainder gone: {n:?}"
    );
}

/// The fuel side of the same rule, at a calibration a thousand times below
/// this host's speed: an `init` that spins its whole frame out in a moment of
/// wall time is charged its fuel at the calibration, the 100 ms of the init
/// frame, which is the whole of a 100 ms decision. One failure on the breaker
/// (the frame it ran out under was its own), and the neighbour's open finds
/// the remainder gone. An `init` the wall clock ended with its thread off
/// the CPU burnt no frame: such a decision says so and is made again,
/// [`TRIES`] at most.
#[test]
fn a_lazy_init_that_burns_its_frame_out_is_charged_its_fuel_at_the_calibration() {
    let mut made = 0;
    loop {
        made += 1;
        let (clock, runtime, cold) = plugin_whose_init_asks(0, |row| {
            row.fuel_per_micro = SLOW_CALIBRATION;
            row.budget_micros = SLOW_FRAME_MICROS;
        });
        let neighbour = load_slow(
            &runtime,
            "after-the-init",
            guest::record_guest(),
            mode::LEAST_IN_FLIGHT,
        );
        let before = cold.stats().fuel_used;
        let held = decide_on_a_cold_slot(&clock, 3, &cold, &neighbour, 100_000);
        let s = cold.stats();
        // The init spins its frame out in a moment of wall time unless the
        // runner keeps its thread off the CPU: then its deadline on the wall
        // clock ends it with its fuel unburnt.
        if s.fuel_used - before < 99_000 && held >= INIT_FRAME / 2 && made < TRIES {
            made_again(
                "a lazy init that burns its frame out",
                "the decision's thread was off the CPU half the init frame, the init's bound on the wall clock",
            );
            continue;
        }
        assert!(
            s.fuel_used - before >= 99_000,
            "the fuel the failed init burnt is in the plugin's count (try {made}, the thread off the CPU {held:?}): {s:?}"
        );
        assert_eq!(
            (s.lazy_instantiations, s.answered, s.breaker_opens),
            (0, 0, 0),
            "the init ran its own frame out: one failure, short of the breaker's five: {s:?}"
        );
        assert!(!cold.is_warm(), "an init that ran out leaves no instance");
        let n = neighbour.stats();
        assert_eq!(
            (n.answered, n.budget_spent),
            (0, 1),
            "the init's fuel at the calibration is the whole 100 ms: {n:?}"
        );
        break;
    }
}

/// The fuel side on the arm that keeps its instance, at a calibration a
/// hundred times below this host's speed: a decision whose whole budget is
/// the fuel an `init` of eight fixed loops takes, which the load measured.
/// The init fits and is kept; its fuel at the calibration is the whole
/// budget, a hundred times its moment on the clock, so the plugin's own
/// constructor and the neighbour behind it find no remainder.
#[test]
fn a_kept_lazy_init_is_charged_its_fuel_at_the_calibration() {
    let runtime = runtime();
    let mut row = spec("kept-init", guest::record_guest(), mode::BURN_IN_INIT);
    row.fuel_per_micro = 10;
    let cold = loaded(&runtime, row);
    // The load ran `init` once; the same code takes the same fuel every time.
    let loaded = cold.stats();
    let init_micros = u32::try_from(loaded.fuel_used.div_ceil(10)).expect("fits");
    let neighbour = load(
        &runtime,
        "after-the-init",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    decide_on_a_cold_slot(&AtomicU8::new(0), 0, &cold, &neighbour, init_micros);
    let s = cold.stats();
    assert_eq!(
        (
            s.instantiations - loaded.instantiations,
            s.lazy_instantiations,
            s.answered,
            s.budget_spent,
            s.breaker_opens
        ),
        (1, 1, 0, 1, 0),
        "the kept init's fuel at the calibration is the whole {init_micros} us budget: {s:?}"
    );
    assert!(cold.is_warm(), "the instance the init made is kept");
    let n = neighbour.stats();
    assert_eq!(
        (n.answered, n.budget_spent),
        (0, 1),
        "the neighbour's open finds the remainder gone: {n:?}"
    );
}

/// The rule's operator on the same arm: the larger of the two terms, not
/// their sum. At a calibration a hundred times below this host's speed an
/// `init` that burns eight fixed loops costs its fuel, far above the 20 ms
/// it waits for the host. Under a budget of that fuel and half the wait the
/// decision has the half left, so the plugin's own calls and the neighbour
/// answer; charged the sum of the two it would have none.
#[test]
fn a_kept_lazy_init_is_charged_the_larger_of_its_terms_not_their_sum() {
    const CASE: &str = "a kept lazy init is charged the larger of its terms";
    const WAIT: std::time::Duration = std::time::Duration::from_millis(20);
    tried(|made| {
        let clock = Arc::new(AtomicU8::new(0));
        let (runtime, naps) = steered_runtime_counting(&clock, WAIT);
        let mut row = spec("burns-and-waits", guest::record_guest(), mode::BURN_IN_INIT);
        row.config.push(mode::INIT_ASKS_THE_CLOCK);
        row.fuel_per_micro = 10;
        let cold = loaded(&runtime, row);
        // The load ran `init` once, the clock answering at once: its fuel at the
        // calibration, the same every time.
        let init_micros = u32::try_from(cold.stats().fuel_used.div_ceil(10)).expect("fits");
        let neighbour = load(
            &runtime,
            "after-the-init",
            guest::record_guest(),
            mode::LEAST_IN_FLIGHT,
        );
        let off_the_cpu = decide_on_a_cold_slot(&clock, 1, &cold, &neighbour, init_micros + 10_000);
        // The init's time on the wall clock is its other term, and a runner that
        // holds the thread adds to it. What the thread was off the CPU past the
        // clock's naps is that hold: half of what the fuel term has over the
        // wait brings the two terms near.
        let over_the_wait =
            std::time::Duration::from_micros(u64::from(init_micros)).saturating_sub(WAIT);
        let held =
            off_the_cpu.saturating_sub(WAIT * naps.load(Ordering::Relaxed)) >= over_the_wait / 2;
        let (s, n) = (cold.stats(), neighbour.stats());
        let read = (
            (
                s.lazy_instantiations,
                s.answered,
                s.budget_spent,
                s.breaker_opens,
            ),
            (n.answered, n.budget_spent),
        );
        if read != ((1, 4, 0, 0), (2, 0)) && held && made < TRIES {
            made_again(
                CASE,
                "the decision's thread was off the CPU half of what the init's fuel has over its wait",
            );
            return false;
        }
        assert_eq!(
            read.0,
            (1, 4, 0, 0),
            "the init's fuel is the larger term and leaves half the wait (try {made}, a hold read: {held}): {s:?}"
        );
        assert_eq!(
            read.1,
            (2, 0),
            "the neighbour answers under what is left (try {made}, a hold read: {held}): {n:?}"
        );
        true
    });
}

/// A lazy init's time is part of its decision's latency sample, however the
/// init ends, at its two ends like a call's. This `init` waits 3 ms on a
/// host call on every cold slot, under a 4 ms budget: nothing watches an
/// `init`, so on a thread nothing armed its sample is the time it took,
/// past half the budget, and its low end is its fuel, nowhere near it. The
/// first is slow by time nothing read alone: counted with the unread calls,
/// it arms the thread, and the thread's clock is read around the sixty-three
/// after it, whose time is what the thread had. The window of sixty-four
/// holds one slow sample and opens nothing, over inits that keep their
/// instance and over inits that refuse (a threshold of a hundred failures
/// keeps their count out of it). Before that rule a wait the host serves
/// was the plugin's sixty-four times, and the window opened. That it stays
/// shut is asserted where no cold decision's thread had half the budget on
/// the CPU: an `init` that long by a reading is slow by it.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_lazy_init_the_host_holds_on_every_cold_slot_is_one_slow_sample_however_it_ends() {
    const BUDGET_MICROS: u32 = 4_000;
    for steer in [1, 2] {
        let (clock, _runtime, plugin) = plugin_whose_init_asks(3, |row| {
            row.breaker = BreakerSpec {
                failures: 100,
                ..BreakerSpec::default()
            };
        });
        let engine = plugin_engine(&plugin);
        let half = std::time::Duration::from_micros(u64::from(BUDGET_MICROS) / 2);
        let mut cheap = true;
        for seed in 1..=64 {
            plugin.forget_instance();
            clock.store(steer, Ordering::Relaxed);
            let cpu = bound_clock();
            let decision = engine.decide(
                &RequestFacts {
                    budget_micros: BUDGET_MICROS,
                    ..facts(seed)
                },
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            cheap &= bound_clock().saturating_sub(cpu) < half;
            clock.store(0, Ordering::Relaxed);
            assert!(matches!(decision, Decision::Pick(_)), "steer {steer}");
        }
        let s = plugin.stats();
        assert!(
            s.unread >= 1,
            "steer {steer}: the first held init's sample was slow by time nothing read: {s:?}"
        );
        if cheap {
            assert_eq!(
                s.breaker_opens, 0,
                "steer {steer}: one slow sample, the first: the inits after it were read: {s:?}"
            );
        } else {
            excused(
                "a lazy init the host holds on every cold slot",
                "a cold decision's thread had half the budget on the CPU",
            );
        }
    }
}

/// The second half of that rule alone: on an armed thread an `init` is read
/// like a call. A score the host holds past its frame is late and unread,
/// and it arms the thread by the late call's own rule, whatever a sample
/// says. Each of the sixty-three cold slots after it waits 3 ms on a host
/// call in `init`, under a 4 ms budget: the thread's clock is read around
/// each, its time is what the thread had, and none is a slow sample. The
/// window of sixty-four opens nothing and the unread tally stays at the one
/// late call. Without the reading each is sampled at the time it took, past
/// half the budget, and the window opens at its first read. That it stays
/// shut is asserted where no cold decision's thread had half the budget on
/// the CPU: an `init` that long by a reading is slow by it.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_held_init_on_a_thread_a_late_call_armed_is_no_slow_sample() {
    const BUDGET_MICROS: u32 = 4_000;
    let name = "init-asks-armed";
    let clock = Arc::new(AtomicU8::new(0));
    let runtime = steered_runtime(&clock, std::time::Duration::from_millis(3));
    let mut row = spec(name, guest::record_guest(), mode::LOG_IN_SCORE);
    row.config.push(mode::INIT_ASKS_THE_CLOCK);
    // The held score's log line reaches the host's subscriber.
    row.log.per_second = 1_000;
    let plugin = loaded(&runtime, row);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    // The first decision, on the warm slot: its score is held for three
    // frames, late, with no reading of its start.
    hold(
        name,
        3 * std::time::Duration::from_micros(u64::from(FRAME_MICROS)),
        1,
    );
    let decision = decide(&engine, &candidates, 0);
    assert!(matches!(decision, Decision::Pick(_)), "the held decision");
    assert_eq!(
        plugin.stats().unread,
        1,
        "the late score was unread, and it armed the thread: {:?}",
        plugin.stats()
    );
    let half = std::time::Duration::from_micros(u64::from(BUDGET_MICROS) / 2);
    let mut cheap = true;
    for seed in 1..=63 {
        plugin.forget_instance();
        clock.store(1, Ordering::Relaxed);
        let cpu = bound_clock();
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: BUDGET_MICROS,
                ..facts(seed)
            },
            &four(),
            &candidates,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        cheap &= bound_clock().saturating_sub(cpu) < half;
        clock.store(0, Ordering::Relaxed);
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    }
    let s = plugin.stats();
    if cheap {
        assert_eq!(
            s.unread, 1,
            "the held inits were read: nothing on an armed thread is counted again: {s:?}"
        );
        assert_eq!(
            s.breaker_opens, 0,
            "no held init was a slow sample: the window of sixty-four opens nothing: {s:?}"
        );
    } else {
        excused(
            "a held init on a thread a late call armed",
            "a cold decision's thread had half the budget on the CPU",
        );
    }
}

/// The other side: an `init` slow by its own fuel. At a calibration that
/// prices this guest's `init` at three quarters of the decision's budget
/// every cold slot's sample is past half the budget at both its ends: no
/// time nothing read made it slow, so it arms nothing, and a reading (where
/// a hold of the runner's armed the thread) says no less than its fuel. The
/// window of sixty-four reads slow at its first evaluation and opens the
/// breaker, as before.
#[test]
fn a_lazy_init_slow_by_its_own_fuel_on_every_cold_slot_opens_the_window() {
    const CALIBRATION: u64 = 10;
    let runtime = runtime();
    let mut row = spec("init-burns", guest::record_guest(), mode::BURN_IN_INIT);
    row.fuel_per_micro = CALIBRATION;
    let plugin = loaded(&runtime, row);
    // The load ran `init` once: its fuel at the calibration, the same every
    // time.
    let init_micros = u32::try_from(plugin.stats().fuel_used.div_ceil(CALIBRATION)).expect("fits");
    let engine = plugin_engine(&plugin);
    let decide_cold = |seed: u64| {
        plugin.forget_instance();
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: init_micros / 3 * 4,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    };
    for seed in 1..=63 {
        decide_cold(seed);
    }
    assert_eq!(
        plugin.stats().breaker_opens,
        0,
        "sixty-three decisions are one sample short of the window: {:?}",
        plugin.stats()
    );
    decide_cold(64);
    let s = plugin.stats();
    assert_eq!(
        s.breaker_opens, 1,
        "the sixty-fourth sample reads the window slow, by its fuel: {s:?}"
    );
}

/// A lazy `init` a decision pays is in the decision's sample, and the init
/// frame is in what the sample is taken over. Under a request that states
/// no budget this guest pays its `init` in every one of sixty-four
/// decisions, and its own frame is about four tenths of that `init`: against
/// the frames of its calls alone each sample would read slow, and with the
/// init frame beside them none does.
#[test]
fn a_lazy_init_is_read_against_the_init_frame_under_a_request_with_no_budget() {
    const CALIBRATION: u64 = 10;
    let runtime = runtime();
    let mut gauge = spec("init-gauge", guest::record_guest(), mode::BURN_IN_INIT);
    gauge.fuel_per_micro = CALIBRATION;
    let gauge = loaded(&runtime, gauge);
    // The load ran `init` once: its fuel at the calibration, the same every
    // time.
    let init_micros = u32::try_from(gauge.stats().fuel_used.div_ceil(CALIBRATION)).expect("fits");
    let mut row = spec(
        "init-under-no-budget",
        guest::record_guest(),
        mode::BURN_IN_INIT,
    );
    row.fuel_per_micro = CALIBRATION;
    row.budget_micros = init_micros / 50 * 21;
    let plugin = loaded(&runtime, row);
    let engine = plugin_engine(&plugin);
    let made = plugin.stats().lazy_instantiations;
    for seed in 1..=64 {
        plugin.forget_instance();
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: u32::MAX,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    }
    let s = plugin.stats();
    assert_eq!(
        s.lazy_instantiations - made,
        64,
        "every decision paid an `init`: {s:?}"
    );
    assert_eq!(
        (s.fuel_exhausted, s.traps, s.budget_spent, s.breaker_opens),
        (0, 0, 0, 0),
        "no sample read slow: {s:?}"
    );
}

/// A lazy `init` is in the sample of the decision that paid it, and in no
/// later one. This `init` is three quarters of the decision's budget by its
/// fuel; one cold decision pays it and sixty-four warm ones follow on the
/// thread. The window of sixty-four holds one slow sample and opens nothing.
/// An `init` time carried from one decision into the next would make every
/// sample slow, and the window's first read would open the breaker.
#[test]
fn a_lazy_inits_time_is_in_the_sample_of_the_decision_that_paid_it_alone() {
    const CALIBRATION: u64 = 10;
    let runtime = runtime();
    let mut row = spec("init-burns", guest::record_guest(), mode::BURN_IN_INIT);
    row.fuel_per_micro = CALIBRATION;
    let plugin = loaded(&runtime, row);
    // The load ran `init` once: its fuel at the calibration, the same every
    // time.
    let init_micros = u32::try_from(plugin.stats().fuel_used.div_ceil(CALIBRATION)).expect("fits");
    let engine = plugin_engine(&plugin);
    let decide_under = |seed: u64| {
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: init_micros / 3 * 4,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    };
    plugin.forget_instance();
    decide_under(1);
    for seed in 2..=65 {
        decide_under(seed);
    }
    let s = plugin.stats();
    assert_eq!(
        (s.lazy_instantiations, s.breaker_opens),
        (1, 0),
        "one decision paid the init and its sample alone is slow: {s:?}"
    );
}

/// A lazy `init`'s frame is in the frames of the decision that paid it, and
/// in no later one. A guest whose three stages each burn nine tenths of
/// their frame pays its `init` in one cold decision, and sixty-four warm
/// ones follow on the thread, under a request that states no budget. Each
/// warm sample is past half of the four frames of its own calls, and the
/// ring's first read opens the breaker. Read over those frames and the first
/// decision's init frame, 100 ms more, each would be under half of them, and
/// nothing would open.
#[test]
fn a_lazy_inits_frame_is_in_the_frames_of_the_decision_that_paid_it_alone() {
    let runtime = runtime();
    // A calibration that prices one stage's burn near 30 ms: four frames of
    // such a guest are well under the 100 ms an init frame adds to them.
    let burn_fuel = u64::from(one_burn_micros(&runtime)) * SLOW_CALIBRATION;
    let calibration = burn_fuel.div_ceil(30_000).max(1);
    let burn_micros = u32::try_from(burn_fuel.div_ceil(calibration)).expect("fits");
    let mut near = spec(
        "near-behind-an-init",
        guest::record_guest(),
        mode::BURN_IN_STAGES,
    );
    near.fuel_per_micro = calibration;
    near.budget_micros = burn_micros * 10 / 9;
    let plugin = loaded(&runtime, near);
    let engine = plugin_engine(&plugin);
    let made = plugin.stats().lazy_instantiations;
    plugin.forget_instance();
    for seed in 1..=65 {
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: u32::MAX,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: {decision:?}"
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        stats.lazy_instantiations - made,
        1,
        "the first decision paid the one `init`: {stats:?}"
    );
    assert_eq!(
        (stats.fuel_exhausted, stats.traps, stats.budget_spent),
        (0, 0, 0),
        "every call fit its frame: {stats:?}"
    );
    assert_eq!(
        stats.breaker_opens, 1,
        "the warm samples read slow over the frames of their own calls: {stats:?}"
    );
}

/// A decision that pays an `init` and charges no call is read over the init
/// frame alone, whatever the calls of an earlier decision left on its
/// thread. One warm decision makes four calls under one-second frames. Then
/// the `init` of sixty-four cold slots spins its frame out, the plugin's own
/// failure each time under a threshold of a hundred: 100 ms of the init
/// frame's 100 ms. The ring's first read opens the breaker by its latency
/// rule. Read over the four seconds of the first decision's frames beside
/// the init frame, no sample would be slow.
#[test]
fn an_inits_sample_is_read_over_the_init_frame_alone_behind_a_decision_that_charged_calls() {
    let (clock, _runtime, plugin) = plugin_whose_init_asks(0, |row| {
        row.fuel_per_micro = SLOW_CALIBRATION;
        row.budget_micros = SLOW_FRAME_MICROS;
        row.breaker = BreakerSpec {
            failures: 100,
            ..BreakerSpec::default()
        };
    });
    let engine = plugin_engine(&plugin);
    let decide_at = |steer: u8, seed: u64| {
        clock.store(steer, Ordering::Relaxed);
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: u32::MAX,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        clock.store(0, Ordering::Relaxed);
        assert!(
            matches!(decision, Decision::Pick(_)),
            "decision {seed}: {decision:?}"
        );
    };
    decide_at(0, 0);
    let warm = plugin.stats();
    assert_eq!(
        (warm.answered, warm.lazy_instantiations),
        (4, 0),
        "four calls on the instance the load made: {warm:?}"
    );
    for seed in 1..=64 {
        plugin.forget_instance();
        // The clock's answer of two makes the guest's `init` spin.
        decide_at(3, seed);
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.lazy_instantiations),
        (4, 0),
        "no cold decision kept an instance or charged a call: {stats:?}"
    );
    assert_eq!(
        stats.breaker_opens, 1,
        "each `init` is the whole of its own frame, and the window reads slow: {stats:?}"
    );
}

/// A call its own bound ended marks its decision, and no later one. A cold
/// slot's first decision ends a score that spins at the plugin's own bound:
/// the plugin's failure. Its sample is under half of its frames at both its
/// ends, the init frame among them, so it arms nothing. The next decision
/// on the thread pays an `init` the host holds past half the budget: its
/// sample is slow by time nothing read alone, and it is counted with the
/// unread calls. That holds where the held `init` keeps its instance and its
/// calls follow under what the budget has left, and where it refuses and is
/// all the decision pays. Read with the earlier decision's mark, neither
/// would be counted.
///
/// The first decision's score ends on the clock. A runner that keeps its
/// thread off the CPU as long as the plugin's frame makes that sample one
/// slow by time nothing read, which arms the thread: such a try says so and
/// is made again, [`TRIES`] at most.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_call_its_own_bound_ended_marks_no_later_decision() {
    const CASE: &str = "a call its own bound ended marks no later decision";
    const OWN_FRAME_MICROS: u32 = 100_000;
    const BUDGET_MICROS: u32 = 50_000;
    const WAIT: std::time::Duration = std::time::Duration::from_millis(30);
    // The clock's answer of zero keeps the held `init`, of one refuses it.
    for steer in [1, 2] {
        tried(|made| {
            let clock = Arc::new(AtomicU8::new(0));
            let runtime = steered_runtime(&clock, WAIT);
            let mut row = spec(
                "ended-then-held",
                guest::record_guest(),
                mode::SPIN_IN_SCORE,
            );
            row.config.push(mode::INIT_ASKS_THE_CLOCK);
            // No call runs out of fuel: the clock ends the score that spins.
            row.fuel_per_micro = 20_000_000;
            row.budget_micros = OWN_FRAME_MICROS;
            let plugin = loaded(&runtime, row);
            let engine = plugin_engine(&plugin);
            let decide_cold = |seed: u64, budget_micros: u32| {
                plugin.forget_instance();
                let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                let decision = engine.decide(
                    &RequestFacts {
                        budget_micros,
                        ..facts(seed)
                    },
                    &four(),
                    &pool(4),
                    &Constraints::default(),
                    &mut Scratch::new(),
                );
                let off_the_cpu = off_cpu_since(wall, cpu);
                assert!(
                    matches!(decision, Decision::Pick(_)),
                    "steer {steer}: decision {seed}: {decision:?}"
                );
                off_the_cpu
            };
            // The score spins a whole frame of its own, under a request
            // that states no budget.
            let off_the_cpu = decide_cold(1, u32::MAX);
            let first = plugin.stats();
            let frame = std::time::Duration::from_micros(u64::from(OWN_FRAME_MICROS));
            if off_the_cpu >= frame && made < TRIES {
                made_again(
                    CASE,
                    "the first decision's thread was off the CPU a whole frame of the plugin",
                );
                return false;
            }
            assert_eq!(
                (first.fuel_exhausted, first.budget_spent, first.unread),
                (1, 0, 0),
                "steer {steer}: the plugin's own bound ended its score (try {made}, the thread off the CPU {off_the_cpu:?}): {first:?}"
            );
            // The `init` waits 30 ms on the host, of a budget of 50 ms.
            clock.store(steer, Ordering::Relaxed);
            decide_cold(2, BUDGET_MICROS);
            clock.store(0, Ordering::Relaxed);
            let s = plugin.stats();
            assert_eq!(
                (s.fuel_exhausted, s.unread),
                (1, 1),
                "steer {steer}: the held decision's sample is its own, and nothing read it (try {made}): {s:?}"
            );
            true
        });
    }
}

/// A sample slow by time nothing read is in its window as it is: the rule
/// keeps a second one out, it excuses none. This `init` burns the same fuel
/// on every cold slot, and a decision's budget sets what share of it that
/// is. The first decision's `init` also waits on a host call past half its
/// budget, with its fuel a third of it: slow at its high end alone, the
/// sample is counted with the unread calls and arms the thread. Thirty-two
/// decisions follow whose `init` is three quarters of the budget by its
/// fuel, then thirty-one where it is a quarter. The window of sixty-four
/// holds thirty-three slow samples and the held one is its median: it
/// reads slow, and the breaker opens at the sixty-fourth. Taken at its low
/// end the held sample would leave thirty-two, and the window would open
/// nothing.
///
/// Every sample of the window is an `init` on the wall clock. A runner that
/// keeps a decision's thread off the CPU half its budget can end that
/// `init` there, or make its sample a second one slow by time nothing read:
/// the window then holds one slow sample less and opens nothing. Such a try
/// says so and is made again, [`TRIES`] at most. The last try asserts
/// whatever it read.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_sample_slow_by_time_nothing_read_is_in_its_window_as_it_is() {
    const CASE: &str = "a sample slow by time nothing read is in its window";
    const CALIBRATION: u64 = 30;
    const WAIT_MICROS: u32 = 60_000;
    let micros = |n: u32| std::time::Duration::from_micros(u64::from(n));
    tried(|made| {
        let clock = Arc::new(AtomicU8::new(0));
        let (runtime, naps) = steered_runtime_counting(&clock, micros(WAIT_MICROS));
        let mut row = spec("init-burns-held", guest::record_guest(), mode::BURN_IN_INIT);
        row.config.push(mode::INIT_ASKS_THE_CLOCK);
        row.fuel_per_micro = CALIBRATION;
        let plugin = loaded(&runtime, row);
        // The load ran `init` once, with no wait: its fuel at the calibration,
        // the same every time.
        let init_micros =
            u32::try_from(plugin.stats().fuel_used.div_ceil(CALIBRATION)).expect("fits");
        assert!(
            init_micros / 2 * 3 < WAIT_MICROS,
            "the wait is past half a budget of three such inits: {init_micros}"
        );
        let engine = plugin_engine(&plugin);
        // One cold decision, and how long its thread was off the CPU in it.
        let decide_cold = |seed: u64, steer: u8, budget_micros: u32| {
            plugin.forget_instance();
            clock.store(steer, Ordering::Relaxed);
            let (wall, cpu) = (std::time::Instant::now(), bound_clock());
            let decision = engine.decide(
                &RequestFacts {
                    budget_micros,
                    ..facts(seed)
                },
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let off_the_cpu = off_cpu_since(wall, cpu);
            clock.store(0, Ordering::Relaxed);
            assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
            off_the_cpu
        };
        // Held: the wait is past half the budget, the fuel a third of it. The
        // runner's own hold is what the thread was off the CPU past the
        // clock's naps: half of what the wait leaves of the budget.
        let first = init_micros * 3;
        let off_the_cpu = decide_cold(1, 1, first);
        let napped = micros(WAIT_MICROS) * naps.load(Ordering::Relaxed);
        let mut held =
            off_the_cpu.saturating_sub(napped) >= micros(first.saturating_sub(WAIT_MICROS) / 2);
        let unread_after_the_first = plugin.stats().unread;
        // Slow by fuel: three quarters of the budget at both ends.
        let slow = init_micros / 3 * 4;
        for seed in 2..=33 {
            held |= decide_cold(seed, 0, slow) >= micros(slow / 2);
        }
        // A quarter of the budget.
        let quarter = init_micros * 4;
        for seed in 34..=63 {
            held |= decide_cold(seed, 0, quarter) >= micros(quarter / 2);
        }
        let opens_before_the_last = plugin.stats().breaker_opens;
        held |= decide_cold(64, 0, quarter) >= micros(quarter / 2);
        let s = plugin.stats();
        if held && made < TRIES {
            made_again(CASE, "a decision's thread was off the CPU half its budget");
            return false;
        }
        assert_eq!(
            unread_after_the_first, 1,
            "the held init's sample was slow by time nothing read (try {made}, a hold read: {held}): {s:?}"
        );
        assert_eq!(
            opens_before_the_last, 0,
            "sixty-three decisions are one sample short of the window (try {made}, a hold read: {held}): {s:?}"
        );
        assert_eq!(
            (s.unread, s.breaker_opens),
            (1, 1),
            "thirty-three slow samples, the held one as it is: the median reads slow (try {made}, a hold read: {held}): {s:?}"
        );
        true
    });
}

/// A cold slot's `init` the remainder clamps, held by the host as well: a
/// 2 ms decision whose `init` waits 4 ms for the host, so the wall bound's
/// tick ends it under its clamp when the wait returns. The draw is the
/// whole remainder; the plugin's sample is the time of this `init` as of
/// any other, at its two ends. On a thread nothing armed that is its clock,
/// twice the budget, over a fuel nowhere near it: slow by time nothing read
/// alone, it is counted with the unread calls and arms the thread. The
/// second such `init` is read, and its time is what its thread had, a
/// fraction of the budget: the wait is the host's. So two such decisions in
/// a window of sixty-four are one sample past the budget, and the ring's
/// tail, its second-slowest sample, opens nothing; before that rule they
/// were two, and it opened. The close's rebuild waits as long and is no
/// sample. That the breaker stays shut is asserted where neither cold
/// decision's thread had the budget on the CPU.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn two_clamped_inits_the_host_holds_are_one_slow_sample() {
    const CASE: &str = "two clamped inits the host holds";
    const NAP: std::time::Duration = std::time::Duration::from_millis(4);
    tried(|made| {
        const BUDGET_MICROS: u32 = 2_000;
        let clock = Arc::new(AtomicU8::new(0));
        let (runtime, naps) = steered_runtime_counting(&clock, NAP);
        let mut heavy = spec(
            "clamped-and-held",
            guest::record_guest(),
            mode::BURN_IN_INIT,
        );
        heavy.config.push(mode::INIT_ASKS_THE_CLOCK);
        // A tenth of the default calibration: in fuel the init's burn outlasts
        // the 2 ms budget, so the remainder clamps it.
        heavy.fuel_per_micro = 100;
        let plugin = loaded(&runtime, heavy);
        let engine = plugin_engine(&plugin);
        let decide = |seed: u64, budget_micros: u32| {
            let decision = engine.decide(
                &RequestFacts {
                    budget_micros,
                    ..facts(seed)
                },
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            assert!(matches!(decision, Decision::Pick(_)), "seed {seed}");
        };
        // Two cold decisions under a budget the init's burn outlasts; the clock
        // takes 4 ms to answer each init, the clamped one and the close's rebuild.
        clock.store(1, Ordering::Relaxed);
        let budget = std::time::Duration::from_micros(u64::from(BUDGET_MICROS));
        let mut cheap = true;
        let mut off_the_cpu = std::time::Duration::ZERO;
        for seed in 1..=2 {
            plugin.forget_instance();
            let (wall, cpu) = (std::time::Instant::now(), bound_clock());
            decide(seed, BUDGET_MICROS);
            cheap &= bound_clock().saturating_sub(cpu) < budget;
            off_the_cpu += off_cpu_since(wall, cpu);
        }
        clock.store(0, Ordering::Relaxed);
        // Each close's rebuild is an `init`, on the wall clock. What the two
        // decisions' thread was off the CPU past the clock's naps is a runner's
        // hold: half the init frame of it can have ended a rebuild.
        let held = off_the_cpu.saturating_sub(NAP * naps.load(Ordering::Relaxed)) >= INIT_FRAME / 2;
        let s = plugin.stats();
        let rebuilt = (s.budget_spent, s.lazy_instantiations, s.breaker_opens);
        if rebuilt != (2, 2, 0) && held && made < TRIES {
            made_again(
                CASE,
                "a cold decision's thread was off the CPU half the init frame, the bound of a close's rebuild",
            );
            return false;
        }
        assert_eq!(
            rebuilt,
            (2, 2, 0),
            "two clamped inits, each rebuilt at its close (try {made}, a hold read: {held}): {s:?}"
        );
        assert!(
            s.unread >= 1,
            "the first clamped init's sample was slow by time nothing read: {s:?}"
        );
        // Sixty-two honest decisions on the warm slot fill the window.
        for seed in 3..=64 {
            decide(seed, FRAME_MICROS);
        }
        let s = plugin.stats();
        if cheap {
            assert_eq!(
                s.breaker_opens, 0,
                "one sample past the budget, the first: the second init was read: {s:?}"
            );
        } else {
            excused(CASE, "a cold decision's thread had the budget on the CPU");
        }
        true
    });
}

/// A clock the test steers for a few asks: while `asks` is above zero an ask
/// takes one from it and reads one less than `steer`; every other ask reads
/// zero. A decision's open reads the clock once before a cold slot's `init`
/// asks it, so two asks reach that `init` and none after it.
fn runtime_steered_for(steer: &Arc<AtomicU8>, asks: &Arc<AtomicU8>) -> Arc<Runtime> {
    let (steer, asks) = (Arc::clone(steer), Arc::clone(asks));
    Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), move || {
            let armed = asks
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                .is_ok();
            match steer.load(Ordering::Relaxed) {
                value if armed && value != 0 => u64::from(value - 1),
                _ => 0,
            }
        })
        .expect("a runtime"),
    )
}

/// A cold slot's `init` that traps at once, and would not trap again, while
/// the decision has less than the init frame left: its charge stays below
/// the remainder, so the clamp did not end it. It is the `init`'s own
/// failure, as under its full frame: one count on the breaker, the charge
/// drawn, and the neighbour opened behind it is asked on what the decision
/// has left. No budget is spent and the close rebuilds nothing, so a trap
/// the rebuild would not meet again is counted all the same. The slack is
/// the 90 ms decision: an instantiation held off its core that long reads
/// as the clamp's.
#[test]
fn a_cold_init_that_traps_short_of_its_clamp_is_its_own_failure() {
    const CASE: &str = "a cold init that traps short of its clamp";
    const HELD: &str = "the decision's thread was off the CPU half its budget, the init's clamp";
    const BUDGET_MICROS: u32 = 90_000;
    // An instantiation held half the decision is near its clamp.
    let half = std::time::Duration::from_micros(u64::from(BUDGET_MICROS) / 2);
    tried(|made| {
        let clock = Arc::new(AtomicU8::new(0));
        let asks = Arc::new(AtomicU8::new(0));
        let runtime = runtime_steered_for(&clock, &asks);
        let mut row = spec("traps-once", guest::record_guest(), mode::LEAST_IN_FLIGHT);
        row.config.push(mode::INIT_ASKS_THE_CLOCK);
        row.breaker = BreakerSpec {
            failures: 2,
            ..BreakerSpec::default()
        };
        let cold = loaded(&runtime, row);
        let neighbour = load(
            &runtime,
            "after-the-init",
            guest::record_guest(),
            mode::LEAST_IN_FLIGHT,
        );
        let loaded = cold.stats().instantiations;
        // The open's read of the clock and its `init`'s ask read three, which
        // makes that `init` trap; an `init` after it would go on. Ninety
        // milliseconds are below the init frame, so the remainder clamps it.
        asks.store(2, Ordering::Relaxed);
        let mut held = decide_on_a_cold_slot(&clock, 4, &cold, &neighbour, BUDGET_MICROS) >= half;
        let (s, n) = (cold.stats(), neighbour.stats());
        let first = (
            (
                s.budget_spent,
                s.instantiations - loaded,
                s.lazy_instantiations,
                s.breaker_opens,
            ),
            cold.is_warm(),
            (n.answered, n.budget_spent),
        );
        if first != ((0, 1, 0, 0), false, (2, 0)) && held && made < TRIES {
            made_again(CASE, HELD);
            return false;
        }
        assert_eq!(
            first.0,
            (0, 1, 0, 0),
            "the trap is the init's own: no budget spent, no rebuild at the close (try {made}, a hold read: {held}): {s:?}"
        );
        assert!(!first.1, "the next decision's open tries the init");
        assert_eq!(
            first.2,
            (2, 0),
            "the neighbour's open is fed and answers under what is left: {n:?}"
        );
        // One failure a decision: the second crosses the threshold of two.
        asks.store(2, Ordering::Relaxed);
        held = decide_on_a_cold_slot(&clock, 4, &cold, &neighbour, BUDGET_MICROS) >= half;
        let s = cold.stats();
        let second = (s.budget_spent, s.instantiations - loaded, s.breaker_opens);
        if second != (0, 2, 1) && held && made < TRIES {
            made_again(CASE, HELD);
            return false;
        }
        assert_eq!(
            second,
            (0, 2, 1),
            "the second decision's trap is the second failure (try {made}, a hold read: {held}): {s:?}"
        );
        true
    });
}

/// The decision is the breaker's unit. A cold slot's `init` that refuses,
/// with the plugin on both sides of a pair: the first side's open counts one
/// failure and the second side's makes no second `init` and no second count,
/// so a threshold of two is two decisions on a pair as on a single placement.
/// The next decision's open tries the `init` again.
#[test]
fn a_refused_cold_init_fails_its_decision_once_on_a_pair() {
    let (clock, _runtime, plugin) = plugin_whose_init_asks(0, |row| {
        row.breaker = BreakerSpec {
            failures: 2,
            ..BreakerSpec::default()
        };
    });
    let pair = pair_engine(
        vec![
            plugin_profile(&plugin, None, None, None),
            plugin_profile(&plugin, None, None, None),
        ],
        1,
    );
    let loaded = plugin.stats().instantiations;
    plugin.forget_instance();
    clock.store(2, Ordering::Relaxed);
    assert!(matches!(pair_decide(&pair), Decision::Pick(_)));
    let s = plugin.stats();
    assert_eq!(
        (s.instantiations - loaded, s.breaker_opens),
        (1, 0),
        "one init and one failure for the decision's two opens: {s:?}"
    );
    assert!(matches!(pair_decide(&pair), Decision::Pick(_)));
    clock.store(0, Ordering::Relaxed);
    let s = plugin.stats();
    assert_eq!(
        (s.instantiations - loaded, s.breaker_opens),
        (2, 1),
        "the next decision tried the init again, and its failure is the second: {s:?}"
    );
}

/// The same unit for a frame budget gone on arrival: both sides of a pair
/// find it gone, and the decision counts once.
#[test]
fn a_gone_budget_fails_its_decision_once_on_a_pair() {
    let runtime = runtime();
    let mut row = spec("gone", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    row.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let plugin = loaded(&runtime, row);
    let pair = pair_engine(
        vec![
            plugin_profile(&plugin, None, None, None),
            plugin_profile(&plugin, None, None, None),
        ],
        1,
    );
    assert!(matches!(pair_decide_under(&pair, 0), Decision::Pick(_)));
    let s = plugin.stats();
    assert_eq!(
        (s.budget_gone, s.breaker_opens),
        (1, 0),
        "one budget hit for the decision's two opens: {s:?}"
    );
    assert!(matches!(pair_decide_under(&pair, 0), Decision::Pick(_)));
    let s = plugin.stats();
    assert_eq!(
        (s.budget_gone, s.breaker_opens),
        (2, 1),
        "the second decision's hit is the second failure: {s:?}"
    );
}

/// The same unit at the close: a stage call traps, which counts, and the
/// rebuild of the slot it emptied is refused, which is no second failure of
/// the same decision. The next decision's open tries the `init` again, and
/// that failure is the second.
#[test]
fn a_rebuild_refused_after_a_counted_trap_is_no_second_failure() {
    tried(a_rebuild_refused_after_a_counted_trap_is_no_second_failure_in_a_try);
}

/// One try of the case above, told its number: whether it stood. A
/// decision's budget is on the wall clock, so a thread held off the CPU
/// half of it can find the budget spent where its call was due.
fn a_rebuild_refused_after_a_counted_trap_is_no_second_failure_in_a_try(made: u32) -> bool {
    const CASE: &str = "a rebuild refused after a counted trap";
    let clock = Arc::new(AtomicU8::new(0));
    let runtime = steered_runtime(&clock, std::time::Duration::ZERO);
    let mut row = spec(
        "trap-then-refuse",
        guest::record_guest(),
        mode::TRAP_IN_FILTER,
    );
    row.config.push(mode::INIT_ASKS_THE_CLOCK);
    row.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let plugin = loaded(&runtime, row);
    let engine = plugin_engine(&plugin);
    let decide = |seed: u64| {
        engine.decide(
            &RequestFacts {
                budget_micros: FRAME_MICROS,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    let half = std::time::Duration::from_micros(u64::from(FRAME_MICROS) / 2);
    let held_in = |seed: u64| {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        assert!(matches!(decide(seed), Decision::Pick(_)));
        off_cpu_since(wall, cpu)
    };
    clock.store(2, Ordering::Relaxed);
    let held = held_in(1);
    let s = plugin.stats();
    if (s.traps, s.breaker_opens) != (1, 0) && held >= half && made < TRIES {
        made_again(
            CASE,
            "the first decision's thread was off the CPU half its budget",
        );
        return false;
    }
    assert_eq!(
        (s.traps, s.breaker_opens),
        (1, 0),
        "the trap counted; the refused rebuild at the same close did not: {s:?}"
    );
    assert!(!plugin.is_warm(), "the rebuild was refused");
    let held = held_in(2);
    clock.store(0, Ordering::Relaxed);
    let s = plugin.stats();
    if (s.traps, s.breaker_opens) != (1, 1) && held >= half && made < TRIES {
        made_again(
            CASE,
            "the second decision's thread was off the CPU half its budget",
        );
        return false;
    }
    assert_eq!(
        (s.traps, s.breaker_opens),
        (1, 1),
        "the next decision's refused init is the second failure: {s:?}"
    );
    true
}

/// A fault in one open ends the plugin's part in the decision. The prefill
/// side's constructor traps and the instance goes; the decode side's open, a
/// moment later on the same thread, makes no new one inside the decision that
/// lost it. One trap, and one rebuild at the close.
#[test]
fn a_decision_that_emptied_the_slot_makes_no_instance_before_its_close() {
    tried(a_decision_that_emptied_the_slot_makes_no_instance_before_its_close_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_decision_that_emptied_the_slot_makes_no_instance_before_its_close_in_a_try(made: u32) -> bool {
    const CASE: &str = "a decision that emptied the slot";
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "constructor-trap",
        guest::record_guest(),
        mode::TRAP_IN_CONSTRUCTOR,
    );
    let prefill = plugin_profile(&plugin, None, None, None);
    let decode = plugin_profile(&plugin, None, None, None);
    let pair = pair_engine(vec![prefill, decode], 1);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    assert!(
        matches!(pair_decide(&pair), Decision::Pick(_)),
        "the fallbacks decide"
    );
    let held = off_cpu_since(wall, cpu);
    let stats = plugin.stats();
    assert_eq!(
        stats.traps, 1,
        "the prefill constructor's trap; the decode side asked nothing: {stats:?}"
    );
    if !the_close_rebuilt(CASE, &plugin, held, made) {
        return false;
    }
    assert_eq!(
        (stats.instantiations, stats.lazy_instantiations),
        (2, 1),
        "the load and the close's rebuild: {stats:?}"
    );
    true
}

/// A lifecycle event's trap, outside any decision, drops the instance; the
/// next decision on the thread arrives with its budget gone and touches no
/// slot, so its close rebuilds nothing: the slot stays cold for the next
/// event or open to make lazily.
#[test]
fn an_observe_fault_leaves_no_mark_for_a_decision_that_never_touched_the_slot() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "observe-trap",
        guest::record_guest(),
        mode::TRAP_IN_OBSERVE,
    );
    let engine = plugin_engine(&plugin);
    let decide = |budget_micros: u32, seed: u64| {
        engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    };
    assert!(matches!(decide(FRAME_MICROS, 1), Decision::Pick(_)));
    assert!(plugin.is_warm(), "a decision ran on the kept instance");
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    let s = plugin.stats();
    assert_eq!(s.traps, 1, "the event's trap: {s:?}");
    assert!(!plugin.is_warm(), "the trap dropped the instance");
    assert!(
        matches!(decide(0, 2), Decision::Pick(_)),
        "the fallbacks decide"
    );
    let s = plugin.stats();
    assert!(
        !plugin.is_warm() && s.instantiations == 1 && s.budget_gone == 1,
        "a decision that touched no slot rebuilt none: {s:?}"
    );
}

/// A destructor that fills its memory without end: little fuel per fill, so
/// its fuel frame never ends it. Its bound on the clock is its own, in
/// milliseconds, not the decision's frame: a decision at a 100 ms budget is
/// not stretched by one; the run-out drops the instance and the close
/// rebuilds it. A destructor is a watched call: its bound counts the time
/// its thread had the CPU, so the case reads the clock the bound counts.
/// It ends at a tick, and a ticker ticks once a millisecond while a call is
/// in flight: a destructor far past its bound had no tick for that long. A
/// ticker the runner kept from the CPU was runnable all that time, and the
/// kernel counts what a thread waits on a runqueue: a decision whose ticker
/// waited half of what the destructor had says so and is made again,
/// [`TRIES`] at most. A ticker its host parked, never woke or left asleep
/// waited on none, and that run is red at once. The last try is red
/// whatever its ticker waited: a ticker that shares its destructor's core
/// waits there and ticks all the same.
#[test]
fn a_destructors_bound_is_its_own_not_the_decisions_frame() {
    tried(|made| {
        let (runtime, born) = runtime_and_the_threads_born_with_it();
        let mut filler = spec("fill-dtor", guest::record_guest(), mode::FILL_IN_DESTRUCTOR);
        // The frame is 100 ms, the default of a plugin that states none and
        // the request's too, and the calibration makes the destructor's
        // fuel frame outlast it: only its bound on the clock ends the fill.
        filler.budget_micros = 0;
        filler.fuel_per_micro = 100_000_000;
        let plugin = loaded(&runtime, filler);
        let engine = plugin_engine(&plugin);
        // A ticker still waiting for the CPU reads short: the baseline is
        // read with the tickers at rest.
        tickers_at_rest(&born);
        let waits = ticker_waits(&born);
        let (wall, started) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: 100_000,
                ..facts(1)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let had = bound_clock().saturating_sub(started);
        let held = off_cpu_since(wall, started);
        assert!(matches!(decision, Decision::Pick(_)), "the decision stands");
        // No tick comes while the runner keeps the ticker from the CPU, and
        // the destructor runs on until one does. A ticker that waited on no
        // runqueue for half of that was free to tick: the time is the
        // host's. So is the time of the last try.
        if had >= std::time::Duration::from_millis(30) {
            let waited = least_ticker_wait_since(&waits, &born);
            assert!(
                waited.is_some_and(|waited| waited >= had / 2) && made < TRIES,
                "the destructor's bound is its own, not the 100 ms frame: {had:?} on the clock it counts in try {made}, and its ticker waited {waited:?} on a runqueue"
            );
            made_again(
                "a destructor's bound",
                &format!(
                    "its ticker waited {:?} on a runqueue of the {had:?} the destructor had",
                    waited.unwrap_or_default()
                ),
            );
            return false;
        }
        if !the_close_rebuilt("a destructor's bound", &plugin, held, made) {
            return false;
        }
        let s = plugin.stats();
        assert_eq!(
            (
                s.fuel_exhausted,
                s.traps,
                s.breaker_opens,
                s.lazy_instantiations
            ),
            (1, 0, 0, 1),
            "the destructor ran out, the instance went, the close rebuilt it: {s:?}"
        );
        true
    });
}

/// Under an operator frame below the budget, the charge is per call. A
/// three-stage plugin whose every call fits its frame, clamped on its last
/// call by what three honest neighbours spent, is not charged for the sum of
/// its calls: the call ran out under the remainder, not under its own frame,
/// and the plugin did not consume the budget alone. Its share of the
/// decision, under half, is no slow sample either: sixty-five decisions open
/// nothing on any side.
#[test]
fn an_honest_multi_stage_plugin_clamped_under_its_own_frame_is_not_charged() {
    tried(an_honest_multi_stage_plugin_clamped_under_its_own_frame_is_not_charged_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn an_honest_multi_stage_plugin_clamped_under_its_own_frame_is_not_charged_in_a_try(
    made: u32,
) -> bool {
    const CASE: &str = "an honest multi-stage plugin clamped under its own frame";
    let runtime = runtime();
    // Every charge at the calibration of record is the fuel, exact.
    let mut gauge = spec("gauge", guest::record_guest(), mode::BURN_IN_STAGES);
    gauge.fuel_per_micro = SLOW_CALIBRATION;
    gauge.budget_micros = SLOW_FRAME_MICROS;
    let fuel_per_micro = gauge.fuel_per_micro;
    let gauge = loaded(&runtime, gauge);
    let before = gauge.stats().fuel_used;
    plugin_engine(&gauge).decide(
        &RequestFacts {
            budget_micros: 1_000_000,
            ..facts(1)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let burn_micros =
        u32::try_from(((gauge.stats().fuel_used - before) / 3).div_ceil(fuel_per_micro))
            .expect("fits");
    // The frame holds one burn and two thirds: each of the three calls fits.
    let mut three_stage = spec("three-stage", guest::record_guest(), mode::BURN_IN_STAGES);
    three_stage.budget_micros = burn_micros * 5 / 3;
    three_stage.fuel_per_micro = SLOW_CALIBRATION;
    let three_stage = loaded(&runtime, three_stage);
    let neighbours: Vec<Arc<Plugin>> = ["neighbour-a", "neighbour-b", "neighbour-c"]
        .into_iter()
        .map(|name| {
            let mut neighbour = spec(name, guest::record_guest(), mode::BURN_IN_STAGES);
            neighbour.fuel_per_micro = SLOW_CALIBRATION;
            neighbour.budget_micros = SLOW_FRAME_MICROS;
            loaded(&runtime, neighbour)
        })
        .collect();
    let bound = three_stage.bind();
    let engine = single_engine(Profile {
        name: "clamped".into(),
        filters: vec![Box::new(WasmFilter::new(bound.clone(), None))],
        scorers: std::iter::once(WeightedScorer {
            scorer: Box::new(WasmScorer::new(bound.clone(), None)),
            weight: 1.0,
        })
        .chain(neighbours.iter().map(|neighbour| WeightedScorer {
            scorer: Box::new(WasmScorer::new(neighbour.bind(), None)),
            weight: 1.0,
        }))
        .collect(),
        picker: Box::new(WasmPicker::new(bound, None)),
    });
    // Five burns and a half, plus the constructors: the filter, the score
    // and the neighbours' scores fit with half a burn of slack, the pick meets
    // half a burn under its frame of a burn and two thirds. The plugin's
    // spend, two burns and a half of the five and a half, stays under half
    // the budget: the decision's sample does not read it slow. A burn is
    // twenty thousand loop turns, near 118k fuel: near 118 milliseconds at
    // this calibration against near a tenth of one on the wall, so the slack
    // holds until one call is held off the core for sixty milliseconds. A
    // tenth of a burn covers the constructors and the lazy init.
    let budget_micros = burn_micros * 11 / 2 + burn_micros / 10;
    let mut worst = std::time::Duration::ZERO;
    for seed in 1..=65 {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        worst = worst.max(off_cpu_since(wall, cpu));
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallback picker decides"
        );
    }
    let a = three_stage.stats();
    let stood = (a.fuel_exhausted, a.traps, a.breaker_opens, a.budget_spent) == (0, 0, 0, 65)
        && neighbours.iter().all(|neighbour| {
            let n = neighbour.stats();
            (n.fuel_exhausted, n.budget_spent, n.breaker_opens) == (0, 0, 0)
        });
    if held_counts_are_made_again(CASE, stood, worst, made) {
        return false;
    }
    assert_eq!(
        (a.fuel_exhausted, a.traps, a.breaker_opens, a.budget_spent),
        (0, 0, 0, 65),
        "the clamped last call is the budget spent, never a charge: {a:?}"
    );
    for neighbour in &neighbours {
        let n = neighbour.stats();
        assert_eq!(
            (n.fuel_exhausted, n.budget_spent, n.breaker_opens),
            (0, 0, 0),
            "a neighbour is untouched: {n:?}"
        );
    }
    true
}

/// The latency sample is the decision's charge over the budget, not its
/// clock.
/// A spinner at the slow calibration of record burns its fuel
/// grant (the whole remainder behind an honest scorer) in a fraction of the
/// budget's wall: sampled by the clock it never reads slow and spends the
/// decision's budget forever; sampled by its charge, the remainder it took,
/// the window opens its breaker. The case decides until it does.
#[test]
fn a_fuel_fast_spinner_is_sampled_by_its_charge_not_its_clock() {
    let runtime = runtime();
    let honest = load(
        &runtime,
        "honest",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let mut slow_calibration = spec("fuel-fast", guest::record_guest(), mode::SPIN_IN_SCORE);
    slow_calibration.fuel_per_micro = SLOW_CALIBRATION;
    let spinner = loaded(&runtime, slow_calibration);
    // What one `init` of the spinner costs: the load made one instance.
    let init_fuel = spinner.stats().fuel_used;
    let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
        scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
        weight: 1.0,
    };
    let engine = single_engine(Profile {
        name: "fuel-fast".into(),
        filters: vec![],
        scorers: vec![scorer(&honest), scorer(&spinner)],
        picker: Box::new(Best),
    });
    let (decisions, on_the_cpu) = decide_until_open(&spinner, |seed| {
        engine.decide(
            &timed(seed),
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        )
    });
    let s = spinner.stats();
    assert_eq!(
        (s.traps, s.breaker_opens),
        (0, 1),
        "the spinner's breaker opened once, on no trap: {s:?}"
    );
    let quiet = no_hold("a fuel-fast spinner", on_the_cpu, &[&spinner, &honest]);
    if quiet {
        assert_eq!(
            s.answered, 64,
            "the window opened at the spinner's 64th sample, one a decision that reached its constructor: {s:?}"
        );
        assert_eq!(
            decisions, 64,
            "no hold moved the run, so every decision reached the spinner"
        );
        assert_eq!(
            s.fuel_exhausted, 0,
            "the spinner's charge, the remainder, opened its breaker at the window: {s:?}"
        );
        assert!(
            s.budget_spent >= 64,
            "every decision was the budget spent: {s:?}"
        );
    }
    the_neighbour_stays("a fuel-fast spinner", quiet, &honest);
    // The spinner is fast by its charge, whatever a clock reads: a decision
    // grants its calls at most the frame's fuel at the slow calibration,
    // ten thousand units, and its rebuild one `init`. A spin burns that in
    // microseconds, so its fuel ended each call long before its bound. The
    // thread's CPU clock is no witness of it: on a loaded runner it read
    // 1.57 s over 64 decisions with no call late, where a quiet one reads
    // 6 ms.
    let frame_fuel = u64::from(FRAME_MICROS) * SLOW_CALIBRATION;
    assert!(
        s.fuel_used <= decisions * (frame_fuel + init_fuel) + init_fuel,
        "a decision granted past its frame or built twice: {s:?} over {decisions} decisions"
    );
    // That bound is the grant's cap and holds however a call ended. Which
    // bound ended the calls is read from below: a spin its fuel ended burnt
    // its whole grant, the remainder behind the honest scorer, and one its
    // clock bound ended first burnt a part of it. On a quiet run the
    // remainder is past half the frame's fuel: a hold under a quarter of
    // the budget is the most that came off it.
    if quiet {
        let spun = s.fuel_used.saturating_sub(s.instantiations * init_fuel);
        assert!(
            spun >= decisions * frame_fuel / 2,
            "the spins burnt {spun} units over {decisions} decisions, short of their grants: {s:?}"
        );
    }
}

/// A plugin that spins in its constructor, behind another plugin's
/// constructor, is the decision's budget spent (the grant is below its cap by
/// the other's spend, its own spend short of the seed) and a constructor is
/// no stage sample: the call the remainder ended is sampled all the same, so
/// its one sample per decision opens the breaker at the window. Its scope
/// first, the constructor had its whole frame and the failure count opens it.
/// Behind the honest constructor the case decides until the opening: a
/// decision whose budget a hold of the runner spent inside the honest call
/// leaves the spinner no sample, or one under half the budget.
#[test]
fn a_constructor_spinner_behind_another_constructor_opens_its_breaker_in_either_order() {
    for spinner_first in [false, true] {
        let runtime = runtime();
        let honest = load(
            &runtime,
            "honest",
            guest::record_guest(),
            mode::LEAST_IN_FLIGHT,
        );
        let spinner = load(
            &runtime,
            "ctor-spinner",
            guest::record_guest(),
            mode::SPIN_IN_CONSTRUCTOR,
        );
        let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
            scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
            weight: 1.0,
        };
        let scorers = if spinner_first {
            vec![scorer(&spinner), scorer(&honest)]
        } else {
            vec![scorer(&honest), scorer(&spinner)]
        };
        let engine = single_engine(Profile {
            name: "ctor".into(),
            filters: vec![],
            scorers,
            picker: Box::new(Best),
        });
        let decide = |seed: u64| {
            engine.decide(
                &timed(seed),
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        let (decisions, on_the_cpu) = if spinner_first {
            let mut on_the_cpu = true;
            for seed in 1..=64 {
                let (decision, kept) = decide_on_the_cpu(|| decide(seed));
                on_the_cpu &= kept;
                assert!(
                    matches!(decision, Decision::Pick(_)),
                    "the fallbacks decide"
                );
            }
            (64, on_the_cpu)
        } else {
            decide_until_open(&spinner, decide)
        };
        let s = spinner.stats();
        assert!(
            s.breaker_opens >= 1,
            "spinner first {spinner_first}: the spinner's breaker opened: {s:?}"
        );
        if spinner_first {
            // A budget a hold of the runner spent before the spinner's
            // constructor is a budget hit, and the breaker counts it in a
            // run-out's place: the floor is asserted where no hold moved
            // the sixty-four decisions.
            let quiet = no_hold(
                "a constructor spinner, its scope first",
                on_the_cpu,
                &[&spinner, &honest],
            );
            if quiet {
                assert!(
                    s.fuel_exhausted >= 5,
                    "its scope first, the constructor had its whole frame: charged: {s:?}"
                );
            }
            the_neighbour_stays("a constructor spinner, its scope first", quiet, &honest);
        } else {
            assert_eq!(s.traps, 0, "behind the honest constructor, no trap: {s:?}");
            let quiet = no_hold("a constructor spinner", on_the_cpu, &[&spinner, &honest]);
            if quiet {
                assert_eq!(
                    decisions, 64,
                    "no hold moved the run: the window opened at the sixty-fourth decision"
                );
                assert_eq!(
                    s.fuel_exhausted, 0,
                    "behind the honest constructor, never charged: {s:?}"
                );
                assert!(
                    s.budget_spent >= 64,
                    "every decision was the budget spent: {s:?}"
                );
            }
            the_neighbour_stays("a constructor spinner", quiet, &honest);
        }
        decide(decisions + 1);
        assert!(
            spinner.stats().bypassed >= 1,
            "spinner first {spinner_first}: open, the spinner is bypassed"
        );
    }
}

/// A constructor that eats most of the budget and answers is no run-out: no
/// call of its own is charged or ends under the remainder, and the honest
/// plugin's stages find the budget gone. The decision's sample is the
/// plugin's whole charge in it, so the window opens the hog's breaker and
/// the honest plugin's never. That opening writes its one line, under
/// the latency rule.
#[test]
fn a_constructor_hog_that_answers_opens_at_the_window() {
    said();
    let runtime = runtime();
    // Every charge at the calibration of record is the fuel, exact.
    let mut gauge = spec("gauge", guest::record_guest(), mode::BURN_IN_STAGES);
    gauge.fuel_per_micro = SLOW_CALIBRATION;
    gauge.budget_micros = SLOW_FRAME_MICROS;
    let fuel_per_micro = gauge.fuel_per_micro;
    let gauge = loaded(&runtime, gauge);
    let before = gauge.stats().fuel_used;
    plugin_engine(&gauge).decide(
        &RequestFacts {
            budget_micros: 1_000_000,
            ..facts(1)
        },
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let burn_micros =
        u32::try_from(((gauge.stats().fuel_used - before) / 3).div_ceil(fuel_per_micro))
            .expect("fits");
    let honest = load(
        &runtime,
        "honest",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let mut hog = spec("ctor-hog", guest::record_guest(), mode::BURN_IN_CONSTRUCTOR);
    hog.fuel_per_micro = SLOW_CALIBRATION;
    hog.budget_micros = SLOW_FRAME_MICROS;
    let hog = loaded(&runtime, hog);
    let scorer = |plugin: &Arc<Plugin>| WeightedScorer {
        scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
        weight: 1.0,
    };
    let engine = single_engine(Profile {
        name: "hog".into(),
        filters: vec![],
        scorers: vec![scorer(&honest), scorer(&hog)],
        picker: Box::new(Best),
    });
    // A burn and a half: the hog's constructor takes two thirds of it and
    // answers, a third of a burn of slack for its stages.
    let budget_micros = burn_micros * 3 / 2;
    for seed in 1..=64 {
        engine.decide(
            &RequestFacts {
                budget_micros,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
    }
    let (g, h) = (hog.stats(), honest.stats());
    assert_eq!(
        (g.fuel_exhausted, g.traps, g.breaker_opens),
        (0, 0, 1),
        "the hog answered every constructor and the window opened it: {g:?}"
    );
    assert_eq!(h.breaker_opens, 0, "the honest plugin stays: {h:?}");
    let lines = openings("ctor-hog");
    assert_eq!(lines.len(), 1, "one opening, one line: {lines:?}");
    assert!(lines[0].contains("rule=\"latency\""), "{}", lines[0]);
}

/// A plugin that spins in its third stage behind its own two cheap stages:
/// per call, two cheap samples a decision would hide the third; per
/// decision, the plugin's whole charge is the remainder's share and the
/// window opens it, whichever scorer runs first. Behind an honest
/// neighbour's spend the spinning pick is the decision's budget spent, never
/// a charge. The case decides until the opening.
#[test]
fn a_third_stage_spinner_behind_its_own_cheap_stages_opens_at_the_window() {
    for spinner_first in [false, true] {
        let runtime = runtime();
        let honest = load(
            &runtime,
            "honest",
            guest::record_guest(),
            mode::LEAST_IN_FLIGHT,
        );
        let spinner = load(
            &runtime,
            "pick-spinner",
            guest::record_guest(),
            mode::SPIN_IN_PICK,
        );
        let bound = spinner.bind();
        let own = WeightedScorer {
            scorer: Box::new(WasmScorer::new(bound.clone(), None)),
            weight: 1.0,
        };
        let other = WeightedScorer {
            scorer: Box::new(WasmScorer::new(honest.bind(), None)),
            weight: 1.0,
        };
        let scorers = if spinner_first {
            vec![own, other]
        } else {
            vec![other, own]
        };
        let engine = single_engine(Profile {
            name: "third-stage".into(),
            filters: vec![Box::new(WasmFilter::new(bound.clone(), None))],
            scorers,
            picker: Box::new(WasmPicker::new(bound, None)),
        });
        let (decisions, on_the_cpu) = decide_until_open(&spinner, |seed| {
            engine.decide(
                &timed(seed),
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            )
        });
        let s = spinner.stats();
        assert_eq!(
            (s.traps, s.breaker_opens),
            (0, 1),
            "spinner first {spinner_first}: the spinner's breaker opened once, on no trap: {s:?}"
        );
        let quiet = no_hold("a third-stage spinner", on_the_cpu, &[&spinner, &honest]);
        if quiet {
            assert_eq!(
                decisions, 64,
                "spinner first {spinner_first}: no hold moved the run: the window opened at the sixty-fourth decision"
            );
            assert_eq!(
                s.fuel_exhausted, 0,
                "spinner first {spinner_first}: the spinning pick was the budget spent and the window opened it: {s:?}"
            );
            assert!(
                s.budget_spent >= 64,
                "spinner first {spinner_first}: every decision was the budget spent: {s:?}"
            );
        }
        the_neighbour_stays("a third-stage spinner", quiet, &honest);
    }
}

/// A pair decision opens both sides' scopes up front and closes them at the
/// end: a plugin bound on both sides closes twice, and its one sample is the
/// two sides' spend summed, once: the window opens at the sixty-fourth
/// decision, not before. Three burns a side at the slow calibration, a burn
/// nine percent of the budget: summed, past half; a side alone, under it.
#[test]
fn a_plugin_on_both_pair_sides_is_sampled_once_with_both_sides_summed() {
    tried(a_plugin_on_both_pair_sides_is_sampled_once_with_both_sides_summed_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_plugin_on_both_pair_sides_is_sampled_once_with_both_sides_summed_in_a_try(made: u32) -> bool {
    const CASE: &str = "a plugin on both pair sides";
    let runtime = runtime();
    let mut both = spec("both-sides", guest::record_guest(), mode::BURN_IN_STAGES);
    both.fuel_per_micro = SLOW_CALIBRATION;
    both.budget_micros = PAIR_BUDGET_MICROS;
    let plugin = loaded(&runtime, both);
    let shared = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![shared], 0);
    let worst = std::cell::Cell::new(std::time::Duration::ZERO);
    let decide_held = || {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let decision = pair_decide_under(&engine, PAIR_BUDGET_MICROS);
        worst.set(worst.get().max(off_cpu_since(wall, cpu)));
        assert!(decision.pick().is_some(), "the pair decides");
    };
    for _ in 0..63 {
        decide_held();
    }
    let opens = plugin.stats().breaker_opens;
    if held_counts_are_made_again(CASE, opens == 0, worst.get(), made) {
        return false;
    }
    assert_eq!(
        opens,
        0,
        "sixty-three decisions are one sample short of the window: {:?}",
        plugin.stats()
    );
    decide_held();
    let s = plugin.stats();
    let read = (s.fuel_exhausted, s.traps, s.budget_spent, s.breaker_opens);
    if held_counts_are_made_again(CASE, read == (0, 0, 0, 1), worst.get(), made) {
        return false;
    }
    assert_eq!(
        read,
        (0, 0, 0, 1),
        "both sides summed into one sample a decision opened the breaker at the window: {s:?}"
    );
    true
}

/// One plugin per side: the decode side's scopes close before the prefill
/// side's, so a decode-only plugin's last close is not the decision's last
/// close. It is sampled at its own, once a decision: heavy on the decode
/// side alone, three burns of near a fifth of its budget each, past half
/// together, it opens at the sixty-fourth decision while the prefill
/// plugin, cheap, stays.
#[test]
fn a_decode_side_plugin_is_sampled_at_its_own_last_close() {
    let runtime = runtime();
    let prefill = load(
        &runtime,
        "prefill",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    let mut heavy = spec("decode-heavy", guest::record_guest(), mode::BURN_IN_STAGES);
    heavy.fuel_per_micro = SLOW_CALIBRATION;
    heavy.budget_micros = PAIR_BUDGET_MICROS / 2;
    let decode = loaded(&runtime, heavy);
    let engine = pair_engine(
        vec![
            plugin_profile(&prefill, None, None, None),
            plugin_profile(&decode, None, None, None),
        ],
        1,
    );
    for _ in 0..63 {
        assert!(
            pair_decide_under(&engine, PAIR_BUDGET_MICROS / 2)
                .pick()
                .is_some(),
            "the pair decides"
        );
    }
    assert_eq!(
        decode.stats().breaker_opens,
        0,
        "sixty-three decisions are one sample short of the window: {:?}",
        decode.stats()
    );
    assert!(
        pair_decide_under(&engine, PAIR_BUDGET_MICROS / 2)
            .pick()
            .is_some(),
        "the pair decides"
    );
    let (d, p) = (decode.stats(), prefill.stats());
    assert_eq!(
        (d.fuel_exhausted, d.traps, d.budget_spent, d.breaker_opens),
        (0, 0, 0, 1),
        "the decode plugin's own last close sampled its decision and the window opened it: {d:?}"
    );
    assert_eq!(p.breaker_opens, 0, "the prefill plugin stays: {p:?}");
}

/// A plugin's declared needs reach the profile's needs whichever stage it is
/// bound as. A plugin that reads the affinity column as the filter alone, or
/// as the picker alone, beside built-in stages that declare nothing, counts
/// as a reader of it: a host that fills the column where it is declared
/// leaves it empty only where nothing reads it.
#[test]
fn a_plugin_bound_as_a_filter_or_a_picker_alone_counts_in_the_profiles_needs() {
    let runtime = runtime();
    let mut reading = spec("reader", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    reading.needs.affinity = true;
    let plugin = loaded(&runtime, reading);
    let scorers = || {
        vec![WeightedScorer {
            scorer: Box::new(LeastRequests),
            weight: 1.0,
        }]
    };
    let as_filter = Profile {
        name: "filter".into(),
        filters: vec![Box::new(WasmFilter::new(plugin.bind(), None))],
        scorers: scorers(),
        picker: Box::new(Best),
    };
    let as_picker = Profile {
        name: "picker".into(),
        filters: vec![Box::new(Healthy)],
        scorers: scorers(),
        picker: Box::new(WasmPicker::new(plugin.bind(), None)),
    };
    assert!(as_filter.needs().affinity, "the plugin as the filter alone");
    assert!(as_picker.needs().affinity, "the plugin as the picker alone");
}

/// A fallback answers when its plugin fails, so what it reads counts as
/// what the plugin reads does: a plugin that declares nothing, bound over a
/// fallback that reads the affinity column, is a reader of it. Each of the
/// three adapters hands its own fallback's needs on, so the same reader is
/// asked behind the filter, behind a scorer and behind the picker.
#[test]
fn a_plugin_stages_fallback_counts_in_the_stages_needs() {
    use smg_policy_controller::engine::{Filter, Picker, Scorer, Targets, TieBand};

    #[derive(Debug)]
    struct ColumnReader;

    fn reads_the_column() -> Needs {
        Needs::NONE.with_affinity()
    }

    impl Lifecycle for ColumnReader {}

    impl Filter for ColumnReader {
        fn name(&self) -> &'static str {
            "column-reader"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }

        fn needs(&self) -> Needs {
            reads_the_column()
        }
    }

    impl Scorer for ColumnReader {
        fn name(&self) -> &'static str {
            "column-reader"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(0.0);
            TieBand::NORMALIZED
        }

        fn needs(&self) -> Needs {
            reads_the_column()
        }
    }

    impl Picker for ColumnReader {
        fn name(&self) -> &'static str {
            "column-reader"
        }

        fn pick(
            &self,
            facts: &RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            scores: &[f32],
            band: TieBand,
            out: &mut Targets,
        ) {
            Best.pick(facts, candidates, kept, scores, band, out);
        }

        fn needs(&self) -> Needs {
            reads_the_column()
        }
    }

    let runtime = runtime();
    let plugin = load(
        &runtime,
        "plain",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    // The plugin declares nothing: alone, no adapter reads the column.
    let alone: [(&str, Needs); 3] = [
        ("the filter", WasmFilter::new(plugin.bind(), None).needs()),
        ("a scorer", WasmScorer::new(plugin.bind(), None).needs()),
        ("the picker", WasmPicker::new(plugin.bind(), None).needs()),
    ];
    // Over the reader, each adapter declares its fallback's needs.
    let over: [(&str, Needs); 3] = [
        (
            "the filter",
            WasmFilter::new(plugin.bind(), Some(Box::new(ColumnReader))).needs(),
        ),
        (
            "a scorer",
            WasmScorer::new(plugin.bind(), Some(Box::new(ColumnReader))).needs(),
        ),
        (
            "the picker",
            WasmPicker::new(plugin.bind(), Some(Box::new(ColumnReader))).needs(),
        ),
    ];
    for (bound_as, needs) in alone {
        assert!(!needs.affinity, "the plugin alone, bound as {bound_as}");
    }
    for (bound_as, needs) in over {
        assert!(needs.affinity, "the fallback's needs, bound as {bound_as}");
    }
}

/// A named decider's kept instance is its name's, on whichever thread acts
/// as it. Three threads act as one decider in turn: the first decides, the
/// second is told a dispatch and decides, the third decides, and each pick
/// names the row at the count the instance heard. A second name has an
/// instance of its own, which heard nothing. The apply's own instance went
/// to the first decider that needed one: the instances made are the deciders.
#[test]
fn a_named_deciders_instance_is_its_names_on_whichever_thread_acts_as_it() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "named",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    // One thread a call, acting as the decider of that number.
    let acting = |number: u8, dispatch: bool| {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _turn = DecisionSlot::new(number).enter();
                    if dispatch {
                        engine.on_dispatched(
                            &Lease::default(),
                            &Target::default(),
                            Side::Single,
                            3,
                        );
                    }
                    pick_ids(&decide(&engine, &pool(4), 1)).first().copied()
                })
                .join()
                .expect("the acting thread")
        })
    };
    assert_eq!(acting(0, false), Some(100), "nothing told yet");
    assert_eq!(
        acting(0, true),
        Some(101),
        "told on a second thread, to the instance the first decided on"
    );
    assert_eq!(
        acting(0, false),
        Some(101),
        "and a third thread decides on it"
    );
    assert_eq!(
        acting(1, false),
        Some(100),
        "another name's instance heard nothing"
    );
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.instantiations,
            stats.lazy_instantiations,
            stats.observed,
            stats.events_dropped
        ),
        (2, 1, 1, 0),
        "two deciders, two instances, the apply's own among them: {stats:?}"
    );
    assert!(!plugin.is_warm(), "the loading thread kept none");
}

/// A thread that decides under no name and also acts as a named decider
/// keeps the two apart: its own instance stands aside while it acts, and is
/// as it left it after.
#[test]
fn a_threads_own_instance_stands_aside_while_it_acts_as_a_named_decider() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "aside",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    let dispatch = || engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    dispatch();
    assert_eq!(pick(), Some(101), "the thread's own instance heard it");
    {
        let _turn = DecisionSlot::new(5).enter();
        assert_eq!(pick(), Some(100), "the named decider's heard nothing");
        dispatch();
        dispatch();
        assert_eq!(pick(), Some(102), "and hears what is told as it");
    }
    assert_eq!(pick(), Some(101), "the thread's own is as it left it");
    assert_eq!(plugin.stats().instantiations, 2);
}

/// A thread that names no decider and never decided with the plugin only
/// tells: its lease event has no instance to reach and makes none. The
/// event is dropped and counted, and the plugin's first drop writes its one
/// warning. The thread that loaded is told as before: the apply's instance
/// is its own.
#[test]
fn a_thread_that_only_tells_makes_no_instance_and_its_events_are_dropped() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "only-told",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    let tell = || engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tell();
                tell();
            })
            .join()
            .expect("the telling thread");
    });
    let stats = plugin.stats();
    assert_eq!(
        (stats.events_dropped, stats.observed, stats.instantiations),
        (2, 0, 1),
        "dropped, and no instance made: {stats:?}"
    );
    let said = warnings(plugin.name());
    assert_eq!(
        said.iter()
            .filter(|line| line.contains("names no decider"))
            .count(),
        1,
        "one warning a plugin, at its first drop: {said:?}"
    );
    tell();
    let stats = plugin.stats();
    assert_eq!(
        (stats.events_dropped, stats.observed, stats.instantiations),
        (2, 1, 1),
        "the loading thread is told: {stats:?}"
    );
    assert_eq!(
        pick_ids(&decide(&engine, &pool(4), 1)).first().copied(),
        Some(101),
        "and decides on the instance that heard"
    );
}

/// A thread that decided or warmed under no name is a decider of its own:
/// what it tells reaches the instance it decides on.
#[test]
fn a_thread_that_decided_or_warmed_under_no_name_hears_what_it_tells() {
    let runtime = runtime();
    // On a thread that did not load: the pick before the dispatch (none
    // where the thread warmed first), the pick after it, the drops.
    let heard_by = |name: &str, warm_first: bool| {
        let plugin = load(
            &runtime,
            name,
            guest::record_guest(),
            mode::COUNT_DISPATCHES,
        );
        let engine = plugin_engine(&plugin);
        let picks = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
                    let before = if warm_first {
                        warmed(&plugin);
                        None
                    } else {
                        pick()
                    };
                    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                    (before, pick())
                })
                .join()
                .expect("the thread")
        });
        (picks, plugin.stats().events_dropped)
    };
    assert_eq!(
        heard_by("own-decided", false),
        ((Some(100), Some(101)), 0),
        "it decided first"
    );
    assert_eq!(
        heard_by("own-warmed", true),
        ((None, Some(101)), 0),
        "it warmed first"
    );
}

/// A pair decided as a named decider keeps the decider's instance on the
/// thread until the plugin's last close of the decision: each side's
/// resource is dropped on it. So two pairs decided by two threads that act
/// as one decider use the fuel two pairs use under no name, to the unit, on
/// one instance.
#[test]
fn a_named_pair_decision_keeps_its_instance_to_the_plugins_last_close() {
    let runtime = runtime();
    // The fuel of two pair decisions, one after the other, and the
    // instances made: on this thread under no name, or on two threads that
    // act as one decider in turn.
    let two_pairs = |name: &str, named: bool| {
        let plugin = load(&runtime, name, guest::record_guest(), mode::LEAST_IN_FLIGHT);
        let prefill = plugin_profile(&plugin, None, None, None);
        let decode = plugin_profile(&plugin, None, None, None);
        let engine = pair_engine(vec![prefill, decode], 1);
        let loaded = plugin.stats().fuel_used;
        let pair = || {
            let decision = pair_decide(&engine);
            assert!(decision.pick().is_some(), "a pair pick, not {decision:?}");
            plugin.stats().fuel_used
        };
        let acting = || {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        let _turn = DecisionSlot::new(4).enter();
                        pair()
                    })
                    .join()
                    .expect("the acting thread")
            })
        };
        let first = if named { acting() } else { pair() };
        let second = if named { acting() } else { pair() };
        (
            first - loaded,
            second - first,
            plugin.stats().instantiations,
        )
    };
    let under_no_name = two_pairs("pair-own", false);
    assert_eq!(under_no_name.2, 1);
    assert_eq!(
        two_pairs("pair-named", true),
        under_no_name,
        "the same calls, the two destructors of each pair among them, on one instance"
    );
}

/// The membership is fed to every instance of a plugin whose row lists it.
/// A change is told once, on a thread that only tells, and makes no guest
/// call there and no instance; each decider's instance hears it before its
/// next decision, one call a change. An instance made after a telling hears
/// the list as it stands first.
///
/// The second decider's instance is made by its first decision: a lazy
/// `init`, on the wall clock. A runner that holds that thread for the init
/// frame has the host end the `init`. The decision then falls back, to the
/// very worker the plugin names there, and the instance the decider's next
/// decision makes hears the changes told so far as one snapshot: one call
/// fewer, on one instance more. A destructor's time costs an instance the
/// same way, and the close rebuilds it. So a try that counts another
/// instance than the two the case makes, where a decision's thread was off
/// the CPU half the init frame, says so and is made again ([`tried`]). Each
/// try reads its picks and its counts before it asserts any, and a try no
/// hold reached asserts what it read.
#[test]
fn a_membership_told_once_reaches_every_deciders_instance_before_it_decides() {
    use smg_policy_controller::tracking::Membership;
    const CASE: &str = "a_membership_told_once_reaches_every_deciders_instance_before_it_decides";
    let runtime = runtime();
    tried(|made| {
        let plugin = load(&runtime, "fed", guest::record_guest(), mode::COUNT_MEMBERS);
        let engine = plugin_engine(&plugin);
        // The guest's pick names the row at the count of members its
        // instance heard: on a thread of its own, acting as the decider of
        // that number. The longest any such thread was off the CPU is kept.
        let worst = std::cell::Cell::new(std::time::Duration::ZERO);
        let acting = |number: u8| {
            let (pick, held) = std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                        let pick = {
                            let _turn = DecisionSlot::new(number).enter();
                            pick_ids(&decide(&engine, &pool(4), 1)).first().copied()
                        };
                        (pick, off_cpu_since(wall, cpu))
                    })
                    .join()
                    .expect("the acting thread")
            });
            worst.set(worst.get().max(held));
            pick
        };
        // Told on a thread that names no decider and never decides.
        let tell = |change: Membership<'_>| {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| engine.on_membership(&change))
                    .join()
                    .expect("the telling thread");
            });
        };
        let untold = acting(0);
        tell(Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
        let told = plugin.stats();
        let heard = acting(0);
        // The second decider's first decision, and the instances made
        // lazily by its end: its own.
        let late = (acting(1), plugin.stats().lazy_instantiations);
        tell(Membership::Added(WorkerId(102)));
        let added = (acting(1), acting(0));
        tell(Membership::Removed(WorkerId(100)));
        tell(Membership::Removed(WorkerId(101)));
        let removed = (acting(0), acting(1));
        let stats = plugin.stats();
        // The witness: the load's instance and the second decider's are the
        // two a try makes. Another one, made or only begun, in a try where
        // a decision's thread was off the CPU half the init frame, is the
        // host's answer to that hold. With no such hold it is the host's
        // own, and the try asserts.
        let another = (stats.instantiations, stats.lazy_instantiations) != (2, 1);
        if another && worst.get() >= INIT_FRAME / 2 && made < TRIES {
            made_again(
                CASE,
                "a decision's thread was off the CPU half the init frame, the bound of a lazy init and of a close's rebuild",
            );
            return false;
        }
        assert_eq!(untold, Some(100), "no member told yet");
        assert_eq!(
            (told.observed, told.instantiations, told.events_dropped),
            (0, 1, 0),
            "a telling calls no guest, makes no instance and drops nothing: {told:?}"
        );
        assert_eq!(heard, Some(102), "the first decider's instance heard");
        assert_eq!(
            late,
            (Some(102), 1),
            "an instance made after the telling, by its decider's first decision, hears the list as it stands"
        );
        assert_eq!(added, (Some(103), Some(103)));
        assert_eq!(removed, (Some(101), Some(101)));
        assert_eq!(
            (
                stats.observed,
                stats.instantiations,
                stats.lazy_instantiations
            ),
            (8, 2, 1),
            "one call a decider a change, the snapshot among them, on the two instances: {stats:?}"
        );
        true
    });
}

/// An instance new to the feed hears the membership as it stands, as one
/// snapshot, whatever the host told: the first instance where only changes
/// were told, an instance remade after a fault, which then counts what one
/// that heard every change counts, and an instance made at the warm-up,
/// which hears there and not at its first decision. A change told before a
/// lease's event is heard before it.
#[test]
fn an_instance_made_anew_hears_the_membership_as_it_stands() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(&runtime, "anew", guest::record_guest(), mode::COUNT_MEMBERS);
    let engine = plugin_engine(&plugin);
    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    // Five workers come, one change each, and no snapshot was ever told: an
    // instance new to the feed still hears them as one snapshot.
    for id in 100..105 {
        engine.on_membership(&Membership::Added(WorkerId(id)));
    }
    // Five members over four rows: the guest's pick names no row, an answer
    // outside the contract, and its instance is dropped.
    let _ = pick();
    let faulted = plugin.stats();
    assert_eq!((faulted.invalid, faulted.observed), (1, 1), "{faulted:?}");
    engine.on_membership(&Membership::Removed(WorkerId(104)));
    engine.on_membership(&Membership::Removed(WorkerId(103)));
    assert_eq!(pick(), Some(103), "three members stand");
    let remade = plugin.stats();
    assert_eq!(
        remade.observed - faulted.observed,
        1,
        "one snapshot, not the two changes: {remade:?}"
    );
    let (at_the_warm_up, at_its_decision) = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                warmed(&plugin);
                let warmed = plugin.stats().observed;
                assert_eq!(pick(), Some(103), "the warmed instance counts three");
                (warmed - remade.observed, plugin.stats().observed - warmed)
            })
            .join()
            .expect("the warming thread")
    });
    assert_eq!((at_the_warm_up, at_its_decision), (1, 0));
    // A change, then a lease's event with no decision between: the instance
    // hears the change first, then the event.
    let before = plugin.stats().observed;
    engine.on_membership(&Membership::Removed(WorkerId(102)));
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    assert_eq!(plugin.stats().observed - before, 2);
    assert_eq!(pick(), Some(102), "two members stand");
}

/// The feed keeps the models a worker left: an instance level with the feed
/// hears one call for the word, and counts the members it counted.
#[test]
fn a_guest_level_with_the_feed_hears_the_models_a_worker_left() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(&runtime, "left", guest::record_guest(), mode::COUNT_MEMBERS);
    let engine = plugin_engine(&plugin);
    let pick = || pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    assert_eq!(pick(), Some(102), "two members stand");
    let level = plugin.stats().observed;
    let models = ["a".to_owned()];
    engine.on_membership(&Membership::left(WorkerId(100), &models));
    assert_eq!(pick(), Some(102), "the same two members");
    assert_eq!(plugin.stats().observed, level + 1, "one call for the word");
}

/// The feed keeps its last 64 changes. An instance behind that many hears
/// each, one call a change; one further behind, or behind a snapshot, hears
/// the list as it stands, as one snapshot.
#[test]
fn an_instance_behind_more_than_the_feed_keeps_hears_one_snapshot() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "behind",
        guest::record_guest(),
        mode::COUNT_MEMBERS,
    );
    let engine = plugin_engine(&plugin);
    // The row the guest picks, and the `observe` calls the decision's
    // catch-up made.
    let pick = || {
        let before = plugin.stats().observed;
        let row = pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
        (row, plugin.stats().observed - before)
    };
    let come_and_go = |times: u32| {
        for _ in 0..times {
            engine.on_membership(&Membership::Added(WorkerId(300)));
            engine.on_membership(&Membership::Removed(WorkerId(300)));
        }
    };
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100)]));
    assert_eq!(pick(), (Some(101), 1), "new to the feed: the snapshot");
    come_and_go(32);
    assert_eq!(pick(), (Some(101), 64), "sixty-four behind: each change");
    come_and_go(32);
    engine.on_membership(&Membership::Added(WorkerId(301)));
    assert_eq!(pick(), (Some(102), 1), "sixty-five behind: one snapshot");
    engine.on_membership(&Membership::Added(WorkerId(302)));
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    engine.on_membership(&Membership::Added(WorkerId(303)));
    assert_eq!(
        pick(),
        (Some(103), 1),
        "behind a snapshot: the list as it stands"
    );
    assert_eq!(pick(), (Some(103), 0), "level: no call");
    // A replacement is folded into the list: an instance made after it
    // counts what stands.
    engine.on_membership(&Membership::replaced(WorkerId(303), WorkerId(304)));
    engine.on_membership(&Membership::Removed(WorkerId(304)));
    plugin.forget_instance();
    assert_eq!(
        pick(),
        (Some(102), 1),
        "the two that stand, as one snapshot"
    );
}

/// A warm-up whose telling faults the instance leaves no mark for a decision:
/// the next decision on that thread makes an instance at its open and is the
/// plugin's own, not the fallback's.
#[test]
fn a_warm_up_whose_telling_faults_leaves_the_next_decision_the_plugins_own() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "cold-warm",
        guest::record_guest(),
        mode::TRAP_ON_FIVE_MEMBERS,
    );
    let engine = plugin_engine(&plugin);
    let five: Vec<WorkerId> = (100..105).map(WorkerId).collect();
    engine.on_membership(&Membership::Snapshot(&five));
    // The warm-up tells the snapshot of five and this guest traps on it: the
    // fault is the breaker's, and the slot is cold.
    plugin.warm().expect("the warm-up's own work is done");
    let warmed = plugin.stats();
    assert_eq!((warmed.traps, warmed.observed), (1, 1), "{warmed:?}");
    assert!(!plugin.is_warm(), "the faulted instance is gone");
    // Two members go. The decision makes an instance at its open, which
    // hears the three that stand, and its pick names the row at that count:
    // not the row the fallback picks for this pool and seed.
    for id in [104, 103] {
        engine.on_membership(&Membership::Removed(WorkerId(id)));
    }
    let picked = pick_ids(&decide(&engine, &pool(4), 1)).first().copied();
    let decided = plugin.stats();
    assert_eq!(picked, Some(103), "the plugin's own pick: {decided:?}");
    assert_eq!(
        decided.observed - warmed.observed,
        1,
        "the instance made at the open hears one snapshot: {decided:?}"
    );
    assert!(decided.answered > warmed.answered, "{decided:?}");
}

/// A thread under no name that came by its first instance at a close's
/// rebuild (its first `init` ended by the decision's clamp) is a decider of
/// its own from there: a lease's event it tells is heard, not dropped.
#[test]
fn a_thread_whose_first_instance_the_close_rebuilt_hears_what_it_tells() {
    tried(a_thread_whose_first_instance_the_close_rebuilt_hears_what_it_tells_in_a_try);
}

/// One try of the case above, told its number: whether it stood.
fn a_thread_whose_first_instance_the_close_rebuilt_hears_what_it_tells_in_a_try(made: u32) -> bool {
    const CASE: &str = "a first instance the close rebuilt";
    // The host's warnings are kept from here on: a rebuild that fails says why.
    said();
    let began = TRY_BEGAN.get();
    let runtime = runtime();
    let heavy = load(
        &runtime,
        "rebuilt-first",
        guest::record_guest(),
        mode::BURN_IN_INIT,
    );
    let engine = plugin_engine(&heavy);
    // A thread that did not load and was not warmed: the load's instance is
    // not its to take, and its first `init` runs under a decision's clamp.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                TRY_BEGAN.set(began);
                let (wall, cpu) = (std::time::Instant::now(), bound_clock());
                let _ = engine.decide(
                    &RequestFacts {
                        budget_micros: 200,
                        ..facts(1)
                    },
                    &four(),
                    &pool(4),
                    &Constraints::default(),
                    &mut Scratch::new(),
                );
                let held = off_cpu_since(wall, cpu);
                if !the_close_rebuilt(CASE, &heavy, held, made) {
                    return false;
                }
                let rebuilt = heavy.stats();
                assert_eq!(rebuilt.budget_spent, 1, "the clamp ended it: {rebuilt:?}");
                // Three tellings: after the rebuild, after a second decision
                // on that instance, and after a warm-up there.
                let tell = || {
                    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                };
                tell();
                let _ = decide(&engine, &pool(4), 2);
                tell();
                heavy.warm().expect("the thread is warm already");
                tell();
                let told = heavy.stats();
                assert_eq!(
                    (told.observed - rebuilt.observed, told.events_dropped),
                    (3, 0),
                    "the thread is a decider of its own: {told:?}"
                );
                true
            })
            .join()
            .expect("the thread that did not load")
    })
}

/// What a host's scope does inside a decision, after the plugin's scope
/// opened and brought its decider's instance.
#[derive(Clone, Copy, Debug)]
enum Inside {
    /// Acts as another decider and warms the plugin there.
    WarmsAs(DecisionSlot),
    /// Ends the thread's name and tells a dispatch under none.
    TellsUnderNoName,
    /// Hands the membership feed one worker that comes.
    HandsAMember(WorkerId),
    /// Moves the case's own clock to a time, in microseconds.
    MovesTheClock(&'static AtomicU64, u64),
    /// Lets another thread take a whole turn as a decider, and exit.
    TurnsElsewhereAs(DecisionSlot),
}

thread_local! {
    /// The name a case's thread holds, where a scope of the host can end it.
    static HELD_NAME: std::cell::RefCell<Option<smg_policy_controller::engine::DecisionSlotGuard>> =
        const { std::cell::RefCell::new(None) };
}

/// A stage of the host's own whose scope acts on the plugin inside a
/// decision, on the deciding thread.
struct ActsInside {
    plugin: Arc<Plugin>,
    tells: Arc<Engine>,
    does: Inside,
}

impl std::fmt::Debug for ActsInside {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActsInside")
            .field("does", &self.does)
            .finish_non_exhaustive()
    }
}

impl smg_policy_controller::engine::Filter for ActsInside {
    fn name(&self) -> &'static str {
        "acts-inside"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_policy_controller::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_policy_controller::engine::DecisionScope for ActsInside {
    fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {
        match self.does {
            Inside::WarmsAs(other) => {
                let _turn = other.enter();
                warmed(&self.plugin);
            }
            Inside::TellsUnderNoName => {
                drop(HELD_NAME.take());
                self.tells
                    .on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
            }
            Inside::HandsAMember(worker) => {
                self.tells
                    .on_membership(&smg_policy_controller::tracking::Membership::Added(worker));
            }
            Inside::MovesTheClock(clock, to) => clock.store(to, Ordering::Relaxed),
            Inside::TurnsElsewhereAs(other) => {
                let tells = &self.tells;
                std::thread::scope(|scope| {
                    scope
                        .spawn(|| {
                            let _turn = other.enter();
                            let _ = decide(tells, &pool(4), 1);
                        })
                        .join()
                        .expect("the thread that takes its turn elsewhere");
                });
            }
        }
    }

    fn close(&self) {}
}

/// An engine over the plugin whose decision, after the plugin's scope
/// opened, runs a scope of the host that acts inside it.
fn engine_that(plugin: &Arc<Plugin>, tells: &Arc<Engine>, does: Inside) -> Engine {
    single_engine(profile_that(plugin, tells, does))
}

/// The plugin's profile with a scope of the host opened after the plugin's.
fn profile_that(plugin: &Arc<Plugin>, tells: &Arc<Engine>, does: Inside) -> Profile {
    let mut profile = plugin_profile(plugin, None, None, None);
    profile.filters.push(Box::new(ActsInside {
        plugin: Arc::clone(plugin),
        tells: Arc::clone(tells),
        does,
    }));
    profile
}

/// What a thread still holds of one named decider goes back to that
/// decider's rest before the thread acts as another: the first decider's
/// instance, with what it heard, is there for whoever acts as it next.
#[test]
fn a_thread_that_turns_to_another_name_gives_the_held_one_back_first() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "turns",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    let turning = engine_that(&plugin, &plain, Inside::WarmsAs(DecisionSlot::new(6)));
    let first = DecisionSlot::new(5);
    {
        let _turn = first.enter();
        for _ in 0..2 {
            plain.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
        }
        // The plugin's scope brings the first decider's instance, and the
        // host's scope then acts as the second, inside the decision.
        let _ = decide(&turning, &pool(4), 1);
    }
    let picked = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _turn = first.enter();
                pick_ids(&decide(&plain, &pool(4), 2)).first().copied()
            })
            .join()
            .expect("the thread that acts as the first decider next")
    });
    let stats = plugin.stats();
    assert_eq!(picked, Some(102), "the instance that heard two: {stats:?}");
    // The warm-up inside the decision ran on this thread: one made again is
    // an instance more.
    assert_eq!(
        stats.instantiations,
        2 + WARMED_AGAIN.get(),
        "one a decider: {stats:?}"
    );
}

/// A thread that still holds a named decider's instance when its name ends
/// gives it back before it acts under no name: what it then tells reaches
/// its own instance and not the decider's.
#[test]
fn a_thread_whose_name_ends_gives_the_held_decider_back_before_it_acts_as_itself() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "ends",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    let ending = engine_that(&plugin, &plain, Inside::TellsUnderNoName);
    let pick = |seed| pick_ids(&decide(&plain, &pool(4), seed)).first().copied();
    // The thread is a decider of its own: its instance heard nothing.
    assert_eq!(pick(1), Some(100));
    let first = DecisionSlot::new(5);
    HELD_NAME.set(Some(first.enter()));
    for _ in 0..2 {
        plain.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    // The plugin's scope brings the first decider's instance. The host's
    // scope then ends the name and tells a dispatch under none.
    let _ = decide(&ending, &pool(4), 2);
    assert_eq!(DecisionSlot::acting(), None, "the scope ended the name");
    assert_eq!(pick(3), Some(101), "the thread's own instance heard it");
    let as_first = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _turn = first.enter();
                pick(4)
            })
            .join()
            .expect("the thread that acts as the first decider")
    });
    assert_eq!(as_first, Some(102), "its two dispatches and no more");
}

/// Both opens of a decision run on the instance its bring brought. A
/// thread acts as a decider whose instance heard two dispatches, and
/// decides a pair. Between the pair's two opens another thread takes a
/// whole turn as that decider: the rest then names that thread's place and
/// keeps an instance that heard none. The pair's second side still decides
/// on the instance its first side brought. An open that checked that hold
/// would find it stale, drop it and decide on the instance at rest.
#[test]
fn both_opens_of_a_pair_decide_on_the_instance_its_bring_brought() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "one-pair",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    let first = DecisionSlot::new(5);
    // The first side's profile: the plugin's scope opens and brings the
    // decider's instance, then the host's scope lets another thread take a
    // turn as that decider. The second side's open follows it.
    let prefill = profile_that(&plugin, &plain, Inside::TurnsElsewhereAs(first));
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    let _turn = first.enter();
    for _ in 0..2 {
        plain.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    }
    let decision = pair_decide(&engine);
    let Some(pick) = decision.pick() else {
        panic!("a pair pick, not {decision:?}")
    };
    let lead = pick.targets.first().expect("a target");
    let stats = plugin.stats();
    assert_eq!(
        (lead.id, lead.pair.map(|p| p.decode_id)),
        (WorkerId(102), Some(WorkerId(202))),
        "each side counted the two dispatches its instance heard: {stats:?}"
    );
    assert_eq!(
        stats.instantiations, 2,
        "the other thread's turn made an instance of its own: {stats:?}"
    );
}

/// An instance hears the feed once a decision, at the first open that
/// reaches it. A change handed between a pair's two opens is not told
/// inside the pair, where the first side's resource is live on the
/// instance: it is heard at the next decision's first open.
#[test]
fn a_change_handed_between_a_pairs_opens_is_heard_at_the_next_decision() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "between-sides",
        guest::record_guest(),
        mode::COUNT_MEMBERS,
    );
    let plain = Arc::new(plugin_engine(&plugin));
    // The first side's profile: the plugin's scope opens, then the host's
    // scope hands a change. The second side's open follows it.
    let prefill = profile_that(&plugin, &plain, Inside::HandsAMember(WorkerId(900)));
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    plain.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin
        .warm()
        .expect("it warms, and hears the two that stand");
    let warmed = plugin.stats().observed;
    let _ = pair_decide(&engine);
    let paired = plugin.stats().observed;
    assert_eq!(paired - warmed, 0, "no telling inside the pair");
    let _ = pair_decide(&engine);
    assert_eq!(
        plugin.stats().observed - paired,
        1,
        "the first pair's change, at the second pair's first open"
    );
}

/// The open that reaches the instance may be a pair's second. A pair whose
/// first side meets the open breaker and whose second side opens at the
/// cool-down's end makes its instance there: that instance hears the list
/// as it stands before the side reads it.
#[test]
fn a_pair_that_straddles_the_cool_down_hears_the_feed_at_the_open_that_reaches_it() {
    use smg_policy_controller::tracking::Membership;
    const START: u64 = 1_000_000;
    const COOL_DOWN: u64 = 1_000_000;
    // The host's clock is the case's own: the cool-down ends when the first
    // side's scope says so, whatever the runner's clock reads.
    static CLOCK: AtomicU64 = AtomicU64::new(START);
    let runtime = Arc::new(
        Runtime::with_clock(&OpenDecisions::new(), || CLOCK.load(Ordering::Relaxed))
            .expect("a runtime"),
    );
    let mut row = spec(
        "straddles",
        guest::record_guest(),
        mode::TRAP_ON_FIVE_MEMBERS,
    );
    row.breaker = BreakerSpec {
        failures: 1,
        window_micros: 60_000_000,
        cooldown_micros: COOL_DOWN,
    };
    let plugin = loaded(&runtime, row);
    let plain = Arc::new(plugin_engine(&plugin));
    // The first side's profile: the plugin's open, then the host's scope
    // moves the clock to the cool-down's end. The second side's open follows.
    let ends = Inside::MovesTheClock(&CLOCK, START + COOL_DOWN);
    let prefill = profile_that(&plugin, &plain, ends);
    let decode = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![prefill, decode], 1);
    // The warm-up tells the snapshot of five and this guest traps on it: the
    // one failure opens the breaker, and the slot is cold.
    let five: Vec<WorkerId> = (100..105).map(WorkerId).collect();
    plain.on_membership(&Membership::Snapshot(&five));
    plugin.warm().expect("the warm-up's own work is done");
    let warmed = plugin.stats();
    assert_eq!((warmed.traps, warmed.breaker_opens), (1, 1), "{warmed:?}");
    // Four go and one stands.
    for id in [104, 103, 102, 101] {
        plain.on_membership(&Membership::Removed(WorkerId(id)));
    }
    let decision = pair_decide(&engine);
    let paired = plugin.stats();
    assert_eq!(
        (
            paired.bypassed - warmed.bypassed,
            paired.lazy_instantiations - warmed.lazy_instantiations
        ),
        (1, 1),
        "the first side met the open breaker, the second made an instance: {paired:?}"
    );
    assert_eq!(
        (paired.observed - warmed.observed, paired.invalid),
        (1, 0),
        "the instance made at the second side's open hears one snapshot: {paired:?}"
    );
    // The second side's pick is the plugin's own: the row at one member.
    let decode_id = decision
        .pick()
        .and_then(|pick| pick.targets.first())
        .and_then(|lead| lead.pair)
        .map(|pair| pair.decode_id);
    assert_eq!(decode_id, Some(WorkerId(201)), "{decision:?}");
}

/// The pick of one decision over the pool of four that the plugin itself
/// answered: its calls answered and its pick within the pool. A decision
/// that went by the plugin, or took an answer out of the pool, names a row
/// too, the fallback's.
fn own_pick(plugin: &Plugin, engine: &Engine) -> Option<u64> {
    let before = plugin.stats();
    let picked = pick_ids(&decide(engine, &pool(4), 1)).first().copied();
    let after = plugin.stats();
    assert!(
        after.answered > before.answered && after.invalid == before.invalid,
        "the plugin's own answer: {before:?}, then {after:?}"
    );
    picked
}

/// In a told event the feed comes first: a change handed ahead of a lease's
/// event is heard ahead of it, so what the event's call reads of the
/// membership is the membership as the host told it.
#[test]
fn a_change_told_ahead_of_a_leases_event_is_heard_ahead_of_it() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "ahead",
        guest::record_guest(),
        mode::MEMBERS_AT_A_DISPATCH,
    );
    let engine = plugin_engine(&plugin);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin.warm().expect("it warms, and hears the two");
    // One comes, then a dispatch with no decision between: the guest keeps
    // the count as it stands at the dispatch. Told the other way round the
    // dispatch reads two, and the pick names the row before.
    engine.on_membership(&Membership::Added(WorkerId(102)));
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(103), "three stood at the dispatch: {stats:?}");
}

/// An instance behind the feed hears a relabelling and a resize as the host
/// told them, each by its own kind.
#[test]
fn an_instance_behind_hears_a_relabelling_and_a_resize_each_as_told() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(&runtime, "kinds", guest::record_guest(), mode::COUNT_KINDS);
    let engine = plugin_engine(&plugin);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin.warm().expect("it warms");
    let warmed = plugin.stats().observed;
    engine.on_membership(&Membership::Relabelled(WorkerId(100)));
    engine.on_membership(&Membership::Resized(WorkerId(101)));
    // One up the relabelling and two up the resize: the row at three.
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(103), "{stats:?}");
    assert_eq!(stats.observed - warmed, 2, "one call a change: {stats:?}");
}

/// The feed's list holds a worker once, and a replacement of a worker it
/// does not hold brings the new one, once: an instance new to the feed
/// hears the list as it stands.
#[test]
fn the_feeds_list_holds_a_worker_once_and_takes_an_unknown_ones_replacement() {
    use smg_policy_controller::tracking::Membership;
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "folds",
        guest::record_guest(),
        mode::COUNT_MEMBERS,
    );
    let engine = plugin_engine(&plugin);
    for _ in 0..2 {
        engine.on_membership(&Membership::Added(WorkerId(100)));
    }
    for old in [900, 901] {
        engine.on_membership(&Membership::replaced(WorkerId(old), WorkerId(101)));
    }
    // Two members: one told twice, and one for a worker the list never
    // held, told twice. A list that held either twice, or left the second
    // out, names another row of the pool.
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(102), "{stats:?}");
}

/// A thread that acted under a name alone is no decider of its own: what it
/// tells under no name afterwards is dropped and counted, and makes no
/// instance.
#[test]
fn a_thread_that_acted_under_a_name_alone_only_tells_under_none() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "named-alone",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                {
                    let _turn = DecisionSlot::new(3).enter();
                    let _ = decide(&engine, &pool(4), 1);
                }
                let decided = plugin.stats();
                engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
                let told = plugin.stats();
                assert_eq!(
                    (
                        told.events_dropped,
                        told.observed - decided.observed,
                        told.instantiations
                    ),
                    (1, 0, 1),
                    "dropped and counted, no instance made: {told:?}"
                );
            })
            .join()
            .expect("the thread that acted under a name alone");
    });
}

/// A named decider's latency window is one, on whichever thread acts as it:
/// the ring rests with the instance. Sixty-three slow samples taken on one
/// thread and the sixty-fourth, a warm decision's on another, read the
/// window slow: the ring came with the instance and went back with it at
/// every turn.
#[test]
fn a_named_deciders_latency_window_is_one_on_whichever_thread_acts_as_it() {
    const CALIBRATION: u64 = 10;
    let runtime = runtime();
    let mut row = spec("ring-rests", guest::record_guest(), mode::BURN_IN_INIT);
    row.fuel_per_micro = CALIBRATION;
    let plugin = loaded(&runtime, row);
    let init_micros = u32::try_from(plugin.stats().fuel_used.div_ceil(CALIBRATION)).expect("fits");
    let engine = plugin_engine(&plugin);
    // A cold decision's `init` is slow by its own fuel. A warm decision
    // brings the ring once and gives it back once.
    let decide_as_eight = |seed: u64, cold: bool| {
        let _turn = DecisionSlot::new(8).enter();
        if cold {
            plugin.forget_instance();
        }
        let decision = engine.decide(
            &RequestFacts {
                budget_micros: init_micros / 3 * 4,
                ..facts(seed)
            },
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(matches!(decision, Decision::Pick(_)), "decision {seed}");
    };
    for seed in 1..=63 {
        decide_as_eight(seed, true);
    }
    let short = plugin.stats();
    assert_eq!(short.breaker_opens, 0, "one sample short: {short:?}");
    std::thread::scope(|scope| {
        scope
            .spawn(|| decide_as_eight(64, false))
            .join()
            .expect("the second thread");
    });
    let s = plugin.stats();
    assert_eq!(
        s.breaker_opens, 1,
        "the sixty-fourth sample, on another thread, completes the window: {s:?}"
    );
}

/// The thread that loaded is told as before, also after it forgot the
/// load's instance: the next event it tells makes an instance and is heard.
#[test]
fn the_thread_that_loaded_hears_what_it_tells_after_it_forgot_its_instance() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "forgot",
        guest::record_guest(),
        mode::COUNT_DISPATCHES,
    );
    let engine = plugin_engine(&plugin);
    plugin.forget_instance();
    engine.on_dispatched(&Lease::default(), &Target::default(), Side::Single, 3);
    let told = plugin.stats();
    assert_eq!(
        (told.observed, told.events_dropped, told.lazy_instantiations),
        (1, 0, 1),
        "an instance made at the telling, and told: {told:?}"
    );
}

/// Under a name the load's instance warms no decider by waiting: it is the
/// first asker's. A warm-up that asks before it warms makes an instance for
/// every decider, the first by taking the load's.
#[test]
fn a_warm_up_that_asks_first_warms_every_named_decider() {
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "asks-first",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    assert!(plugin.is_warm(), "the thread that loaded, by the load's");
    for name in 0..3 {
        let _turn = DecisionSlot::new(name).enter();
        assert!(!plugin.is_warm(), "decider {name} holds none yet");
        plugin.warm().expect("it warms");
        assert!(plugin.is_warm(), "decider {name} holds its own");
    }
    let stats = plugin.stats();
    assert_eq!(
        stats.instantiations, 3,
        "one a decider, the load's among them: {stats:?}"
    );
    assert!(!plugin.is_warm(), "the load's instance went to a decider");
}

/// The models a worker left reach a guest as one word, with its names, ahead
/// of the relabelling of the same write: on either world an instance behind
/// the feed hears the two in the host's order, one call each.
#[test]
fn an_instance_behind_hears_the_models_a_worker_left_ahead_of_its_relabelling() {
    use smg_policy_controller::tracking::Membership;
    let models = ["m-chat".to_owned(), "m-embed".to_owned()];
    for (world, component) in [
        ("left-record", guest::record_guest()),
        ("left-packed", guest::packed_guest()),
    ] {
        let runtime = runtime();
        let plugin = load(&runtime, world, component, mode::COUNT_LEFT);
        let engine = plugin_engine(&plugin);
        engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
        plugin.warm().expect("it warms");
        let warmed = plugin.stats().observed;
        engine.on_membership(&Membership::left(WorkerId(100), &models));
        engine.on_membership(&Membership::Relabelled(WorkerId(100)));
        // One up each name of the word and one more up the relabelling
        // behind it: the row at three. A word not told, or told behind its
        // relabelling, names another row; one told twice names none of the
        // pool; a name the guest cannot read is its fault.
        let picked = own_pick(&plugin, &engine);
        let stats = plugin.stats();
        assert_eq!(picked, Some(103), "{world}: {stats:?}");
        assert_eq!(stats.observed - warmed, 2, "{world}: {stats:?}");
        assert_eq!(stats.traps, 0, "{world}: {stats:?}");
    }
}

/// A module built against an older minor of the world is refused by the
/// host's range, both versions named, before anything of it is linked, in
/// either world. Its shapes are today's here, so nothing but its version
/// stops it.
#[test]
fn a_module_of_an_older_minor_is_refused_by_the_range() {
    let runtime = runtime();
    for world in ["policy", "policy-packed"] {
        let older = spec(
            "older",
            guest::older_minor_guest(world),
            mode::LEAST_IN_FLIGHT,
        );
        let err = Plugin::load(&runtime, older).expect_err("refused");
        assert!(matches!(err, LoadError::Abi { .. }), "{world}: {err}");
        assert_eq!(
            err.to_string(),
            "plugin older: targets smg:policy@1.1.0; this host accepts 1.2",
            "{world}"
        );
    }
}

/// A snapshot names ids and no pool. An instance further behind than the
/// feed keeps hears the list as it stands, in one call, and no word of the
/// models a worker left among the changes it missed.
#[test]
fn an_instance_that_hears_a_snapshot_for_what_it_missed_hears_no_left_word() {
    use smg_policy_controller::tracking::Membership;
    let models = ["m-chat".to_owned(), "m-embed".to_owned()];
    let runtime = runtime();
    let plugin = load(&runtime, "behind", guest::record_guest(), mode::COUNT_LEFT);
    let engine = plugin_engine(&plugin);
    engine.on_membership(&Membership::Snapshot(&[WorkerId(100), WorkerId(101)]));
    plugin.warm().expect("it warms");
    let warmed = plugin.stats().observed;
    engine.on_membership(&Membership::left(WorkerId(100), &models));
    // The feed keeps 64 changes, and the word is the oldest of 65.
    for _ in 0..64 {
        engine.on_membership(&Membership::Resized(WorkerId(101)));
    }
    let picked = own_pick(&plugin, &engine);
    let stats = plugin.stats();
    assert_eq!(picked, Some(100), "no name was counted: {stats:?}");
    assert_eq!(stats.observed - warmed, 1, "one snapshot: {stats:?}");
}
