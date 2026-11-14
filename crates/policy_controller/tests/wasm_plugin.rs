//! The WebAssembly plugin host against the test guests: what loads and what
//! is refused, the honest pick through both worlds, the four host imports,
//! and the failure rules (a decline keeps the instance, a trap, an exhausted
//! budget or an invalid answer drops it and counts on the breaker, the
//! breaker bypasses, the fallbacks and the closed policy), one resource per
//! decision and per side of a pair, the lifecycle once per engine build, and
//! the instances kept per thread.
#![cfg(feature = "wasm")]
#![expect(
    clippy::expect_used,
    reason = "the guests' assembly and the loads run inside the tests; a failure there is the test's failure"
)]

mod support {
    pub mod wasm_guest;
}

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use smg_policy_controller::{
    builtins::{Best, Healthy, LeastRequests},
    wasm::{
        Abi, Bound, BreakerSpec, Failure, LoadError, Plugin, PluginSpec, Reveal, Runtime,
        WasmFilter, WasmPicker, WasmScorer, World,
    },
    Binding, Candidate, Constraints, Decision, Draw, Engine, Health, Lease, Lifecycle, Needs,
    PairPools, Pick, Plan, PoolFacts, Profile, RequestFacts, Scratch, ShedReason, Side, Target,
    Weighted, WorkerId,
};
use support::wasm_guest::{self as guest, canonical, counter, mode};
use wit_component::{ComponentEncoder, StringEncoding};
use wit_parser::Resolve;

const FRAME_MICROS: u32 = 10_000;
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
    Arc::new(Runtime::new().expect("a runtime"))
}

fn spec(name: &str, component: Vec<u8>, mode: u8) -> PluginSpec {
    let mut spec = PluginSpec::new(name, component);
    spec.config = vec![mode];
    spec.budget_micros = FRAME_MICROS;
    spec.metrics = COUNTERS.iter().map(|s| s.to_string()).collect();
    spec
}

fn load(runtime: &Arc<Runtime>, name: &str, component: Vec<u8>, mode: u8) -> Arc<Plugin> {
    Plugin::load(runtime, spec(name, component, mode)).expect("the guest loads")
}

/// `load` at the slow calibration, with its frame.
fn load_slow(runtime: &Arc<Runtime>, name: &str, component: Vec<u8>, mode: u8) -> Arc<Plugin> {
    let mut spec = spec(name, component, mode);
    spec.fuel_per_micro = SLOW_CALIBRATION;
    spec.budget_micros = SLOW_FRAME_MICROS;
    Plugin::load(runtime, spec).expect("the guest loads")
}

/// A profile whose three stages are the plugin's, each with the given
/// fallback: the one profile of its build that binds the plugin.
fn plugin_profile(
    plugin: &Arc<Plugin>,
    filter_fallback: Option<Box<dyn smg_policy_controller::Filter>>,
    scorer_fallback: Option<Box<dyn smg_policy_controller::Scorer>>,
    picker_fallback: Option<Box<dyn smg_policy_controller::Picker>>,
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
    bound: Arc<Bound>,
    filter_fallback: Option<Box<dyn smg_policy_controller::Filter>>,
    scorer_fallback: Option<Box<dyn smg_policy_controller::Scorer>>,
    picker_fallback: Option<Box<dyn smg_policy_controller::Picker>>,
) -> Profile {
    Profile {
        name: "plugin".into(),
        filters: vec![Box::new(WasmFilter::new(
            Arc::clone(&bound),
            filter_fallback,
        ))],
        scorers: vec![Weighted {
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
            sessions: None,
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

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        budget_micros: FRAME_MICROS,
        seed,
        now_micros: 777_000,
        ..RequestFacts::default()
    }
}

fn decide(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    let pool_facts = PoolFacts {
        pool_size: candidates.len() as u32,
        generation: 1,
        ..PoolFacts::default()
    };
    engine.decide(
        &facts(seed),
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
fn the_record_guests_offsets_match_the_canonical_layout() {
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
    assert_eq!(record.abi(), Abi::V1_0);
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
fn the_abi_pin_the_memory_floor_the_init_failure_and_bad_bytes_are_refused() {
    let runtime = runtime();
    let mut pinned = spec("pinned", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    pinned.abi = Some(Abi { major: 1, minor: 1 });
    let err = Plugin::load(&runtime, pinned).expect_err("refused");
    assert!(
        matches!(
            err,
            LoadError::AbiPin {
                found: Abi::V1_0,
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

    let err = Plugin::load(
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
        assert_eq!(pick.trace.survivors, 8);
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
    assert_eq!(pick.trace.survivors, 4);
    assert_eq!(plugin.stats().answered, 4, "filter asked once");
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
    let closed = Plugin::load(&runtime, closed).expect("loads");
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
    let runtime = runtime();
    let plugin = load(
        &runtime,
        "trapping",
        guest::record_guest(),
        mode::TRAP_IN_FILTER,
    );
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    let decision = decide(&engine, &candidates, 1);
    let mut ids = pick_ids(&decision);
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![100, 101, 102, 103],
        "the filter fell back to keep-all; the flat scores tie and the seed orders them"
    );
    let stats = plugin.stats();
    assert_eq!(stats.traps, 1);
    assert!(
        plugin.is_warm(),
        "the faulted instance went and the close rebuilt the slot"
    );
    assert_eq!((stats.instantiations, stats.lazy_instantiations), (2, 1));
    decide(&engine, &candidates, 2);
    let stats = plugin.stats();
    assert_eq!(
        stats.instantiations, 3,
        "rebuilt after each fault, never reused"
    );
    assert_eq!(stats.lazy_instantiations, 2);
    assert_eq!(stats.traps, 2);
}

#[test]
fn an_exhausted_budget_is_a_fuel_fault_and_the_scorer_falls_back() {
    let runtime = runtime();
    let mut spinning = spec("spinning", guest::record_guest(), mode::SPIN_IN_SCORE);
    spinning.budget_micros = 200;
    let plugin = Plugin::load(&runtime, spinning).expect("loads");
    let engine = single_engine(plugin_profile(
        &plugin,
        None,
        Some(Box::new(LeastRequests)),
        None,
    ));
    let candidates = pool(4);
    let decision = decide(&engine, &candidates, 1);
    assert_eq!(
        pick_ids(&decision).first().copied(),
        Some(103),
        "the fallback scorer decided"
    );
    let stats = plugin.stats();
    assert_eq!(stats.fuel_exhausted, 1);
    assert_eq!(stats.traps, 0);
    assert!(stats.fuel_used >= 200 * PluginSpec::DEFAULT_FUEL_PER_MICRO);
    assert!(
        plugin.is_warm(),
        "the faulted instance went and the close rebuilt the slot"
    );
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
        let plugin = load(&runtime, name, guest::record_guest(), mode);
        let engine = plugin_engine(&plugin);
        let decision = decide(&engine, &pool(4), 1);
        assert!(
            matches!(decision, Decision::Pick(_)),
            "{name}: the fallbacks decide"
        );
        let stats = plugin.stats();
        assert_eq!(stats.invalid, 1, "{name}");
        assert!(
            plugin.is_warm(),
            "{name}: the instance is dropped and the close rebuilds the slot"
        );
        assert_eq!(plugin.stats().lazy_instantiations, 1, "{name}");
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
    let plugin = Plugin::load(&runtime, closed).expect("loads");
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
        let plugin = Plugin::load(&runtime, declining).expect("loads");
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
    let plugin = Plugin::load(&runtime, spec).expect("loads");
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
/// as its level and its fields.
struct Said(Arc<std::sync::Mutex<Vec<String>>>);

impl tracing::Subscriber for Said {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        *metadata.level() <= tracing::Level::WARN
            && metadata.target().starts_with("smg_policy_controller")
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
        self.0.lock().expect("the kept lines").push(fields.0);
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
    let plugin = Plugin::load(&runtime, spec).expect("loads");
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
    let roomy = Plugin::load(&runtime, roomy).expect("loads");
    decide(&plugin_engine(&roomy), &pool(4), 1);
    let stats = roomy.stats();
    assert_eq!(
        (stats.traps, stats.answered),
        (0, 4),
        "under 64 MiB the growth fits"
    );
}

#[test]
fn the_lifecycle_reaches_the_plugin_once_per_profile_on_the_hosts_clock() {
    let runtime = Arc::new(Runtime::with_clock(|| 424_242).expect("a runtime"));
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
    engine.on_released(&lease, smg_policy_controller::ReleaseReason::ClientGone);
    engine.on_membership(&smg_policy_controller::Membership::Snapshot(&[
        WorkerId(1),
        WorkerId(2),
    ]));
    engine.on_signal(&smg_policy_controller::Signal::GatewayVeto {
        worker: WorkerId(1),
        until_micros: 500_000,
    });
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
            sessions: None,
        }],
        1,
        Needs::default(),
    )
}

/// Three prefills (100..102) and four decodes (200..203) of one pair class,
/// each pool's last row the least loaded.
fn pair_decide(engine: &Engine) -> Decision {
    pair_decide_under(engine, FRAME_MICROS)
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
        scorers: vec![Weighted {
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
        let withheld = Plugin::load(&runtime, withheld).expect("the guest loads");
        let hashed = load(&runtime, name, component, mode::LEAST_IN_FLIGHT);
        let lease = Lease {
            session_key: 0xBEEF,
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
            0xBEEF,
            "{name}: the fingerprint, as in the request"
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

    let plugin = load(
        &runtime,
        "table-grower",
        guest::record_guest(),
        mode::GROW_TABLE,
    );
    let engine = plugin_engine(&plugin);
    assert!(matches!(decide(&engine, &pool(4), 1), Decision::Pick(_)));
    let stats = plugin.stats();
    assert_eq!(
        (stats.traps, stats.answered),
        (1, 1),
        "the constructor answered, the filter trapped"
    );
    assert!(
        plugin.is_warm(),
        "the instance went and the close rebuilt the slot"
    );

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
    let runtime = runtime();
    let mut burner = spec("burner", guest::record_guest(), mode::BURN_IN_STAGES);
    // The frame is the request's: each call may take all of what is left.
    burner.budget_micros = 0;
    // The budgets below are burns of fuel: at the slow calibration a call
    // held off its core is still charged its fuel.
    burner.fuel_per_micro = SLOW_CALIBRATION;
    let fuel_per_micro = burner.fuel_per_micro;
    let plugin = Plugin::load(&runtime, burner).expect("the guest loads");
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
    let budget_micros = u32::try_from((per_call * 5 / 2).div_ceil(fuel_per_micro)).expect("fits");
    assert!(
        matches!(decide_under(budget_micros, 2), Decision::Pick(_)),
        "the fallbacks decide"
    );
    let two = plugin.stats();
    assert_eq!(
        two.answered - one.answered,
        3,
        "the constructor, the filter and the scorer answered"
    );
    assert_eq!(
        (two.budget_gone, two.fuel_exhausted, two.traps),
        (0, 1, 0),
        "the picker's grant was the remainder and ran out: the calls together exceeded the budget"
    );
    // The run-out emptied the slot and the close rebuilt it: one `init`, the
    // load's cost, burnt outside the decision's budget.
    let init = before;
    assert!(
        two.fuel_used - one.fuel_used <= u64::from(budget_micros) * fuel_per_micro + init,
        "no decision burns past its budget: {two:?}"
    );
    assert!(
        plugin.is_warm(),
        "the interrupted instance went and the close rebuilt the slot"
    );
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
    assert_eq!(counter_value(&plugin, counter::RANDOM), p1.wrapping_add(d1));
    assert_eq!(
        counter_value(&plugin, counter::PICK_RANDOM),
        p2.wrapping_add(d2),
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

/// Fuel prices operators, not bytes: a loop of whole-memory fills burns a
/// few units per millisecond. The frame's epoch deadline stops it on the
/// clock, counted as the frame running out.
#[test]
fn a_bulk_memory_loop_meets_the_frames_wall_bound() {
    let runtime = runtime();
    // A calibration no host has: at it the frame's fuel lasts seconds of
    // fills (a fill is charged by its length), so only the clock can end the
    // loop inside the bound below; with the deadline late or the ticker
    // asleep the case reads seconds and fails.
    let mut filler = spec("filler", guest::record_guest(), mode::FILL_IN_SCORE);
    filler.fuel_per_micro = 20_000_000;
    let plugin = Plugin::load(&runtime, filler).expect("the guest loads");
    let engine = plugin_engine(&plugin);
    let started = std::time::Instant::now();
    assert!(
        matches!(decide(&engine, &pool(4), 1), Decision::Pick(_)),
        "the fallback scorer decided"
    );
    let elapsed = started.elapsed();
    let stats = plugin.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.traps),
        (1, 0),
        "the frame ran out on the clock, not on fuel: {stats:?}"
    );
    // The fills cost a few units each under a frame of ten thousand million
    // units, so the loop would run for seconds on fuel alone; the deadline
    // ends it within a few ticks of the frame (a trapped call's fuel reads
    // as spent: the store's counter is not synced on an interrupt).
    let bound = std::time::Duration::from_micros(u64::from(FRAME_MICROS) * 20);
    assert!(
        elapsed < bound,
        "the wall bound held: {elapsed:?} for a {FRAME_MICROS} us frame"
    );
    assert!(
        plugin.is_warm(),
        "the interrupted instance went and the close rebuilt the slot"
    );
}

/// One plugin on both sides of a pair draws on one decision budget: the
/// sides' calls together never exceed the request's.
#[test]
fn a_pair_shares_one_decision_budget() {
    let runtime = runtime();
    let mut burner = spec("pair-burner", guest::record_guest(), mode::BURN_IN_STAGES);
    burner.budget_micros = 0;
    // Budgets in burns of fuel, at the slow calibration.
    burner.fuel_per_micro = SLOW_CALIBRATION;
    let fuel_per_micro = burner.fuel_per_micro;
    let plugin = Plugin::load(&runtime, burner).expect("the guest loads");
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
    assert_eq!(
        stats.traps, 1,
        "the decode filter's trap and no second one from a stale handle: {stats:?}"
    );
    assert_eq!(
        (stats.instantiations, stats.lazy_instantiations),
        (2, 1),
        "rebuilt once: {stats:?}"
    );
    assert!(plugin.is_warm());
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

/// The budget is wall time: a long call that answers, then an endless one,
/// end at the budget together, because the second call's deadline is what
/// the first one left, not its whole frame. Fuel cannot end either call
/// inside the bound at this calibration; the clock does.
#[test]
fn the_decisions_wall_is_bounded_by_its_budget_whatever_the_fuel() {
    let runtime = runtime();
    let mut spec = spec(
        "fill-then-fill",
        guest::record_guest(),
        mode::FILL_IN_FILTER_THEN_FILL,
    );
    spec.fuel_per_micro = 20_000_000;
    spec.budget_micros = 1_000_000;
    let plugin = Plugin::load(&runtime, spec).expect("the guest loads");
    let engine = plugin_engine(&plugin);
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
    let stats = plugin.stats();
    assert_eq!(
        (stats.fuel_exhausted, stats.traps),
        (1, 0),
        "the endless fill ran out on the clock: {stats:?}"
    );
    // Eight hundred fills take near four hundred milliseconds and answer;
    // the endless fill then gets what the second has left, plus a tick, the
    // fill in flight and the runner's slack (a fifth of a hundred-millisecond
    // budget seen with sibling test threads on a loaded host). A whole-frame
    // deadline would run the endless fill the frame's whole second on top:
    // near one and a half.
    assert!(
        elapsed < std::time::Duration::from_millis(1_200),
        "the decision ended at its budget: {elapsed:?}"
    );
    assert!(
        plugin.is_warm(),
        "the interrupted instance went and the close rebuilt the slot"
    );
}

/// Two plugins in one profile: the scorer spends most of the budget, the
/// picker runs out under what is left. The decision's budget is spent, the
/// picker's instance goes, and the picker's breaker is untouched.
#[test]
fn an_exhaustion_under_another_plugins_clamp_is_the_decisions_budget_spent() {
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
    let picker = Plugin::load(&runtime, picker_spec).expect("the guest loads");
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
        scorers: vec![Weighted {
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
        "the fallback picker decides"
    );
    let (s, p) = (scorer.stats(), picker.stats());
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
        (p.instantiations, p.lazy_instantiations),
        (2, 1),
        "the picker's interrupted instance went and the close rebuilt it: {p:?}"
    );
    assert_eq!(p.breaker_opens, 0);
}

/// The decision's first open seeds its budget and every later open draws on
/// it: one plugin on both sides of a pair, a constructor that burns, a budget
/// of a burn and a half. The second constructor gets what the first left and
/// runs out; a budget re-seeded per open would let it through.
#[test]
fn the_first_open_seeds_the_budget_and_later_opens_draw_on_it() {
    let runtime = runtime();
    let mut spec = spec(
        "ctor-burner",
        guest::record_guest(),
        mode::BURN_IN_CONSTRUCTOR,
    );
    spec.budget_micros = 0;
    // Budgets in burns of fuel, at the slow calibration.
    spec.fuel_per_micro = SLOW_CALIBRATION;
    let fuel_per_micro = spec.fuel_per_micro;
    let plugin = Plugin::load(&runtime, spec).expect("the guest loads");
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
    assert_eq!(
        (
            stats.fuel_exhausted - before.fuel_exhausted,
            stats.answered - before.answered
        ),
        (1, 1),
        "the first constructor answered and the second ran out on what it left: {stats:?}"
    );
    assert!(plugin.is_warm(), "the close rebuilt the slot");
}

/// A destructor long on the clock but cheap in fuel runs under its own wall
/// bound of two milliseconds and is no latency sample: sixty-five decisions
/// of it open no breaker. Its fifty host calls cost a few microseconds each,
/// a small fraction of the wall; one held off the core past the wall on a
/// loaded runner is dropped and rebuilt, the mechanism's answer, no failure.
#[test]
fn a_slow_destructor_within_its_own_wall_opens_nothing() {
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

/// Case (a) of the shared budget, in both scope orders: an honest scorer and
/// a scorer that spins, both with a frame of the whole budget. Neither
/// order charges the spinner on the breaker's failure count: it never consumed
/// the budget alone (the other's constructor spent first), so each decision
/// is the decision's budget spent. Its stage samples score the remainder it
/// took of the budget, the ring survives its rebuilds, and the window (64
/// stage calls) opens its breaker; the honest scorer's never opens.
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
        let scorer = |plugin: &Arc<Plugin>| Weighted {
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
                &facts(seed),
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        for seed in 1..=64 {
            assert!(
                matches!(decide(seed), Decision::Pick(_)),
                "the fallbacks decide"
            );
        }
        let (s, h) = (spinner.stats(), honest.stats());
        assert_eq!(
            (s.fuel_exhausted, s.traps, s.breaker_opens),
            (0, 0, 1),
            "spinner first {spinner_first}: the window opened the spinner's breaker, no charge: {s:?}"
        );
        assert!(
            s.budget_spent >= 64,
            "spinner first {spinner_first}: every decision was the budget spent: {s:?}"
        );
        assert_eq!(
            h.breaker_opens, 0,
            "spinner first {spinner_first}: the honest scorer stays: {h:?}"
        );
        decide(65);
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
    let spinner = Plugin::load(&runtime, spinner_spec).expect("the guest loads");
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
    let scorer = |plugin: &Arc<Plugin>| Weighted {
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
    for seed in 1..=65 {
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
            "the fallbacks decide"
        );
    }
    let s = spinner.stats();
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
}

/// Case (e): a plugin whose `init` outlasts a small decision budget. The
/// remainder clamps the cold slot's `init` below its cost: the decision's
/// budget is spent (the clamp drawn whole, so a neighbour opened after it
/// finds the remainder gone), the breaker (which judges `init` under its own
/// frame) is untouched, and the close rebuilds the slot under the full frame,
/// so the next decision finds it warm. Five such decisions open nothing.
#[test]
fn an_init_the_remainder_clamps_is_the_decisions_budget_spent_and_the_close_rebuilds() {
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
            Weighted {
                scorer: Box::new(WasmScorer::new(bound.clone(), None)),
                weight: 1.0,
            },
            Weighted {
                scorer: Box::new(WasmScorer::new(neighbour.bind(), None)),
                weight: 1.0,
            },
        ],
        picker: Box::new(WasmPicker::new(bound, None)),
    });
    for seed in 1..=5 {
        heavy.forget_instance();
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
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
        assert!(
            heavy.is_warm(),
            "the close rebuilt the slot under the full init frame"
        );
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
}

/// A clock the test steers: at `0` it answers zero at once; at any other
/// value it naps, then answers one less than the value. The nap is a host
/// call's wall time, which costs the guest that waits for it no fuel.
fn steered_runtime(steer: &Arc<AtomicU8>, nap: std::time::Duration) -> Arc<Runtime> {
    let steer = Arc::clone(steer);
    Arc::new(
        Runtime::with_clock(move || match steer.load(Ordering::Relaxed) {
            0 => 0,
            value => {
                std::thread::sleep(nap);
                u64::from(value - 1)
            }
        })
        .expect("a runtime"),
    )
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
    let plugin = Plugin::load(&runtime, asks).expect("the guest loads");
    (clock, runtime, plugin)
}

/// One decision under `budget_micros` on a cold slot of `cold` while the
/// clock is at `steer`: the plugin's three stages with a neighbour's score
/// behind them, no fallback named.
fn decide_on_a_cold_slot(
    clock: &AtomicU8,
    steer: u8,
    cold: &Arc<Plugin>,
    neighbour: &Arc<Plugin>,
    budget_micros: u32,
) {
    let bound = cold.bind();
    let engine = single_engine(Profile {
        name: "cold-init".into(),
        filters: vec![Box::new(WasmFilter::new(bound.clone(), None))],
        scorers: vec![
            Weighted {
                scorer: Box::new(WasmScorer::new(bound.clone(), None)),
                weight: 1.0,
            },
            Weighted {
                scorer: Box::new(WasmScorer::new(neighbour.bind(), None)),
                weight: 1.0,
            },
        ],
        picker: Box::new(WasmPicker::new(bound, None)),
    });
    cold.forget_instance();
    clock.store(steer, Ordering::Relaxed);
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
    clock.store(0, Ordering::Relaxed);
    assert!(
        matches!(decision, Decision::Pick(_)),
        "the fallbacks decide"
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
/// the remainder gone.
#[test]
fn a_lazy_init_that_burns_its_frame_out_is_charged_its_fuel_at_the_calibration() {
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
    decide_on_a_cold_slot(&clock, 3, &cold, &neighbour, 100_000);
    let s = cold.stats();
    assert!(
        s.fuel_used - before >= 99_000,
        "the fuel the failed init burnt is in the plugin's count: {s:?}"
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
    let cold = Plugin::load(&runtime, row).expect("the guest loads");
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
    let clock = Arc::new(AtomicU8::new(0));
    let runtime = steered_runtime(&clock, std::time::Duration::from_millis(20));
    let mut row = spec("burns-and-waits", guest::record_guest(), mode::BURN_IN_INIT);
    row.config.push(mode::INIT_ASKS_THE_CLOCK);
    row.fuel_per_micro = 10;
    let cold = Plugin::load(&runtime, row).expect("the guest loads");
    // The load ran `init` once, the clock answering at once: its fuel at the
    // calibration, the same every time.
    let init_micros = u32::try_from(cold.stats().fuel_used.div_ceil(10)).expect("fits");
    let neighbour = load(
        &runtime,
        "after-the-init",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    decide_on_a_cold_slot(&clock, 1, &cold, &neighbour, init_micros + 10_000);
    let s = cold.stats();
    assert_eq!(
        (
            s.lazy_instantiations,
            s.answered,
            s.budget_spent,
            s.breaker_opens
        ),
        (1, 4, 0, 0),
        "the init's fuel is the larger term and leaves half the wait: {s:?}"
    );
    let n = neighbour.stats();
    assert_eq!(
        (n.answered, n.budget_spent),
        (2, 0),
        "the neighbour answers under what is left: {n:?}"
    );
}

/// The lazy init's charge is the plugin's latency sample of its decision,
/// however the init ends: 3 ms on every cold slot under a 4 ms budget is past
/// half of it, so the window of sixty-four reads slow at its first evaluation
/// and opens the breaker, over inits that keep their instance and over inits
/// that refuse (a threshold of a hundred failures keeps their count out of
/// it).
#[test]
fn a_lazy_init_long_on_the_clock_is_sampled_however_it_ends() {
    for steer in [1, 2] {
        let (clock, _runtime, plugin) = plugin_whose_init_asks(3, |row| {
            row.breaker = BreakerSpec {
                failures: 100,
                ..BreakerSpec::default()
            };
        });
        let engine = plugin_engine(&plugin);
        let decide_cold = |seed: u64| {
            plugin.forget_instance();
            clock.store(steer, Ordering::Relaxed);
            let decision = engine.decide(
                &RequestFacts {
                    budget_micros: 4_000,
                    ..facts(seed)
                },
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            clock.store(0, Ordering::Relaxed);
            assert!(matches!(decision, Decision::Pick(_)), "steer {steer}");
        };
        for seed in 1..=63 {
            decide_cold(seed);
        }
        assert_eq!(
            plugin.stats().breaker_opens,
            0,
            "steer {steer}: sixty-three decisions are one sample short of the window: {:?}",
            plugin.stats()
        );
        decide_cold(64);
        let s = plugin.stats();
        assert_eq!(
            s.breaker_opens, 1,
            "steer {steer}: the sixty-fourth sample reads the window slow: {s:?}"
        );
    }
}

/// A cold slot's `init` the remainder clamps, long on the clock as well: a
/// 200 us decision whose `init` waits 2 ms for the host and is then ended by
/// its clamp, the burn out of fuel or the wall bound's tick, whichever is
/// first (both are the clamp's, and the sample is the clock either way).
/// The draw is the whole remainder; the plugin's sample is the larger of
/// that and the charge of the rule, here the clock. Two such decisions in a
/// window of sixty-four are its second-slowest sample past the budget, which
/// is slow. Sampled at the remainder alone they would read at the budget,
/// never past it.
#[test]
fn a_clamped_init_long_on_the_clock_is_sampled_by_its_clock() {
    let clock = Arc::new(AtomicU8::new(0));
    let runtime = steered_runtime(&clock, std::time::Duration::from_millis(2));
    let mut heavy = spec(
        "clamped-and-slow",
        guest::record_guest(),
        mode::BURN_IN_INIT,
    );
    heavy.config.push(mode::INIT_ASKS_THE_CLOCK);
    let plugin = Plugin::load(&runtime, heavy).expect("the guest loads");
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
    // takes 2 ms to answer each init, the clamped one and the close's rebuild.
    clock.store(1, Ordering::Relaxed);
    for seed in 1..=2 {
        plugin.forget_instance();
        decide(seed, 200);
    }
    clock.store(0, Ordering::Relaxed);
    let s = plugin.stats();
    assert_eq!(
        (s.budget_spent, s.lazy_instantiations, s.breaker_opens),
        (2, 2, 0),
        "two clamped inits, each rebuilt at its close: {s:?}"
    );
    // Sixty-two honest decisions on the warm slot fill the window.
    for seed in 3..=64 {
        decide(seed, FRAME_MICROS);
    }
    let s = plugin.stats();
    assert_eq!(
        s.breaker_opens, 1,
        "the two clamped inits are the window's tail, past the budget by their clock: {s:?}"
    );
}

/// A clock the test steers for a few asks: while `asks` is above zero an ask
/// takes one from it and reads one less than `steer`; every other ask reads
/// zero. A decision's open reads the clock once before a cold slot's `init`
/// asks it, so two asks reach that `init` and none after it.
fn runtime_steered_for(steer: &Arc<AtomicU8>, asks: &Arc<AtomicU8>) -> Arc<Runtime> {
    let (steer, asks) = (Arc::clone(steer), Arc::clone(asks));
    Arc::new(
        Runtime::with_clock(move || {
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
    let clock = Arc::new(AtomicU8::new(0));
    let asks = Arc::new(AtomicU8::new(0));
    let runtime = runtime_steered_for(&clock, &asks);
    let mut row = spec("traps-once", guest::record_guest(), mode::LEAST_IN_FLIGHT);
    row.config.push(mode::INIT_ASKS_THE_CLOCK);
    row.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let cold = Plugin::load(&runtime, row).expect("the guest loads");
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
    decide_on_a_cold_slot(&clock, 4, &cold, &neighbour, 90_000);
    let s = cold.stats();
    assert_eq!(
        (
            s.budget_spent,
            s.instantiations - loaded,
            s.lazy_instantiations,
            s.breaker_opens
        ),
        (0, 1, 0, 0),
        "the trap is the init's own: no budget spent, no rebuild at the close: {s:?}"
    );
    assert!(!cold.is_warm(), "the next decision's open tries the init");
    let n = neighbour.stats();
    assert_eq!(
        (n.answered, n.budget_spent),
        (2, 0),
        "the neighbour's open is fed and answers under what is left: {n:?}"
    );
    // One failure a decision: the second crosses the threshold of two.
    asks.store(2, Ordering::Relaxed);
    decide_on_a_cold_slot(&clock, 4, &cold, &neighbour, 90_000);
    let s = cold.stats();
    assert_eq!(
        (s.budget_spent, s.instantiations - loaded, s.breaker_opens),
        (0, 2, 1),
        "the second decision's trap is the second failure: {s:?}"
    );
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
    let plugin = Plugin::load(&runtime, row).expect("the guest loads");
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
    let plugin = Plugin::load(&runtime, row).expect("the guest loads");
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
    clock.store(2, Ordering::Relaxed);
    assert!(matches!(decide(1), Decision::Pick(_)));
    let s = plugin.stats();
    assert_eq!(
        (s.traps, s.breaker_opens),
        (1, 0),
        "the trap counted; the refused rebuild at the same close did not: {s:?}"
    );
    assert!(!plugin.is_warm(), "the rebuild was refused");
    assert!(matches!(decide(2), Decision::Pick(_)));
    clock.store(0, Ordering::Relaxed);
    let s = plugin.stats();
    assert_eq!(
        (s.traps, s.breaker_opens),
        (1, 1),
        "the next decision's refused init is the second failure: {s:?}"
    );
}

/// A fault in one open ends the plugin's part in the decision. The prefill
/// side's constructor traps and the instance goes; the decode side's open, a
/// moment later on the same thread, makes no new one inside the decision that
/// lost it. One trap, and one rebuild at the close.
#[test]
fn a_decision_that_emptied_the_slot_makes_no_instance_before_its_close() {
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
    assert!(
        matches!(pair_decide(&pair), Decision::Pick(_)),
        "the fallbacks decide"
    );
    let stats = plugin.stats();
    assert_eq!(
        stats.traps, 1,
        "the prefill constructor's trap; the decode side asked nothing: {stats:?}"
    );
    assert_eq!(
        (stats.instantiations, stats.lazy_instantiations),
        (2, 1),
        "the load and the close's rebuild: {stats:?}"
    );
    assert!(plugin.is_warm(), "the close rebuilt the slot");
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
/// its fuel frame never ends it. Its wall bound is its own, in milliseconds,
/// not the decision's frame: a decision at a 100 ms budget is not stretched
/// by one; the run-out drops the instance and the close rebuilds it.
#[test]
fn a_destructors_wall_bound_is_its_own_not_the_decisions_frame() {
    let runtime = runtime();
    let mut filler = spec("fill-dtor", guest::record_guest(), mode::FILL_IN_DESTRUCTOR);
    // The frame is the request's 100 ms, and the calibration makes the
    // destructor's fuel frame outlast it: only a wall bound ends the fill.
    filler.budget_micros = 0;
    filler.fuel_per_micro = 100_000_000;
    let plugin = Plugin::load(&runtime, filler).expect("the guest loads");
    let engine = plugin_engine(&plugin);
    let started = std::time::Instant::now();
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
    let elapsed = started.elapsed();
    assert!(matches!(decision, Decision::Pick(_)), "the decision stands");
    assert!(
        elapsed < std::time::Duration::from_millis(30),
        "the destructor's wall bound is its own, not the 100 ms frame: {elapsed:?}"
    );
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
    assert!(plugin.is_warm(), "the close rebuilt the dropped instance");
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
    let runtime = runtime();
    // Every charge at the calibration of record is the fuel, exact.
    let mut gauge = spec("gauge", guest::record_guest(), mode::BURN_IN_STAGES);
    gauge.fuel_per_micro = SLOW_CALIBRATION;
    gauge.budget_micros = SLOW_FRAME_MICROS;
    let fuel_per_micro = gauge.fuel_per_micro;
    let gauge = Plugin::load(&runtime, gauge).expect("the guest loads");
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
    let three_stage = Plugin::load(&runtime, three_stage).expect("the guest loads");
    let neighbours: Vec<Arc<Plugin>> = ["neighbour-a", "neighbour-b", "neighbour-c"]
        .into_iter()
        .map(|name| {
            let mut neighbour = spec(name, guest::record_guest(), mode::BURN_IN_STAGES);
            neighbour.fuel_per_micro = SLOW_CALIBRATION;
            neighbour.budget_micros = SLOW_FRAME_MICROS;
            Plugin::load(&runtime, neighbour).expect("the guest loads")
        })
        .collect();
    let bound = three_stage.bind();
    let engine = single_engine(Profile {
        name: "clamped".into(),
        filters: vec![Box::new(WasmFilter::new(bound.clone(), None))],
        scorers: std::iter::once(Weighted {
            scorer: Box::new(WasmScorer::new(bound.clone(), None)),
            weight: 1.0,
        })
        .chain(neighbours.iter().map(|neighbour| Weighted {
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
    for seed in 1..=65 {
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
            "the fallback picker decides"
        );
    }
    let a = three_stage.stats();
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
}

/// The latency sample is the decision's charge over the budget, not its
/// clock.
/// A spinner at the slow calibration of record burns its fuel
/// grant (the whole remainder behind an honest scorer) in a fraction of the
/// budget's wall: sampled by the clock it never reads slow and spends the
/// decision's budget forever; sampled by its charge, the remainder it took,
/// the window opens its breaker.
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
    let spinner = Plugin::load(&runtime, slow_calibration).expect("the guest loads");
    let scorer = |plugin: &Arc<Plugin>| Weighted {
        scorer: Box::new(WasmScorer::new(plugin.bind(), None)),
        weight: 1.0,
    };
    let engine = single_engine(Profile {
        name: "fuel-fast".into(),
        filters: vec![],
        scorers: vec![scorer(&honest), scorer(&spinner)],
        picker: Box::new(Best),
    });
    let started = std::time::Instant::now();
    for seed in 1..=64 {
        let decision = engine.decide(
            &facts(seed),
            &four(),
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(
            matches!(decision, Decision::Pick(_)),
            "the fallbacks decide"
        );
    }
    let wall = started.elapsed();
    let s = spinner.stats();
    assert_eq!(
        (s.fuel_exhausted, s.traps, s.breaker_opens),
        (0, 0, 1),
        "the spinner's charge, the remainder, opened its breaker at the window: {s:?}"
    );
    assert!(
        s.budget_spent >= 64,
        "every decision was the budget spent: {s:?}"
    );
    // Sixty-four decisions at a 10 ms budget would be 640 ms on the clock:
    // the spinner's fuel ran out long before its wall.
    assert!(
        wall < std::time::Duration::from_millis(400),
        "the fuel ended the calls early, not the wall: {wall:?}"
    );
    assert_eq!(honest.stats().breaker_opens, 0, "the honest scorer stays");
}

/// A plugin that spins in its constructor, behind another plugin's
/// constructor, is the decision's budget spent (the grant is below its cap by
/// the other's spend, its own spend short of the seed) and a constructor is
/// no stage sample: the call the remainder ended is sampled all the same, so
/// its one sample per decision opens the breaker at the window. Its scope
/// first, the constructor had its whole frame and the failure count opens it.
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
        let scorer = |plugin: &Arc<Plugin>| Weighted {
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
                &facts(seed),
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            )
        };
        for seed in 1..=64 {
            assert!(
                matches!(decide(seed), Decision::Pick(_)),
                "the fallbacks decide"
            );
        }
        let (s, h) = (spinner.stats(), honest.stats());
        assert!(
            s.breaker_opens >= 1,
            "spinner first {spinner_first}: the spinner's breaker opened: {s:?}"
        );
        if spinner_first {
            assert!(
                s.fuel_exhausted >= 5,
                "its scope first, the constructor had its whole frame: charged: {s:?}"
            );
        } else {
            assert_eq!(
                (s.fuel_exhausted, s.traps),
                (0, 0),
                "behind the honest constructor, never charged: {s:?}"
            );
            assert!(
                s.budget_spent >= 64,
                "every decision was the budget spent: {s:?}"
            );
        }
        assert_eq!(
            h.breaker_opens, 0,
            "spinner first {spinner_first}: the honest plugin stays: {h:?}"
        );
        decide(65);
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
    let gauge = Plugin::load(&runtime, gauge).expect("the guest loads");
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
    let hog = Plugin::load(&runtime, hog).expect("the guest loads");
    let scorer = |plugin: &Arc<Plugin>| Weighted {
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
/// a charge.
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
        let own = Weighted {
            scorer: Box::new(WasmScorer::new(bound.clone(), None)),
            weight: 1.0,
        };
        let other = Weighted {
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
        for seed in 1..=64 {
            let decision = engine.decide(
                &facts(seed),
                &four(),
                &pool(4),
                &Constraints::default(),
                &mut Scratch::new(),
            );
            assert!(
                matches!(decision, Decision::Pick(_)),
                "the fallback picker decides"
            );
        }
        let (s, h) = (spinner.stats(), honest.stats());
        assert_eq!(
            (s.fuel_exhausted, s.traps, s.breaker_opens),
            (0, 0, 1),
            "spinner first {spinner_first}: the spinning pick was the budget spent and the window opened it: {s:?}"
        );
        assert!(
            s.budget_spent >= 64,
            "spinner first {spinner_first}: every decision was the budget spent: {s:?}"
        );
        assert_eq!(
            h.breaker_opens, 0,
            "spinner first {spinner_first}: the honest plugin stays: {h:?}"
        );
    }
}

/// A pair decision opens both sides' scopes up front and closes them at the
/// end: a plugin bound on both sides closes twice, and its one sample is the
/// two sides' spend summed, once: the window opens at the sixty-fourth
/// decision, not before. Three burns a side at the slow calibration, a burn
/// nine percent of the budget: summed, past half; a side alone, under it.
#[test]
fn a_plugin_on_both_pair_sides_is_sampled_once_with_both_sides_summed() {
    let runtime = runtime();
    let mut both = spec("both-sides", guest::record_guest(), mode::BURN_IN_STAGES);
    both.fuel_per_micro = SLOW_CALIBRATION;
    both.budget_micros = PAIR_BUDGET_MICROS;
    let plugin = Plugin::load(&runtime, both).expect("the guest loads");
    let shared = plugin_profile(&plugin, None, None, None);
    let engine = pair_engine(vec![shared], 0);
    for _ in 0..63 {
        assert!(
            pair_decide_under(&engine, PAIR_BUDGET_MICROS)
                .pick()
                .is_some(),
            "the pair decides"
        );
    }
    assert_eq!(
        plugin.stats().breaker_opens,
        0,
        "sixty-three decisions are one sample short of the window: {:?}",
        plugin.stats()
    );
    assert!(
        pair_decide_under(&engine, PAIR_BUDGET_MICROS)
            .pick()
            .is_some(),
        "the pair decides"
    );
    let s = plugin.stats();
    assert_eq!(
        (s.fuel_exhausted, s.traps, s.budget_spent, s.breaker_opens),
        (0, 0, 0, 1),
        "both sides summed into one sample a decision opened the breaker at the window: {s:?}"
    );
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
    let decode = Plugin::load(&runtime, heavy).expect("the guest loads");
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
