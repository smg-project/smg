//! The WebAssembly plugin host against the test guests: what loads and what
//! is refused, the honest pick through both worlds, the four host imports,
//! and the failure rules (a decline keeps the instance, a trap, an exhausted
//! budget or an invalid answer drops it and counts on the breaker, the
//! breaker bypasses, the fallbacks and the closed policy), one resource per
//! decision and per side of a pair, the lifecycle once per engine build, and
//! the instances kept per decider.
//!
//! One module a subject holds the cases, each a file under
//! `wasm_plugin_cases/`: `loading.rs`, `stages.rs`, `budgets.rs`,
//! `unwinding.rs`, `timing.rs`, `samples.rs`, `init.rs`, `instances.rs`,
//! `lifecycle.rs`, `tries.rs` and `measurement.rs`. This file holds what
//! two of them read: the loads, the profiles and the decisions over the
//! guests, the host's lines a case reads, the tries of a case that a hold
//! of the runner can move, and the clocks a case steers.
#![cfg(feature = "wasm")]
#![expect(
    clippy::expect_used,
    reason = "the guests' assembly and the loads run inside the tests; a failure there is the test's failure"
)]

#[path = "../support"]
mod support {
    pub mod wasm_guest;
}

use std::sync::{
    atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering},
    Arc,
};

use smg_grand_central::{
    document::Needs,
    engine::{
        Binding, Candidate, Constraints, Decision, DecisionSlot, Engine, PairPools, Plan,
        PoolFacts, Profile, RequestFacts, Scratch, Target, WeightedScorer, WorkerId,
    },
    plugins::{
        shared::OpenDecisions,
        wasm::{
            LoadError, Plugin, PluginBinding, PluginSpec, Runtime, WasmFilter, WasmPicker,
            WasmScorer,
        },
    },
    policies::power_of_two::PowerOfTwo,
    stages::Lifecycle,
    tracking::{EventSet, Lease, Side},
};
use support::wasm_guest::{self as guest, mode};

#[path = "wasm_plugin_cases/budgets.rs"]
mod budgets;
#[path = "wasm_plugin_cases/init.rs"]
mod init;
#[path = "wasm_plugin_cases/instances.rs"]
mod instances;
#[path = "wasm_plugin_cases/lifecycle.rs"]
mod lifecycle;
#[path = "wasm_plugin_cases/loading.rs"]
mod loading;
#[path = "wasm_plugin_cases/measurement.rs"]
mod measurement;
#[path = "wasm_plugin_cases/samples.rs"]
mod samples;
#[path = "wasm_plugin_cases/stages.rs"]
mod stages;
#[path = "wasm_plugin_cases/timing.rs"]
mod timing;
#[path = "wasm_plugin_cases/tries.rs"]
mod tries;
#[path = "wasm_plugin_cases/unwinding.rs"]
mod unwinding;

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
    filter_fallback: Option<Box<dyn smg_grand_central::engine::Filter>>,
    scorer_fallback: Option<Box<dyn smg_grand_central::engine::Scorer>>,
    picker_fallback: Option<Box<dyn smg_grand_central::engine::Picker>>,
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
    filter_fallback: Option<Box<dyn smg_grand_central::engine::Filter>>,
    scorer_fallback: Option<Box<dyn smg_grand_central::engine::Scorer>>,
    picker_fallback: Option<Box<dyn smg_grand_central::engine::Picker>>,
) -> Profile {
    let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> = vec![Box::new(WasmFilter::new(
        Arc::clone(&bound),
        filter_fallback,
    ))];
    let scorers = vec![WeightedScorer::of(
        Box::new(WasmScorer::new(Arc::clone(&bound), scorer_fallback)),
        1.0,
    )];
    let mut profile = Profile::of(
        "plugin".into(),
        Box::new(WasmPicker::new(bound, picker_fallback)),
    );
    profile.filters = filters;
    profile.scorers = scorers;
    profile
}

fn single_engine(profile: Profile) -> Engine {
    let mut binding = Binding::of("*".into(), 0);
    binding.phases = vec![];
    binding.plan = Plan::Single;
    binding.decode_profile = None;
    binding.shadow = None;
    binding.decode_shadow = None;
    binding.sessions = None;
    Engine::new(vec![profile], vec![binding], 1, Needs::default())
}

fn plugin_engine(plugin: &Arc<Plugin>) -> Engine {
    single_engine(plugin_profile(plugin, None, None, None))
}

/// `n` healthy candidates whose in-flight counts descend, so the least
/// loaded is the last one.
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

/// A request under the untimed budget. A case that tests the decision's
/// budget names its own (`timed`, or the field).
fn facts(seed: u64) -> RequestFacts<'static> {
    let mut facts = RequestFacts::default();
    facts.model = "m";
    facts.budget_micros = UNTIMED_BUDGET_MICROS;
    facts.seed = seed;
    facts.now_micros = 777_000;
    facts
}

/// A request whose decision budget is one frame: for the cases that assert
/// what a spent budget does.
fn timed(seed: u64) -> RequestFacts<'static> {
    let mut facts = facts(seed);
    facts.budget_micros = FRAME_MICROS;
    facts
}

fn decide(engine: &Engine, candidates: &[Candidate], seed: u64) -> Decision {
    decide_with(engine, candidates, &facts(seed))
}

fn decide_with(engine: &Engine, candidates: &[Candidate], facts: &RequestFacts<'_>) -> Decision {
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = candidates.len() as u32;
    pool_facts.generation = 1;
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
        let mut profile = Profile::of("two-draw".into(), Box::new(PowerOfTwo));
        profile.filters = vec![];
        profile.scorers = vec![WeightedScorer::of(
            Box::new(WasmScorer::new(plugin.bind(), None)),
            1.0,
        )];
        let engine = single_engine(profile);
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

impl tracing::Subscriber for Said {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        (*metadata.level() <= tracing::Level::WARN
            && metadata.target().starts_with("smg_grand_central"))
            || (*metadata.level() == tracing::Level::INFO
                && metadata.target() == "smg_grand_central::plugins::wasm::bindings")
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

fn pair_engine(profiles: Vec<Profile>, decode_profile: u16) -> Engine {
    let mut binding = Binding::of("*".into(), 0);
    binding.phases = vec![];
    binding.plan = Plan::PdPair;
    binding.decode_profile = Some(decode_profile);
    binding.shadow = None;
    binding.decode_shadow = None;
    binding.sessions = None;
    Engine::new(profiles, vec![binding], 1, Needs::default())
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
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 7;
    pool_facts.generation = 1;
    let mut request_facts = facts(9);
    request_facts.budget_micros = budget_micros;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    engine.decide_pair(
        &request_facts,
        &pool_facts,
        pools,
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    )
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
    let mut request_facts = facts(1);
    request_facts.budget_micros = SLOW_FRAME_MICROS;
    let decision = plugin_engine(&gauge).decide(
        &request_facts,
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(matches!(decision, Decision::Pick(_)), "{decision:?}");
    let fuel = gauge.stats().fuel_used - before;
    u32::try_from((fuel / 3).div_ceil(SLOW_CALIBRATION)).expect("fits")
}

/// The pool of four every shared-budget case decides over.
fn four() -> PoolFacts<'static> {
    let mut pool = PoolFacts::default();
    pool.pool_size = 4;
    pool.generation = 1;
    pool
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
    static HELD_NAME: std::cell::RefCell<Option<smg_grand_central::engine::DecisionSlotGuard>> =
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

impl smg_grand_central::engine::Filter for ActsInside {
    fn name(&self) -> &'static str {
        "acts-inside"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn smg_grand_central::engine::DecisionScope> {
        Some(self)
    }
}

impl smg_grand_central::engine::DecisionScope for ActsInside {
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
                    .on_membership(&smg_grand_central::tracking::Membership::Added(worker));
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
