//! The decision's budget: one remainder its calls, its sides and its plugins
//! draw on, the frame that bounds a call on the clock, and who is charged
//! when the budget is spent.

use std::sync::Arc;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use smg_grand_central::{
    engine::{Candidate, RequestFacts},
    stages::Lifecycle,
};
use smg_grand_central::{
    engine::{
        Constraints, Decision, PairPools, PoolFacts, Profile, Scratch, WeightedScorer, WorkerId,
    },
    plugins::{
        shared::OpenDecisions,
        wasm::{Plugin, Runtime, WasmFilter, WasmPicker, WasmScorer},
    },
    stages::Best,
};

use crate::{
    bound_clock, decide, decide_until_open, excused, facts, four, held_counts_are_made_again, load,
    load_slow, loaded, made_again, no_hold, off_cpu_since, pair_engine, plugin_engine,
    plugin_profile, pool, runtime, said, single_engine, spec,
    support::wasm_guest::{self as guest, mode},
    the_close_rebuilt, the_neighbour_stays, timed, tried, FRAME_MICROS, SLOW_CALIBRATION,
    SLOW_FRAME_MICROS, TRIES,
};

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
        let mut pool_facts = PoolFacts::default();
        pool_facts.pool_size = 4;
        pool_facts.generation = 1;
        let decide_under = |budget_micros: u32, seed: u64| {
            let mut facts = facts(seed);
            facts.budget_micros = budget_micros;
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
    let facts_of = |budget_micros: u32| {
        let mut facts = facts(1);
        facts.budget_micros = budget_micros;
        facts
    };
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
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
    let mut record = PoolFacts::default();
    record.pool_size = 7;
    record.generation = 1;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    let decision = pair.decide_pair(
        &facts_of(budget_micros),
        &record,
        pools,
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
    let mut facts = facts(1);
    facts.budget_micros = 3;
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
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
        profile.scorers.push(WeightedScorer::of(
            Box::new(ReadsTheClockWhenAsked(Arc::clone(&scored))),
            1.0,
        ));
        let engine = single_engine(profile);
        let cpu_before = bound_clock();
        let started = std::time::Instant::now();
        let mut request_facts = facts(1);
        request_facts.budget_micros = 1_000_000;
        let mut pool_facts = PoolFacts::default();
        pool_facts.pool_size = 4;
        pool_facts.generation = 1;
        let decision = engine.decide(
            &request_facts,
            &pool_facts,
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
impl smg_grand_central::engine::Filter for SaysWhenAsked {
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
impl smg_grand_central::engine::Scorer for ReadsTheClockWhenAsked {
    fn name(&self) -> &'static str {
        "reads-the-clock-when-asked"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        _: &[Candidate],
        _: &[u32],
        out: &mut [f32],
    ) -> smg_grand_central::engine::TieBand {
        self.0.get_or_init(bound_clock);
        out.fill(0.0);
        smg_grand_central::engine::TieBand::NORMALIZED
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
    let mut request_facts = facts(1);
    request_facts.budget_micros = 1_000_000;
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
    plugin_engine(&scorer).decide(
        &request_facts,
        &pool_facts,
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let burn = (scorer.stats().fuel_used - before) / 3;
    let mut profile = Profile::of("two".into(), Box::new(WasmPicker::new(picker.bind(), None)));
    profile.filters = vec![];
    profile.scorers = vec![WeightedScorer::of(
        Box::new(WasmScorer::new(scorer.bind(), None)),
        1.0,
    )];
    let engine = single_engine(profile);
    let gauged = scorer.stats().answered;
    // A burn and a half, plus a little for the constructors: the scorer's
    // burn fits, the picker's does not.
    let budget_micros = u32::try_from((burn * 3 / 2).div_ceil(fuel_per_micro) + 20).expect("fits");
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let mut request_facts = facts(1);
    request_facts.budget_micros = budget_micros;
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
    let decision = engine.decide(
        &request_facts,
        &pool_facts,
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
    let mut request_facts = facts(1);
    request_facts.budget_micros = 1_000_000;
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
    plugin_engine(&plugin).decide(
        &request_facts,
        &pool_facts,
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
    let mut request_facts = facts(1);
    request_facts.budget_micros = budget_micros;
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 7;
    pool_facts.generation = 1;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    let decision = pair.decide_pair(
        &request_facts,
        &pool_facts,
        pools,
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
        let mut request_facts = facts(1);
        request_facts.budget_micros = 1_000_000;
        let mut pool_facts = PoolFacts::default();
        pool_facts.pool_size = 4;
        pool_facts.generation = 1;
        plugin_engine(&scorer).decide(
            &request_facts,
            &pool_facts,
            &pool(4),
            &Constraints::default(),
            &mut Scratch::new(),
        );
        let burn = (scorer.stats().fuel_used - before) / 3;
        let mut profile = Profile::of("two".into(), Box::new(WasmPicker::new(picker.bind(), None)));
        profile.filters = vec![];
        profile.scorers = vec![WeightedScorer::of(
            Box::new(WasmScorer::new(scorer.bind(), None)),
            1.0,
        )];
        let engine = single_engine(profile);
        let gauged = scorer.stats().answered;
        // A burn and a half, plus a little for the constructors: the
        // scorer's burn fits, and the picker's fits only in a remainder the
        // scorer did not draw.
        let budget_micros =
            u32::try_from((burn * 3 / 2).div_ceil(SLOW_CALIBRATION) + 20).expect("fits");
        let mut request_facts = facts(1);
        request_facts.budget_micros = budget_micros;
        let mut pool_facts = PoolFacts::default();
        pool_facts.pool_size = 4;
        pool_facts.generation = 1;
        let decision = engine.decide(
            &request_facts,
            &pool_facts,
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
        let scorer = |plugin: &Arc<Plugin>| {
            WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
        };
        let scorers = if spinner_first {
            vec![scorer(&spinner), scorer(&honest)]
        } else {
            vec![scorer(&honest), scorer(&spinner)]
        };
        let mut profile = Profile::of("shared".into(), Box::new(Best));
        profile.filters = vec![];
        profile.scorers = scorers;
        let engine = single_engine(profile);
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
    let mut request_facts = facts(1);
    request_facts.budget_micros = 1_000_000;
    plugin_engine(&burners[0]).decide(
        &request_facts,
        &four(),
        &pool(4),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let burn = (burners[0].stats().fuel_used - before) / 3;
    let scorer = |plugin: &Arc<Plugin>| {
        WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
    };
    let mut profile = Profile::of("quarters".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![
        scorer(&burners[0]),
        scorer(&burners[1]),
        scorer(&burners[2]),
        scorer(&spinner),
    ];
    let engine = single_engine(profile);
    // Four burns and a half, plus a little for the constructors.
    let budget_micros = u32::try_from((burn * 9 / 2).div_ceil(fuel_per_micro) + 40).expect("fits");
    let mut worst = std::time::Duration::ZERO;
    for seed in 1..=65 {
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let mut record = facts(seed);
        record.budget_micros = budget_micros;
        let decision = engine.decide(
            &record,
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
    let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> =
        vec![Box::new(WasmFilter::new(bound.clone(), None))];
    let scorers = vec![
        WeightedScorer::of(Box::new(WasmScorer::new(bound.clone(), None)), 1.0),
        WeightedScorer::of(Box::new(WasmScorer::new(neighbour.bind(), None)), 1.0),
    ];
    let mut profile = Profile::of(
        "clamped-init".into(),
        Box::new(WasmPicker::new(bound, None)),
    );
    profile.filters = filters;
    profile.scorers = scorers;
    let engine = single_engine(profile);
    for seed in 1..=5 {
        heavy.forget_instance();
        let (wall, cpu) = (std::time::Instant::now(), bound_clock());
        let mut request_facts = facts(seed);
        request_facts.budget_micros = 200;
        let decision = engine.decide(
            &request_facts,
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
    let mut request_facts = facts(1);
    request_facts.budget_micros = 1_000_000;
    plugin_engine(&gauge).decide(
        &request_facts,
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
    let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> =
        vec![Box::new(WasmFilter::new(bound.clone(), None))];
    let scorers = std::iter::once(WeightedScorer::of(
        Box::new(WasmScorer::new(bound.clone(), None)),
        1.0,
    ))
    .chain(neighbours.iter().map(|neighbour| {
        WeightedScorer::of(Box::new(WasmScorer::new(neighbour.bind(), None)), 1.0)
    }))
    .collect();
    let mut profile = Profile::of("clamped".into(), Box::new(WasmPicker::new(bound, None)));
    profile.filters = filters;
    profile.scorers = scorers;
    let engine = single_engine(profile);
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
        let mut record = facts(seed);
        record.budget_micros = budget_micros;
        let decision = engine.decide(
            &record,
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
