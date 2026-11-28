//! The stages' answers and the failure rules: the honest pick through both
//! worlds, a guest's band, what a request's view carries, a decline and its
//! fallback, a fault that drops the instance and counts on the breaker, the
//! breaker's unit and its opening, a pair's two sides, and what a plugin's
//! stages need.

use std::{
    ops::Range,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
};

use smg_grand_central::{
    document::Needs,
    engine::{
        Candidate, Constraints, Decision, Engine, PairPools, Pick, PoolFacts, Profile,
        RequestFacts, Scorer as _, Scratch, ShedReason, WeightedScorer, WorkerId,
    },
    plugins::{
        shared::{BreakerSpec, Failure, FrameBudget},
        wasm::{Plugin, WasmFilter, WasmPicker, WasmScorer},
    },
    stages::{Best, Healthy, LeastRequests, Lifecycle, SessionAffinity},
    Draw,
};

use crate::{
    bound_clock, counter_value, decide, decide_with, facts, four, load, load_slow, loaded,
    made_again, off_cpu_since, openings, pair_decide, pair_decide_under, pair_engine, pick_ids,
    plugin_engine, plugin_profile, plugin_whose_init_asks, pool, runtime, said, single_engine,
    spec, steered_runtime,
    support::wasm_guest::{self as guest, counter, mode},
    the_close_rebuilt, tried, FRAME_MICROS, SLOW_CALIBRATION, SLOW_FRAME_MICROS, TRIES,
};

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
    fallback: Option<Box<dyn smg_grand_central::engine::Scorer>>,
) -> Engine {
    let mut profile = Profile::of("band".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![WeightedScorer::of(
        Box::new(WasmScorer::new(plugin.bind(), fallback)),
        1.0,
    )];
    single_engine(profile)
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
            let mut without_band = facts(seed);
            without_band.now_micros = 0;
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
    let mut pool = PoolFacts::default();
    pool.pool_size = candidates.len() as u32;
    for band in [Some(0.0), Some(1.0), None] {
        let guest_mode = if band.is_some() {
            mode::SCORE_BAND
        } else {
            mode::LEAST_IN_FLIGHT
        };
        let expected = band.unwrap_or(smg_grand_central::engine::TieBand::NORMALIZED.0);
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
    let mut widest = facts(1);
    widest.budget_micros = u32::MAX;
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
    spec.breaker = BreakerSpec::default();
    spec.breaker.failures = 2;
    spec.breaker.window_micros = 60_000_000;
    spec.breaker.cooldown_micros = 60_000_000;
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

/// An opening is otherwise silent (the stages take their fallback): under
/// the failure rule it writes one line with the plugin, the rule and the
/// cool-down, and the decisions it bypasses write none.
#[test]
fn an_opening_under_the_failure_rule_writes_one_line() {
    said();
    let runtime = runtime();
    let mut spec = spec("trapper", guest::record_guest(), mode::TRAP_IN_FILTER);
    spec.breaker = BreakerSpec::default();
    spec.breaker.failures = 2;
    spec.breaker.window_micros = 60_000_000;
    spec.breaker.cooldown_micros = 60_000_000;
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
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
    let decide_gone = |seed: u64| {
        let mut gone = facts(seed);
        gone.budget_micros = 0;
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
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 4;
    pool_facts.generation = 1;
    let mut long_model = facts(1);
    long_model.model = &long;
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
            let mut request_facts = facts(1);
            request_facts.block_hashes = &hashes;
            let mut pool = PoolFacts::default();
            pool.pool_size = 4;
            pool.generation = 1;
            let decision = engine.decide(
                &request_facts,
                &pool,
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
fn a_profile_of_built_ins_holds_no_scope_and_a_plugin_profile_does() {
    let native = {
        let mut profile = Profile::of("native".into(), Box::new(Best));
        profile.filters = vec![Box::new(Healthy)];
        profile.scorers = vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
        profile
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
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 7;
    pool_facts.generation = 1;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    let decision = pair.decide_pair(
        &facts(1),
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

/// The decision is the breaker's unit. A cold slot's `init` that refuses,
/// with the plugin on both sides of a pair: the first side's open counts one
/// failure and the second side's makes no second `init` and no second count,
/// so a threshold of two is two decisions on a pair as on a single placement.
/// The next decision's open tries the `init` again.
#[test]
fn a_refused_cold_init_fails_its_decision_once_on_a_pair() {
    let (clock, _runtime, plugin) = plugin_whose_init_asks(0, |row| {
        row.breaker = BreakerSpec::default();
        row.breaker.failures = 2;
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
    row.breaker = BreakerSpec::default();
    row.breaker.failures = 2;
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
    row.breaker = BreakerSpec::default();
    row.breaker.failures = 2;
    let plugin = loaded(&runtime, row);
    let engine = plugin_engine(&plugin);
    let decide = |seed: u64| {
        let mut request_facts = facts(seed);
        request_facts.budget_micros = FRAME_MICROS;
        engine.decide(
            &request_facts,
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
    let scorers = || vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
    let mut as_filter = Profile::of("filter".into(), Box::new(Best));
    as_filter.filters = vec![Box::new(WasmFilter::new(plugin.bind(), None))];
    as_filter.scorers = scorers();
    let mut as_picker = Profile::of(
        "picker".into(),
        Box::new(WasmPicker::new(plugin.bind(), None)),
    );
    as_picker.filters = vec![Box::new(Healthy)];
    as_picker.scorers = scorers();
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
    use smg_grand_central::engine::{Filter, Picker, Scorer, Targets, TieBand};

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
