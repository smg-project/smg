//! What each binding asks, and the answer's validity rules: a decline and an
//! invalid answer.

#[cfg(feature = "wasm")]
use std::sync::Arc;
use std::time::Duration;

use smg_grand_central::{
    document::Needs,
    engine::{Constraints, Engine, PairPools, Plan, PoolFacts, Scratch, WorkerId},
    plugins::{
        grpc::{Plugin, Runtime},
        shared::{BreakerSpec, OpenDecisions},
    },
    service::proto,
};
#[cfg(feature = "wasm")]
use smg_grand_central::{
    engine::{Filter, Profile, Scorer, WeightedScorer},
    plugins::grpc::GrpcScorer,
    stages::{Best, Healthy, LeastRequests},
};

use crate::{
    binding, decide, decide_pair, decide_with, facts, lead, load, pool, profile, runtime,
    single_engine, spec,
    support::{
        grpc_plugin::{eventually, Behaviour, TestPlugin, ALL, FILTER, PICK, SCORE},
        said::{lines, openings, said},
    },
    BUDGET_MICROS,
};
#[cfg(feature = "wasm")]
use crate::{guest_loaded, support};

#[test]
fn one_frame_answers_the_three_stages() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "whole");
    let engine = single_engine(profile(&plugin, ALL));
    let mut candidates = pool(4);
    candidates[3].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(102),
        "the least loaded of the rows the plugin's filter kept"
    );
    let seen = served.seen();
    assert_eq!(seen.frames.len(), 1, "one frame for the decision");
    let frame = &seen.frames[0];
    assert_eq!(frame.stages, ALL);
    assert_eq!(frame.candidates.len(), 4, "the whole view");
    assert!(
        frame.scores_in.is_empty(),
        "asked at the open: no scores yet"
    );
    let request = frame.request.as_ref().expect("the request");
    assert_eq!(
        (frame.budget_micros, request.budget_micros),
        (BUDGET_MICROS, BUDGET_MICROS),
        "the plugin's deadline is what the decision has left, never more"
    );
    assert_eq!((request.model.as_str(), request.seed), ("m", 1));
    assert_eq!(
        frame.pool.as_ref().map(|p| (p.pool_size, p.generation)),
        Some((4, 1))
    );
    drop(seen);
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.declined, stats.invalid, stats.missed),
        (1, 0, 0, 0)
    );
}

#[test]
fn a_filter_alone_narrows_the_pool() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::KeepFirst);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "filter");
    let engine = single_engine(profile(&plugin, FILTER));
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(100),
        "the one row the plugin kept, the most loaded: no built-in stage leaves it"
    );
    assert_eq!(served.seen().frames[0].stages, FILTER);
    assert_eq!(plugin.stats().answered, 1);
}

#[test]
fn a_scorer_alone_is_asked_for_scores_alone() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::ScoreMostLoaded);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(100),
        "the built-in picker follows the plugin's scores to the most loaded row"
    );
    assert_eq!(served.seen().frames[0].stages, SCORE);
    assert_eq!(plugin.stats().answered, 1);
}

// The best scores of the four rows are 1/2 and 1/3, followed by 1/4.
const SCORES_WITH_BAND: Behaviour = Behaviour::ScoreBand(210);
const BAND_ROUNDS: u64 = 64;

#[test]
fn a_plugins_band_widens_the_tie_the_built_in_picker_draws_from() {
    let served = TestPlugin::serve();
    served.behave(SCORES_WITH_BAND);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    let candidates = pool(4);
    let mut led = [false; 2];
    for seed in 0..BAND_ROUNDS {
        let picked = lead(&decide(&engine, &candidates, seed));
        assert!(matches!(picked, Some(102 | 103)), "seed {seed}: {picked:?}");
        led[0] |= picked == Some(102);
        led[1] |= picked == Some(103);
    }
    assert_eq!(led, [true; 2], "each row in the band led");
    assert_eq!(plugin.stats().answered, BAND_ROUNDS);
}

#[test]
fn a_plugin_that_sends_no_band_decides_as_before() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    let candidates = pool(4);
    for seed in 0..BAND_ROUNDS {
        assert_eq!(
            lead(&decide(&engine, &candidates, seed)),
            Some(103),
            "seed {seed}"
        );
    }
    assert_eq!(plugin.stats().answered, BAND_ROUNDS);
}

#[test]
fn a_band_is_not_carried_to_the_next_decision() {
    let served = TestPlugin::serve();
    served.behave(SCORES_WITH_BAND);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scorer");
    let engine = single_engine(profile(&plugin, SCORE));
    let candidates = pool(4);
    let seed = 13;
    assert_eq!(lead(&decide(&engine, &candidates, seed)), Some(102));
    served.behave(Behaviour::Honest);
    assert_eq!(lead(&decide(&engine, &candidates, seed)), Some(103));
    assert_eq!(plugin.stats().answered, 2);
}

#[test]
fn a_picker_alone_is_asked_at_the_pick_with_the_combined_scores() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "picker");
    let engine = single_engine(profile(&plugin, PICK));
    let mut candidates = pool(4);
    candidates[0].health.healthy = false;
    // The scripted picker takes the surviving row with the LOWEST combined
    // score: the most loaded healthy row, which the built-in picker (the
    // highest score) would not choose.
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(101));
    let seen = served.seen();
    assert_eq!(seen.frames.len(), 1);
    let frame = &seen.frames[0];
    assert_eq!(frame.stages, PICK);
    assert_eq!(frame.scores_in.len(), 4, "one combined score per candidate");
    assert!(
        frame.scores_in[0] < 0.0,
        "the row the filter dropped scores below zero"
    );
    assert!(
        frame.scores_in[1..].iter().all(|s| *s >= 0.0),
        "a surviving row never does: {:?}",
        frame.scores_in
    );
    drop(seen);
    assert_eq!(plugin.stats().answered, 1);
}

#[test]
fn a_pick_of_a_row_the_frame_showed_as_dropped_is_the_plugins_fault() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::PickDroppedRow);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "blind-picker");
    let engine = single_engine(profile(&plugin, PICK));
    let mut candidates = pool(4);
    candidates[0].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(103),
        "the fallback picker decides"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.answered),
        (1, 0),
        "the frame named its survivors: the answer is refused whole"
    );
}

#[test]
fn a_pick_the_native_filters_dropped_since_the_open_is_no_fault() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::PickFirst);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "early-picker");
    // Asked at the open over the whole view (the scorer is bound): the
    // plugin cannot know what the built-in filter drops meanwhile.
    let engine = single_engine(profile(&plugin, SCORE | PICK));
    let mut candidates = pool(4);
    candidates[0].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(103),
        "the fallback picker decides over the plugin's scores"
    );
    assert_eq!(served.seen().frames[0].stages, SCORE | PICK);
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.answered, stats.breaker_opens),
        (0, 1, 0),
        "a pick within the view it was shown is within the contract"
    );
}

#[test]
fn a_plugin_is_asked_for_the_stages_it_implements_alone() {
    let served = TestPlugin::serve();
    served.script(|s| s.implements = SCORE);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "scores-only");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert_eq!(
        served.seen().frames[0].stages,
        SCORE,
        "the profile binds three stages, the ack implements one"
    );
    assert_eq!(plugin.stats().answered, 1);

    let idle = TestPlugin::serve();
    idle.script(|s| s.implements = 0);
    let plugin = load(&runtime, &idle, "implements-nothing");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert_eq!(idle.frames(), 0, "nothing to ask: no frame");
}

/// A stage the profile binds and the plugin's ack leaves out takes its
/// fallback on every decision, and no counter moves for it: the host says
/// so once a bound set, which is what names the mismatch.
#[test]
fn a_stage_bound_and_not_implemented_says_so_once() {
    said();
    let served = TestPlugin::serve();
    served.script(|s| s.implements = SCORE);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "partly-implemented");
    let engine = single_engine(profile(&plugin, ALL));
    for seed in 1..=3 {
        assert_eq!(lead(&decide(&engine, &pool(4), seed)), Some(103));
    }
    let said_once = lines("partly-implemented", "does not implement");
    assert_eq!(said_once.len(), 1, "one line a bound set: {said_once:?}");
    assert!(
        said_once[0].contains(&format!("bound_stages={ALL}"))
            && said_once[0].contains(&format!("implemented_stages={SCORE}")),
        "{}",
        said_once[0]
    );

    // An ack that implements no stage loads (the plugin may observe alone)
    // and is named the same way.
    let idle = TestPlugin::serve();
    idle.script(|s| s.implements = 0);
    let plugin = load(&runtime, &idle, "observes-alone");
    let engine = single_engine(profile(&plugin, ALL));
    decide(&engine, &pool(4), 1);
    decide(&engine, &pool(4), 2);
    assert_eq!(lines("observes-alone", "does not implement").len(), 1);
}

#[test]
fn a_pair_asks_both_sides_at_once_on_one_stream() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "pair");
    let engine = Engine::new(
        vec![profile(&plugin, ALL)],
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
    let decision = engine.decide_pair(
        &facts(9),
        &pool_facts,
        pools,
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    let pick = decision.pick().expect("a pick");
    let lead = pick.targets.first().expect("a target");
    assert_eq!(lead.id, WorkerId(102), "the least loaded prefill");
    assert_eq!(
        lead.pair.map(|p| p.decode_id),
        Some(WorkerId(203)),
        "the least loaded decode of its class"
    );
    let seen = served.seen();
    assert_eq!(seen.frames.len(), 2, "one frame per side");
    let phases: Vec<i32> = seen
        .frames
        .iter()
        .map(|f| f.request.as_ref().map_or(0, |r| r.phase))
        .collect();
    assert_eq!(
        phases,
        [proto::Phase::Prefill as i32, proto::Phase::Decode as i32]
    );
    assert_ne!(
        seen.frames[0].decision_id, seen.frames[1].decision_id,
        "each frame its own id on the stream"
    );
    assert_eq!(
        (
            seen.frames[0].candidates.len(),
            seen.frames[1].candidates.len()
        ),
        (3, 4)
    );
    drop(seen);
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.late_answers, stats.missed),
        (2, 0, 0),
        "both frames were in flight together and each answer found its own"
    );
}

/// The decision is the breaker's unit in this kind too. A plugin that
/// answers outside the contract on both sides of a pair fails the decision
/// once: both frames left at the opens, the second answer is not read.
#[test]
fn an_invalid_answer_fails_its_decision_once_on_a_pair() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "pair-sloppy");
    spec.breaker = BreakerSpec::default();
    spec.breaker.failures = 2;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    served.behave(Behaviour::ScoreNan);
    let decision = decide_pair(&plugin, &facts(9));
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    assert!(
        eventually(|| served.frames() == 2),
        "both frames left at the opens"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.breaker_opens),
        (1, 0),
        "one failure a decision: the other side's answer was not read"
    );
    // The next such decision is the threshold's second failure.
    decide_pair(&plugin, &facts(10));
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.breaker_opens), (2, 1));
}

/// A plugin asked for the pick alone on both sides of a pair, whose first
/// pick is a row its frame showed as dropped: the decision is failed once
/// and the other side's pick sends no frame, as an open after the failure
/// sends none.
#[test]
fn a_pick_only_frame_is_not_sent_once_the_plugin_failed_its_decision() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::PickDroppedRow);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "pair-blind-picker");
    let engine = Engine::new(
        vec![profile(&plugin, PICK)],
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
    // A row the built-in filter drops on each side: each frame names it.
    prefills[0].health.healthy = false;
    decodes[0].health.healthy = false;
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 7;
    pool_facts.generation = 1;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    let decision = engine.decide_pair(
        &facts(9),
        &pool_facts,
        pools,
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.answered), (1, 0), "{stats:?}");
    // A second frame, were it sent, is answered into nothing.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        (served.frames(), plugin.stats().late_answers),
        (1, 0),
        "the failed decision's other side asked nothing"
    );
}

/// One decision over both kinds: a WebAssembly plugin filters and a remote
/// plugin scores on both sides of a pair, their runtimes built over the same
/// open decisions. The decision is seeded once and has one remainder: each
/// remote frame carries what the WebAssembly plugin's constructors left of
/// the budget by its open, less than the budget and less again on the second
/// side. And the decision is one unit to each plugin: the WebAssembly filter
/// traps and the remote scores are not numbers, one failure each, however
/// many sides opened them.
#[cfg(feature = "wasm")]
#[test]
fn a_pair_decision_over_both_kinds_is_one_decision_to_every_plugin() {
    use smg_grand_central::plugins::wasm;
    use support::wasm_guest::{mode, record_guest};

    let open = OpenDecisions::new();
    let guests = Arc::new(wasm::Runtime::new(&open).expect("a runtime"));
    let runtime = Runtime::new(&open).expect("a runtime");
    let mut row = wasm::PluginSpec::new("trapping", record_guest());
    row.config = vec![mode::TRAP_IN_FILTER];
    row.budget_micros = 10_000;
    let guest = guest_loaded(&guests, &row);
    let served = TestPlugin::serve();
    let remote = load(&runtime, &served, "sloppy");
    served.behave(Behaviour::ScoreNan);
    let healthy: Option<Box<dyn Filter>> = Some(Box::new(Healthy));
    let least: Option<Box<dyn Scorer>> = Some(Box::new(LeastRequests));
    let mut profile = Profile::of("both-kinds".into(), Box::new(Best));
    profile.filters = vec![Box::new(wasm::WasmFilter::new(guest.bind(), healthy))];
    profile.scorers = vec![WeightedScorer::of(
        Box::new(GrpcScorer::new(remote.bind(), least)),
        1.0,
    )];
    let engine = Engine::new(
        vec![profile],
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
    let request = facts(9);
    let mut pool_facts = PoolFacts::default();
    pool_facts.pool_size = 7;
    pool_facts.generation = 1;
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
    let decision = engine.decide_pair(
        &request,
        &pool_facts,
        pools,
        &Constraints::default(),
        &Constraints::default(),
        &mut Scratch::new(),
    );
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    assert!(
        eventually(|| served.frames() == 2),
        "both remote frames left at the opens"
    );
    let (g, r) = (guest.stats(), remote.stats());
    assert_eq!(
        (g.traps, r.invalid),
        (1, 1),
        "one failure a decision for each plugin: {g:?} {r:?}"
    );
    let left: Vec<u32> = served
        .seen()
        .frames
        .iter()
        .map(|frame| frame.request.as_ref().map_or(0, |r| r.budget_micros))
        .collect();
    assert!(
        left[1] < left[0] && left[0] < request.budget_micros,
        "one remainder: each frame carries what the WebAssembly constructors left of {}: {left:?}",
        request.budget_micros
    );
}

/// A frame budget gone on arrival is one budget hit a decision, on a pair
/// as on a single placement: the second side falls back without a count.
#[test]
fn a_gone_budget_fails_its_decision_once_on_a_pair() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "pair-budgetless");
    spec.breaker = BreakerSpec::default();
    spec.breaker.failures = 2;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let mut gone = facts(9);
    gone.budget_micros = 0;
    let decision = decide_pair(&plugin, &gone);
    assert!(decision.pick().is_some(), "the fallbacks placed the pair");
    let stats = plugin.stats();
    assert_eq!(
        (stats.budget_gone, stats.breaker_opens),
        (1, 0),
        "one budget hit a decision"
    );
    decide_pair(&plugin, &gone);
    let stats = plugin.stats();
    assert_eq!((stats.budget_gone, stats.breaker_opens), (2, 1));
    assert_eq!(served.frames(), 0, "no frame left the host");
}

/// An opening is otherwise silent (the stages take their fallback): in
/// this kind too it writes one line, with the plugin's own name, the rule
/// and the cool-down.
#[test]
fn an_opening_writes_one_line_under_the_plugins_name() {
    said();
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "grpc-opener");
    spec.breaker = BreakerSpec::default();
    spec.breaker.failures = 2;
    spec.breaker.window_micros = 60_000_000;
    spec.breaker.cooldown_micros = 60_000_000;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let mut gone = facts(1);
    gone.budget_micros = 0;
    let candidates = pool(4);
    decide_with(&engine, &gone, &candidates);
    assert!(
        openings("grpc-opener").is_empty(),
        "one failure under a threshold of two opens nothing"
    );
    decide_with(&engine, &gone, &candidates);
    decide_with(&engine, &gone, &candidates);
    assert_eq!(plugin.stats().bypassed, 1);
    let lines = openings("grpc-opener");
    assert_eq!(lines.len(), 1, "one opening, one line: {lines:?}");
    assert!(
        lines[0].contains("rule=\"failures\"") && lines[0].contains("cooldown_micros=60000000"),
        "{}",
        lines[0]
    );
}

#[test]
fn every_decision_of_a_thread_rides_its_one_stream() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "steady");
    let engine = single_engine(profile(&plugin, ALL));
    for seed in 1..=5 {
        assert_eq!(lead(&decide(&engine, &pool(4), seed)), Some(103));
    }
    let stats = plugin.stats();
    assert_eq!((stats.connects, stats.answered), (1, 5));
    let seen = served.seen();
    assert_eq!(seen.hellos.len(), 1);
    let ids: Vec<u64> = seen.frames.iter().map(|f| f.decision_id).collect();
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]) && ids[0] != 0,
        "the thread numbers its frames, never zero: {ids:?}"
    );
}

#[test]
fn a_declined_stage_takes_its_fallback_and_the_others_answer() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Decline {
        stages: FILTER,
        reason: proto::decline::Reason::NotApplicable as i32,
    });
    let runtime = runtime();
    let plugin = load(&runtime, &served, "no-filter");
    let engine = single_engine(profile(&plugin, ALL));
    let mut candidates = pool(4);
    candidates[3].health.healthy = false;
    assert_eq!(
        lead(&decide(&engine, &candidates, 1)),
        Some(102),
        "the fallback filter drops the unhealthy row, the plugin's pick stands"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.declined, stats.busy, stats.answered, stats.invalid),
        (1, 0, 1, 0)
    );
}

#[test]
fn a_budget_decline_is_busy_and_a_frame_that_answers_nothing_declines_every_stage() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Decline {
        stages: ALL,
        reason: proto::decline::Reason::Budget as i32,
    });
    let runtime = runtime();
    let plugin = load(&runtime, &served, "busy");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    let stats = plugin.stats();
    assert_eq!(
        (stats.busy, stats.declined, stats.answered),
        (1, 3, 0),
        "the plugin's own budget: every stage falls back at once"
    );

    served.behave(Behaviour::Unanswered);
    assert_eq!(lead(&decide(&engine, &pool(4), 2)), Some(103));
    let stats = plugin.stats();
    assert_eq!(
        (stats.busy, stats.declined, stats.answered),
        (1, 6, 0),
        "an asked stage left unanswered is declined"
    );
    assert_eq!(
        (stats.invalid, stats.breaker_opens),
        (0, 0),
        "a decline is no fault"
    );
}

#[test]
fn each_invalid_answer_is_one_fault_and_the_frame_falls_back_whole() {
    let behaviours = [
        Behaviour::SurvivorOutOfRange,
        Behaviour::SurvivorTwice,
        Behaviour::ScoreCount,
        Behaviour::ScoreNan,
        Behaviour::PickOutsideOwnSurvivors,
        Behaviour::DeclinedAndAnswered,
        Behaviour::BandNan,
        Behaviour::BandAboveOne,
        Behaviour::BandBelowZero,
        Behaviour::BandWithoutScores,
        Behaviour::BandWithScoreDecline,
    ];
    let failures = behaviours.len();
    let served = TestPlugin::serve();
    served.script(|s| s.counters = vec![(0, 1)]);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), || 1_234).expect("a runtime");
    let mut spec = spec(&served, "sloppy");
    spec.breaker = BreakerSpec::default();
    spec.breaker.failures = failures as u32;
    spec.metrics = vec!["answers".to_owned()];
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    for (i, behaviour) in behaviours.into_iter().enumerate() {
        served.behave(behaviour);
        assert_eq!(
            lead(&decide(&engine, &pool(4), i as u64 + 1)),
            Some(103),
            "{behaviour:?}: the fallbacks decide"
        );
        let stats = plugin.stats();
        assert_eq!(
            (
                stats.invalid,
                stats.answered,
                stats.declined,
                stats.breaker_opens
            ),
            (i as u64 + 1, 0, 0, u64::from(i + 1 == failures)),
            "{behaviour:?}"
        );
    }
    assert!(plugin.counters().read().all(|(_, value)| value == 0));
}

#[test]
fn the_counters_an_answer_carries_are_added_by_declared_id() {
    let served = TestPlugin::serve();
    // Id 7 is not declared: the host ignores it. An answer holds one
    // increment a declared counter at most, so three are declared.
    served.script(|s| s.counters = vec![(0, 2), (1, 5), (7, 1)]);
    let runtime = runtime();
    let mut spec = spec(&served, "counting");
    spec.metrics = vec!["hits".to_owned(), "skips".to_owned(), "spare".to_owned()];
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    decide(&engine, &pool(4), 1);
    decide(&engine, &pool(4), 2);
    let read: Vec<(String, u64)> = plugin
        .counters()
        .read()
        .map(|(name, value)| (name.to_owned(), value))
        .collect();
    assert_eq!(
        read,
        [
            ("hits".to_owned(), 4),
            ("skips".to_owned(), 10),
            ("spare".to_owned(), 0)
        ],
        "two answers"
    );
}

/// An answer holds no more increments than the plugin declared counters:
/// a longer list is outside the contract, one fault, and none of it is
/// added.
#[test]
fn more_increments_than_declared_counters_is_one_fault() {
    let served = TestPlugin::serve();
    served.script(|s| s.counters = vec![(0, 2), (1, 5), (0, 1)]);
    let runtime = runtime();
    let mut spec = spec(&served, "over-counting");
    spec.metrics = vec!["first".to_owned(), "second".to_owned()];
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.answered), (1, 0), "{stats:?}");
    assert!(
        plugin.counters().read().all(|(_, value)| value == 0),
        "nothing of the list is added"
    );
}

/// An answer is bounded in bytes by what its row lets it hold: a message
/// past that ends the stream before it is decoded.
#[test]
fn an_answer_past_the_message_bound_ends_its_stream() {
    let served = TestPlugin::serve();
    // Hundreds of kilobytes of increments under a row of 128 candidates.
    served.script(|s| s.counters = vec![(1, 1); 50_000]);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "flooding");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(103));
    assert!(
        eventually(|| plugin.stats().failed_streams == 1),
        "the stream ends at the message: {:?}",
        plugin.stats()
    );
    let stats = plugin.stats();
    assert_eq!((stats.invalid, stats.answered), (0, 0), "{stats:?}");
}
