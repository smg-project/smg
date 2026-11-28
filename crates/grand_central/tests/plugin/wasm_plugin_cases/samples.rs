//! The latency sample of a decision and the window that opens a slow plugin:
//! what a sample is read against, which call is the plugin's own, and when a
//! decision is sampled.

use std::sync::Arc;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use smg_grand_central::engine::Engine;
use smg_grand_central::{
    engine::{Constraints, Decision, Profile, Scratch, WeightedScorer},
    plugins::{
        shared::BreakerSpec,
        wasm::{Plugin, WasmFilter, WasmPicker, WasmScorer},
    },
    stages::Best,
};

#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::hold;
use crate::{
    bound_clock, decide_on_the_cpu, decide_until_open, excused, facts, four,
    held_counts_are_made_again, load, load_slow, loaded, no_hold, off_cpu_since, one_burn_micros,
    openings, pair_decide_under, pair_engine, plugin_engine, plugin_profile, pool, runtime, said,
    single_engine, spec,
    support::wasm_guest::{self as guest, mode},
    the_neighbour_stays, timed, tried, FRAME_MICROS, PAIR_BUDGET_MICROS, SLOW_CALIBRATION,
    SLOW_FRAME_MICROS,
};

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
    slow.breaker = BreakerSpec::default();
    slow.breaker.failures = 1_000;
    slow.breaker.window_micros = 60_000_000;
    slow.breaker.cooldown_micros = 60_000_000;
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
        let mut request_facts = facts(seed);
        request_facts.budget_micros = BUDGET_MICROS;
        let decision = engine.decide(
            &request_facts,
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

/// Sixty-four decisions of `plugin`, bound as the three stages, under a
/// request that states no budget: a stage is shown the widest one there.
fn decide_64_with_no_budget(plugin: &Arc<Plugin>) {
    let engine = plugin_engine(plugin);
    for seed in 1..=64 {
        let mut request_facts = facts(seed);
        request_facts.budget_micros = u32::MAX;
        let decision = engine.decide(
            &request_facts,
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
    let mut profile = Profile::of("one-stage".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![WeightedScorer::of(
        Box::new(WasmScorer::new(plugin.bind(), None)),
        1.0,
    )];
    let engine = single_engine(profile);
    for seed in 1..=64 {
        let mut request_facts = facts(seed);
        request_facts.budget_micros = u32::MAX;
        let decision = engine.decide(
            &request_facts,
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
    let scorer = |plugin: &Arc<Plugin>| {
        WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
    };
    let decide_with_budget = |engine: &Engine, budget_micros: u32, seed: u64| {
        let mut request_facts = facts(seed);
        request_facts.budget_micros = budget_micros;
        let decision = engine.decide(
            &request_facts,
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
    let mut profile = Profile::of("front-alone".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![scorer(&front)];
    let alone = single_engine(profile);
    let before = front.stats().fuel_used;
    decide_with_budget(&alone, SLOW_FRAME_MICROS, 0);
    let front_micros =
        u32::try_from((front.stats().fuel_used - before).div_ceil(SLOW_CALIBRATION)).expect("fits");
    let budget_micros = front_micros + LEFT_MICROS;
    let mut profile = Profile::of("front-then-held".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![scorer(&front), scorer(&held)];
    let engine = single_engine(profile);
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
    let mut profile = Profile::of("held-alone".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![WeightedScorer::of(
        Box::new(WasmScorer::new(held.bind(), None)),
        1.0,
    )];
    let engine = single_engine(profile);
    hold(
        name,
        std::time::Duration::from_micros(u64::from(OWN_FRAME_MICROS) * 2),
        1,
    );
    let mut request_facts = facts(1);
    request_facts.budget_micros = u32::MAX;
    let decision = engine.decide(
        &request_facts,
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
    let mut profile = Profile::of("two".into(), Box::new(WasmPicker::new(picker.bind(), None)));
    profile.filters = vec![];
    profile.scorers = vec![WeightedScorer::of(
        Box::new(WasmScorer::new(scorer.bind(), None)),
        1.0,
    )];
    let engine = single_engine(profile);
    let decide_under = |budget_micros: u32, seed: u64| {
        let mut request_facts = facts(seed);
        request_facts.budget_micros = budget_micros;
        let decision = engine.decide(
            &request_facts,
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
    let scorer = |plugin: &Arc<Plugin>| {
        WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
    };
    let mut profile = Profile::of("fuel-fast".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![scorer(&honest), scorer(&spinner)];
    let engine = single_engine(profile);
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
        let scorer = |plugin: &Arc<Plugin>| {
            WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
        };
        let scorers = if spinner_first {
            vec![scorer(&spinner), scorer(&honest)]
        } else {
            vec![scorer(&honest), scorer(&spinner)]
        };
        let mut profile = Profile::of("ctor".into(), Box::new(Best));
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
    let scorer = |plugin: &Arc<Plugin>| {
        WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
    };
    let mut profile = Profile::of("hog".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![scorer(&honest), scorer(&hog)];
    let engine = single_engine(profile);
    // A burn and a half: the hog's constructor takes two thirds of it and
    // answers, a third of a burn of slack for its stages.
    let budget_micros = burn_micros * 3 / 2;
    for seed in 1..=64 {
        let mut record = facts(seed);
        record.budget_micros = budget_micros;
        engine.decide(
            &record,
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
        let own = WeightedScorer::of(Box::new(WasmScorer::new(bound.clone(), None)), 1.0);
        let other = WeightedScorer::of(Box::new(WasmScorer::new(honest.bind(), None)), 1.0);
        let scorers = if spinner_first {
            vec![own, other]
        } else {
            vec![other, own]
        };
        let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> =
            vec![Box::new(WasmFilter::new(bound.clone(), None))];
        let mut profile = Profile::of("third-stage".into(), Box::new(WasmPicker::new(bound, None)));
        profile.filters = filters;
        profile.scorers = scorers;
        let engine = single_engine(profile);
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
