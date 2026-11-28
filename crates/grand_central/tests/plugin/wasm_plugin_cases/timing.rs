//! The clock on a call: a hold of the host excuses lateness and never a
//! fault, what a hold short of late leaves of the budget and of a window, an
//! arming and its length, and a destructor's own bound.

use std::sync::Arc;

use smg_grand_central::{
    engine::{Constraints, Decision, Scratch},
    plugins::wasm::Runtime,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use smg_grand_central::{
    engine::{Engine, PoolFacts, Profile, WeightedScorer},
    plugins::{
        shared::BreakerSpec,
        wasm::{Plugin, WasmScorer},
    },
    stages::Best,
};

use crate::{
    bound_clock, decide, facts, four, load, loaded, made_again, off_cpu_since, plugin_engine, pool,
    runtime, spec,
    support::wasm_guest::{self as guest, mode},
    the_close_rebuilt, tried, TRIES,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::{
    excused, hold, pick_ids, said, single_engine, timed, FRAME_MICROS, SLOW_CALIBRATION,
    SLOW_FRAME_MICROS,
};

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
    held.breaker = BreakerSpec::default();
    held.breaker.failures = 1;
    held.breaker.window_micros = 60_000_000;
    held.breaker.cooldown_micros = 60_000_000;
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
    held.breaker = BreakerSpec::default();
    held.breaker.failures = 2;
    held.breaker.window_micros = 60_000_000;
    held.breaker.cooldown_micros = 60_000_000;
    let plugin = loaded(&runtime, held);
    let engine = plugin_engine(&plugin);
    // On three candidates the guest's score traps after its log line.
    let candidates = pool(3);
    let mut three = PoolFacts::default();
    three.pool_size = 3;
    three.generation = 1;
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
    slow.breaker = BreakerSpec::default();
    slow.breaker.failures = 1;
    slow.breaker.window_micros = 60_000_000;
    slow.breaker.cooldown_micros = 60_000_000;
    let plugin = loaded(&runtime, slow);
    let engine = plugin_engine(&plugin);
    let candidates = pool(4);
    hold(name, held, 1);
    let wall = std::time::Instant::now();
    let mut request_facts = facts(1);
    request_facts.budget_micros = budget_micros;
    let decision = engine.decide(
        &request_facts,
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
    let mut request_facts = facts(2);
    request_facts.budget_micros = 20_000;
    let decision = engine.decide(
        &request_facts,
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
    let scorer = |plugin: &Arc<Plugin>| {
        WeightedScorer::of(Box::new(WasmScorer::new(plugin.bind(), None)), 1.0)
    };
    let mut profile = Profile::of("held-first".into(), Box::new(Best));
    profile.filters = vec![];
    profile.scorers = vec![scorer(&held), scorer(&spinner)];
    let engine = single_engine(profile);
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
        let mut request_facts = facts(seed);
        request_facts.budget_micros = BUDGET_MICROS;
        let decision = engine.decide(
            &request_facts,
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
        let mut request_facts = facts(1);
        request_facts.budget_micros = 100_000;
        let decision = engine.decide(
            &request_facts,
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
