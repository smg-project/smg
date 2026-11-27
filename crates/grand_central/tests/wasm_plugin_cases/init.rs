//! The lazy `init` of a cold slot: its bound on the wall clock, what it
//! charges the decision that pays it, and where its time and its frame stand
//! in that decision's sample.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

use smg_grand_central::{
    engine::{Constraints, Decision, Profile, Scratch, WeightedScorer},
    plugins::{
        shared::{BreakerSpec, OpenDecisions},
        wasm::{LoadError, Plugin, Runtime, WasmFilter, WasmPicker, WasmScorer},
    },
};

use crate::{
    bound_clock, facts, four, load, load_slow, loaded, made_again, off_cpu_since, one_burn_micros,
    plugin_engine, plugin_whose_init_asks, pool, runtime, single_engine, spec, steered_runtime,
    steered_runtime_counting,
    support::wasm_guest::{self as guest, mode},
    tried, FRAME_MICROS, INIT_FRAME, SLOW_CALIBRATION, SLOW_FRAME_MICROS, TRIES,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::{decide, excused, hold};

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
    let filters: Vec<Box<dyn smg_grand_central::engine::Filter>> =
        vec![Box::new(WasmFilter::new(bound.clone(), None))];
    let scorers = vec![
        WeightedScorer::of(Box::new(WasmScorer::new(bound.clone(), None)), 1.0),
        WeightedScorer::of(Box::new(WasmScorer::new(neighbour.bind(), None)), 1.0),
    ];
    let mut profile = Profile::of("cold-init".into(), Box::new(WasmPicker::new(bound, None)));
    profile.filters = filters;
    profile.scorers = scorers;
    let engine = single_engine(profile);
    cold.forget_instance();
    clock.store(steer, Ordering::Relaxed);
    let (wall, cpu) = (std::time::Instant::now(), bound_clock());
    let mut request_facts = facts(1);
    request_facts.budget_micros = budget_micros;
    let decision = engine.decide(
        &request_facts,
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

/// A cold slot opened under a budget an earlier open spent makes no
/// instance. The plugin in front is the one of the case above, whose `init`
/// outlasts the 10 ms decision. The neighbour's slot is cold at its open:
/// the open is the decision's budget spent, no `init` runs under nothing,
/// and the close has no slot of the decision's to rebuild. Nothing counts
/// on the breaker, and the slot stays as cold as the open found it.
#[test]
fn a_cold_slot_opened_under_a_spent_budget_makes_no_instance() {
    let (clock, runtime, cold) = plugin_whose_init_asks(20, |_| {});
    let neighbour = load(
        &runtime,
        "after-the-init",
        guest::record_guest(),
        mode::LEAST_IN_FLIGHT,
    );
    neighbour.forget_instance();
    let made = neighbour.stats().instantiations;
    decide_on_a_cold_slot(&clock, 1, &cold, &neighbour, FRAME_MICROS);
    let n = neighbour.stats();
    assert_eq!(
        (n.budget_spent, n.answered, n.breaker_opens),
        (1, 0, 0),
        "the open is the decision's budget spent and no failure: {n:?}"
    );
    assert_eq!(
        (n.instantiations - made, n.lazy_instantiations),
        (0, 0),
        "no `init` ran under nothing and the close rebuilt nothing: {n:?}"
    );
    assert!(
        !neighbour.is_warm(),
        "the slot is as cold as its open found it"
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
/// the remainder gone. The plugin itself counts no budget spent: its `init`
/// was granted its whole frame, so no clamp ended it, though its charge took
/// all that was left. An `init` the wall clock ended with its thread off
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
            (
                s.lazy_instantiations,
                s.answered,
                s.breaker_opens,
                s.budget_spent
            ),
            (0, 0, 0, 0),
            "the init ran its own frame out: one failure, short of the breaker's five, and no budget spent of its own: {s:?}"
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
            row.breaker = BreakerSpec::default();
            row.breaker.failures = 100;
        });
        let engine = plugin_engine(&plugin);
        let half = std::time::Duration::from_micros(u64::from(BUDGET_MICROS) / 2);
        let mut cheap = true;
        for seed in 1..=64 {
            plugin.forget_instance();
            clock.store(steer, Ordering::Relaxed);
            let cpu = bound_clock();
            let mut request_facts = facts(seed);
            request_facts.budget_micros = BUDGET_MICROS;
            let decision = engine.decide(
                &request_facts,
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
        let mut request_facts = facts(seed);
        request_facts.budget_micros = init_micros / 3 * 4;
        let decision = engine.decide(
            &request_facts,
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
        let mut request_facts = facts(seed);
        request_facts.budget_micros = u32::MAX;
        let decision = engine.decide(
            &request_facts,
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
        let mut request_facts = facts(seed);
        request_facts.budget_micros = init_micros / 3 * 4;
        let decision = engine.decide(
            &request_facts,
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
        row.breaker = BreakerSpec::default();
        row.breaker.failures = 100;
    });
    let engine = plugin_engine(&plugin);
    let decide_at = |steer: u8, seed: u64| {
        clock.store(steer, Ordering::Relaxed);
        let mut request_facts = facts(seed);
        request_facts.budget_micros = u32::MAX;
        let decision = engine.decide(
            &request_facts,
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
                let mut request_facts = facts(seed);
                request_facts.budget_micros = budget_micros;
                let decision = engine.decide(
                    &request_facts,
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
            let mut request_facts = facts(seed);
            request_facts.budget_micros = budget_micros;
            let decision = engine.decide(
                &request_facts,
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
            let mut request_facts = facts(seed);
            request_facts.budget_micros = budget_micros;
            let decision = engine.decide(
                &request_facts,
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
        row.breaker = BreakerSpec::default();
        row.breaker.failures = 2;
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
