//! The budget and the breaker.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use smg_grand_central::{
    engine::{
        Candidate, Constraints, Decision, Filter, Picker, Profile, RequestFacts, Scorer,
        ShedReason, Targets, TieBand, WeightedScorer,
    },
    plugins::{
        grpc::{GrpcFilter, GrpcScorer, Plugin, Runtime},
        shared::{BreakerSpec, Failure, FrameBudget, OpenDecisions},
    },
    stages::{Best, Healthy, LeastRequests, Lifecycle},
};

use crate::{
    decide, decide_short, decide_until_judged, decide_with, facts, filter_then_picker, lead, load,
    made_again, pool, profile, profile_of, runtime, single_engine, spec,
    support::{
        grpc_plugin::{eventually, Behaviour, TestPlugin, ALL, FILTER, PICK, SCORE},
        said::{openings, said},
    },
    BUDGET_MICROS, SHORT_MICROS,
};

/// A deadline missed under the plugin's own bound, on an I/O thread seen
/// running inside a flight longer than its promise, is the plugin's
/// failure: one of its decision, counted on the breaker like a fault.
#[test]
fn a_missed_deadline_under_the_plugins_own_bound_is_one_failure_of_its_decision() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let mut spec = spec(&served, "silent");
    // One failure opens the breaker: the miss is one.
    spec.breaker = BreakerSpec {
        failures: 1,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let started = Instant::now();
    assert_eq!(
        lead(&decide_short(&engine, &pool(4), 1)),
        Some(103),
        "the fallbacks decide"
    );
    assert!(
        started.elapsed() >= Duration::from_micros(u64::from(SHORT_MICROS)),
        "the first stage waited out the frame's deadline"
    );
    assert!(eventually(|| served.frames() == 1));
    assert_eq!(
        served.seen().frames[0].budget_micros,
        SHORT_MICROS,
        "the deadline the frame named"
    );
    // A loaded host can hold the I/O thread under any one frame, and a held
    // miss is the host's: the next frame that leaves is judged.
    let mut short = facts(1);
    short.budget_micros = SHORT_MICROS;
    decide_until_judged(&engine, &plugin, &short, &pool(4), 1);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.own_misses,
            stats.unread,
            stats.answered
        ),
        (1, 1, 0, 0),
        "{stats:?}"
    );
    assert_eq!(
        stats.budget_spent, stats.stalls,
        "a miss of its own is not the decision's budget spent: {stats:?}"
    );
    assert_eq!(
        (stats.invalid, stats.failed_streams, stats.breaker_opens),
        (0, 0, 1),
        "the miss is the plugin's failure: {stats:?}"
    );
}

/// A plugin that states no budget takes the default frame, under a request
/// that states none too: a stage is shown the widest budget there, the
/// frame names the default as the plugin's deadline, and a plugin that
/// answers nothing is waited for that long and no longer. The fallbacks
/// decide, and the missed frame is the plugin's own failure.
#[test]
fn a_plugin_with_no_budget_is_waited_for_the_default_frame_under_a_request_with_none() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let mut unset = spec(&served, "unbudgeted");
    unset.budget_micros = 0;
    let plugin = Plugin::load(&runtime, unset).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    // What a stage is shown for a request that states no budget.
    let mut widest = facts(1);
    widest.budget_micros = u32::MAX;
    let default = FrameBudget::DEFAULT_MICROS;
    let started = Instant::now();
    assert_eq!(
        lead(&decide_with(&engine, &widest, &pool(4))),
        Some(103),
        "the fallbacks decide"
    );
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_micros(u64::from(default)),
        "the first stage waited out the default frame: {waited:?}"
    );
    assert!(
        waited < Duration::from_micros(u64::from(BUDGET_MICROS)),
        "and nothing like a request's budget: {waited:?}"
    );
    assert!(eventually(|| served.frames() == 1));
    let seen = served.seen();
    let frame = &seen.frames[0];
    assert_eq!(frame.budget_micros, default, "the deadline the frame named");
    assert_eq!(
        frame.request.as_ref().map(|request| request.budget_micros),
        Some(u32::MAX),
        "the request's own budget, as a stage is shown it"
    );
    // A loaded host can hold the I/O thread under any one frame, and a held
    // miss is the host's: the next frame that leaves is judged.
    decide_until_judged(&engine, &plugin, &widest, &pool(4), 1);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.own_misses,
            stats.unread,
            stats.answered
        ),
        (1, 1, 0, 0),
        "the missed default frame is the plugin's own: {stats:?}"
    );
    assert_eq!(
        stats.budget_spent, stats.stalls,
        "no budget of the request's was spent: {stats:?}"
    );
}

#[test]
fn a_plugin_that_misses_every_deadline_opens_its_breaker_through_the_latency_sample() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "always-late");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let mut quick = facts(1);
    quick.budget_micros = 2_000;
    // One sample per frame that left, each the whole budget: the window of
    // 64 reads slow at its first evaluation.
    decide_until_judged(&engine, &plugin, &quick, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 1, 0),
        "{stats:?}"
    );
    // A budget no longer than the I/O thread's promise: no miss of it is
    // the plugin's failure, the ring alone judged these.
    assert_eq!(
        (stats.own_misses, stats.unread),
        (0, stats.missed - stats.stalls),
        "{stats:?}"
    );
    assert!(matches!(
        decide_with(&engine, &quick, &candidates),
        Decision::Pick(_)
    ));
    let after = plugin.stats();
    assert_eq!(
        (after.missed, after.bypassed),
        (stats.missed, 1),
        "inside the cool-down the plugin is not asked"
    );
}

/// The same plugin under a request that states no budget, with a budget of
/// its own that it misses every time: a stage is shown the widest budget
/// there, and the plugin's time is set against its own frames, not against
/// that budget. So the ring still judges it, and opens the breaker at its
/// first window.
#[test]
fn a_plugin_that_misses_every_deadline_of_its_own_is_judged_under_a_request_with_no_budget() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    let mut tight = spec(&served, "always-late-of-its-own");
    // A budget no longer than the I/O thread's promise: no miss of it is
    // the plugin's failure, so the ring alone can turn the plugin off.
    tight.budget_micros = 2_000;
    let plugin = Plugin::load(&runtime, tight).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    // What a stage is shown for a request that states no budget.
    let mut widest = facts(1);
    widest.budget_micros = u32::MAX;
    decide_until_judged(&engine, &plugin, &widest, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 1, 0),
        "{stats:?}"
    );
    assert_eq!(
        (stats.own_misses, stats.unread),
        (0, stats.missed - stats.stalls),
        "the ring alone judged these: {stats:?}"
    );
    assert!(matches!(
        decide_with(&engine, &widest, &candidates),
        Decision::Pick(_)
    ));
    let after = plugin.stats();
    assert_eq!(
        (after.missed, after.bypassed),
        (stats.missed, 1),
        "inside the cool-down the plugin is not asked"
    );
}

#[test]
fn a_plugin_that_answers_past_half_the_budget_opens_its_breaker_through_the_latency_sample() {
    let served = TestPlugin::serve();
    // Six tenths of the budget before each answer: a loaded host only adds
    // to it, and a frame that misses samples the whole budget.
    served.script(|s| s.delay = Duration::from_millis(6));
    let runtime = runtime();
    let mut spec = spec(&served, "slow-answers");
    // A loaded host turns some of these answers into misses of the plugin's
    // own, and five of those would open the breaker by the failure rule
    // first. That rule is set out of reach: the latency sample alone opens.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let mut tight = facts(1);
    tight.budget_micros = 10_000;
    // One sample per frame that left, each a round trip past half the
    // budget: the window of 64 reads slow at its first evaluation.
    decide_until_judged(&engine, &plugin, &tight, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        (
            stats.answered + stats.missed - stats.stalls,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 1, 0),
        "an answer in time is still judged by its round trip: {stats:?}"
    );
}

/// The I/O thread is the host's part of a round trip. Held from before a
/// frame leaves until past its deadline, an honest plugin misses every
/// frame and none of the misses is its own: each frame falls back, each is
/// counted as a stall, the ring takes no sample and the breaker stays shut.
/// Without the witness every miss samples its whole budget, and the window
/// reads slow at its first read.
#[test]
fn frames_missed_across_a_hold_of_the_io_thread_are_the_hosts_and_open_nothing() {
    const HOLD: Duration = Duration::from_millis(25);
    let served = TestPlugin::serve();
    let runtime = runtime();
    let plugin = load(&runtime, &served, "held-io");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let mut tight = facts(1);
    tight.budget_micros = 10_000;
    let mut rounds = 0_u64;
    let more = |rounds: u64| {
        let stats = plugin.stats();
        stats.stalls < 64 && stats.breaker_opens == 0 && rounds < 96
    };
    while more(rounds) {
        rounds += 1;
        runtime.hold_io(HOLD);
        assert!(matches!(
            decide_with(&engine, &tight, &candidates),
            Decision::Pick(_)
        ));
        // Released, the thread hands the frame over and the plugin answers:
        // late, or in time when this thread itself was held off the send.
        // Either way nothing queues behind the next hold.
        assert!(
            eventually(|| {
                let stats = plugin.stats();
                stats.late_answers + stats.answered == rounds
            }),
            "round {rounds}: {:?}",
            plugin.stats()
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.stalls, stats.breaker_opens, stats.bypassed),
        (64, 0, 0),
        "after {rounds} rounds: {stats:?}"
    );
    assert_eq!(
        stats.missed + stats.answered,
        rounds,
        "every frame left and was judged: {stats:?}"
    );
}

/// An answer that arrives in time across a stall of the I/O thread is read
/// and used, and its round trip is no sample: the time is the host's. Each
/// answer here waits out a hold of more than half the budget: without the
/// witness the window reads slow by its median at its first read.
#[test]
fn answers_that_cross_a_hold_of_the_io_thread_are_used_and_are_no_sample() {
    const HOLD: Duration = Duration::from_millis(20);
    let served = TestPlugin::serve();
    // Row 0, which no built-in stage would leave: the answer is seen used.
    served.behave(Behaviour::KeepFirst);
    let runtime = runtime();
    let plugin = load(&runtime, &served, "held-answers");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let mut snug = facts(1);
    snug.budget_micros = 36_000;
    let mut used = 0_u64;
    for round in 1..=64_u64 {
        runtime.hold_io(HOLD);
        let decision = decide_with(&engine, &snug, &candidates);
        // A fallback's pick is a frame a loaded host made miss after the
        // hold: a stall all the same.
        used += u64::from(lead(&decision) == Some(100));
        assert!(
            eventually(|| {
                let stats = plugin.stats();
                stats.answered + stats.late_answers == round
            }),
            "round {round}: {:?}",
            plugin.stats()
        );
    }
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.breaker_opens, stats.bypassed),
        (used, 0, 0),
        "every answer in time is used and none is a slow sample: {stats:?}"
    );
    assert_eq!(stats.answered + stats.missed, 64, "{stats:?}");
    assert!(
        used >= 48 && stats.stalls >= 60,
        "the held round trips are stalls, but for a frame this thread sent late: {stats:?}"
    );
}

/// A slow answer on an I/O thread that is seen running is the plugin's
/// time, never a stall: the thread ticks while the frame is in flight.
#[test]
fn a_slow_answer_on_a_thread_seen_running_is_the_plugins_time() {
    let served = TestPlugin::serve();
    // Three times the I/O thread's promise before each answer.
    served.script(|s| s.delay = Duration::from_millis(12));
    let runtime = runtime();
    let plugin = load(&runtime, &served, "slow-not-stalled");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    // A loaded host can hold the I/O thread under any one frame: one round
    // trip seen whole is the proof.
    let mut rounds = 0;
    loop {
        rounds += 1;
        assert!(matches!(decide(&engine, &candidates, 1), Decision::Pick(_)));
        let stats = plugin.stats();
        if stats.answered > stats.stalls {
            break;
        }
        assert!(rounds < 50, "every slow answer read as a stall: {stats:?}");
    }
}

/// A missed frame samples its grant, the budget it was given and used up,
/// never the wait the deciding thread measured: the grant is the whole
/// budget and not past it, so two misses in a window of fast answers do not
/// open by the tail. Sampled at the wait, each would read past the budget by
/// the thread's own lateness at its wake, and the second-slowest of the
/// window would be one of them.
#[test]
fn two_missed_frames_in_a_window_of_fast_answers_do_not_open_by_the_tail() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "two-misses");
    // The window's tail alone is read here. An honest answer a loaded host
    // delivers late is one more miss, and five misses in the window open
    // the breaker by the failure rule before the window is read.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let mut short = facts(1);
    short.budget_micros = SHORT_MICROS;
    served.behave(Behaviour::Silent);
    decide_until_judged(&engine, &plugin, &short, &candidates, 2);
    served.behave(Behaviour::Honest);
    // The window's first read is at its 64th sample.
    decide_until_judged(&engine, &plugin, &short, &candidates, 64);
    // Both made misses are the plugin's own and in its window. An answer
    // the I/O thread stalled under is a stall too, so the misses less the
    // stalls say nothing here.
    let stats = plugin.stats();
    assert_eq!(
        (stats.own_misses >= 2, stats.breaker_opens),
        (true, 0),
        "{stats:?}"
    );
}

/// A filter of the host's own that holds its decision's thread once, for
/// the milliseconds it is told: a plugin's frame sent at the open is in
/// flight meanwhile.
#[derive(Debug)]
struct HoldsTheThread(Arc<AtomicU64>);

impl Filter for HoldsTheThread {
    fn name(&self) -> &'static str {
        "holds-the-thread"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        let hold = self.0.swap(0, Ordering::Relaxed);
        if hold != 0 {
            std::thread::sleep(Duration::from_millis(hold));
        }
        true
    }
}

/// An answer that arrives after its frame's deadline, while the decision's
/// thread is elsewhere, is used when the thread comes to read it, and it is
/// sampled at its whole flight. A miss is a frame the thread stopped
/// waiting for; the lateness is the plugin's and the latency rule reads it.
/// So two such answers in a window of fast ones open the breaker by the
/// tail, where two missed frames, sampled at their grant, do not.
#[test]
fn late_answers_the_thread_finds_without_waiting_are_used_and_sampled_at_their_flight() {
    said();
    let served = TestPlugin::serve();
    // Row 0, which no built-in stage would leave: the answer is seen used.
    served.behave(Behaviour::KeepFirst);
    let runtime = runtime();
    let mut spec = spec(&served, "late-answers");
    // The latency rule alone opens here.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let hold = Arc::new(AtomicU64::new(0));
    // The host's filter runs ahead of the plugin's: the frame left at the
    // open, and the thread reads its answer after the hold.
    let mut held = profile(&plugin, ALL);
    held.filters
        .insert(0, Box::new(HoldsTheThread(Arc::clone(&hold))));
    let engine = single_engine(held);
    let candidates = pool(4);
    let mut short = facts(1);
    short.budget_micros = SHORT_MICROS;
    // The answer leaves three budgets after its frame, and the thread is
    // held ten.
    served.script(|s| s.delay = Duration::from_millis(60));
    let mut late = 0;
    for _ in 0..16 {
        if late == 2 {
            break;
        }
        let before = plugin.stats();
        hold.store(200, Ordering::Relaxed);
        let decision = decide_with(&engine, &short, &candidates);
        let after = plugin.stats();
        if after.stalls != before.stalls || after.missed != before.missed {
            made_again(
                "late answers",
                "the host held the answer past the thread's hold, or held its I/O thread",
            );
            continue;
        }
        assert_eq!(lead(&decision), Some(100), "the late answer is used");
        assert_eq!(after.answered - before.answered, 1, "{after:?}");
        late += 1;
    }
    assert_eq!(late, 2, "two late answers were read: {:?}", plugin.stats());
    served.script(|s| s.delay = Duration::ZERO);
    // The window's first read is at its 64th sample.
    decide_until_judged(&engine, &plugin, &short, &candidates, 64);
    let stats = plugin.stats();
    assert_eq!(
        stats.breaker_opens, 1,
        "two flights past the budget are the window's tail: {stats:?}"
    );
    let opened = openings("late-answers");
    assert!(
        opened.len() == 1 && opened[0].contains("rule=\"latency\""),
        "{opened:?}"
    );
}

/// A frame the decision's remainder clamped at its send is not under the
/// plugin's own bound: its miss is the decision's budget spent, whoever
/// spent it, and no failure.
#[test]
fn a_miss_under_the_remainders_clamp_is_the_decisions_budget_spent_and_no_failure() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let mute = TestPlugin::serve();
    mute.behave(Behaviour::Silent);
    let runtime = runtime();
    // The silent filter's own budget is a quarter of the decision's, and
    // its own misses never open it here.
    let mut quarter = spec(&silent, "silent-filter");
    quarter.budget_micros = SHORT_MICROS / 4;
    quarter.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let filter = Plugin::load(&runtime, quarter).expect("the plugin loads");
    // One failure would open the picker's breaker: a clamped miss is none.
    let mut clamped = spec(&mute, "clamped-picker");
    clamped.breaker = BreakerSpec {
        failures: 1,
        ..BreakerSpec::default()
    };
    let picker = Plugin::load(&runtime, clamped).expect("the plugin loads");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    let mut short = facts(1);
    short.budget_micros = SHORT_MICROS;
    // The picker is asked at the pick under the rest, three quarters of the
    // budget: a flight far longer than the I/O thread's promise.
    decide_until_judged(&engine, &picker, &short, &pool(4), 1);
    let stats = picker.stats();
    assert_eq!(
        (
            stats.missed - stats.stalls,
            stats.own_misses,
            stats.unread,
            stats.breaker_opens
        ),
        (1, 0, 0, 0),
        "{stats:?}"
    );
    assert_eq!(stats.budget_spent, stats.missed, "{stats:?}");
}

/// A profile whose filters are these plugins in order, each with its
/// built-in fallback: every frame leaves at the decision's open.
fn filters_of(plugins: &[&Arc<Plugin>]) -> Profile {
    let mut profile = Profile::of("filters".into(), Box::new(Best));
    profile.filters = plugins
        .iter()
        .map(|plugin| {
            let healthy: Option<Box<dyn Filter>> = Some(Box::new(Healthy));
            Box::new(GrpcFilter::new(plugin.bind(), healthy)) as Box<dyn Filter>
        })
        .collect();
    profile.scorers = vec![WeightedScorer::of(Box::new(LeastRequests), 1.0)];
    profile
}

/// A frame sent with its whole budget is still the remainder's where the
/// remainder ended its wait first. Three silent plugins in front draw the
/// whole budget while their frames fly together, so the last plugin's
/// wait ends at about a third of the frame it was sent with: its miss is
/// the decision's budget spent, no failure, and its sample is what it was
/// given, about a third of the budget, which no window reads slow. Judged
/// by the frame as sent it would fail every decision; sampled at the frame
/// as sent its window would read slow at its first read.
#[test]
fn a_wait_the_remainder_ended_first_is_the_decisions_spend_at_what_the_frame_was_given() {
    let runtime = runtime();
    let fronts: Vec<TestPlugin> = (0..3).map(|_| TestPlugin::serve()).collect();
    let front: Vec<Arc<Plugin>> = fronts
        .iter()
        .enumerate()
        .map(|(i, served)| {
            served.behave(Behaviour::Silent);
            // Seven of the decision's twenty milliseconds each, their own
            // misses never opening them here. Each sample of theirs is a
            // whole frame, so their windows read slow: under a cool-down of
            // zero that opening bypasses nothing.
            let mut spec = spec(served, &format!("front-{i}"));
            spec.budget_micros = 7_000;
            spec.breaker = BreakerSpec {
                failures: u32::MAX,
                cooldown_micros: 0,
                ..BreakerSpec::default()
            };
            Plugin::load(&runtime, spec).expect("the plugin loads")
        })
        .collect();
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let last = load(&runtime, &served, "starved");
    let engine = single_engine(filters_of(&[&front[0], &front[1], &front[2], &last]));
    let candidates = pool(4);
    let mut short = facts(1);
    short.budget_micros = SHORT_MICROS;
    // The window's first read is at its 64th sample. A round whose frame
    // the stream did not take (the I/O thread held, its window full) is no
    // sample and costs microseconds, so the loop is bounded on the clock:
    // a count of rounds runs out before a held thread is seen again.
    let started = Instant::now();
    let mut rounds = 0_u32;
    loop {
        let stats = last.stats();
        if stats.missed - stats.stalls >= 64 || stats.breaker_opens != 0 {
            break;
        }
        rounds += 1;
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the frames do not leave: {stats:?}"
        );
        assert!(matches!(
            decide_with(&engine, &short, &candidates),
            Decision::Pick(_)
        ));
    }
    let stats = last.stats();
    assert_eq!(
        (stats.unread, stats.breaker_opens, stats.bypassed),
        (0, 0, 0),
        "after {rounds} rounds: {stats:?}"
    );
    assert_eq!(
        stats.budget_spent + stats.own_misses,
        stats.missed,
        "{stats:?}"
    );
    assert!(
        stats.own_misses <= 2,
        "a wait the remainder ended is not the plugin's own: {stats:?}"
    );
}

/// A frame sent under what a decision had left is read against its plugin's
/// whole frame. A silent filter draws three quarters of the budget, so the
/// picker's frame leaves with the last quarter as its deadline and misses
/// it: the decision's budget spent, whoever spent it, and no failure. Its
/// sample is a quarter of the plugin's own frame, which no window reads
/// slow. Read against the frame as sent, every sample would be a whole
/// frame and the window would open at its first read.
#[test]
fn a_frame_sent_under_the_rest_is_read_against_its_plugins_whole_frame() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let mute = TestPlugin::serve();
    mute.behave(Behaviour::Silent);
    let runtime = runtime();
    // The silent filter's own budget is three quarters of the decision's.
    // Nothing bypasses it here: its misses never count to an opening, and
    // an opening by its own slow samples lasts no time.
    let mut front = spec(&silent, "silent-front");
    front.budget_micros = SHORT_MICROS / 4 * 3;
    front.breaker = BreakerSpec {
        failures: u32::MAX,
        cooldown_micros: 0,
        ..BreakerSpec::default()
    };
    let filter = Plugin::load(&runtime, front).expect("the plugin loads");
    // The picker's own frame is the whole budget. A round whose filter frame
    // did not leave asks it under that, and such a miss is its own: its
    // failures never open it here.
    let mut rest = spec(&mute, "under-the-rest");
    rest.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let picker = Plugin::load(&runtime, rest).expect("the plugin loads");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    let mut short = facts(1);
    short.budget_micros = SHORT_MICROS;
    // The window's first read is at its 64th sample.
    decide_until_judged(&engine, &picker, &short, &pool(4), 64);
    let stats = picker.stats();
    assert_eq!(
        (stats.unread, stats.breaker_opens, stats.bypassed),
        (0, 0, 0),
        "{stats:?}"
    );
    assert_eq!(
        stats.budget_spent + stats.own_misses,
        stats.missed,
        "{stats:?}"
    );
    assert!(
        stats.own_misses <= 2,
        "a frame the remainder clamped is not the plugin's own: {stats:?}"
    );
}

/// The second arm of the plugin's own bound: a frame the remainder clamped
/// is still the plugin's own where the plugin alone spent the budget. One
/// plugin is the filter and the picker of a decision and answers after six
/// tenths of the budget: the filter's frame is answered, the picker's frame
/// leaves under the four tenths left, a flight longer than the I/O thread's
/// promise, and misses. Nothing but the plugin drew the budget: the miss is
/// its failure.
#[test]
fn a_miss_under_a_remainder_the_plugin_alone_spent_is_its_own() {
    let served = TestPlugin::serve();
    served.script(|s| s.delay = Duration::from_millis(12));
    let runtime = runtime();
    let mut spec = spec(&served, "spends-alone");
    // Its failures never open it here: a loaded host takes more than one
    // decision to show one whole.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(filter_then_picker(&plugin, &plugin));
    let candidates = pool(4);
    // A loaded host makes another round of it, each told by a count of the
    // plugin's own: the first frame's answer past the budget (the plugin
    // answers at six tenths of it), the I/O thread stalled, or a remainder
    // too short for the witness. Such a round shows nothing of the rule: it
    // says so and is not one of the twenty.
    let (mut rounds, mut held) = (0_u32, 0_u32);
    loop {
        rounds += 1;
        let before = plugin.stats();
        assert!(matches!(
            decide_short(&engine, &candidates, 1),
            Decision::Pick(_)
        ));
        let after = plugin.stats();
        let seen = (
            after.answered - before.answered,
            after.own_misses - before.own_misses,
            after.budget_spent - before.budget_spent,
        );
        if seen == (1, 1, 0) {
            break;
        }
        if after.answered == before.answered
            || after.stalls != before.stalls
            || after.unread != before.unread
        {
            held += 1;
            made_again(
                "a miss under a remainder the plugin alone spent",
                &format!("the host held round {rounds}, which read {seen:?}"),
            );
        }
        assert!(
            rounds - held < 20 && rounds < 400,
            "no decision answered its first frame and missed its second as its own in {rounds} rounds, {held} of them held: {after:?}"
        );
        // The answer of the missed frame is late: nothing queues behind.
        std::thread::sleep(Duration::from_millis(15));
    }
}

/// A frame whose flight is no longer than the I/O thread's promise ends
/// before that thread can be seen late: a hold shorter than the promise
/// across its deadline leaves no gap for the witness. An honest plugin
/// under such a budget, the I/O thread held across each deadline for less
/// than the promise, misses every frame, and nothing says whose the miss
/// is: each is the decision's budget spent, counted unread, and no miss is
/// the plugin's failure. Judged as its own, the fifth would open the
/// breaker. The latency rule alone judges such frames: a window of them
/// used up whole reads slow at its first read, whoever was late.
#[test]
fn a_miss_too_short_for_the_witness_is_unread_and_the_decisions_budget_spent() {
    // Longer than the plugin's budget, shorter than the promise.
    const HOLD: Duration = Duration::from_millis(3);
    let served = TestPlugin::serve();
    // The plugin's usual time: inside its budget below, and over half of it.
    // So a round answered in time is a slow sample too, like a miss.
    served.script(|s| s.delay = Duration::from_micros(1_200));
    let runtime = runtime();
    // The plugin's own budget: a fifth of the decision's, half the promise.
    let mut spec = spec(&served, "short-frames");
    spec.budget_micros = 2_000;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    let mut tight = facts(1);
    tight.budget_micros = 10_000;
    // Up to the window's first read, at its 64th sample: an unread miss, or
    // a round answered in time. A round the I/O thread was seen late in is
    // charged nothing, answered or missed, so it is no sample.
    let mut rounds = 0_u64;
    let more = |rounds: u64| {
        let stats = plugin.stats();
        stats.answered + stats.missed - stats.stalls < 64
            && stats.breaker_opens == 0
            && rounds < 192
    };
    while more(rounds) {
        rounds += 1;
        runtime.hold_io(HOLD);
        assert!(matches!(
            decide_with(&engine, &tight, &candidates),
            Decision::Pick(_)
        ));
        // Released, the thread hands the frame over and the plugin answers:
        // late, or in time when this thread itself was held off the send.
        // Either way nothing queues behind the next hold.
        assert!(
            eventually(|| {
                let stats = plugin.stats();
                stats.late_answers + stats.answered == rounds
            }),
            "round {rounds}: {:?}",
            plugin.stats()
        );
    }
    let stats = plugin.stats();
    // The window's sixty-four samples: every one an unread miss, but a
    // round answered in time, where this thread was held off its send. Each
    // is past half of the plugin's frame, so the window's first read opens.
    assert_eq!(
        (
            stats.answered + stats.missed - stats.stalls,
            stats.own_misses,
            stats.breaker_opens,
            stats.bypassed
        ),
        (64, 0, 1, 0),
        "after {rounds} rounds: {stats:?}"
    );
    assert_eq!(
        (stats.missed + stats.answered, stats.budget_spent),
        (rounds, stats.missed),
        "every frame left, and every miss is the decision's budget spent: {stats:?}"
    );
}

#[test]
fn an_answered_wait_draws_the_time_the_thread_blocked() {
    let slow = TestPlugin::serve();
    // Far inside the answered cases' budget: the answer always arrives.
    slow.script(|s| s.delay = Duration::from_millis(200));
    let quick = TestPlugin::serve();
    let runtime = runtime();
    let filter = load(&runtime, &slow, "slow-filter");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    // The scripted picker, asked alone, takes the most loaded row.
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(100));
    assert_eq!((filter.stats().answered, picker.stats().answered), (1, 1));
    // The picker's frame left after the filter's wait and names what the
    // decision had left. Half the 200 ms the thread blocked must be gone
    // from it; the other half is what a stalled thread may lose between
    // the send and the wait.
    let named = quick.seen().frames[0].budget_micros;
    assert!(
        named <= BUDGET_MICROS - 100_000,
        "the wait drew nothing like its 200 ms: {named} of {BUDGET_MICROS} left"
    );
}

/// A stream that goes while its decision waits ends the wait, and the time
/// the thread blocked is drawn like an answered wait's: a plugin asked
/// later in the decision is asked under the rest.
#[test]
fn a_wait_its_stream_broke_draws_the_time_the_thread_blocked() {
    let leaving = TestPlugin::serve();
    // The stream ends 200 ms after the frame, far inside the budget.
    leaving.script(|s| {
        s.behaviour = Behaviour::EndStream;
        s.delay = Duration::from_millis(200);
    });
    let quick = TestPlugin::serve();
    let runtime = runtime();
    let filter = load(&runtime, &leaving, "leaving-filter");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    // The filter falls back; the scripted picker, asked alone, takes the
    // most loaded row.
    assert_eq!(lead(&decide(&engine, &pool(4), 1)), Some(100));
    assert!(
        eventually(|| filter.stats().failed_streams == 1),
        "the stream went under the wait"
    );
    assert_eq!((filter.stats().missed, picker.stats().answered), (0, 1));
    // As for an answered wait: half the 200 ms must be gone from what the
    // picker's frame names.
    let named = quick.seen().frames[0].budget_micros;
    assert!(
        named <= BUDGET_MICROS - 100_000,
        "the broken wait drew nothing like its 200 ms: {named} of {BUDGET_MICROS} left"
    );
}

#[test]
fn a_missed_frame_draws_its_grant_and_the_next_plugin_is_asked_under_the_rest() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let quick = TestPlugin::serve();
    let runtime = runtime();
    // The silent filter's own budget is a quarter of the decision's.
    let mut quarter = spec(&silent, "silent-filter");
    quarter.budget_micros = SHORT_MICROS / 4;
    let filter = Plugin::load(&runtime, quarter).expect("the plugin loads");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    assert!(matches!(
        decide_short(&engine, &pool(4), 1),
        Decision::Pick(_)
    ));
    let stats = filter.stats();
    assert_eq!(
        (stats.missed, stats.own_misses + stats.budget_spent),
        (1, 1),
        "its own miss, or the host's on a stalled thread: {stats:?}"
    );
    // The miss drew the frame's grant, not a clock: the picker's frame
    // names exactly the rest, whatever the host's load.
    assert!(eventually(|| quick.frames() == 1));
    assert_eq!(
        quick.seen().frames[0].budget_micros,
        SHORT_MICROS - SHORT_MICROS / 4
    );
    // The silent filter's frame named two budgets: its own quarter as the
    // frame's deadline, the decision's remainder as the request's. Its
    // server records the frame on a thread of its own: the read waits.
    assert!(eventually(|| silent.frames() == 1));
    let sent = silent.seen();
    assert_eq!(
        (
            sent.frames[0].budget_micros,
            sent.frames[0].request.as_ref().map(|r| r.budget_micros)
        ),
        (SHORT_MICROS / 4, Some(SHORT_MICROS))
    );
}

#[test]
fn a_pick_only_ask_after_the_budget_is_spent_sends_no_frame() {
    let silent = TestPlugin::serve();
    silent.behave(Behaviour::Silent);
    let quick = TestPlugin::serve();
    let runtime = runtime();
    // The silent filter's grant is the whole budget: its miss leaves none.
    let filter = load(&runtime, &silent, "silent-filter");
    let picker = load(&runtime, &quick, "picker");
    let engine = single_engine(filter_then_picker(&filter, &picker));
    assert_eq!(
        lead(&decide_short(&engine, &pool(4), 1)),
        Some(103),
        "the fallbacks decide"
    );
    assert_eq!(filter.stats().missed, 1);
    let stats = picker.stats();
    assert_eq!(
        (stats.budget_spent, stats.missed, stats.answered),
        (1, 0, 0),
        "the picker is not asked under a spent budget"
    );
    assert_eq!(stats.breaker_opens, 0, "and it is not its fault");
    assert_eq!(quick.frames(), 0);
}

#[test]
fn an_answer_after_its_deadline_is_late() {
    let served = TestPlugin::serve();
    // Far past the short budget: the wait cannot end with the answer.
    served.script(|s| s.delay = Duration::from_millis(1_500));
    let runtime = runtime();
    let plugin = load(&runtime, &served, "slow");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(lead(&decide_short(&engine, &pool(4), 1)), Some(103));
    assert_eq!(plugin.stats().missed, 1);
    assert!(
        eventually(|| plugin.stats().late_answers == 1),
        "the answer arrived for a decision nobody waits for"
    );
    assert_eq!(plugin.stats().answered, 0);
}

#[test]
fn a_frame_its_decision_never_read_is_forgotten_at_the_close() {
    let served = TestPlugin::serve();
    // The answer leaves the plugin long after the decision closed.
    served.script(|s| s.delay = Duration::from_millis(1_000));
    let runtime = runtime();
    let plugin = load(&runtime, &served, "unread");
    // The scorer is bound, so the frame leaves at the open; the built-in
    // filter then empties the pool and no stage of the plugin's runs.
    let engine = single_engine(profile(&plugin, SCORE | PICK));
    let mut candidates = pool(3);
    for c in &mut candidates {
        c.health.healthy = false;
    }
    assert!(matches!(decide(&engine, &candidates, 1), Decision::Shed(_)));
    assert!(
        eventually(|| plugin.stats().late_answers == 1),
        "the answer found no frame waiting"
    );
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.missed), (0, 0));
    // The place it held is free: the next decision is answered.
    served.script(|s| s.delay = Duration::ZERO);
    assert_eq!(lead(&decide(&engine, &pool(4), 2)), Some(103));
    assert_eq!(plugin.stats().answered, 1);
}

#[test]
fn a_frame_past_the_window_is_refused_without_a_fault() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "nine");
    // One failure would open the breaker: a refused frame is none.
    spec.breaker = BreakerSpec {
        failures: 1,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    // Nine bound sets of one build score in one profile: each opens its own
    // frame at the decision's open, on the deciding thread's one stream,
    // before any of them is read.
    let binder = plugin.binder();
    let scorers = (0..9)
        .map(|_| {
            let fallback: Option<Box<dyn Scorer>> = Some(Box::new(LeastRequests));
            WeightedScorer::of(
                Box::new(GrpcScorer::new(binder.bind(), fallback)) as Box<dyn Scorer>,
                1.0,
            )
        })
        .collect();
    let mut profile = Profile::of("nine".into(), Box::new(Best));
    profile.filters = vec![Box::new(Healthy)];
    profile.scorers = scorers;
    let engine = single_engine(profile);
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(103),
        "eight answers and one fallback agree"
    );
    let stats = plugin.stats();
    assert_eq!(
        (stats.answered, stats.refused),
        (8, 1),
        "the ninth frame found the window full"
    );
    assert_eq!(
        (stats.breaker_opens, stats.bypassed, stats.missed),
        (0, 0, 0),
        "a full window is not the plugin's fault"
    );
    assert_eq!(served.frames(), 8, "the refused frame never left");
    // Every place is free again at the close: the next decision's first
    // eight frames are answered.
    assert_eq!(lead(&decide(&engine, &pool(4), 2)), Some(103));
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.refused), (16, 2));
}

#[test]
fn a_gone_budget_sends_no_frame_and_counts_on_the_breaker() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "budgetless");
    spec.breaker = BreakerSpec {
        failures: 2,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let mut gone = facts(1);
    gone.budget_micros = 0;
    let candidates = pool(4);
    assert!(matches!(
        decide_with(&engine, &gone, &candidates),
        Decision::Pick(_)
    ));
    let stats = plugin.stats();
    assert_eq!((stats.budget_gone, stats.breaker_opens), (1, 0));
    decide_with(&engine, &gone, &candidates);
    let stats = plugin.stats();
    assert_eq!(
        (stats.budget_gone, stats.breaker_opens),
        (2, 1),
        "a budget hit each, to the threshold"
    );
    assert_eq!(lead(&decide(&engine, &candidates, 3)), Some(103));
    assert_eq!(plugin.stats().bypassed, 1);
    assert_eq!(served.frames(), 0, "no frame left the host");
}

#[test]
fn a_pool_past_max_candidates_sends_no_frame() {
    let served = TestPlugin::serve();
    let runtime = runtime();
    let mut spec = spec(&served, "small-view");
    spec.max_candidates = 3;
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    assert_eq!(
        lead(&decide(&engine, &pool(4), 1)),
        Some(103),
        "whole view or none: the fallbacks decide"
    );
    assert_eq!(plugin.stats().too_many, 1);
    assert_eq!(served.frames(), 0);
    assert_eq!(lead(&decide(&engine, &pool(3), 2)), Some(102));
    assert_eq!((plugin.stats().too_many, plugin.stats().answered), (1, 1));
}

/// The plugin's own open marks its phase open and counts the decision
/// before it reads the runtime's clock, and the engine does not close a
/// scope whose open did not return. So an open that stops half-way undoes
/// itself: here the runtime's clock panics inside it, and the thread's next
/// decision is a decision of its own that asks the plugin. Left half-open,
/// the phase would read as open to the next decision, and it would not ask
/// the plugin.
#[test]
fn a_plugins_open_that_stops_half_way_undoes_itself() {
    let served = TestPlugin::serve();
    let clock_panics = Arc::new(AtomicBool::new(false));
    let told = Arc::clone(&clock_panics);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        assert!(
            !told.swap(false, Ordering::Relaxed),
            "the runtime's clock panics, once"
        );
        1_000_000
    })
    .expect("a runtime");
    let plugin = load(&runtime, &served, "half-open");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(3);
    // A whole decision first: the stream is settled, and what stops the next
    // open is the clock alone.
    assert_eq!(lead(&decide(&engine, &candidates, 1)), Some(102));
    let before = plugin.stats();
    clock_panics.store(true, Ordering::Relaxed);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decide(&engine, &candidates, 2)
    }));
    assert!(caught.is_err(), "the plugin's open panicked at the clock");
    assert_eq!(
        plugin.stats().answered,
        before.answered,
        "the open stopped before its frame"
    );
    let next = decide(&engine, &candidates, 3);
    let asked = plugin.stats();
    assert_eq!(
        asked.answered - before.answered,
        1,
        "the next decision asked the plugin: {asked:?}"
    );
    assert_eq!(lead(&next), Some(102), "the plugin's pick leads");
}

/// What the case below shares between its picker and the runtime's clock.
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

impl Picker for TellsThePick {
    fn name(&self) -> &'static str {
        "tells-the-pick"
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
        if self.0.armed.load(Ordering::Relaxed) {
            self.0.picked.store(true, Ordering::Relaxed);
        }
        Best.pick(facts, candidates, kept, scores, band, out);
    }
}

/// The plugin's own close takes its decision's latency sample, and a window
/// that reads slow asks the runtime's clock: the host's code, inside the
/// close. So a close that stops half-way leaves its scope closed: here the
/// clock panics inside it, and the thread's next decision is a decision of
/// its own, asked under its own budget. Left open, the phase would read as
/// open to the next decision, and it would not ask the plugin. With its
/// place in the thread's decision not given back, the next decision would
/// run on what the stopped one had left.
#[test]
fn a_plugins_close_that_stops_half_way_leaves_its_scope_closed() {
    let served = TestPlugin::serve();
    let shared = Arc::new(PastThePick::default());
    let told = Arc::clone(&shared);
    let deciding = std::thread::current().id();
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || {
        // Past an armed pick the deciding thread reads the clock in a close
        // alone: that read panics, once.
        assert!(
            std::thread::current().id() != deciding || !told.picked.swap(false, Ordering::Relaxed),
            "the runtime's clock panics inside a close"
        );
        1_000_000
    })
    .expect("a runtime");
    let mut spec = spec(&served, "half-closed");
    // Every answer below is slow or a miss. Five misses of the plugin's own
    // would open the breaker by the failure rule first, so that rule is set
    // out of reach: the window's read is what asks the clock.
    spec.breaker = BreakerSpec {
        failures: u32::MAX,
        ..BreakerSpec::default()
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    // The plugin is bound at one stage, so a decision closes its scope
    // once. Bound at two, the engine closes the same bound set twice, and
    // the second close forgets what the first left open.
    let mut bound = profile(&plugin, FILTER);
    bound.picker = Box::new(TellsThePick(Arc::clone(&shared)));
    let engine = single_engine(bound);
    // A whole decision first: the stream is settled, and its lead is what
    // the plugin makes of this pool.
    let want = lead(&decide(&engine, &pool(3), 1));
    // Six tenths of the budget before each answer, and a miss samples the
    // whole budget: the window reads slow at its first read, which is at
    // the close of the decision that fills it.
    served.script(|s| s.delay = Duration::from_millis(6));
    let mut tight = facts(2);
    tight.budget_micros = 10_000;
    let candidates = pool(4);
    shared.armed.store(true, Ordering::Relaxed);
    let mut stopped = false;
    for _ in 0..1_024 {
        shared.picked.store(false, Ordering::Relaxed);
        let decided = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decide_with(&engine, &tight, &candidates)
        }));
        if decided.is_err() {
            stopped = true;
            break;
        }
    }
    shared.armed.store(false, Ordering::Relaxed);
    shared.picked.store(false, Ordering::Relaxed);
    assert!(
        stopped,
        "a close read the clock and stopped there: {:?}",
        plugin.stats()
    );
    // The next decision's answer takes twice the stopped decision's whole
    // budget: it is read only under a budget of the decision's own.
    served.script(|s| s.delay = Duration::from_millis(20));
    let before = plugin.stats();
    let next = decide(&engine, &pool(3), 1);
    let asked = plugin.stats();
    assert_eq!(
        asked.answered - before.answered,
        1,
        "the next decision asked the plugin under its own budget: {asked:?}"
    );
    assert_eq!(lead(&next), want, "and picks from its own view");
}

#[test]
fn the_breaker_opens_on_a_run_of_faults_and_closes_after_the_cool_down() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::ScoreNan);
    let clock = Arc::new(AtomicU64::new(1_000));
    let reads = Arc::clone(&clock);
    let runtime = Runtime::with_clock(&OpenDecisions::new(), move || reads.load(Ordering::Relaxed))
        .expect("a runtime");
    let mut spec = spec(&served, "breaking");
    spec.breaker = BreakerSpec {
        failures: 2,
        window_micros: 1_000_000,
        cooldown_micros: 5_000_000,
    };
    let plugin = Plugin::load(&runtime, spec).expect("the plugin loads");
    let engine = single_engine(profile(&plugin, ALL));
    let candidates = pool(4);
    decide(&engine, &candidates, 1);
    decide(&engine, &candidates, 2);
    let stats = plugin.stats();
    assert_eq!(
        (stats.invalid, stats.breaker_opens, stats.bypassed),
        (2, 1, 0)
    );
    assert_eq!(lead(&decide(&engine, &candidates, 3)), Some(103));
    assert_eq!(plugin.stats().bypassed, 1);
    assert!(
        eventually(|| served.frames() == 2),
        "the bypassed decision sent nothing"
    );

    // Past the cool-down on the runtime's clock the plugin is asked again.
    clock.store(6_100_000, Ordering::Relaxed);
    served.behave(Behaviour::Honest);
    assert_eq!(lead(&decide(&engine, &candidates, 4)), Some(103));
    let stats = plugin.stats();
    assert_eq!((stats.answered, stats.bypassed), (1, 1));
}

#[test]
fn without_a_fallback_the_failure_policy_decides() {
    let served = TestPlugin::serve();
    served.behave(Behaviour::Silent);
    let runtime = runtime();
    // Open, the default: every stage steps aside.
    let plugin = load(&runtime, &served, "lenient");
    let engine = single_engine(profile_of(&plugin, ALL, false));
    assert!(matches!(
        decide_short(&engine, &pool(4), 1),
        Decision::Pick(_)
    ));

    let mut closed = spec(&served, "strict");
    closed.failure = Failure::Closed;
    let plugin = Plugin::load(&runtime, closed).expect("the plugin loads");
    let engine = single_engine(profile_of(&plugin, ALL, false));
    match decide_short(&engine, &pool(4), 2) {
        Decision::Shed(shed) => {
            assert_eq!(shed.emptied_by, "strict");
            assert_eq!(shed.reason, ShedReason::NoEndpoints);
        }
        other => panic!("a shed naming the plugin, not {other:?}"),
    }
}
