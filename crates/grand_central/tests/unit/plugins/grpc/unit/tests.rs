use std::time::{Duration, Instant};

use super::Unit;
use crate::plugins::shared::{breaker::Latency, config::BreakerSpec};

const BUDGET: u64 = 1_000;

fn unit() -> Unit {
    Unit::new("unit", BreakerSpec::default())
}

/// The charges of a decision are one sample, taken at the plugin's last
/// close of it: on a pair neither side is past half the budget, the
/// decision is, and a window of them reads slow at its first read.
#[test]
fn the_last_close_of_a_decision_takes_its_charges_as_one_sample() {
    let unit = unit();
    for decision in 1..=Latency::EVERY as u64 {
        assert!(!unit.enter(decision));
        assert!(!unit.enter(decision), "the other side of the pair");
        unit.charge(decision, 300, BUDGET);
        unit.charge(decision, 300, BUDGET);
        unit.leave(decision, BUDGET, || 5);
        assert_eq!(unit.opens(), 0, "one open is still to close");
        unit.leave(decision, BUDGET, || 5);
    }
    assert_eq!(unit.opens(), 1, "the window's first read");
}

/// Under a budget wider than the frames of a decision's round trips, the
/// sample is taken over those frames: a request in process may state no
/// budget, and its decision is then seeded with the widest. A plugin that
/// used six tenths of its one frame reads slow there, where its charge
/// over that budget read as nothing.
#[test]
fn a_charge_is_read_against_its_frame_under_a_wider_budget() {
    let unit = unit();
    let widest = u64::from(u32::MAX);
    for decision in 1..=Latency::EVERY as u64 {
        assert!(!unit.enter(decision));
        unit.charge(decision, 600, BUDGET);
        unit.leave(decision, widest, || 5);
    }
    assert_eq!(unit.opens(), 1, "the window's first read");
}

/// The frames of a decision add up: three round trips that each used
/// four tenths of their frame are four tenths of the decision's frames,
/// not twelve tenths of one.
#[test]
fn three_calls_under_half_their_frames_are_no_slow_sample() {
    let unit = unit();
    let widest = u64::from(u32::MAX);
    for decision in 1..=Latency::EVERY as u64 {
        assert!(!unit.enter(decision));
        for _ in 0..3 {
            unit.charge(decision, 400, BUDGET);
        }
        unit.leave(decision, widest, || 5);
    }
    assert_eq!(unit.opens(), 0);
}

/// A decision's charges start over with the decision: one slow decision
/// in front of a window of fast ones is one slow sample.
#[test]
fn the_charges_of_a_decision_do_not_carry_into_the_next() {
    let unit = unit();
    for decision in 1..=Latency::EVERY as u64 {
        assert!(!unit.enter(decision));
        unit.charge(decision, if decision == 1 { 900 } else { 100 }, BUDGET);
        unit.leave(decision, BUDGET, || 5);
    }
    assert_eq!(unit.opens(), 0);
}

/// A charge is the decision it names: one that names another decision
/// (a frame read after its decision closed) is nobody's sample.
#[test]
fn a_charge_that_names_another_decision_is_dropped() {
    let unit = unit();
    for decision in 1..=Latency::EVERY as u64 {
        assert!(!unit.enter(decision));
        unit.charge(decision + 1_000, 900, BUDGET);
        unit.leave(decision, BUDGET, || 5);
    }
    assert_eq!(unit.opens(), 0);
}

/// A decision that charged the plugin nothing is no sample: it does not
/// thin a slow window out.
#[test]
fn a_decision_that_charged_nothing_is_no_sample() {
    let unit = unit();
    let idle = Latency::EVERY as u64 / 2;
    for decision in 1..=idle {
        assert!(!unit.enter(decision));
        unit.leave(decision, BUDGET, || 5);
    }
    for decision in idle + 1..=idle + Latency::EVERY as u64 {
        assert!(!unit.enter(decision));
        unit.charge(decision, 600, BUDGET);
        unit.leave(decision, BUDGET, || 5);
    }
    assert_eq!(unit.opens(), 1, "the 64 charged decisions are the window");
}

/// What a decision charged so far is the plugin's own spend in it, read
/// by the decision's number: another decision's reads nothing, and the
/// next decision starts over.
#[test]
fn what_a_decision_charged_so_far_is_read_by_its_number() {
    let unit = unit();
    assert!(!unit.enter(1));
    assert_eq!(unit.charged(1), 0);
    unit.charge(1, 300, BUDGET);
    unit.charge(1, 200, BUDGET);
    assert_eq!((unit.charged(1), unit.charged(2)), (500, 0));
    unit.leave(1, BUDGET, || 5);
    assert!(!unit.enter(2));
    assert_eq!(unit.charged(2), 0, "the next decision starts over");
    unit.leave(2, BUDGET, || 5);
}

/// A failure marks its decision for the plugin's later opens of it and
/// for nothing after; a failure with no decision to mark counts alone.
#[test]
fn a_failure_marks_its_decision_and_a_report_marks_none() {
    let unit = unit();
    assert!(!unit.enter(1));
    unit.fail(5);
    assert!(unit.failed_in(1));
    assert!(unit.enter(1), "the other side finds the decision failed");
    unit.leave(1, BUDGET, || 5);
    unit.leave(1, BUDGET, || 5);
    assert!(!unit.enter(2), "the next decision starts clean");
    unit.report(5);
    assert!(!unit.failed_in(2), "a stream's failure marks no decision");
    unit.leave(2, BUDGET, || 5);
    // Five failures in the window: the default threshold.
    for _ in 0..3 {
        assert!(!unit.is_open(5));
        unit.report(5);
    }
    assert!(unit.is_open(5));
    assert_eq!(unit.opens(), 1);
}

/// The breaker hears a plugin's streams as one lane: at most one failure
/// a pause. Two threads' streams end together, and each is asked for
/// again a pause after its own failure: the first at 100, 200 and 300
/// ms, the second, whose decisions come later, at 151 and 251. Under a
/// threshold of four the count reads one thread's: closed through 251,
/// open at 300. Counted by half a pause, the second thread's reopens
/// count too and the breaker opens at 251.
#[test]
fn the_streams_of_one_plugin_count_as_one_lane_at_any_phases() {
    let unit = Unit::new(
        "unit",
        BreakerSpec {
            failures: 4,
            window_micros: 60_000_000,
            cooldown_micros: 60_000_000,
        },
    );
    let start = Instant::now();
    for millis in [0, 1, 100, 151, 200, 251] {
        unit.stream_failed(start + Duration::from_millis(millis), || 5);
    }
    assert_eq!(unit.opens(), 0, "three pauses, three failures");
    unit.stream_failed(start + Duration::from_millis(300), || 5);
    assert_eq!(unit.opens(), 1, "the fourth pause's failure opens");
}
