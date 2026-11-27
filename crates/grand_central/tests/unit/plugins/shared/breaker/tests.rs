use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Breaker, Latency, Recorded};
use crate::plugins::shared::config::BreakerSpec;

fn spec() -> BreakerSpec {
    BreakerSpec {
        failures: 3,
        window_micros: 1_000,
        cooldown_micros: 5_000,
    }
}

/// Whether the call the ring answers for completed a slow window.
fn slow(recorded: Recorded) -> bool {
    recorded == Recorded::SlowWindow
}

#[test]
fn the_threshold_of_failures_within_a_window_opens_for_the_cooldown() {
    let b = Breaker::new(spec());
    b.failure(10);
    b.failure(20);
    assert!(!b.is_open(30));
    b.failure(30);
    assert!(b.is_open(31));
    assert!(b.is_open(5_029));
    assert!(!b.is_open(5_030), "the cool-down passed: closed again");
    assert_eq!(b.opens(), 1);
}

#[test]
fn failures_in_separate_windows_do_not_add_up() {
    let b = Breaker::new(spec());
    b.failure(0);
    b.failure(500);
    b.failure(2_000);
    assert!(!b.is_open(2_001), "the third failure opened a new window");
    b.failure(2_100);
    b.failure(2_200);
    assert!(b.is_open(2_201));
}

#[test]
fn an_older_failure_outside_the_current_window_does_not_open() {
    let b = Breaker::new(BreakerSpec {
        failures: 2,
        window_micros: 1_000,
        cooldown_micros: 5_000,
    });
    assert!(!b.failure(2_000));
    assert!(!b.failure(0));
    assert_eq!(b.opens(), 0);
    assert!(!b.is_open(2_001));
}

#[test]
fn an_older_failure_inside_the_window_counts_from_its_own_instant() {
    let b = Breaker::new(spec());
    assert!(!b.failure(1_000));
    assert!(!b.failure(500), "the older failure fits the window");
    assert!(b.failure(1_400), "three failures within 900 us open");

    let b = Breaker::new(spec());
    assert!(!b.failure(1_000));
    assert!(!b.failure(500));
    assert!(
        !b.failure(1_900),
        "a failure outside the window starts a new window"
    );
    assert_eq!(b.opens(), 0);
}

#[test]
fn slowness_opens_at_once_and_a_failure_after_the_cooldown_reopens() {
    let b = Breaker::new(spec());
    b.slow(100);
    assert!(b.is_open(101));
    assert!(!b.is_open(6_000));
    b.failure(6_001);
    b.failure(6_002);
    b.failure(6_003);
    assert!(b.is_open(6_004));
    assert_eq!(b.opens(), 2);
}

/// A failure that finds none counted starts a window of its own, the
/// first after an open like the first ever: a burst is not split at the
/// end of a window that began before it.
#[test]
fn a_failure_that_finds_none_counted_starts_its_own_window() {
    // A cool-down shorter than the window.
    let spec = BreakerSpec {
        failures: 3,
        window_micros: 1_000,
        cooldown_micros: 100,
    };
    let b = Breaker::new(spec);
    // Past the clock's first window: this burst's window starts at 5,000.
    b.failure(5_000);
    b.failure(5_100);
    b.failure(5_200);
    assert!(b.is_open(5_201));
    assert!(!b.is_open(5_300), "the cool-down passed: closed again");
    // Three failures in 800 us, across the end of that window (6,000) and
    // of one counted from the open (6,200).
    b.failure(5_500);
    b.failure(5_900);
    assert!(!b.is_open(5_901));
    b.failure(6_300);
    assert!(
        b.is_open(6_301),
        "a burst after the open is counted in a window of its own"
    );
    assert_eq!(b.opens(), 2);

    // The first failure ever, inside the clock's first window: its window
    // starts at the failure, not at the clock's origin.
    let fresh = Breaker::new(spec);
    fresh.failure(600);
    fresh.failure(900);
    fresh.failure(1_100);
    assert!(
        fresh.is_open(1_101),
        "the first burst is counted in a window of its own"
    );
}

/// A failure reported while the breaker is open, by a call admitted
/// before it opened, counts in no window and opens nothing again.
#[test]
fn a_failure_reported_while_open_neither_counts_nor_opens() {
    // A cool-down shorter than the window.
    let b = Breaker::new(BreakerSpec {
        failures: 3,
        window_micros: 1_000,
        cooldown_micros: 100,
    });
    b.failure(0);
    b.failure(0);
    b.failure(0);
    assert!(b.is_open(1));
    // Three calls already out at the open fail during the cool-down.
    b.failure(10);
    b.failure(20);
    b.failure(30);
    assert_eq!(b.opens(), 1, "a report while open opens nothing again");
    assert!(!b.is_open(100), "the cool-down was not restarted");
    // The count starts afresh after the cool-down: one failure is the
    // first of three, not the third.
    b.failure(110);
    assert!(
        !b.is_open(111),
        "the reports made while open were not counted"
    );
    b.failure(120);
    b.failure(130);
    assert!(b.is_open(131));
    assert_eq!(b.opens(), 2);
}

/// A slow window reported while the breaker is open, completed by
/// decisions in flight at the open, opens nothing again.
#[test]
fn a_slow_window_reported_while_open_does_not_open_again() {
    let b = Breaker::new(spec());
    b.slow(100);
    b.slow(2_000);
    assert_eq!(b.opens(), 1);
    assert!(!b.is_open(5_100), "the cool-down was not restarted");
}

/// A report stamped inside a cool-down is dropped whenever it arrives:
/// after the cool-down's end too, and after another thread read the
/// breaker closed there. The opening's end stays where it was, so the
/// guard still finds it.
#[test]
fn a_report_stamped_inside_the_cooldown_is_dropped_after_a_closed_read() {
    let b = Breaker::new(BreakerSpec {
        failures: 1,
        window_micros: 1_000,
        cooldown_micros: 100,
    });
    assert!(b.failure(100), "the threshold of one opens until 200");
    // Another thread reads the breaker closed at the cool-down's end;
    // then a call that was in flight at the opening reports, with the
    // instant it read before that end.
    assert!(!b.is_open(200));
    assert!(!b.failure(199), "stamped inside the cool-down: dropped");
    assert!(!b.slow(199), "and a slow window stamped there");
    assert_eq!(b.opens(), 1, "nothing opened again");
    assert!(!b.is_open(200), "and the breaker reads closed");
    // A report stamped at the end is counted.
    assert!(b.failure(200));
    assert_eq!(b.opens(), 2);
}

#[test]
fn a_report_says_whether_it_opened_the_breaker() {
    let b = Breaker::new(spec());
    assert!(!b.failure(10));
    assert!(!b.failure(20));
    assert!(b.failure(30), "the third failure opened it");
    assert!(!b.failure(40), "a report while open opened nothing");
    assert!(!b.slow(50), "nor did a slow window");
    assert!(b.slow(5_030), "after the cool-down a slow window opens");
    // The same edge for a failure: the cool-down's last instant is
    // inside it, its end is not.
    let one = Breaker::new(BreakerSpec {
        failures: 1,
        window_micros: 1_000,
        cooldown_micros: 100,
    });
    assert!(one.failure(0));
    assert!(!one.failure(99), "the cool-down's last instant");
    assert!(one.failure(100), "the cool-down has passed");
}

/// Spins, then yields, until `ready`.
fn wait(ready: impl Fn() -> bool) {
    let mut spins = 0_u32;
    while !ready() {
        spins += 1;
        if spins < 128 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

/// `racers` threads fail at the same instant past the window of a
/// breaker that failed once, under a threshold of `failures`, round
/// after round: the breakers as the rounds left them.
fn race(racers: usize, failures: u32) -> Vec<Breaker> {
    const ROUNDS: usize = 4_000;
    let spec = BreakerSpec {
        failures,
        window_micros: 10,
        cooldown_micros: 1_000,
    };
    let breakers: Vec<Breaker> = (0..ROUNDS)
        .map(|_| {
            let breaker = Breaker::new(spec);
            // One failure: its window, ten microseconds from zero, has
            // run out by the instant of the race.
            breaker.failure(0);
            breaker
        })
        .collect();
    // The rounds started so far, and the failures reported in all.
    let started = AtomicUsize::new(0);
    let reported = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..racers {
            scope.spawn(|| {
                for (round, breaker) in breakers.iter().enumerate() {
                    wait(|| started.load(Ordering::Acquire) > round);
                    // Past the window the breaker's one failure started.
                    breaker.failure(1_000);
                    reported.fetch_add(1, Ordering::Release);
                }
            });
        }
        for round in 0..ROUNDS {
            started.store(round + 1, Ordering::Release);
            wait(|| reported.load(Ordering::Acquire) == racers * (round + 1));
        }
    });
    breakers
}

/// Two failures that race a window's rollover both count: under a
/// threshold of two the breaker opens in every round.
#[test]
fn two_failures_racing_a_window_rollover_both_count() {
    let breakers = race(2, 2);
    let closed = breakers.iter().filter(|b| !b.is_open(1_001)).count();
    assert_eq!(
        closed,
        0,
        "{closed} of {} rollovers lost a failure",
        breakers.len()
    );
}

/// Three failures at once under a threshold of two cross it once: the
/// breaker opens, and one open is counted.
#[test]
fn failures_racing_past_the_threshold_open_once() {
    let breakers = race(3, 2);
    let wrong = breakers.iter().filter(|b| b.opens() != 1).count();
    assert_eq!(
        wrong,
        0,
        "{wrong} of {} rounds did not open once",
        breakers.len()
    );
}

#[test]
fn the_latency_window_reads_slow_on_p50_or_p99_past_the_frame() {
    let mut fast = Latency::new();
    let mut tripped = false;
    for _ in 0..Latency::SAMPLES {
        tripped |= slow(fast.record(10, 100));
    }
    assert!(!tripped, "10 us calls under a 100 us frame are not slow");

    let mut half_slow = Latency::new();
    let mut tripped = false;
    for i in 0..Latency::SAMPLES {
        // One more than half the calls past half the frame: the median
        // lands on them.
        let micros = if i % 2 == 0 || i == 1 { 60 } else { 10 };
        tripped |= slow(half_slow.record(micros, 100));
    }
    assert!(tripped, "a p50 past half the frame is slow");

    let mut one_outlier = Latency::new();
    let mut tripped = false;
    for i in 0..Latency::SAMPLES {
        let micros = if i == 5 { 500 } else { 10 };
        tripped |= slow(one_outlier.record(micros, 100));
    }
    assert!(
        !tripped,
        "one call past the frame is not the p99 of the ring"
    );

    let mut at_the_frame = Latency::new();
    let mut tripped = false;
    for i in 0..Latency::SAMPLES {
        let micros = if i == 5 || i == 9 { 100 } else { 10 };
        tripped |= slow(at_the_frame.record(micros, 100));
    }
    assert!(
        !tripped,
        "two calls at the frame, past its half, are not a slow p99"
    );

    let mut tail = Latency::new();
    let mut tripped = false;
    for i in 0..Latency::SAMPLES {
        let micros = if i == 5 || i == 9 { 500 } else { 10 };
        tripped |= slow(tail.record(micros, 100));
    }
    assert!(tripped, "two calls past the frame are the p99");
    let mut tripped_again = false;
    for _ in 0..Latency::SAMPLES {
        tripped_again |= slow(tail.record(10, 100));
    }
    assert!(!tripped_again, "the window that read slow is forgotten");

    let mut mixed = Latency::new();
    let mut tripped = false;
    for i in 0..Latency::SAMPLES {
        // Sixty microseconds is slow under a 100 us frame and nothing
        // under a 1,000 us one: each call against its own frame.
        let frame = if i % 2 == 0 { 1_000 } else { 100 };
        tripped |= slow(mixed.record(if frame == 1_000 { 60 } else { 10 }, frame));
    }
    assert!(!tripped, "each sample is judged against its own frame");

    let mut unbudgeted = Latency::new();
    assert!((0..Latency::SAMPLES).all(|_| unbudgeted.record(1_000, 0) == Recorded::Nothing));
}

/// The ring says what it did with a call: nothing where the call has no
/// frame, a sample where it took one, and a slow window at the call that
/// completed one. The slow window is forgotten: the next call is a
/// sample of another.
#[test]
fn a_record_says_what_the_ring_did_with_the_call() {
    let mut ring = Latency::new();
    assert_eq!(ring.record(900, 0), Recorded::Nothing);
    // Nine times its frame, each call: sixty-three samples of a window
    // still open, and the sixty-fourth completes it slow.
    for sample in 1..Latency::EVERY {
        assert_eq!(ring.record(900, 100), Recorded::Sample, "{sample}");
    }
    assert_eq!(ring.record(900, 100), Recorded::SlowWindow);
    assert_eq!(ring.record(10, 100), Recorded::Sample);
}

/// A mark stands between two charges where one is past it and the other
/// is not, whichever of the two is named first: half the frame and the
/// frame, each at its edge.
#[cfg(feature = "wasm")]
#[test]
fn a_mark_stands_between_two_charges_on_its_two_sides() {
    // Two charges, their frame, and whether a mark stands between them.
    let table = [
        (500, 0, 1_000, false),
        (501, 500, 1_000, true),
        (1_000, 501, 1_000, false),
        (1_001, 1_000, 1_000, true),
        (9_000, 1_001, 1_000, false),
        (1_001, 0, 1_000, true),
        // No frame, no sample, no mark.
        (700, 5, 0, false),
    ];
    for (micros, other_micros, frame_micros, between) in table {
        for (one, other) in [(micros, other_micros), (other_micros, micros)] {
            assert_eq!(
                Latency::mark_between(one, other, frame_micros),
                between,
                "{one} and {other} under a frame of {frame_micros}"
            );
        }
    }
}

/// A sample's frame is the decision's budget or the frames of the
/// plugin's calls in the decision, whichever is less. A request that
/// states no budget seeds its decision with the widest one: the frames
/// are the lesser there.
#[test]
fn a_samples_frame_is_the_lesser_of_the_budget_and_the_calls_frames() {
    assert_eq!(Latency::frame(1_000, 3_000), 1_000, "the budget");
    assert_eq!(Latency::frame(10_000, 3_000), 3_000, "the frames");
    let widest = u64::from(u32::MAX);
    assert_eq!(Latency::frame(widest, 300), 300, "no budget stated");
    assert_eq!(
        Latency::frame(u64::MAX, u64::MAX),
        u32::MAX,
        "past the ring's range both: the widest frame"
    );
}
