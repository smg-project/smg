use std::cell::Cell;

use wasmtime::{Trap, UpdateDeadline};

use super::{
    deadline_ticks, first_deadline, import_version, judge, thread_cpu_micros, tick, End, Latency,
    ThreadRing, Verdict, Watch, EPOCH_TICK_MICROS,
};

/// Whether a tick ended the call; a tick that lets it run asks for the
/// next one.
fn ended(tick: wasmtime::Result<UpdateDeadline>) -> bool {
    match tick {
        Ok(UpdateDeadline::Continue(1)) => false,
        Ok(_) => panic!("a tick that lets a call run asks for the next tick"),
        Err(error) => {
            assert_eq!(error.downcast_ref::<Trap>(), Some(&Trap::Interrupt));
            true
        }
    }
}

/// Keeps the calling thread on the CPU for `micros` of its own clock.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn burn(micros: u64) {
    let from = thread_cpu_micros().expect("a thread clock");
    while thread_cpu_micros().expect("a thread clock") < from + micros {
        std::hint::spin_loop();
    }
}

/// A tick ends a call on the CPU its thread had since a reading, never
/// on the wall clock: a thread that slept past the bound (the host held
/// it) runs on, and the same thread once it has burned the bound is
/// ended. With no reading of the start the first tick takes one and ends
/// nothing; with one, no stand-in is taken.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_tick_ends_a_call_on_the_cpu_its_thread_had_since_a_reading() {
    const BOUND: u64 = 2_000;
    let held = std::time::Duration::from_micros(3 * BOUND);

    let mut unread = Watch {
        bound_micros: BOUND,
        watched: true,
        start: None,
        first_tick: None,
    };
    assert!(!ended(tick(&mut unread)), "a first tick ends nothing");
    assert!(unread.first_tick.is_some(), "it took the reading");
    std::thread::sleep(held);
    assert!(
        !ended(tick(&mut unread)),
        "three bounds on the wall clock, off the CPU: not ended"
    );
    burn(BOUND);
    assert!(
        ended(tick(&mut unread)),
        "the bound had on the CPU since the first tick"
    );

    let mut read = Watch {
        bound_micros: BOUND,
        watched: true,
        start: thread_cpu_micros(),
        first_tick: None,
    };
    std::thread::sleep(held);
    assert!(
        !ended(tick(&mut read)),
        "held since its start: short of the bound on the CPU"
    );
    assert!(
        read.first_tick.is_none(),
        "a start reading needs no stand-in"
    );
    burn(BOUND);
    assert!(
        ended(tick(&mut read)),
        "the bound had on the CPU since the start"
    );
}

/// A call nothing watches ends at its deadline on the wall clock: an
/// `init`, and every call on a host with no thread clock, which takes
/// the same arm.
#[test]
fn a_tick_ends_a_call_nothing_watches() {
    assert!(ended(tick(&mut Watch::default())), "an init's watch");
    let mut unwatched = Watch {
        bound_micros: u64::MAX,
        watched: false,
        start: None,
        first_tick: None,
    };
    assert!(ended(tick(&mut unwatched)), "whatever its bound");
}

/// A watched call with no reading of its start meets its next tick,
/// which takes one. A call whose start was read meets the tick its
/// deadline on the wall clock falls on and none before it, like a call
/// nothing watches: a reading never ends a call the wall clock would let
/// run.
#[test]
fn a_start_reading_meets_no_tick_before_the_wall_clocks_deadline() {
    assert_eq!(
        (
            deadline_ticks(1),
            deadline_ticks(1_000),
            deadline_ticks(1_001)
        ),
        (2, 2, 3),
        "the bound in whole ticks and the tick in flight"
    );
    for bound in [1, 20, 1_000, 1_001, 10_000] {
        assert_eq!(first_deadline(true, false, bound), 1, "{bound}: unread");
        assert_eq!(
            first_deadline(true, true, bound),
            deadline_ticks(bound),
            "{bound}: its start read"
        );
        assert_eq!(
            first_deadline(false, false, bound),
            deadline_ticks(bound),
            "{bound}: nothing watched"
        );
    }
}

/// The judgement of a call at its end is a function of its readings and
/// of how it ended: the wall time, the reading at its start or none, the
/// reading at its end, its fuel and the bound. Every branch, each at its
/// boundary, and the calls that read the clock again: every call whose
/// start was read, but one that is not late and whose fuel covers the
/// time it took. With no reading a call is sampled at its charge and
/// spends its fuel (its charge on a host with no thread clock); a
/// reading gives it one time, late or not.
#[test]
fn a_late_call_is_judged_once_on_its_readings() {
    // A bound that is no whole number of ticks, and the wall time that
    // makes its call late: the bound in whole ticks.
    const BOUND: u64 = 1_500;
    const LATE: u64 = 2 * EPOCH_TICK_MICROS;
    const START: u64 = 100;
    let watch = |start: Option<u64>| Watch {
        bound_micros: BOUND,
        watched: true,
        start,
        first_tick: None,
    };
    let read = watch(Some(START));
    let unread = watch(None);
    let ticked = Watch {
        first_tick: Some(START),
        ..unread
    };
    let unwatched = Watch {
        watched: false,
        ..read
    };
    // The case; the watch; the CPU clock at the end; the wall time; the
    // fuel in microseconds; how the call ended; the own time (its
    // sample, its spend), the verdict and whether the clock was read
    // again.
    let rows = [
            (
                "past its bound, a microsecond short of late, read: no stall, and no time but its fuel",
                read,
                START,
                LATE - 1,
                7,
                End::Answered,
                (7, 7, Verdict::Own, 1),
            ),
            (
                "not late, read, the CPU at the wall time: its charge",
                read,
                START + 900,
                900,
                7,
                End::Answered,
                (900, 900, Verdict::Own, 1),
            ),
            (
                "not late, read, the CPU short of the wall time: a hold is no time of the call",
                read,
                START + 40,
                900,
                7,
                End::Answered,
                (40, 40, Verdict::Own, 1),
            ),
            (
                "not late, read, the fuel above the CPU: its fuel",
                read,
                START + 40,
                900,
                600,
                End::Answered,
                (600, 600, Verdict::Own, 1),
            ),
            (
                "not late, read, the CPU above the wall time: never more than the charge",
                read,
                START + 5 * BOUND,
                900,
                7,
                End::Failed,
                (900, 900, Verdict::Own, 1),
            ),
            (
                "not late, read, the fuel at the wall time: its fuel, and the clock is not read again",
                read,
                START + 40,
                600,
                600,
                End::Answered,
                (600, 600, Verdict::Own, 0),
            ),
            (
                "not late, read, the fuel above the wall time: its fuel, and no reading",
                read,
                START + 40,
                10,
                900,
                End::Failed,
                (900, 900, Verdict::Own, 0),
            ),
            (
                "not late, read, the wall time a microsecond past the fuel: read again",
                read,
                START + 40,
                601,
                600,
                End::Answered,
                (600, 600, Verdict::Own, 1),
            ),
            (
                "late, read, the fuel above the wall time: read again, for a reading judges lateness",
                read,
                START + 40,
                LATE,
                LATE + 5,
                End::Answered,
                (LATE + 5, LATE + 5, Verdict::Held, 1),
            ),
            (
                "late at its bound in whole ticks, the CPU one short of the bound: held",
                read,
                START + BOUND - 1,
                LATE,
                7,
                End::Answered,
                (BOUND - 1, BOUND - 1, Verdict::Held, 1),
            ),
            (
                "a bound of whole ticks: late at the bound itself",
                Watch {
                    bound_micros: 10 * EPOCH_TICK_MICROS,
                    ..unread
                },
                START,
                10 * EPOCH_TICK_MICROS,
                7,
                End::Answered,
                (10 * EPOCH_TICK_MICROS, 7, Verdict::Unread, 0),
            ),
            (
                "a bound under a tick, twice had on the wall clock: not late",
                Watch {
                    bound_micros: 20,
                    ..unread
                },
                START,
                45,
                7,
                End::Answered,
                (45, 7, Verdict::Own, 0),
            ),
            (
                "late, the CPU at the bound: the plugin's, at its CPU",
                read,
                START + BOUND,
                LATE,
                7,
                End::Answered,
                (BOUND, BOUND, Verdict::Own, 1),
            ),
            (
                "held, the fuel above the CPU: its own time is its fuel",
                read,
                START + 40,
                9 * BOUND,
                600,
                End::Answered,
                (600, 600, Verdict::Held, 1),
            ),
            (
                "held on the CPU, failed by itself: the plugin's, at its CPU",
                read,
                START + BOUND - 1,
                LATE,
                7,
                End::Failed,
                (BOUND - 1, BOUND - 1, Verdict::Own, 1),
            ),
            (
                "held on the CPU, ended on the clock: the plugin's",
                read,
                START + BOUND - 1,
                LATE,
                7,
                End::OnTheClock,
                (BOUND - 1, BOUND - 1, Verdict::Own, 1),
            ),
            (
                "the CPU above the wall time: never more than the charge",
                read,
                START + 5 * BOUND,
                LATE,
                7,
                End::Answered,
                (LATE, LATE, Verdict::Own, 1),
            ),
            (
                "late, read, ended on the clock: the plugin's, never the hold",
                read,
                START + BOUND + 300,
                9 * BOUND,
                7,
                End::OnTheClock,
                (BOUND + 300, BOUND + 300, Verdict::Own, 1),
            ),
            (
                "late, no reading of its start, not ended: unread, sampled at its charge, spent at its fuel",
                unread,
                START,
                9 * BOUND,
                7,
                End::Answered,
                (9 * BOUND, 7, Verdict::Unread, 0),
            ),
            (
                "late, no reading of its start, failed by itself: unread all the same",
                unread,
                START,
                9 * BOUND,
                7,
                End::Failed,
                (9 * BOUND, 7, Verdict::Unread, 0),
            ),
            (
                "late, no reading of its start, ended on the clock: the plugin's",
                unread,
                START,
                9 * BOUND,
                7,
                End::OnTheClock,
                (9 * BOUND, 7, Verdict::Own, 0),
            ),
            (
                "a first tick's reading is no reading of the start",
                ticked,
                START + 40,
                9 * BOUND,
                7,
                End::Answered,
                (9 * BOUND, 7, Verdict::Unread, 0),
            ),
            (
                "nothing watched: the wall clock's call, its charge twice, and no reading",
                unwatched,
                START,
                9 * BOUND,
                7,
                End::Answered,
                (9 * BOUND, 9 * BOUND, Verdict::Own, 0),
            ),
            (
                "not late, no reading, the fuel above the wall time: the charge is the fuel",
                unread,
                START,
                10,
                900,
                End::Answered,
                (900, 900, Verdict::Own, 0),
            ),
        ];
    for (case, watch, cpu_at_end, elapsed, fuel, end, wanted) in rows {
        let reads = Cell::new(0);
        let (own, verdict) = judge(&watch, elapsed, fuel, end, || {
            reads.set(reads.get() + 1);
            Some(cpu_at_end)
        });
        assert_eq!(
            (own.sample, own.spend, verdict, reads.get()),
            wanted,
            "{case}"
        );
    }
}

/// The engine the host asks for keeps no state for concurrent calls and
/// admits no component that needs it, in every build. The runtime's
/// switch defaults to on wherever its async component support is
/// compiled in, which any crate of a build can ask for: there this case
/// is red without the switch's line. A build without that support passes
/// with or without the line.
#[test]
fn the_runtime_keeps_no_state_for_concurrent_calls() {
    // An engine as the runtime builds its own: a config, then the
    // host's settings.
    let mut config = wasmtime::Config::new();
    super::configure(&mut config);
    let engine = wasmtime::Engine::new(&config).expect("an engine");
    // The least component that needs the support: one export lifted the
    // async way.
    let needs_it = wat::parse_str(
        r#"(component
                (core module $m
                    (func (export "run") (result i32) i32.const 0)
                    (func (export "cb") (param i32 i32 i32) (result i32) i32.const 0))
                (core instance $i (instantiate $m))
                (type $run (func async))
                (func (export "run") (type $run)
                    (canon lift (core func $i "run") async (callback (core func $i "cb")))))"#,
    )
    .expect("the component assembles");
    assert!(
        wasmtime::component::Component::new(&engine, needs_it).is_err(),
        "a component with an async export compiled"
    );
    assert!(
        !engine.get_concurrency_support(),
        "the runtime keeps state for concurrent calls"
    );
}

#[test]
fn an_interface_name_yields_its_version_and_nothing_else_matches() {
    assert_eq!(
        import_version("smg:policy/plugin@1.0.0", "smg:policy/plugin"),
        Some("1.0.0")
    );
    assert_eq!(
        import_version("smg:policy/plugin-packed@1.0.0", "smg:policy/plugin"),
        None,
        "the packed world is not the record world"
    );
    assert_eq!(
        import_version("wasi:io/poll@0.2.0", "smg:policy/host"),
        None
    );
    assert_eq!(import_version("smg:policy/host", "smg:policy/host"), None);
}

/// A sample is uncovered where it is past a mark of the ring that its
/// low end is short of: each mark at its edge, in the ring's own unit.
#[test]
fn a_sample_past_a_mark_its_low_end_is_short_of_is_uncovered() {
    // A sample, its low end and the frame, and whether no reading
    // covers the sample.
    let table = [
        // At half the frame the high end is past no mark.
        (500, 0, 1_000, false),
        // Past half the frame by the high end alone.
        (501, 0, 1_000, true),
        (501, 500, 1_000, true),
        // Past half the frame by both ends, and short of the frame by
        // both.
        (501, 501, 1_000, false),
        (1_000, 501, 1_000, false),
        // Past the frame by the high end alone, whatever half says.
        (1_001, 0, 1_000, true),
        (1_001, 501, 1_000, true),
        (1_001, 1_000, 1_000, true),
        // Past the frame by both ends.
        (1_001, 1_001, 1_000, false),
        (9_000, 1_001, 1_000, false),
        // The share is the ring's: a frame that is no thousand.
        (700, 5, 400, true),
        (200, 5, 400, false),
        (201, 5, 400, true),
        (2, 1, 3, true),
        // No frame, no sample.
        (700, 5, 0, false),
    ];
    for (micros, low_micros, frame_micros, uncovered) in table {
        assert_eq!(
            ThreadRing::uncovered(micros, low_micros, frame_micros),
            uncovered,
            "{micros} over {low_micros} under a frame of {frame_micros}"
        );
    }
}

/// A parted sample, past the half mark at its high end alone, arms a
/// ring nothing armed and is counted unread. The same sample of a
/// decision in which the bound ended a call as the plugin's own arms the
/// ring all the same and is not counted: at its bound a call is the
/// plugin's.
#[test]
fn a_parted_sample_is_unread_unless_the_bound_ended_a_call_of_its_decision() {
    // Six tenths of the frame by the time the calls took, a tenth by
    // their fuel.
    let (micros, low_micros, frame_micros) = (600, 100, 1_000);
    assert!(ThreadRing::uncovered(micros, low_micros, frame_micros));
    let mut ring = ThreadRing::new();
    assert_eq!(
        ring.take(micros, low_micros, frame_micros, false),
        (true, false),
        "no call was ended: slow by time nothing read alone"
    );
    assert!(ring.armed(), "and the ring is armed");
    assert_eq!(
        ring.take(micros, low_micros, frame_micros, false),
        (false, false),
        "an armed ring counts no second one"
    );
    let mut ring = ThreadRing::new();
    assert_eq!(
        ring.take(micros, low_micros, frame_micros, true),
        (false, false),
        "the bound ended a call: the sample is the plugin's, never unread"
    );
    assert!(ring.armed(), "and the ring is armed all the same");
    // Recorded like any other: sixty-four of them are a slow window.
    let last = (1..Latency::EVERY)
        .map(|_| ring.take(micros, low_micros, frame_micros, true))
        .last();
    assert_eq!(last, Some((false, true)));
}

/// An armed ring stays armed for its own length in samples and no
/// longer: through the sample of the decision that armed it and the 127
/// after it, so the next sample a late call can leave unread overwrites
/// the first one's. A decision that records nothing does not count.
#[test]
fn an_armed_ring_stays_armed_for_its_length_in_samples() {
    let mut ring = ThreadRing::new();
    assert!(!ring.armed(), "a new ring is not armed");
    ring.arm();
    for sample in 0..Latency::SAMPLES - 1 {
        assert!(ring.armed(), "armed before sample {sample}");
        assert!(!ring.record(10, 0), "no frame: not a sample");
        assert!(
            ring.armed(),
            "a decision that records nothing does not count"
        );
        ring.record(10, 100);
    }
    assert!(
        ring.armed(),
        "the call 127 samples after the first is still read"
    );
    ring.record(10, 100);
    assert!(!ring.armed(), "the arm fell with the ring's length");
}

/// One slow sample a window is never its p99. A ring armed by each late
/// call no reading covers takes such a sample once a length at most, so
/// those samples alone never read slow.
#[test]
fn one_unread_sample_a_ring_length_never_reads_slow() {
    let mut ring = ThreadRing::new();
    let mut tripped = false;
    for round in 0..4 {
        assert!(!ring.armed(), "round {round}: the last arm has fallen");
        // The unread call's decision: its sample is the time the call
        // took, five hundred times its budget. It arms the ring.
        ring.arm();
        tripped |= ring.record(50_000, 100);
        // Armed, a late call is read: its sample is the plugin's own.
        for _ in 1..Latency::SAMPLES {
            tripped |= ring.record(10, 100);
        }
    }
    assert!(!tripped, "one slow sample in every window of the ring");
}
