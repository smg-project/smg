use super::{reaches_the_budget, Budget, OpenDecisions, Part};

/// What is left of a budget, and the budget as seeded.
fn left_and_seed(budget: &Budget) -> (u64, u64) {
    (budget.remaining_micros, budget.seed_micros)
}

/// A budget seeded with `seed` that has `left` of it.
fn budget(seed: u32, left: u64) -> Budget {
    let mut budget = Budget::default();
    budget.seed(seed);
    budget.draw(u64::from(seed) - left);
    budget
}

#[test]
fn the_first_open_numbers_and_seeds_and_later_opens_draw_the_same_remainder() {
    let open = OpenDecisions::new();
    let mut decision = open.on_this_thread();
    let first = decision.open(900);
    assert_ne!(first, 0, "a decision's number is never zero");
    assert_eq!(left_and_seed(decision.budget()), (900, 900));
    decision.budget().draw(400);
    // The other side of a pair, or another plugin: the same decision,
    // whatever budget its facts carry, and the remainder as drawn.
    assert_eq!(decision.open(700), first);
    assert_eq!(left_and_seed(decision.budget()), (500, 900));
    decision.close();
    assert_eq!(decision.open(700), first, "an open is still to close");
    decision.close();
    decision.close();
    let second = decision.open(300);
    assert_ne!(second, first, "the last close ended the decision");
    assert_eq!(left_and_seed(decision.budget()), (300, 300));
}

/// The remainder only falls, and never below zero: a draw past it
/// leaves nothing, and the seed stands.
#[test]
fn a_draw_past_the_remainder_leaves_nothing() {
    let open = OpenDecisions::new();
    let mut decision = open.on_this_thread();
    decision.open(900);
    let budget = decision.budget();
    assert!(!budget.is_spent());
    budget.draw(899);
    assert_eq!(left_and_seed(budget), (1, 900));
    assert_eq!(budget.remaining_micros_u32(), 1);
    budget.draw(5);
    assert!(
        budget.is_spent(),
        "a draw past the remainder leaves nothing"
    );
    budget.draw(u64::MAX);
    assert_eq!(left_and_seed(budget), (0, 900));
}

/// The widest budget a request states is read whole in the width a
/// frame carries: nothing is cut off.
#[test]
fn the_widest_budget_is_read_whole_in_a_frames_width() {
    let open = OpenDecisions::new();
    let mut decision = open.on_this_thread();
    decision.open(u32::MAX);
    let budget = decision.budget();
    assert_eq!(
        left_and_seed(budget),
        (u64::from(u32::MAX), u64::from(u32::MAX))
    );
    assert_eq!(budget.remaining_micros_u32(), u32::MAX);
}

/// A budget no decision seeded holds its own bound and a seed of zero,
/// so every bound under it is its plugin's own; its remainder as fuel
/// saturates.
#[cfg(feature = "wasm")]
#[test]
fn a_budget_alone_holds_its_bound_and_a_seed_of_zero() {
    let mut alone = Budget::alone(250);
    assert_eq!(left_and_seed(&alone), (250, 0));
    assert!(alone.own_bound(false, 0, 0));
    assert_eq!(alone.grant_call(u64::MAX, u32::MAX, 4), (1_000, 250));
    assert_eq!(
        alone.grant_call(u64::MAX, u32::MAX, u64::MAX).0,
        u64::MAX,
        "the fuel saturates"
    );
    alone.draw(250);
    assert!(alone.is_spent());
    assert_eq!(alone.grant_call(u64::MAX, u32::MAX, 4), (0, 0));
}

#[test]
fn each_thread_holds_a_decision_of_its_own() {
    let open = OpenDecisions::new();
    let here = open.on_this_thread().open(900);
    open.on_this_thread().budget().draw(899);
    let there = std::thread::scope(|scope| {
        let other = scope.spawn(|| {
            let mut decision = open.on_this_thread();
            let number = decision.open(500);
            (number, decision.budget().remaining_micros)
        });
        other.join()
    });
    assert!(
        matches!(there, Ok((_, 500))),
        "another thread's open seeds its own remainder: {there:?}"
    );
    assert_eq!(
        open.on_this_thread().open(700),
        here,
        "this thread's decision is the one it opened"
    );
    assert_eq!(open.on_this_thread().budget().remaining_micros, 1);
}

#[test]
fn a_part_marks_the_decision_its_plugin_failed_until_the_next_one_opens_it() {
    let mut part = Part::default();
    assert!(!part.open(7), "a fresh decision was not failed");
    part.fail();
    assert!(part.failed_in(7));
    assert!(part.open(7), "the other side finds the decision failed");
    assert!(!part.close(7), "one open is still to close");
    assert!(part.close(7), "the last close");
    assert!(part.failed_in(7), "read at the close's cleanup too");
    assert!(!part.open(8), "the next decision starts the part over");
    assert!(part.is_of(8) && !part.is_of(7));
    assert!(!part.failed_in(7));
    part.fail();
    assert!(
        part.failed_in(8) && !part.failed_in(7),
        "the mark is the open decision's"
    );
    assert!(part.close(8));
}

#[test]
fn a_failure_outside_a_decision_marks_nothing() {
    let mut part = Part::default();
    assert!(!part.open(3));
    assert!(part.close(3));
    part.fail();
    assert!(!part.failed_in(3), "a lifecycle failure after the close");
    assert!(
        !part.close(4),
        "a close of a decision the part does not hold"
    );
}

#[test]
fn equal_numbers_are_one_decision_to_a_part() {
    // Why one issuer hands out every number a plugin sees: a part cannot
    // tell two decisions with one number apart.
    let mut part = Part::default();
    assert!(!part.open(1));
    part.fail();
    assert!(part.close(1));
    assert!(part.is_of(1));
    assert!(
        part.open(1),
        "the same number reads as the decision already failed"
    );
}

/// The run-out's rule is exact: a plugin's own time and the grant in
/// full reach the budget or they do not, to the microsecond.
#[test]
fn a_spend_and_a_grant_reach_the_budget_to_the_microsecond() {
    // A plugin's time before the call, the grant, the budget as seeded,
    // and whether the plugin alone consumed it.
    let table = [
        (400, 599, 1_000, false),
        (400, 600, 1_000, true),
        (400, 601, 1_000, true),
        (0, 999, 1_000, false),
        (0, 1_000, 1_000, true),
        (999, 0, 1_000, false),
        (1_000, 0, 1_000, true),
        // The sum saturates, and reaches.
        (u64::MAX, 1, u64::MAX, true),
        (u64::MAX - 1, 0, u64::MAX, false),
    ];
    for (spent, granted, seed, reached) in table {
        assert_eq!(
            reaches_the_budget(spent, granted, seed),
            reached,
            "{spent} and {granted} under {seed}"
        );
    }
}

/// A bound is the plugin's own by either arm: the call had its whole
/// frame, however small beside the budget, or the plugin's own spend
/// and the grant reach the budget. Short of both it is the remainder's.
#[test]
fn a_bound_is_the_plugins_own_by_its_whole_frame_or_by_its_own_spend() {
    // The whole frame, the spend before, the grant, the budget as
    // seeded, and whether the bound was the plugin's own.
    let table = [
        (true, 0, 1, 1_000, true),
        (true, 0, 0, 1_000, true),
        (false, 0, 999, 1_000, false),
        (false, 400, 599, 1_000, false),
        (false, 400, 600, 1_000, true),
        (false, 0, 1_000, 1_000, true),
    ];
    for (whole, spent, granted, seed, own) in table {
        // The answer reads the budget as seeded, whatever is left of it.
        for left in [0, u64::from(seed)] {
            assert_eq!(
                budget(seed, left).own_bound(whole, spent, granted),
                own,
                "whole {whole}, {spent} and {granted} under {seed}, {left} left"
            );
        }
    }
}

/// A call of the WebAssembly kind may spend its cap of fuel and its frame
/// on the clock, or what is left where that is less: each of the two by
/// itself, to the microsecond, and nothing of a spent budget.
#[cfg(feature = "wasm")]
#[test]
fn a_call_may_spend_its_frames_or_what_is_left_where_that_is_less() {
    // What is left of a budget of 1,000, the cap of fuel, the frame on
    // the clock, the fuel a microsecond, and the grant's two parts.
    let table = [
        (1_000, 4_000, 400, 10, (4_000, 400)),
        (400, 4_000, 400, 10, (4_000, 400)),
        (399, 4_000, 400, 10, (3_990, 399)),
        // The two frames need not agree: each is held by itself.
        (500, 9_000, 100, 10, (5_000, 100)),
        (500, 1_000, 900, 10, (1_000, 500)),
        (1, 4_000, 400, 10, (10, 1)),
        (0, 4_000, 400, 10, (0, 0)),
        (1_000, 0, 0, 10, (0, 0)),
        // The fuel saturates.
        (1_000, u64::MAX, u32::MAX, u64::MAX, (u64::MAX, 1_000)),
    ];
    for (left, cap, frame, per_micro, grant) in table {
        assert_eq!(
            budget(1_000, left).grant_call(cap, frame, per_micro),
            grant,
            "a cap of {cap} at {per_micro} and a frame of {frame} under {left}"
        );
    }
    // The widest budget grants the widest frame whole.
    let widest = budget(u32::MAX, u64::from(u32::MAX));
    assert_eq!(widest.grant_call(7, u32::MAX, 1), (7, u32::MAX));
}

/// A decision's sample is read over the frames its plugin was given, at
/// most the budget as seeded, whatever is left of it.
#[test]
fn a_sample_is_read_over_the_frames_at_most_the_budget_as_seeded() {
    for left in [0, 1_000] {
        let budget = budget(1_000, left);
        assert_eq!(budget.sample_frame(300), 300, "the frames");
        assert_eq!(budget.sample_frame(1_000), 1_000);
        assert_eq!(budget.sample_frame(3_000), 1_000, "the budget");
    }
}

/// A frame of the gRPC kind is sent with the plugin's own frame, or what
/// is left where that is less, and with what is left.
#[cfg(feature = "controller")]
#[test]
fn a_frame_is_sent_with_its_grant_and_what_is_left() {
    use super::FrameBudget;
    let stated = |micros: u32| FrameBudget {
        micros,
        fuel_per_micro: 10,
    };
    // What is left of a budget of 1,000, the plugin's stated frame, and
    // the two a frame is sent with (its grant, what is left).
    let table = [
        (1_000, 400, (400, 1_000)),
        (400, 400, (400, 400)),
        (399, 400, (399, 399)),
        (0, 400, (0, 0)),
        // A frame wider than the budget is granted the budget at most.
        (1_000, 5_000, (1_000, 1_000)),
        (300, 5_000, (300, 300)),
    ];
    for (left, frame, sent) in table {
        let under = budget(1_000, left);
        assert_eq!(
            under.grant_frame(stated(frame)),
            sent,
            "a frame of {frame} under {left}"
        );
    }
    // A plugin that states no frame takes the default one.
    let default = FrameBudget::DEFAULT_MICROS;
    let widest = budget(u32::MAX, u64::from(default) + 5);
    assert_eq!(widest.grant_frame(stated(0)), (default, default + 5));
}

/// A wait ends at its frame's deadline, or where what is left runs out
/// if that is sooner. A frame with no answer was given the frame as
/// sent, or the time ahead of the wait and what is left where that is
/// less.
#[cfg(feature = "controller")]
#[test]
fn a_wait_ends_at_its_deadline_or_with_what_is_left_and_a_miss_was_given_no_more() {
    use std::time::{Duration, Instant};
    let started = Instant::now();
    let after = |micros: u64| started + Duration::from_micros(micros);
    // What is left of a budget of 1,000, the frame as sent, the time
    // between its send and the wait, then the wait's end in
    // microseconds behind its start and what a miss was given. The
    // frame's deadline stands the rest of the frame behind the wait's
    // start.
    let table = [
        (1_000, 400, 0, (400, 400)),
        (400, 400, 0, (400, 400)),
        (399, 400, 0, (399, 399)),
        (300, 400, 99, (300, 399)),
        (300, 400, 100, (300, 400)),
        (0, 400, 150, (0, 150)),
        // A wait that began behind the frame's deadline ends at once.
        (0, 400, 900, (0, 400)),
        (1_000, 400, 900, (0, 400)),
        (1_000, 400, u64::MAX, (0, 400)),
    ];
    for (left, frame, before, (end, given)) in table {
        let deadline = after(u64::from(frame).saturating_sub(before));
        let under = budget(1_000, left);
        assert_eq!(
            (
                under.grant_wait(started, deadline),
                under.grant_missed(frame, before)
            ),
            (after(end), given),
            "a frame of {frame}, {before} ahead of the wait, under {left}"
        );
    }
}
