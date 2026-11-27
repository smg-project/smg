//! The cut of the ranges at an interval's end: when it is due, when it
//! waits, and how the size classes fall to the workers.

use super::*;

#[test]
fn the_first_decision_at_an_interval_s_end_cuts_the_ranges_before_it_is_placed() {
    let (scorer, pool) = three_classes();
    assert_eq!(
        led(&scorer, &pool, 400, 15 * SECOND - 1),
        Some(0),
        "the even split"
    );
    assert_eq!(ranges(&scorer)[0], (0, 0, 1_364));
    // The model's first decision came at ten seconds: the cut is due at
    // fifteen, and the decision that reaches it is placed by the new
    // ranges. Each worker but the last takes whole classes: the first
    // stops at one class, since the two left are owed to the two workers
    // after it, and the second reaches its share with the next.
    assert_eq!(led(&scorer, &pool, 400, 15 * SECOND), Some(1));
    assert_eq!(
        ranges(&scorer),
        [(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)]
    );
    assert_eq!(led(&scorer, &pool, 100, 15 * SECOND), Some(0));
    assert_eq!(led(&scorer, &pool, 101, 15 * SECOND), Some(1));
    assert_eq!(led(&scorer, &pool, 401, 15 * SECOND), Some(2));
    let partition = of(&scorer, "m");
    let held = partition.lock();
    assert_eq!((held.cut_for, held.next_cut_micros), (2_000, 20 * SECOND));
}

#[test]
fn a_cut_waits_until_a_worker_s_share_moved_by_more_than_a_factor_of_two() {
    let (scorer, pool) = three_classes();
    led(&scorer, &pool, 0, 15 * SECOND);
    let cut = [(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)];
    assert_eq!(ranges(&scorer), cut);
    // A new class and 1,000 tokens more: 7,000 in the window, 2,333 a
    // worker against 2,000 at the last cut. The ranges stand.
    for _ in 0..20 {
        book(&scorer, &pool, 0, 50, 16 * SECOND);
    }
    assert_eq!(loads(&scorer).1, 7_000);
    led(&scorer, &pool, 0, 20 * SECOND);
    assert_eq!(ranges(&scorer), cut);
    // The bookings of eleven seconds leave and 18,000 tokens join: 6,333
    // a worker is more than twice 2,000, and the next cut is made.
    book(&scorer, &pool, 2, 9_000, 20 * SECOND + SECOND / 2);
    book(&scorer, &pool, 2, 9_000, 20 * SECOND + SECOND / 2);
    assert_eq!(loads(&scorer), (vec![1_000, 0, 18_000], 19_000));
    led(&scorer, &pool, 0, 25 * SECOND - 1);
    assert_eq!(ranges(&scorer), cut, "not before the interval's end");
    led(&scorer, &pool, 0, 25 * SECOND);
    assert_eq!(
        ranges(&scorer),
        [(0, 0, 50), (1, 51, 9_000), (2, 9_001, u32::MAX)]
    );
    // A share that fell by more than half is cut for too.
    book(&scorer, &pool, 0, 30, 31 * SECOND);
    book(&scorer, &pool, 1, 60, 31 * SECOND);
    book(&scorer, &pool, 2, 90, 31 * SECOND);
    assert_eq!(loads(&scorer), (vec![30, 60, 90], 180));
    led(&scorer, &pool, 0, 35 * SECOND);
    assert_eq!(
        ranges(&scorer),
        [(0, 0, 30), (1, 31, 60), (2, 61, u32::MAX)]
    );
}

#[test]
fn fewer_size_classes_than_workers_leave_a_worker_without_a_range() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(3);
    led(&scorer, &pool, 0, SECOND);
    for _ in 0..3 {
        book(&scorer, &pool, 0, 200, SECOND);
    }
    led(&scorer, &pool, 0, 6 * SECOND);
    assert_eq!(ranges(&scorer), [(0, 0, 200), (2, 201, u32::MAX)]);
    let over =
        |tokens, kept: &[u32]| led_by(&scorer, &prefill("m", tokens, 6 * SECOND), &pool, kept);
    assert_eq!(over(150, &[0, 1, 2]), Some(0));
    assert_eq!(
        over(150, &[1, 2]),
        Some(2),
        "the middle worker owns no range"
    );
    assert_eq!(over(5_000, &[0, 1]), Some(0), "the nearest owner below");
    assert_eq!(
        over(5_000, &[1]),
        None,
        "no owner survives: the picker's draw"
    );
}

#[test]
fn a_decision_several_intervals_late_makes_one_cut_and_keeps_the_cadence() {
    let (scorer, pool) = three_classes();
    led(&scorer, &pool, 0, 32 * SECOND + 7);
    assert_eq!(ranges(&scorer)[0], (0, 0, 100));
    assert_eq!(
        of(&scorer, "m").lock().next_cut_micros,
        35 * SECOND,
        "the cadence still counts from the model's first decision"
    );
}

#[test]
fn a_share_exactly_twice_or_half_the_last_cut_s_is_not_cut_again() {
    // The ranges at twenty seconds, the window's tokens and the share
    // the ranges were cut for, after the cut for 2,000 a worker at
    // fifteen and `sizes` booked at `at_micros`.
    let second_cut = |at_micros: u64, sizes: &[u32]| {
        let (scorer, pool) = three_classes();
        led(&scorer, &pool, 0, 15 * SECOND);
        for &tokens in sizes {
            book(&scorer, &pool, 0, tokens, at_micros);
        }
        led(&scorer, &pool, 0, 20 * SECOND);
        let cut_for = of(&scorer, "m").lock().cut_for;
        (ranges(&scorer), loads(&scorer).1, cut_for)
    };
    let first = vec![(0, 0, 100), (1, 101, 400), (2, 401, u32::MAX)];
    // At sixteen seconds the bookings of eleven still count: 6,000 more
    // make 12,000, and 4,000 a worker is exactly twice 2,000.
    assert_eq!(
        second_cut(16 * SECOND, &[1_000; 6]),
        (first.clone(), 12_000, 2_000)
    );
    // Three tokens more make it 4,001.
    assert_eq!(
        second_cut(16 * SECOND, &[1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 3]),
        (
            vec![(0, 0, 400), (1, 401, 1_000), (2, 1_001, u32::MAX)],
            12_003,
            4_001
        )
    );
    // A microsecond later the bookings of eleven are gone: 3,000 in the
    // window, and 1,000 a worker is exactly half of 2,000.
    assert_eq!(
        second_cut(16 * SECOND + 1, &[1_000; 3]),
        (first, 3_000, 2_000)
    );
    // Three tokens fewer make it 999.
    assert_eq!(
        second_cut(16 * SECOND + 1, &[1_000, 1_000, 997]),
        (
            vec![(0, 0, 997), (1, 998, 1_000), (2, 1_001, u32::MAX)],
            2_997,
            999
        )
    );
}

#[test]
fn a_worker_whose_classes_add_up_to_exactly_its_share_takes_no_class_more() {
    // Three workers' ranges at the first cut over `sizes`, each one
    // booking, and the share they were cut for.
    let first_cut = |sizes: &[u32]| {
        let scorer = scorer(NEVER, 1.5, 5);
        let pool = pool(3);
        led(&scorer, &pool, 0, SECOND);
        for &tokens in sizes {
            book(&scorer, &pool, 0, tokens, SECOND);
        }
        led(&scorer, &pool, 0, 6 * SECOND);
        let cut_for = of(&scorer, "m").lock().cut_for;
        (ranges(&scorer), cut_for)
    };
    // Ten bookings of 100 tokens are the first class: 1,000 of the
    // window's 3,000, exactly a worker's share, and the first range ends
    // there. The second worker's two classes add up to its share too.
    let mut sizes = vec![100; 10];
    sizes.extend([200, 300, 500, 1_000]);
    assert_eq!(
        first_cut(&sizes),
        (vec![(0, 0, 100), (1, 101, 500), (2, 501, u32::MAX)], 1_000)
    );
    // Three tokens more in the window make the share 1,001: the first
    // class is one token short of it, and its worker takes the next.
    sizes[13] = 1_003;
    assert_eq!(
        first_cut(&sizes),
        (vec![(0, 0, 200), (1, 201, 500), (2, 501, u32::MAX)], 1_001)
    );
}
