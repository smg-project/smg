//! How long a booking counts for its worker: the interval, the order
//! of two clocks and the window's cap.

use super::*;

#[test]
fn a_booking_counts_for_its_worker_until_it_is_older_than_the_interval() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    // A booking before any listing names no member of any partition.
    book(&scorer, &pool, 0, 9, SECOND);
    led(&scorer, &pool, 0, SECOND);
    assert_eq!(loads(&scorer), (vec![0, 0], 0));
    book(&scorer, &pool, 0, 100, SECOND);
    assert_eq!(loads(&scorer), (vec![100, 0], 100));
    // Exactly one interval old is not older than the interval.
    book(&scorer, &pool, 1, 40, 6 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 40], 140));
    book(&scorer, &pool, 1, 10, 6 * SECOND + 1);
    assert_eq!(loads(&scorer), (vec![0, 50], 50), "the first booking left");
    // A booking dated a little before the window's oldest arrived out
    // of its clock's order: both stand.
    book(&scorer, &pool, 0, 7, 6 * SECOND - 5);
    assert_eq!(loads(&scorer), (vec![7, 50], 57));
    // One dated more than an interval before it is a clock that went
    // back: it ends what it passed.
    book(&scorer, &pool, 0, 3, SECOND / 2);
    assert_eq!(loads(&scorer), (vec![3, 0], 3));
    // A worker no partition lists takes no booking.
    let stranger = self::pool(3);
    book(&scorer, &stranger, 2, 1_000, 2 * SECOND);
    assert_eq!(loads(&scorer), (vec![3, 0], 3));
}

#[test]
fn two_bookings_that_arrive_out_of_their_clocks_order_both_count() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    // Two deciders read the clock five microseconds apart and book in
    // the reverse order, into a window that holds nothing older.
    book(&scorer, &pool, 0, 100, SECOND + 5);
    book(&scorer, &pool, 1, 40, SECOND);
    assert_eq!(loads(&scorer), (vec![100, 40], 140));
    // The older one waits behind the newer and leaves with it.
    book(&scorer, &pool, 1, 10, 6 * SECOND + 1);
    assert_eq!(loads(&scorer), (vec![100, 50], 150));
    book(&scorer, &pool, 1, 10, 6 * SECOND + 6);
    assert_eq!(loads(&scorer), (vec![0, 20], 20), "both left");
}

#[test]
fn a_stage_that_names_no_interval_keeps_a_booking_five_seconds() {
    let scorer = Bucket::new(&BucketParams::default());
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    book(&scorer, &pool, 0, 100, SECOND);
    // Exactly five seconds old is not older than the default interval.
    book(&scorer, &pool, 1, 40, 6 * SECOND);
    assert_eq!(loads(&scorer), (vec![100, 40], 140));
    book(&scorer, &pool, 1, 10, 6 * SECOND + 1);
    assert_eq!(loads(&scorer), (vec![0, 50], 50), "the first booking left");
}

#[test]
fn a_window_past_its_cap_drops_its_oldest_booking() {
    let scorer = scorer(NEVER, 1.5, 5);
    let pool = pool(2);
    led(&scorer, &pool, 0, SECOND);
    book(&scorer, &pool, 1, 1_000, SECOND);
    for _ in 0..WINDOW_CAP {
        book(&scorer, &pool, 0, 3, SECOND);
    }
    // The clock stood still and the first booking left all the same.
    assert_eq!(
        loads(&scorer),
        (vec![3 * WINDOW_CAP as u64, 0], 3 * WINDOW_CAP as u64)
    );
    assert_eq!(of(&scorer, "m").lock().window.len(), WINDOW_CAP);
}
