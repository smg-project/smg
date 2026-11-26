use super::Miss;
use crate::plugins::grpc::runtime::PROMISE_MICROS;

/// The table of a miss, row by row: the witness's excuse first, then
/// whose bound it was, then whether the witness can vouch for a flight
/// that long.
#[test]
fn the_table_of_a_miss_row_by_row() {
    const LONG: u64 = PROMISE_MICROS + 1;
    // Stalled, the plugin's own bound, the flight given, the verdict.
    let table = [
        (true, true, LONG, Miss::Held),
        (true, false, LONG, Miss::Held),
        (true, true, 1, Miss::Held),
        (false, false, LONG, Miss::Clamped),
        (false, false, 1, Miss::Clamped),
        (false, true, LONG, Miss::Own),
        // No longer than the promise: the thread cannot be seen late.
        (false, true, PROMISE_MICROS, Miss::Unread),
        (false, true, 1, Miss::Unread),
    ];
    for (stalled, own_bound, given, verdict) in table {
        assert_eq!(
            Miss::of(stalled, own_bound, given),
            verdict,
            "stalled {stalled}, own {own_bound}, given {given}"
        );
    }
}
