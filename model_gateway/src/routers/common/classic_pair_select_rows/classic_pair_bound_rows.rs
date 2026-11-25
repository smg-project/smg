//! The recorder of the gateway's pair placement behind the prefill
//! admission, where a prefill that holds its bound has an open partner or
//! has none: rows of the pair placement recorder's own shape ([`super`]),
//! on a sheet of their own. Nothing in them draws, and recording them
//! draws no row of the pair placement sheet again.
//!
//! The placement asks the admission's bound of the prefills that still
//! have a partner to go to (`pair_pass` in `placement.rs`). Where no open
//! prefill has an open partner it answers no pair, ahead of the bound.
//! Where each open prefill with an open partner holds its bound it answers
//! `at_capacity`, on which a route waits, whatever room a prefill with no
//! open partner has.
//!
//! To record this sheet again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib classic_pair_select_rows::classic_pair_bound_rows -- --ignored --nocapture
//! ```

use super::{fleet, write, Row, State};

/// The most requests the rows' prefill admission lets one prefill hold.
const BOUND: u32 = 4;

fn rows() -> Vec<Row> {
    use State::{Down, Free, Over, Stalled};
    // The fewest requests in flight decides: no draw, no state.
    const BY_COUNT: &str = "power_of_two:least_requests";
    let bound = |name, shed, prefills, decodes| {
        fleet(name, (BY_COUNT, BY_COUNT), shed, prefills, decodes)
            .bound(BOUND)
            .sequence(2)
    };
    let at_bound = [(BOUND, Free), (BOUND + 1, Free)];
    let down = [(0, Down), (0, Down), (0, Down)];
    let over = [(2, Over), (1, Over), (0, Over)];
    vec![
        // No open prefill has an open partner: no pair, ahead of the bound.
        bound(
            "bound-every-prefill-at-capacity-and-no-decode-ready",
            false,
            at_bound,
            down,
        ),
        // The prefill at its bound has the one open partner, and the
        // prefill with room has none: the request waits for the first.
        bound(
            "bound-the-prefill-at-capacity-has-the-only-open-partner",
            false,
            [(BOUND, Free), (1, Free)],
            [(0, Free), (0, Down), (0, Down)],
        ),
        // The same the other way round.
        bound(
            "bound-the-second-prefill-at-capacity-has-the-only-open-partner",
            false,
            [(1, Free), (BOUND, Free)],
            [(0, Down), (0, Free), (0, Down)],
        ),
        // A decode pool over its thresholds whole is open by load, so each
        // prefill has an open partner and the request waits.
        bound(
            "bound-every-prefill-at-capacity-and-every-decode-over",
            false,
            at_bound,
            over,
        ),
        // Under shedding that pool opens nobody: no pair, by the overload
        // shed, ahead of the bound.
        bound(
            "bound-every-prefill-at-capacity-and-every-decode-over-under-shedding",
            true,
            at_bound,
            over,
        ),
        // A decode pool doubted whole pairs by the least-loaded default,
        // which asks the bound of the one prefill it took.
        bound(
            "bound-every-prefill-at-capacity-and-every-decode-stalled",
            false,
            at_bound,
            [(2, Stalled), (1, Stalled), (0, Stalled)],
        ),
        // The one ready prefill holds its bound beside open partners.
        bound(
            "bound-the-only-ready-prefill-at-capacity",
            false,
            [(BOUND, Free), (0, Down)],
            [(2, Free), (1, Free), (0, Free)],
        ),
        // A prefill with room and an open partner takes the request past
        // the prefill at its bound: nothing waits.
        bound(
            "bound-a-prefill-with-room-and-an-open-partner-takes-the-request",
            false,
            [(BOUND, Free), (1, Free)],
            [(0, Free), (1, Free), (2, Free)],
        ),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_pair_bound_rows() {
    write("pair_bound.json", "pair-bound", rows());
}
