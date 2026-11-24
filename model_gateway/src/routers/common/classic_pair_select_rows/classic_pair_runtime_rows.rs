//! The recorder of the gateway's pair placement on the wire that carries
//! one runtime's request, where the workers of one known runtime stand
//! beside workers of another or of none: rows of the pair placement
//! recorder's own shape ([`super`]), on a sheet of their own. Nothing in
//! them draws, and recording them draws no row of the pair placement sheet
//! again.
//!
//! The pair index keeps two known runtimes apart on that wire and pairs a
//! worker of no known runtime with any (`PdWire::mismatch`). The placement
//! then decides one order (`pair_pass` in `placement.rs`). A prefill of a
//! known runtime with an open partner on that runtime is a candidate ahead
//! of every other prefill, the prefill admission's bound is asked of those
//! candidates alone, and the pick's decode is one of its partners on its
//! runtime. Where no prefill has such a partner open, every prefill with an
//! open partner stands, with any of them. The least-loaded default reads no
//! such order.
//!
//! To record this sheet again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib classic_pair_select_rows::classic_pair_runtime_rows -- --ignored --nocapture
//! ```

use super::{member, write, Cell, Member, Row, State};

/// The fewest requests in flight decides: no draw, no state.
const BY_COUNT: &str = "power_of_two:least_requests";

/// The most requests the bound row's prefill admission lets one prefill
/// hold.
const BOUND: u32 = 4;

/// A row on the wire that carries one runtime's request.
fn cell(name: &'static str, shed: bool, prefills: Vec<Member>, decodes: Vec<Member>) -> Cell {
    Cell {
        name,
        prefill_policy: BY_COUNT,
        decode_policy: BY_COUNT,
        shed,
        prefill_bound: None,
        wire: Some("grpc"),
        prefills,
        decodes,
    }
}

/// Two prefills and two decodes that name no transport. Place 0 of each
/// side is on no known runtime and place 1 on the sheet's own, so each
/// prefill pairs with both decodes.
fn mixed(
    name: &'static str,
    shed: bool,
    prefills: [(u32, State); 2],
    decodes: [(u32, State); 2],
) -> Cell {
    let [(p0, s0), (p1, s1)] = prefills;
    let [(d0, t0), (d1, t1)] = decodes;
    cell(
        name,
        shed,
        vec![member(p0, s0, None).on("unknown"), member(p1, s1, None)],
        vec![member(d0, t0, None).on("unknown"), member(d1, t1, None)],
    )
}

fn rows() -> Vec<Row> {
    use State::{Down, Free, Over};
    // The workers on the known runtime hold more requests than the others.
    let busier = [(0, Free), (3, Free)];
    let over = [(0, Over), (3, Over)];
    vec![
        // The pair on the known runtime is placed, past two workers that
        // hold fewer requests.
        mixed(
            "runtime-a-pair-on-one-known-runtime-goes-first",
            false,
            busier,
            busier,
        )
        .sequence(2),
        // At its bound the prefill with a partner on its runtime keeps the
        // request: it waits, beside a prefill with room.
        mixed(
            "runtime-the-known-prefill-at-its-bound-keeps-the-request",
            false,
            [(0, Free), (BOUND, Free)],
            [(0, Free), (0, Free)],
        )
        .bound(BOUND)
        .sequence(2),
        // Not ready, or over its thresholds, that prefill keeps no request:
        // every prefill with an open partner stands, with any of them.
        mixed(
            "runtime-the-known-prefill-not-ready-leaves-every-pair",
            false,
            [(0, Free), (0, Down)],
            busier,
        )
        .sequence(2),
        mixed(
            "runtime-the-known-prefill-over-leaves-every-pair",
            false,
            [(1, Free), (0, Over)],
            busier,
        )
        .sequence(2),
        // Its partner on its runtime is not ready: it pairs with its
        // partner of no known runtime.
        mixed(
            "runtime-the-known-decode-not-ready-pairs-with-any-partner",
            false,
            [(5, Free), (1, Free)],
            [(0, Free), (0, Down)],
        )
        .sequence(2),
        // A decode pool over its thresholds whole is open by load: the pair
        // on the known runtime still goes first, its decode the busier one.
        mixed(
            "runtime-every-decode-over-keeps-the-known-pair",
            false,
            busier,
            over,
        )
        .sequence(2),
        // Under shedding that pool opens nobody.
        mixed(
            "runtime-every-decode-over-under-shedding",
            true,
            busier,
            over,
        )
        .sequence(2),
        // A prefill pool over its thresholds whole pairs by the least-loaded
        // default, which reads every partner of a prefill alike.
        mixed(
            "runtime-every-prefill-over-pairs-by-load-with-any-partner",
            false,
            [(5, Over), (0, Over)],
            busier,
        )
        .sequence(2),
        // Two known runtimes pair apart, and neither goes ahead of the
        // other.
        cell(
            "runtime-two-known-runtimes-pair-apart",
            false,
            vec![member(2, Free, None).on("other"), member(1, Free, None)],
            vec![member(0, Free, None), member(0, Free, None).on("other")],
        )
        .sequence(2),
        // A prefill of a known runtime with no partner on it stands behind
        // the pair on one runtime, though it holds fewer requests.
        cell(
            "runtime-a-known-prefill-with-no-partner-on-its-runtime-stands-behind",
            false,
            vec![member(0, Free, None).on("other"), member(3, Free, None)],
            vec![member(0, Free, None).on("unknown"), member(3, Free, None)],
        )
        .sequence(2),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_pair_runtime_rows() {
    write("pair_runtime.json", "pair-runtime", rows());
}
