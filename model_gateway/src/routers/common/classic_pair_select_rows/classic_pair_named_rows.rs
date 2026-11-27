//! The recorder of the gateway's pair placement for a request that names
//! its worker under `x-smg-target-worker`: rows of the pair placement
//! recorder's own shape ([`super`]), each with the text its requests carry
//! under that header, on a sheet of their own. Nothing in them draws, and
//! recording them draws no row of the pair placement sheet again.
//!
//! The hash policy alone reads the header (`consistent_hashing`), once a
//! side, over the workers the placement hands that side's policy
//! (`pair_pass` in `placement.rs`, `target_among` in
//! `policies/consistent_hashing.rs`). The prefill side is handed the
//! prefills that can be sent to and have a partner that can, and the
//! decode side the partners of the picked prefill that can be sent to: the
//! number counts those, from 0. A side under the hash policy is narrowed by
//! no admission filter and by no retry. A number that reaches no worker of
//! a side is that policy's refusal, and no other worker takes the request.
//! A side under another policy reads no header. Where a pool is over its
//! thresholds on every worker the placement pairs it by load, and reads no
//! header for it.
//!
//! To record this sheet again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib classic_pair_select_rows::classic_pair_named_rows -- --ignored --nocapture
//! ```

use axum::http::HeaderMap;

use super::{member, write, Answer, Bed, Cell, Member, Outcome, Row, State};

/// The hash policy: the one that reads the header.
const RING: &str = "consistent_hashing";

/// The fewest requests in flight decides: no draw, no state.
const BY_COUNT: &str = "power_of_two:least_requests";

/// The most requests the bound row's prefill admission lets one prefill
/// hold.
const BOUND: u32 = 4;

/// A row whose requests name a worker.
struct Named {
    cell: Cell,
    /// The text of the header.
    target: &'static str,
    /// The pair a failed attempt of each request was sent to, where the
    /// row's requests are retries: the prefill's place, then the decode's.
    tried: Option<(usize, usize)>,
}

impl Named {
    /// The row's requests as retries of an attempt sent to that pair.
    fn retried(mut self, prefill: usize, decode: usize) -> Self {
        self.tried = Some((prefill, decode));
        self
    }

    fn run(&self, requests: u32) -> Vec<Answer> {
        let bed = Bed::new(&self.cell);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-smg-target-worker",
            self.target.parse().expect("the header's text"),
        );
        let tried: Vec<String> = self.tried.map_or_else(Vec::new, |(prefill, decode)| {
            vec![
                bed.prefills[prefill].base_url().to_string(),
                bed.decodes[decode].base_url().to_string(),
            ]
        });
        (0..requests)
            .map(|_| bed.pick_with(Some(&headers), &tried))
            .collect()
    }

    /// `requests` requests, one after another: the same answers in each of
    /// twenty recordings.
    fn sequence(self, requests: u32) -> Row {
        let first = self.run(requests);
        for _ in 0..19 {
            assert!(
                self.run(requests) == first,
                "{}: a sequence row must not depend on a draw",
                self.cell.name
            );
        }
        let (target, tried) = (self.target, self.tried);
        let mut row = self.cell.row(requests, Outcome::Sequence(first));
        row.target = Some(target);
        row.tried = tried;
        row
    }
}

/// A row on the plain wire whose requests carry `target`. Each worker is
/// the requests the router holds on it and its state, and names no
/// transport: each prefill pairs with every decode.
fn named(
    name: &'static str,
    policies: (&'static str, &'static str),
    target: &'static str,
    prefills: &[(u32, State)],
    decodes: &[(u32, State)],
) -> Named {
    let members = |side: &[(u32, State)]| -> Vec<Member> {
        side.iter()
            .map(|&(held, state)| member(held, state, None))
            .collect()
    };
    Named {
        cell: Cell {
            name,
            prefill_policy: policies.0,
            decode_policy: policies.1,
            shed: false,
            prefill_bound: None,
            wire: None,
            prefills: members(prefills),
            decodes: members(decodes),
        },
        target,
        tried: None,
    }
}

fn rows() -> Vec<Row> {
    use State::{Down, Free, Over};
    // Both sides read the header, the prefill side alone, the decode side
    // alone.
    const BOTH: (&str, &str) = (RING, RING);
    const PREFILL: (&str, &str) = (RING, BY_COUNT);
    const DECODE: (&str, &str) = (BY_COUNT, RING);
    let free = [(0, Free), (0, Free), (0, Free)];
    let pair = [(0, Free), (0, Free)];
    // Three decodes that hold three, one and two requests.
    let held = [(3, Free), (1, Free), (2, Free)];
    // Two decodes over their thresholds: the second holds fewer requests.
    let over = [(3, Over), (1, Over)];
    // A prefill and a decode of each of two transports, a decode that names
    // none and a second decode of the first: the first prefill pairs with
    // decodes 0, 2 and 3, the second with decodes 1 and 2. The second
    // prefill holds fewer requests.
    let apart = |name: &'static str, target: &'static str| Named {
        cell: Cell {
            name,
            prefill_policy: BY_COUNT,
            decode_policy: RING,
            shed: false,
            prefill_bound: None,
            wire: None,
            prefills: vec![member(2, Free, Some("a")), member(0, Free, Some("b"))],
            decodes: vec![
                member(0, Free, Some("a")),
                member(0, Free, Some("b")),
                member(0, Free, None),
                member(0, Free, Some("a")),
            ],
        },
        target,
        tried: None,
    };
    let mut shedding = named(
        "named-both-every-decode-over-under-shedding",
        BOTH,
        "0",
        &pair,
        &over,
    );
    shedding.cell.shed = true;
    let mut at_bound = named(
        "named-both-the-named-prefill-at-its-bound-keeps-the-request",
        BOTH,
        "0",
        &[(BOUND, Free), (0, Free)],
        &pair,
    );
    at_bound.cell = at_bound.cell.bound(BOUND);
    let mut full = named(
        "named-both-the-named-decode-with-no-free-room-is-taken",
        BOTH,
        "0",
        &pair,
        &pair,
    );
    full.cell = full.cell.rooms([2, 0, 0]);
    let mut roomy = named(
        "named-prefill-the-decode-side-keeps-the-decodes-with-a-room",
        PREFILL,
        "1",
        &free,
        &held,
    );
    roomy.cell = roomy.cell.rooms([0, 2, 0]);
    let mut full_alone = named(
        "named-decode-the-named-decode-with-no-free-room-is-taken",
        DECODE,
        "0",
        &[(1, Free), (0, Free)],
        &pair,
    );
    full_alone.cell = full_alone.cell.rooms([2, 0, 0]);
    // On the wire that carries one runtime's request: place 0 of each side
    // is on no known runtime, the others on the sheet's own. The prefill on
    // the known runtime holds more requests.
    let first_partners = Named {
        cell: Cell {
            name: "named-decode-is-counted-among-the-first-partners",
            prefill_policy: BY_COUNT,
            decode_policy: RING,
            shed: false,
            prefill_bound: None,
            wire: Some("grpc"),
            prefills: vec![member(0, Free, None).on("unknown"), member(3, Free, None)],
            decodes: vec![
                member(0, Free, None).on("unknown"),
                member(0, Free, None),
                member(0, Free, None),
            ],
        },
        target: "1",
        tried: None,
    };
    vec![
        // Each side's number counts the workers of its side that can be
        // sent to: a worker ahead of it that is not ready is not counted.
        named(
            "named-both-each-side-counts-the-workers-that-can-be-sent-to",
            BOTH,
            "1",
            &[(0, Free), (0, Down), (0, Free), (0, Free)],
            &[(0, Free), (0, Free), (0, Down), (0, Free)],
        )
        .sequence(2),
        // A prefill over its thresholds is not counted either.
        named(
            "named-both-a-prefill-over-its-thresholds-is-not-counted",
            BOTH,
            "0",
            &[(0, Over), (0, Free), (0, Free)],
            &pair,
        )
        .sequence(2),
        // A number past the prefills that are counted: the prefill side's
        // refusal, and no other prefill takes the request.
        named(
            "named-both-a-number-past-the-prefills-is-refused",
            BOTH,
            "2",
            &pair,
            &free,
        )
        .sequence(2),
        // A number past the partners of the prefill it reached: the decode
        // side's refusal.
        named(
            "named-both-a-number-past-the-partners-is-refused",
            BOTH,
            "2",
            &free,
            &pair,
        )
        .sequence(2),
        // A text that is no number reaches no worker.
        named(
            "named-both-a-text-that-is-no-number-is-refused",
            BOTH,
            "first",
            &pair,
            &pair,
        )
        .sequence(2),
        // The named prefill holds the admission's bound beside a prefill
        // with room: the request waits for the one it names.
        at_bound.sequence(2),
        // The named decode's rooms are all claimed beside a decode with a
        // free one: it is taken.
        full.sequence(2),
        // A decode pool over its thresholds whole pairs by load: the named
        // prefill, and the decode that holds fewer requests, not the one
        // the number reaches.
        named(
            "named-both-every-decode-over-goes-by-load",
            BOTH,
            "0",
            &pair,
            &over,
        )
        .sequence(2),
        // Under shedding that pool opens nobody.
        shedding.sequence(2),
        // A prefill pool over its thresholds whole pairs by the least-loaded
        // default, which reads no header: neither worker is the named one.
        named(
            "named-both-every-prefill-over-pairs-by-load",
            BOTH,
            "0",
            &[(5, Over), (0, Over)],
            &[(2, Free), (0, Free)],
        )
        .sequence(2),
        // A retry of an attempt sent to the named pair is handed that pair
        // again.
        named(
            "named-both-a-retry-keeps-the-named-pair",
            BOTH,
            "1",
            &free,
            &free,
        )
        .retried(1, 1)
        .sequence(2),
        // The prefill side alone reads the header: the decode is its own
        // policy's pick among the named prefill's partners.
        named(
            "named-prefill-the-decode-is-its-own-policys-pick",
            PREFILL,
            "2",
            &free,
            &held,
        )
        .sequence(2),
        named(
            "named-prefill-a-number-past-the-prefills-is-refused",
            PREFILL,
            "3",
            &free,
            &held,
        )
        .sequence(2),
        // That decode side keeps its admission filter: the decode that
        // holds the fewest requests has no free room, and the next takes
        // the request.
        roomy.sequence(2),
        // And it steps aside from the decode a failed attempt was sent to,
        // while the named prefill is handed the retry again.
        named(
            "named-prefill-a-retry-keeps-the-prefill-and-leaves-the-decode",
            PREFILL,
            "0",
            &free,
            &held,
        )
        .retried(0, 1)
        .sequence(2),
        // The decode side alone reads the header: the prefill is its own
        // policy's pick, and the number counts that pick's partners that
        // can be sent to.
        named(
            "named-decode-is-counted-behind-the-prefill-its-policy-picks",
            DECODE,
            "1",
            &[(2, Free), (0, Free), (1, Free)],
            &[(0, Free), (0, Down), (0, Free), (0, Free)],
        )
        .sequence(2),
        // The number counts the picked prefill's partners, not the pool:
        // the second of the pool pairs with the other prefill alone.
        apart("named-decode-is-counted-among-the-picks-partners", "1").sequence(2),
        // A number past the pick's partners is refused, though the other
        // prefill has a third partner: no other prefill is asked.
        apart(
            "named-decode-a-number-past-the-picks-partners-is-refused",
            "2",
        )
        .sequence(2),
        // A decode pool over its thresholds whole pairs by load with the
        // picked prefill, and reads no header.
        named(
            "named-decode-every-decode-over-goes-by-load",
            DECODE,
            "0",
            &[(1, Free), (0, Free)],
            &over,
        )
        .sequence(2),
        // The named decode's rooms are all claimed: it is taken.
        full_alone.sequence(2),
        // A pair on one known runtime goes first, so the number counts the
        // picked prefill's partners on its runtime.
        first_partners.sequence(2),
        // A retry leaves the prefill a failed attempt was sent to, and is
        // handed the named decode again.
        named(
            "named-decode-a-retry-leaves-the-prefill-and-keeps-the-decode",
            DECODE,
            "0",
            &pair,
            &pair,
        )
        .retried(0, 0)
        .sequence(2),
        // No prefill is open: the first is over its thresholds beside its
        // one partner, which is over them too, and the second is not ready
        // beside the one decode that can be sent to. The placement pairs
        // such a prefill pool by load and reads no header: the first prefill
        // goes with its partner, not with the decode the number reaches.
        Named {
            cell: Cell {
                name: "named-decode-no-open-prefill-pairs-by-load",
                prefill_policy: BY_COUNT,
                decode_policy: RING,
                shed: false,
                prefill_bound: None,
                wire: None,
                prefills: vec![member(0, Over, Some("a")), member(0, Down, Some("b"))],
                decodes: vec![member(0, Over, Some("a")), member(0, Free, Some("b"))],
            },
            target: "0",
            tried: None,
        }
        .sequence(2),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_pair_named_rows() {
    write("pair_named.json", "pair-named", rows());
}
