//! The recorder of the gateway's pair placement under the sticky
//! routing-key override: a prefill pool and a decode pool of the pair
//! placement recorder's own shape ([`super`]), the override's assignment
//! mode, and a sequence of steps: a request with a routing key or with
//! none, a worker that stops or starts being routable, the requests a pool
//! holds, and the requests of one key in flight on a pool's workers. Each
//! request is answered by the pair it was placed on, on a sheet of its own.
//!
//! The pair placement asks the registry once a role: the prefill over the
//! prefills with an open partner, then the decode over the picked prefill's
//! open partners (`pair_pass` in `placement.rs`). The override places a role
//! only where that role's policy does not read the key itself
//! (`PolicyRegistry::sticky_for`): `consistent_hashing` and `manual` do. It
//! tags the key by its model and its role, so the two roles of one key pin
//! apart in one map. A pin with fewer than two of the key's requests in
//! flight on its worker takes the request and asks the role's policy
//! nothing. At that cap the role is placed again, and the pin moves only to
//! another worker under the cap. A key with no usable pin is placed by the
//! role's policy under the `delegate` mode, or by the map's own assignment,
//! and pinned at that selection.
//!
//! A request's key travels in the header `x-smg-routing-key`, or is the key
//! its caller derived from the request's own id, which the header's key
//! goes ahead of. A header's key loses its turn and retry suffixes
//! (`conv_t2_r1` keys as `conv`), and an id's key is stated as its caller
//! derived it. The count a cap reads is the worker's own count of the key's
//! text, the same on both sides: a step states it a side at a time.
//!
//! A sequence row is recorded twenty times from fresh pools, fresh policies
//! and a fresh map and must come out the same each time; a drawn row
//! records how often each pair took the row's last request over its draws.
//!
//! To record this sheet again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib classic_pair_select_rows::classic_pair_sticky_rows -- --ignored --nocapture
//! ```

use axum::http::HeaderMap;
use openai_protocol::worker::WorkerStatus;
use serde::Serialize;

use super::{config_of, member, write, Answer, Bed, Cell, Member, State};
use crate::{
    config::types::{ManualAssignmentMode, RoutingKeyOverrideConfig},
    policies::{PolicyFactory, PolicyRegistry},
    routers::common::placement::{select_pair, PlacementInputs, Stated},
};

/// The fewest requests in flight decides: no draw, no state.
const BY_COUNT: &str = "power_of_two:least_requests";

/// The hash policy: it reads the routing key itself.
const RING: &str = "consistent_hashing";

/// A side of the pair.
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Side {
    Prefill,
    Decode,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: the routing key its header carries, if it carries one,
    /// the key of its own id, if it states one, its model, and the pair a
    /// failed attempt of it was sent to, where the request is a retry: the
    /// prefill's place, then the decode's.
    Pick {
        key: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<&'static str>,
        model: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        tried: Option<(usize, usize)>,
    },
    /// A worker of a side stops being routable.
    Down(Side, usize),
    /// The worker is routable again.
    Up(Side, usize),
    /// The requests the router holds on each worker of a side, from here
    /// on.
    Loads(Side, Vec<u32>),
    /// The requests of one routing key in flight on each worker of a side,
    /// from here on: the count the override's cap reads.
    Held {
        key: &'static str,
        side: Side,
        on: Vec<u32>,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    /// The answer to each `Pick` step, in order.
    Sequence(Vec<Answer>),
    /// How often each pair took the row's last request over `draws` runs of
    /// the row: prefill, decode, count; a pair never placed is not listed.
    Counts {
        draws: u32,
        pairs: Vec<(usize, usize, u32)>,
    },
}

#[derive(Serialize)]
struct Row {
    name: &'static str,
    /// Each pool's policy, in its configuration spelling.
    prefill_policy: &'static str,
    decode_policy: &'static str,
    /// The sticky map's assignment mode, in its configuration spelling.
    assignment: &'static str,
    prefills: Vec<Member>,
    decodes: Vec<Member>,
    steps: Vec<Step>,
    outcome: Outcome,
}

/// A row before its outcome is recorded.
struct Sticky {
    cell: Cell,
    assignment: &'static str,
    steps: Vec<Step>,
}

impl Sticky {
    /// The pair placement recorder's bed in the row's starting state, its
    /// registry built again with the override on: a fresh map under the
    /// row's assignment mode, and the row's two side policies.
    fn bed(&self) -> Bed {
        let mut bed = Bed::new(&self.cell);
        let assignment_mode = match self.assignment {
            "random" => ManualAssignmentMode::Random,
            "min_load" => ManualAssignmentMode::MinLoad,
            "min_group" => ManualAssignmentMode::MinGroup,
            "delegate" => ManualAssignmentMode::Delegate,
            other => panic!("{}: no assignment mode {other}", self.cell.name),
        };
        // No eviction task: a row runs far inside any idle time.
        bed.policies = PolicyRegistry::with_override(
            config_of("random"),
            RoutingKeyOverrideConfig {
                enabled: true,
                eviction_interval_secs: 0,
                max_idle_secs: 0,
                assignment_mode,
                ..RoutingKeyOverrideConfig::default()
            },
        );
        let side = |policy: &str| PolicyFactory::create_from_config(&config_of(policy));
        bed.policies
            .set_prefill_policy(side(self.cell.prefill_policy));
        bed.policies
            .set_decode_policy(side(self.cell.decode_policy));
        bed
    }

    /// One step; a request answers the pair it was placed on.
    fn step(bed: &Bed, step: &Step) -> Option<Answer> {
        let pool = |side: Side| match side {
            Side::Prefill => &bed.prefills,
            Side::Decode => &bed.decodes,
        };
        match step {
            Step::Pick {
                key,
                id,
                model,
                tried,
            } => {
                let mut headers = HeaderMap::new();
                if let Some(key) = key {
                    headers.insert("x-smg-routing-key", key.parse().expect("the key's text"));
                }
                let tried: Vec<String> = tried.map_or_else(Vec::new, |(prefill, decode)| {
                    vec![
                        bed.prefills[prefill].base_url().to_string(),
                        bed.decodes[decode].base_url().to_string(),
                    ]
                });
                let inputs = PlacementInputs {
                    headers: Some(&headers),
                    rid_key: *id,
                    tried: &tried,
                    ..PlacementInputs::default()
                };
                Some(bed.answer(&select_pair(
                    &bed.registry,
                    &bed.policies,
                    model,
                    &bed.pairs,
                    None,
                    None,
                    inputs,
                    Stated::nothing,
                )))
            }
            Step::Down(side, place) => {
                pool(*side)[*place].set_status(WorkerStatus::NotReady);
                None
            }
            Step::Up(side, place) => {
                pool(*side)[*place].set_status(WorkerStatus::Ready);
                None
            }
            Step::Loads(side, loads) => {
                for (worker, &want) in pool(*side).iter().zip(loads) {
                    while worker.load() < want as usize {
                        worker.increment_load();
                    }
                    while worker.load() > want as usize {
                        worker.decrement_load();
                    }
                }
                None
            }
            Step::Held { key, side, on } => {
                for (worker, &want) in pool(*side).iter().zip(on) {
                    while worker.routing_key_inflight(key) < want as usize {
                        worker.increment_routing_key_load(key);
                    }
                    while worker.routing_key_inflight(key) > want as usize {
                        worker.decrement_routing_key_load(key);
                    }
                }
                None
            }
        }
    }

    fn run(&self) -> Vec<Answer> {
        let bed = self.bed();
        self.steps
            .iter()
            .filter_map(|step| Self::step(&bed, step))
            .collect()
    }

    fn row(self, outcome: Outcome) -> Row {
        Row {
            name: self.cell.name,
            prefill_policy: self.cell.prefill_policy,
            decode_policy: self.cell.decode_policy,
            assignment: self.assignment,
            prefills: self.cell.prefills,
            decodes: self.cell.decodes,
            steps: self.steps,
            outcome,
        }
    }

    /// The row's requests, one after another: the same answers in each of
    /// twenty recordings.
    fn sequence(self) -> Row {
        let first = self.run();
        for _ in 0..19 {
            assert!(
                self.run() == first,
                "{}: a sequence row must not depend on a draw",
                self.cell.name
            );
        }
        self.row(Outcome::Sequence(first))
    }

    /// The row drawn 600,000 times: how often each pair took its last
    /// request. A replay draws a hundredth of that, so one deviation of
    /// the recording is a tenth of a replay's. A recording that falls far
    /// from its mean then fails no replay by chance.
    fn drawn(self) -> Row {
        const DRAWS: u32 = 600_000;
        let mut counts = vec![vec![0u32; self.cell.decodes.len()]; self.cell.prefills.len()];
        for _ in 0..DRAWS {
            match self.run().pop() {
                Some(Answer::Pair(prefill, decode)) => counts[prefill][decode] += 1,
                _ => panic!("{}: a drawn row places its last request", self.cell.name),
            }
        }
        let pairs = counts
            .iter()
            .enumerate()
            .flat_map(|(prefill, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(_, &count)| count > 0)
                    .map(move |(decode, &count)| (prefill, decode, count))
            })
            .collect();
        self.row(Outcome::Counts {
            draws: DRAWS,
            pairs,
        })
    }
}

/// Three prefills and three decodes that name no transport, each by its
/// requests in flight: every prefill pairs with every decode.
fn open(
    name: &'static str,
    policies: (&'static str, &'static str),
    assignment: &'static str,
    prefills: [u32; 3],
    decodes: [u32; 3],
    steps: Vec<Step>,
) -> Sticky {
    let side = |loads: [u32; 3]| {
        loads
            .iter()
            .map(|&in_flight| member(in_flight, State::Free, None))
            .collect()
    };
    Sticky {
        cell: Cell {
            name,
            prefill_policy: policies.0,
            decode_policy: policies.1,
            shed: false,
            prefill_bound: None,
            wire: None,
            prefills: side(prefills),
            decodes: side(decodes),
        },
        assignment,
        steps,
    }
}

/// Two prefills of two transports and three decodes: one of each transport
/// and one that names none. The first prefill pairs with decodes 0 and 2,
/// the second with decodes 1 and 2. Under the `delegate` mode, each side by
/// its fewest requests.
fn two_transports(
    name: &'static str,
    prefills: [u32; 2],
    decodes: [u32; 3],
    steps: Vec<Step>,
) -> Sticky {
    let [p0, p1] = prefills;
    let [d0, d1, d2] = decodes;
    Sticky {
        cell: Cell {
            name,
            prefill_policy: BY_COUNT,
            decode_policy: BY_COUNT,
            shed: false,
            prefill_bound: None,
            wire: None,
            prefills: vec![
                member(p0, State::Free, Some("a")),
                member(p1, State::Free, Some("b")),
            ],
            decodes: vec![
                member(d0, State::Free, Some("a")),
                member(d1, State::Free, Some("b")),
                member(d2, State::Free, None),
            ],
        },
        assignment: "delegate",
        steps,
    }
}

/// A request of `key` for the model every row but one asks for.
fn keyed(key: &'static str) -> Step {
    under(key, "m")
}

fn under(key: &'static str, model: &'static str) -> Step {
    asked(Some(key), None, model)
}

fn keyless() -> Step {
    asked(None, None, "m")
}

/// A request with no key in its header, whose own id gives `key`.
fn by_id(key: &'static str) -> Step {
    asked(None, Some(key), "m")
}

/// A request whose header carries `key` and whose own id gives `id`.
fn with_id(key: &'static str, id: &'static str) -> Step {
    asked(Some(key), Some(id), "m")
}

fn asked(key: Option<&'static str>, id: Option<&'static str>, model: &'static str) -> Step {
    Step::Pick {
        key,
        id,
        model,
        tried: None,
    }
}

/// A retry of `key`'s request whose failed attempt went to that pair.
fn retried(key: &'static str, prefill: usize, decode: usize) -> Step {
    Step::Pick {
        key: Some(key),
        id: None,
        model: "m",
        tried: Some((prefill, decode)),
    }
}

fn loads(side: Side, loads: [u32; 3]) -> Step {
    Step::Loads(side, loads.to_vec())
}

fn held(key: &'static str, side: Side, on: [u32; 3]) -> Step {
    Step::Held {
        key,
        side,
        on: on.to_vec(),
    }
}

fn rows() -> Vec<Row> {
    use Side::{Decode, Prefill};
    use Step::{Down, Up};
    const DELEGATE: &str = "delegate";
    const COUNTS: (&str, &str) = (BY_COUNT, BY_COUNT);
    // A side policy that draws: a row whose answers do not draw shows that
    // the policy was not asked.
    const DRAWS: (&str, &str) = ("random", "random");
    vec![
        // The pins: a key keeps its pair, each side under its own pin,
        // whatever the loads say behind its first request.
        open(
            "sticky-pair-delegate-a-key-keeps-its-pair-past-the-loads",
            COUNTS,
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyed("k1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                keyed("k1"),
                keyed("k2"),
                keyed("k1"),
            ],
        )
        .sequence(),
        // A request with no key is each side's policy's own, and pins
        // nothing.
        open(
            "sticky-pair-a-request-with-no-key-is-each-policys-own",
            COUNTS,
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyless(),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                keyless(),
            ],
        )
        .sequence(),
        // One key under two models pins a pair for each.
        open(
            "sticky-pair-one-key-pins-apart-under-two-models",
            COUNTS,
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                under("k1", "m1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                under("k1", "m2"),
                loads(Prefill, [9, 9, 0]),
                loads(Decode, [9, 9, 0]),
                under("k1", "m1"),
                under("k1", "m2"),
            ],
        )
        .sequence(),
        // A key from the request's own id pins as a header's key does, and
        // a header's key goes ahead of the id's.
        open(
            "sticky-pair-a-key-from-the-requests-id-pins-as-a-headers-does-and-behind-it",
            COUNTS,
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyed("k1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                by_id("k2"),
                loads(Prefill, [9, 9, 0]),
                loads(Decode, [9, 9, 0]),
                by_id("k1"),
                with_id("k1", "k2"),
                with_id("k3", "k1"),
            ],
        )
        .sequence(),
        // The turns and the retries of one conversation share its key: its
        // pair, and the count its cap reads, under the key with no suffix.
        open(
            "sticky-pair-a-keys-turns-and-retries-share-its-pair-and-its-cap",
            COUNTS,
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyed("conv_t1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                keyed("conv_t2_r1"),
                held("conv", Prefill, [0, 2, 0]),
                keyed("conv_t3"),
            ],
        )
        .sequence(),
        // The map's own modes place a key seen first on each side, and the
        // side policies, which draw, are asked nothing.
        open(
            "sticky-pair-min-load-a-new-key-takes-each-sides-least-load-and-keeps-it",
            DRAWS,
            "min_load",
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyed("k1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                keyed("k1"),
                keyed("k2"),
                keyed("k1"),
            ],
        )
        .sequence(),
        // The fewest keys in flight decide, whatever the loads: the keys
        // other requests hold on each worker are stated first.
        open(
            "sticky-pair-min-group-a-new-key-takes-each-sides-fewest-keys",
            DRAWS,
            "min_group",
            [0, 5, 0],
            [5, 0, 0],
            vec![
                held("x1", Prefill, [1, 0, 1]),
                held("x2", Prefill, [1, 0, 0]),
                held("x1", Decode, [1, 1, 0]),
                held("x2", Decode, [0, 1, 0]),
                keyed("k1"),
                held("k1", Prefill, [0, 1, 0]),
                held("x3", Prefill, [0, 1, 0]),
                held("x4", Prefill, [0, 1, 0]),
                held("k1", Decode, [0, 0, 1]),
                held("x3", Decode, [0, 0, 1]),
                held("x4", Decode, [0, 0, 1]),
                keyed("k2"),
                keyed("k1"),
            ],
        )
        .sequence(),
        open(
            "sticky-pair-random-a-new-key-is-drawn-on-each-side",
            COUNTS,
            "random",
            [3, 1, 2],
            [2, 0, 1],
            vec![keyed("k1"), keyed("k1")],
        )
        .drawn(),
        // The cap, a side at a time: two of the key's requests in flight
        // on its prefill place the prefill again, and the pin moves to the
        // policy's pick, which holds none of them. The decode keeps its
        // pin.
        open(
            "sticky-pair-the-prefills-cap-moves-the-prefill-alone",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                held("k1", Prefill, [0, 2, 0]),
                loads(Prefill, [0, 3, 2]),
                keyed("k1"),
                held("k1", Prefill, [0, 0, 0]),
                loads(Prefill, [9, 0, 9]),
                keyed("k1"),
            ],
        )
        .sequence(),
        open(
            "sticky-pair-the-decodes-cap-moves-the-decode-alone",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                held("k1", Decode, [0, 2, 0]),
                loads(Decode, [0, 3, 2]),
                keyed("k1"),
                held("k1", Decode, [0, 0, 0]),
                loads(Decode, [9, 0, 9]),
                keyed("k1"),
            ],
        )
        .sequence(),
        // At the cap the policy picks the pinned prefill again: the pin
        // stays.
        open(
            "sticky-pair-at-the-cap-a-pinned-prefill-picked-again-keeps-its-pin",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                held("k1", Prefill, [0, 2, 0]),
                keyed("k1"),
                held("k1", Prefill, [0, 0, 0]),
                loads(Prefill, [0, 5, 9]),
                keyed("k1"),
            ],
        )
        .sequence(),
        // At the cap the policy's pick is at the key's cap too: it takes
        // the request and no pin moves.
        open(
            "sticky-pair-at-the-cap-a-pick-at-the-cap-too-moves-no-pin",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                held("k1", Prefill, [2, 2, 0]),
                loads(Prefill, [0, 3, 2]),
                keyed("k1"),
                held("k1", Prefill, [0, 0, 0]),
                keyed("k1"),
            ],
        )
        .sequence(),
        // A pinned worker that stops being routable: under `delegate` the
        // side is placed again and the new worker leads the key's pins for
        // good.
        open(
            "sticky-pair-delegate-a-prefill-down-moves-its-pin-for-good",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                Down(Prefill, 1),
                keyed("k1"),
                Up(Prefill, 1),
                keyed("k1"),
            ],
        )
        .sequence(),
        open(
            "sticky-pair-delegate-a-decode-down-moves-its-pin-for-good",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                Down(Decode, 1),
                keyed("k1"),
                Up(Decode, 1),
                keyed("k1"),
            ],
        )
        .sequence(),
        // The worker a side left stays behind its new one, among that
        // side's own pins: where the new one stops being routable in its
        // turn the side goes back, whatever the loads say then, and the
        // other side's pin is no part of it.
        open(
            "sticky-pair-delegate-a-prefill-down-again-goes-back-to-the-one-before",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                Down(Prefill, 1),
                keyed("k1"),
                Up(Prefill, 1),
                loads(Prefill, [1, 5, 0]),
                Down(Prefill, 0),
                keyed("k1"),
                Up(Prefill, 0),
                keyed("k1"),
            ],
        )
        .sequence(),
        open(
            "sticky-pair-delegate-a-decode-down-again-goes-back-to-the-one-before",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                Down(Decode, 1),
                keyed("k1"),
                Up(Decode, 1),
                loads(Decode, [1, 5, 0]),
                Down(Decode, 0),
                keyed("k1"),
                Up(Decode, 0),
                keyed("k1"),
            ],
        )
        .sequence(),
        // Under the map's own assignment the worker placed next joins
        // behind the first, which leads again once it is routable.
        open(
            "sticky-pair-min-load-a-prefill-down-lends-the-key-another-until-it-is-back",
            DRAWS,
            "min_load",
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                Down(Prefill, 1),
                keyed("k1"),
                Up(Prefill, 1),
                keyed("k1"),
            ],
        )
        .sequence(),
        open(
            "sticky-pair-min-load-a-decode-down-lends-the-key-another-until-it-is-back",
            DRAWS,
            "min_load",
            [1, 0, 2],
            [1, 0, 2],
            vec![
                keyed("k1"),
                Down(Decode, 1),
                keyed("k1"),
                Up(Decode, 1),
                keyed("k1"),
            ],
        )
        .sequence(),
        // The decode's pin is read among the picked prefill's partners
        // alone. The key's prefill moves to the other transport: its
        // pinned decode pairs with that prefill no more and the decode is
        // placed again.
        two_transports(
            "sticky-pair-a-pinned-decode-no-partner-of-the-new-prefill-is-placed-again",
            [0, 1],
            [0, 1, 2],
            vec![
                keyed("k1"),
                Down(Prefill, 0),
                keyed("k1"),
                Up(Prefill, 0),
                keyed("k1"),
            ],
        )
        .sequence(),
        // A pinned decode that pairs with both prefills stays.
        two_transports(
            "sticky-pair-a-pinned-decode-that-pairs-with-both-prefills-stays",
            [0, 1],
            [5, 5, 0],
            vec![
                keyed("k1"),
                Down(Prefill, 0),
                keyed("k1"),
                Up(Prefill, 0),
                keyed("k1"),
            ],
        )
        .sequence(),
        // A pinned prefill whose partners are all down is no candidate:
        // the key is placed again on the other transport.
        two_transports(
            "sticky-pair-a-pinned-prefill-with-no-open-partner-is-placed-again",
            [0, 1],
            [0, 1, 9],
            vec![
                keyed("k1"),
                Down(Decode, 0),
                Down(Decode, 2),
                keyed("k1"),
                Up(Decode, 0),
                Up(Decode, 2),
                keyed("k1"),
            ],
        )
        .sequence(),
        // A retry steps aside from the pair its failed attempt went to,
        // each side while another worker is left: the pins move with it.
        open(
            "sticky-pair-delegate-a-retry-steps-off-its-pair-and-moves-both-pins",
            COUNTS,
            DELEGATE,
            [1, 0, 2],
            [1, 0, 2],
            vec![keyed("k1"), retried("k1", 1, 1), keyed("k1")],
        )
        .sequence(),
        open(
            "sticky-pair-min-load-a-retry-steps-off-its-pair-for-that-request",
            DRAWS,
            "min_load",
            [1, 0, 2],
            [1, 0, 2],
            vec![keyed("k1"), retried("k1", 1, 1), keyed("k1")],
        )
        .sequence(),
        // A side under the hash policy reads the key itself: the override
        // places the other side alone.
        open(
            "sticky-pair-a-hash-prefill-is-the-rings-beside-a-pinned-decode",
            (RING, BY_COUNT),
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyed("k1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                keyed("k1"),
                keyed("k2"),
                keyed("k1"),
            ],
        )
        .sequence(),
        open(
            "sticky-pair-a-hash-decode-is-the-rings-beside-a-pinned-prefill",
            (BY_COUNT, RING),
            DELEGATE,
            [3, 1, 2],
            [2, 0, 1],
            vec![
                keyed("k1"),
                loads(Prefill, [0, 5, 2]),
                loads(Decode, [0, 5, 1]),
                keyed("k1"),
                keyed("k2"),
                keyed("k1"),
            ],
        )
        .sequence(),
    ]
}

#[test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
fn record_the_classic_pair_sticky_rows() {
    write("pair_sticky.json", "pair-sticky", rows());
}
