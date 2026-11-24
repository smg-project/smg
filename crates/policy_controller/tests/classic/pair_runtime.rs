//! The classic rows of a pair's placement on the wire that carries one
//! runtime's request, where the workers of one known runtime stand beside
//! workers of another or of none. They are rows of the pair placement's own
//! shape, on a sheet of their own, and replay through that module's replay
//! ([`pair_select`](super::pair_select)): the pair's decision of the engine
//! a document builds, over the first partners a row's runtimes state. No
//! row draws, and none is decided apart.
//!
//! The gateway keeps two known runtimes apart on that wire and decides one
//! order: a prefill of a known runtime with an open partner on that runtime
//! is a candidate ahead of every other prefill
//! (`model_gateway/src/routers/common/placement.rs:2479-2502`). It asks the
//! prefill admission's bound of those candidates alone (`:2532-2545`), and
//! takes the pick's decode among its partners on its runtime
//! (`:2615-2618`). Where no prefill has such a partner open, every prefill
//! with an open partner stands, with any of them. Its least-loaded default
//! reads every partner of a prefill alike (`:1804-1813`). The lines are
//! those of the commit the sheet was recorded from.

pub use super::pair_select::replay_rows;

/// What a sheet's head names these rows.
pub const POLICY: &str = "pair-runtime";

/// The rows replay answer for answer: the pair on one known runtime ahead
/// of every other, its wait at the prefill admission's bound, every pair
/// where no such pair is open, and the default's pair by load.
#[test]
fn the_runtime_rows_replay_answer_for_answer() {
    let (rows, found) = replay_rows(INLINE, "inline");
    assert_eq!(rows, 10);
    assert_eq!(found, Vec::<String>::new());
}

/// The recorded sheet's rows, whole.
pub const INLINE: &str = r#"{
  "recorded_from": "adf621bcfd7c1a7921435a008a58fc297eff6747",
  "policy": "pair-runtime",
  "rows": [
    {
      "name": "runtime-a-pair-on-one-known-runtime-goes-first",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "runtime-the-known-prefill-at-its-bound-keeps-the-request",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 4, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "runtime-the-known-prefill-not-ready-leaves-every-pair",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 0, "state": "down", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 0]}, {"pair": [0, 0]}]}
    },
    {
      "name": "runtime-the-known-prefill-over-leaves-every-pair",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 0, "state": "over", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 0]}, {"pair": [0, 0]}]}
    },
    {
      "name": "runtime-the-known-decode-not-ready-pairs-with-any-partner",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 5, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 0, "state": "down", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 0]}]}
    },
    {
      "name": "runtime-every-decode-over-keeps-the-known-pair",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "over", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "runtime-every-decode-over-under-shedding",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": true,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "over", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "overloaded"}}, {"none": {"role": "decode", "why": "overloaded"}}]}
    },
    {
      "name": "runtime-every-prefill-over-pairs-by-load-with-any-partner",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 5, "state": "over", "transport": null, "runtime": "unknown"},
        {"in_flight": 0, "state": "over", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 0]}]}
    },
    {
      "name": "runtime-two-known-runtimes-pair-apart",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 2, "state": "free", "transport": null, "runtime": "other"},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "other"}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 0]}]}
    },
    {
      "name": "runtime-a-known-prefill-with-no-partner-on-its-runtime-stands-behind",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "wire": "grpc",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "other"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    }
  ]
}"#;
