//! The classic rows of a pair's placement behind the prefill admission,
//! where a prefill that holds the admission's bound has an open partner or
//! has none. They are rows of the pair placement's own shape, on a sheet of
//! their own, and replay through that module's replay
//! ([`pair_select`](super::pair_select)): the pair's decision of the engine
//! a document builds. No row draws, and none is decided apart.
//!
//! The gateway asks the bound of the prefills that still have a partner to
//! go to, and of no other
//! (`model_gateway/src/routers/common/placement.rs:2462-2485`). Where no
//! open prefill has an open partner it answers no pair, ahead of the bound
//! (`:2486-2513`). Where each open prefill with an open partner holds its
//! bound it answers `at_capacity`, on which a route waits, whatever room a
//! prefill with no open partner has (`:2515-2528`). Its least-loaded
//! default asks the bound of the one prefill it took (`:1827-1832`). The
//! lines are those of the commit the sheet was recorded from.

pub use super::pair_select::replay_rows;

/// What a sheet's head names these rows.
pub const POLICY: &str = "pair-bound";

/// The rows replay answer for answer: no pair where no prefill at its bound
/// has an open partner, the wait where one has, and the pair of a prefill
/// with room and an open partner.
#[test]
fn the_bound_rows_replay_answer_for_answer() {
    let (rows, found) = replay_rows(INLINE, "inline");
    assert_eq!(rows, 8);
    assert_eq!(found, Vec::<String>::new());
}

/// The recorded sheet's rows, whole.
pub const INLINE: &str = r#"{
  "recorded_from": "7526c6664e4916620fad715f204d35968568e3ee",
  "policy": "pair-bound",
  "rows": [
    {
      "name": "bound-every-prefill-at-capacity-and-no-decode-ready",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 5, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "down", "transport": "a"},
        {"in_flight": 0, "state": "down", "transport": "b"},
        {"in_flight": 0, "state": "down", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "unavailable"}}, {"none": {"role": "decode", "why": "unavailable"}}]}
    },
    {
      "name": "bound-the-prefill-at-capacity-has-the-only-open-partner",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "down", "transport": "b"},
        {"in_flight": 0, "state": "down", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "bound-the-second-prefill-at-capacity-has-the-only-open-partner",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 4, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "down", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "down", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "bound-every-prefill-at-capacity-and-every-decode-over",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 5, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "over", "transport": "a"},
        {"in_flight": 1, "state": "over", "transport": "b"},
        {"in_flight": 0, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "bound-every-prefill-at-capacity-and-every-decode-over-under-shedding",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": true,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 5, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "over", "transport": "a"},
        {"in_flight": 1, "state": "over", "transport": "b"},
        {"in_flight": 0, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "overloaded"}}, {"none": {"role": "decode", "why": "overloaded"}}]}
    },
    {
      "name": "bound-every-prefill-at-capacity-and-every-decode-stalled",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 5, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "stalled", "transport": "a"},
        {"in_flight": 1, "state": "stalled", "transport": "b"},
        {"in_flight": 0, "state": "stalled", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "bound-the-only-ready-prefill-at-capacity",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "down", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "bound-a-prefill-with-room-and-an-open-partner-takes-the-request",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    }
  ]
}"#;
