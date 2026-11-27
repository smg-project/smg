//! A pair plan over the fleet: a request at either phase is answered its
//! prefill for both sides.

use std::time::Instant;

use openai_protocol::worker::RuntimeType;
use smg_grand_central::{
    engine::Role,
    service::{
        client::Answer,
        proto::{self, decision::Outcome, DecideRequest, Decision, Side},
    },
};

use crate::{
    fleet::{within, Harness, Shape, GATEWAY, POLL_INTERVAL, PROBE_INTERVAL, ROOM},
    for_model, given_up, monitored, nth, row_of, Picked, ASKS, CASE, HELD,
};

/// The model the pair's engines serve.
const PAIRED: &str = "fleet-paired-model";

/// The pair's fleet: two engines prefill and two decode, each of one rank
/// on the first wire.
const PAIR_FLEET: [Shape; 4] = [
    Shape {
        model: PAIRED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: PAIRED,
        role: Role::Prefill,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: PAIRED,
        role: Role::Decode,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
    Shape {
        model: PAIRED,
        role: Role::Decode,
        ranks: 1,
        runtime: RuntimeType::Vllm,
    },
];

/// The pair plan's document, as an operator writes it: a request at either
/// phase of a pair is decided under the plan, each side by the least load.
const PAIR_PLAN: &str = r#"{
  "apiVersion": "smg.policy/v1",
  "profiles": {
    "side": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
  },
  "bindings": [
    {
      "match": { "phase": ["prefill", "decode"] },
      "plan": "pd-pair",
      "profiles": { "prefill": "side", "decode": "side" }
    }
  ]
}"#;

/// The `n`th ask of the pair's run: for the pair's model, at the plan's
/// prefill.
fn at_prefill(n: u64) -> DecideRequest {
    let mut ask = for_model(nth("pair", n), PAIRED);
    if let Some(facts) = &mut ask.facts {
        facts.phase = proto::Phase::Prefill.into();
    }
    ask
}

/// Asks at a pair's prefill, and reads the pick's first target as the two
/// rows of the fleet it names: the prefill side's, then the decode side's.
/// The answer carries a lease a side under one id, the prefill side's
/// first. Both are given up by that id, and have left the store's file
/// before the next ask reads the rows.
async fn paired(harness: &mut Harness, ask: DecideRequest) -> Result<(Picked, Picked), String> {
    let before = harness.leases(GATEWAY);
    let answer = harness.client().decide(ask).await;
    let Ok(Answer::Decision(
        Decision {
            leases,
            outcome: Some(Outcome::Pick(pick)),
            ..
        },
        _,
    )) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let sides: Vec<(u64, Side)> = leases
        .iter()
        .map(|lease| (lease.id, lease.side()))
        .collect();
    let &[(id, Side::Prefill), (decode, Side::Decode)] = sides.as_slice() else {
        return Err(format!(
            "a pair's pick carries a lease a side, the prefill side's first: {leases:?}"
        ));
    };
    if id != decode {
        return Err(format!("a pair's leases are one id's: {leases:?}"));
    }
    let target = pick.targets.first().ok_or("a pick with no target")?;
    let pair = target
        .pair
        .as_ref()
        .ok_or("a pair's target names no decode side")?;
    let rows = (
        row_of(harness, target.id, target.dp_rank)?,
        row_of(harness, pair.decode_id, pair.decode_dp_rank)?,
    );
    given_up(harness, &[id], before).await?;
    Ok(rows)
}

/// A pair plan answers a request that enters at its prefill once, for both
/// sides, over a fleet of its own: two engines that prefill and two that
/// decode. Each side's worker is the one whose engine reports no requests,
/// and the answer's two leases are one id's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pair_plan_answers_its_prefill_for_both_sides_by_the_loads_their_engines_told() {
    let case = async {
        let mut harness = Harness::start().await.expect("the harness starts");
        let engines = harness
            .register(&PAIR_FLEET)
            .await
            .expect("the pair's fleet registers");
        let (held_prefill, idle_prefill) = (engines[0], engines[1]);
        let (idle_decode, held_decode) = (engines[2], engines[3]);
        let sides = [(Role::Prefill, held_prefill), (Role::Decode, held_decode)];
        harness.feed();
        within(
            "each side's rows are healthy and hold a load report",
            PROBE_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                sides.iter().all(|&(side, _)| {
                    harness.pool(PAIRED, side).is_ok_and(|pool| {
                        pool.len() == 2 && pool.iter().all(|row| row.healthy && row.load.is_some())
                    })
                })
            },
        )
        .await
        .expect("the pair's fleet turns ready");

        // One engine of each side holds requests: its own report says so,
        // and no other row of its side reads them.
        for (_, held) in sides {
            harness
                .hold(held, HELD as usize)
                .await
                .expect("the held requests");
        }
        within(
            "each side's held engine reads its requests, and no other",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                sides.iter().all(|&(side, held)| {
                    harness.pool(PAIRED, side).is_ok_and(|pool| {
                        pool.iter().all(|row| {
                            row.load
                                .is_some_and(|load| (load.running == HELD) == (row.engine == held))
                        })
                    })
                })
            },
        )
        .await
        .expect("the loads follow on both sides");

        // A decision is given the load monitor's snapshot, where the pools
        // above are the store's table. The monitor polls each engine on its
        // own clock: it has polled all four since the table read so.
        let told = Instant::now();
        monitored(&harness, &engines, told)
            .await
            .expect("the monitor's reports follow");

        // The pair plan's document is applied in the case's process. A
        // request enters the plan at its prefill and is answered once, for
        // both sides, and neither names the engine that holds requests.
        // Unread, a side's loads leave its pick to a draw of two: 64 asks
        // miss it less than once in 10^19 runs.
        let document = serde_json::from_str(PAIR_PLAN).expect("a well-formed document");
        harness
            .apply_document(document)
            .expect("the pair plan's document applies");
        for n in 0..ASKS {
            let (prefill, decode) = paired(&mut harness, at_prefill(n))
                .await
                .expect("a pair's pick answers a request at its prefill");
            assert_eq!(
                prefill.engine, idle_prefill,
                "a pair's prefill side leaves the engine that holds requests: {prefill:?}"
            );
            assert_eq!(
                decode.engine, idle_decode,
                "a pair's decode side leaves the engine that holds requests: {decode:?}"
            );
        }
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}
