//! The classic rows of a pair whose request names its worker under
//! `x-smg-target-worker`. They are rows of the pair placement's own shape
//! ([`pair_select`](super::pair_select)) on a sheet of their own. Each
//! states the text its requests carry under that header (`target`) and,
//! where its requests are retries, the pair a failed attempt was sent to
//! (`tried`). No row draws.
//!
//! The gateway's hash policy alone reads the header, once a side, over the
//! workers the placement hands that side's policy
//! (`model_gateway/src/routers/common/placement.rs:2900-2901`,
//! `:2960-2966`; `model_gateway/src/policies/consistent_hashing.rs:150-162`).
//! The prefill side is handed the prefills that can be sent to and have a
//! partner that can (`:2800-2817`), and the decode side the partners of
//! the picked prefill that can be sent to (`:2930-2933`). A side under that
//! policy is narrowed by no admission filter and by no retry
//! (`:2850-2860`, `:2872`, `:2941`, `:2948`). A number that reaches no
//! worker of a side is the policy's refusal (`:2919`, `:2965`). A pool over
//! its thresholds on every worker is paired by load, and no header is read
//! for it (`:2699-2701`, `:2775-2793`, `:2955-2958`). The lines are those
//! of the commit the sheet was recorded from.
//!
//! The replay places each request through the one call of a host in
//! process ([`InProcess::place`]), under the engine a document builds of
//! the row's two presets. A row's workers are worker objects in the row's
//! state. They are lent as a pair's two lists, with the classes of the
//! row's partner lists, each prefill's first partners on the wire that
//! carries one runtime's request, and a host's admission that answers the
//! row's bound and rooms. A retry states the two workers its failed attempt
//! was sent to. Of a refusal the replay holds why and the role, as the
//! pair placement's replay does: the engine sheds a number that reaches no
//! worker under `target-worker`, which is the gateway's `declined`, on the
//! side the worker was missed on, which is the role of the policy that
//! refused. The wait at the prefill admission's bound is the prefill's.

use std::sync::Arc;

use serde::Deserialize;
use smg_grand_central::{
    document::{AppliedSpec, OverloadSpec, PolicySpec},
    engine::{
        InProcess, LentList, ListCell, PairAdmission, PairLists, PairSide, PartnerClasses,
        PlaceRequest, ShedReason, Unplaced, WorkerId, WorkerLists,
    },
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus, WorkerType},
};

/// What a sheet's head names these rows.
pub const POLICY: &str = "pair-named";

/// The model every worker of a row serves.
const MODEL: &str = "m";

/// The header a request names its worker under.
const TARGET: &str = "x-smg-target-worker";

/// The stage a request is shed under where the worker it names cannot take
/// it.
const TARGET_STAGE: &str = "target-worker";

/// The policy that takes the fewest requests in flight, with no draw and no
/// state, in the gateway's spelling.
const FEWEST: &str = "power_of_two:least_requests";

/// The gateway's hash policy: the one that reads the header.
const RING: &str = "consistent_hashing";

/// What the router holds of a worker besides its requests.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Not ready.
    Down,
}

/// A decode's running window and the rooms of it the admission has claimed.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Rooms {
    window: u32,
    admitted: u32,
}

/// One worker of a row.
#[derive(Clone, Debug, Deserialize)]
struct Member {
    in_flight: u32,
    state: State,
    transport: Option<String>,
    #[serde(default)]
    rooms: Option<Rooms>,
    /// `other` or `unknown`, where it is not on the sheet's one runtime.
    #[serde(default)]
    runtime: Option<String>,
}

impl Member {
    /// The runtime the worker is known to be on, none for `unknown`.
    fn runtime(&self) -> Option<&str> {
        match self.runtime.as_deref() {
            None => Some("own"),
            Some("unknown") => None,
            named => named,
        }
    }
}

/// The gateway's answer to one request: the pair, the prefill's place and
/// then the decode's, or no pair, with the role it refuses for (`prefill`
/// or `decode`) and why.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    Pair([usize; 2]),
    None { role: String, why: String },
}

/// What a row recorded: each request's answer, in order.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Sequence(Vec<Answer>),
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    prefill_policy: String,
    decode_policy: String,
    /// Whether a pool over on every worker is shed: the binding's
    /// `overload.shed_when_all_vetoed`.
    shed: bool,
    /// The most requests the prefill admission lets one prefill hold.
    #[serde(default)]
    prefill_bound: Option<u32>,
    /// `grpc`, where the pair is placed on the wire that carries one
    /// runtime's request.
    #[serde(default)]
    wire: Option<String>,
    /// The text the row's requests carry under the header.
    target: String,
    /// The pair a failed attempt of each request was sent to, where the
    /// row's requests are retries: the prefill's place, then the decode's.
    #[serde(default)]
    tried: Option<[usize; 2]>,
    prefills: Vec<Member>,
    decodes: Vec<Member>,
    requests: usize,
    outcome: Outcome,
}

#[derive(Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// The engine of a row, as a host holds it: its two presets under the pair
/// plan, a side that takes the fewest requests picking by count with no
/// draw, and the row's word for a pool vetoed on every worker. A side under
/// the hash policy brings the binding its target source.
fn engine(row: &Row) -> Result<&'static InProcess, String> {
    let preset = |policy: &str| match policy {
        FEWEST => Ok("power_of_two"),
        RING => Ok(RING),
        other => Err(format!("a policy this replay does not build: {other}")),
    };
    let (prefill, decode) = (preset(&row.prefill_policy)?, preset(&row.decode_policy)?);
    let mut spec = PolicySpec::pair_presets(prefill, decode).map_err(|err| err.to_string())?;
    spec.generation = 1;
    for (side, policy) in [("prefill", prefill), ("decode", decode)] {
        if policy != RING {
            let profile = spec.profiles.get_mut(side).ok_or("a side has no profile")?;
            profile.pick = "count-first".into();
        }
    }
    let binding = spec
        .bindings
        .first_mut()
        .ok_or("the presets bind nothing")?;
    binding.overload = Some(OverloadSpec {
        shed_when_all_vetoed: row.shed,
        ..OverloadSpec::default()
    });
    let applied = AppliedSpec::validate(&spec).map_err(|err| err.to_string())?;
    // An engine as its host holds it: for the life of the process.
    Ok(Box::leak(Box::new(InProcess::new(applied))))
}

/// One worker object of a row's list, in the row's state, with the requests
/// the router holds on it as its object's own count.
fn object(side: &str, at: usize, role: WorkerType, member: &Member) -> Arc<dyn Worker> {
    let object = <BasicWorkerBuilder>::new(format!("http://{side}-{at}:8000"))
        .worker_type(role)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        })
        .build();
    object.set_status(match member.state {
        State::Down => WorkerStatus::NotReady,
        State::Free | State::Over => WorkerStatus::Ready,
    });
    if member.state == State::Over {
        object.set_overloaded(true);
    }
    for _ in 0..member.in_flight {
        object.increment_load();
    }
    Arc::new(object)
}

/// Each prefill's partners as places of the decode list, ascending, and
/// those of them it pairs with first: its partners on its own known
/// runtime, on the wire that carries one runtime's request. Two workers
/// pair when their transports are equal or one of them names none, and
/// never on two known runtimes.
fn partners(row: &Row) -> Vec<[Vec<u32>; 2]> {
    let apart = row.wire.as_deref() == Some("grpc");
    let known = |one: Option<&str>, other: Option<&str>| match (one, other) {
        (Some(ours), Some(theirs)) => ours == theirs,
        _ => true,
    };
    (row.prefills.iter())
        .map(|prefill| {
            let pairs = |decode: &Member| {
                known(prefill.transport.as_deref(), decode.transport.as_deref())
                    && known(prefill.runtime(), decode.runtime())
            };
            let own = prefill.runtime().filter(|_| apart);
            let places = |first: bool| -> Vec<u32> {
                (0u32..)
                    .zip(&row.decodes)
                    .filter(|(_, decode)| pairs(decode))
                    .filter(|(_, decode)| !first || (own.is_some() && decode.runtime() == own))
                    .map(|(place, _)| place)
                    .collect()
            };
            [places(false), places(true)]
        })
        .collect()
}

/// A row's host: the prefill admission's bound and the decodes' rooms, as
/// the row states them.
#[derive(Debug)]
struct Gates<'r>(&'r Row);

impl PairAdmission for Gates<'_> {
    fn has_capacity(&self, prefill: usize) -> bool {
        let Self(row) = self;
        (row.prefill_bound).is_none_or(|bound| row.prefills[prefill].in_flight < bound)
    }

    fn has_room(&self, decode: usize) -> bool {
        let Self(row) = self;
        (row.decodes[decode].rooms).is_none_or(|rooms| rooms.admitted < rooms.window)
    }
}

/// What this crate answers one request of a row: the pair, or why none
/// and for which side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Here {
    Pair([usize; 2]),
    None(&'static str, Option<PairSide>),
}

/// The side the call's answer names: a shed's own, and the prefills' for
/// the wait at the prefill admission's bound, which alone left the request
/// no prefill.
fn side(unplaced: Unplaced) -> Option<PairSide> {
    match unplaced {
        Unplaced::PrefillsFull => Some(PairSide::Prefill),
        Unplaced::Shed { side, .. } => side,
        _ => None,
    }
}

/// Why the gateway answers no pair, read from the call's answer: the wait
/// at the prefill admission's bound, an overload shed, the refusal of a
/// number that reaches no worker, and `unavailable` for every other.
fn why(unplaced: Unplaced) -> &'static str {
    match unplaced {
        Unplaced::PrefillsFull => "at_capacity",
        Unplaced::Shed {
            reason: ShedReason::Saturated,
            ..
        } => "overloaded",
        Unplaced::Shed {
            emptied_by: TARGET_STAGE,
            ..
        } => "declined",
        _ => "unavailable",
    }
}

/// The row's requests placed one after another through the one call. A
/// placement is let go unsent, so the next request meets the lists as the
/// row states them.
fn run(row: &Row) -> Result<Vec<Here>, String> {
    let engine = engine(row)?;
    let list = |side: &str, role: WorkerType, members: &[Member]| -> Arc<[Arc<dyn Worker>]> {
        (members.iter().enumerate())
            .map(|(at, member)| object(side, at, role, member))
            .collect()
    };
    let prefills = list("prefill", WorkerType::Prefill, &row.prefills);
    let decodes = list("decode", WorkerType::Decode, &row.decodes);
    let paired = partners(row);
    let classes = PartnerClasses::number_first(
        decodes.len(),
        paired.iter().map(|[all, first]| (&all[..], &first[..])),
    );
    let (prefill_cell, decode_cell) = (ListCell::default(), ListCell::default());
    let gates = Gates(row);
    let lists = PairLists {
        prefill: LentList {
            workers: &prefills,
            cell: &prefill_cell,
            facts: None,
        },
        decode: LentList {
            workers: &decodes,
            cell: &decode_cell,
            facts: None,
        },
        classes: &classes,
        admission: &gates,
        prefill_places: None,
        decode_places: None,
        prefill_stalled: &[],
        prefill_held: &[],
        decode_stalled: &[],
        decode_held: &[],
    };
    let headers = [(TARGET, row.target.as_str())];
    let tried: Vec<WorkerId> = row.tried.map_or_else(Vec::new, |[prefill, decode]| {
        [&prefills[prefill], &decodes[decode]]
            .map(|worker| WorkerId::from_address(worker.url()))
            .to_vec()
    });
    let request = PlaceRequest {
        headers: &headers,
        tried: &tried,
        ..PlaceRequest::of(MODEL, WorkerLists::Pair(lists))
    };
    let at = |list: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>| {
        let place = list.iter().position(|listed| Arc::ptr_eq(listed, worker));
        place.ok_or("a placement names a worker of no list")
    };
    (0..row.requests)
        .map(|_| match engine.place(&request) {
            Ok(placement) => {
                let decode = placement.decode.as_ref().ok_or("a pair with no decode")?;
                Ok(Here::Pair([
                    at(&prefills, &placement.worker)?,
                    at(&decodes, decode)?,
                ]))
            }
            Err(unplaced) => Ok(Here::None(why(unplaced), side(unplaced))),
        })
        .collect::<Result<Vec<Here>, &str>>()
        .map_err(str::to_string)
}

/// Whether an answer of this crate is the recorded one: the pair, or no
/// pair for the recorded reason and for the recorded role, which is the
/// side the answer names.
fn same(recorded: &Answer, here: &Here) -> bool {
    match (recorded, here) {
        (Answer::Pair(pair), Here::Pair(ours)) => pair == ours,
        (Answer::None { role, why }, Here::None(ours, side)) => {
            why == ours && side.is_some_and(|side| side.as_str() == role)
        }
        _ => false,
    }
}

/// What of one row's outcome this crate does not replay: nothing when its
/// answers are the recorded ones.
fn replay(row: &Row) -> Vec<String> {
    let here = match run(row) {
        Ok(here) => here,
        Err(refused) => return vec![format!("refused here: {refused}")],
    };
    let Outcome::Sequence(recorded) = &row.outcome;
    if recorded.len() != here.len() {
        return vec![format!("recorded {recorded:?}, replayed {here:?}")];
    }
    (recorded.iter().zip(&here).enumerate())
        .filter(|(_, (recorded, here))| !same(recorded, here))
        .map(|(at, (recorded, here))| {
            format!("request {at}: recorded {recorded:?}, replayed {here:?}")
        })
        .collect()
}

/// The rows of a sheet replayed: how many, and what did not replay.
pub fn replay_rows(text: &str, source: &str) -> (usize, Vec<String>) {
    let sheet: Sheet = match serde_json::from_str(text) {
        Ok(sheet) => sheet,
        Err(err) => {
            return (
                0,
                vec![format!("{source} is not a sheet of {POLICY} rows: {err}")],
            )
        }
    };
    let defects = sheet
        .rows
        .iter()
        .flat_map(|row| {
            let name = &row.name;
            replay(row)
                .into_iter()
                .map(move |why| format!("{source}: {name}: {why}"))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// The rows replay answer for answer: each side's count of the workers that
/// can be sent to, the refusal of a number that reaches none, the named
/// prefill's wait at the admission's bound, the named decode taken with no
/// free room, the pair by load where a pool is over its thresholds whole, a
/// retry handed the named worker again, and the decode counted behind the
/// prefill its own policy picks.
#[test]
fn the_named_rows_replay_answer_for_answer() {
    let (rows, found) = replay_rows(INLINE, "inline");
    assert_eq!(rows, 23);
    assert_eq!(found, Vec::<String>::new());
}

/// The recorded sheet's rows, whole.
pub const INLINE: &str = r#"{
  "recorded_from": "4541c6c312394e13b49ae89b91c7321a78ddd270",
  "policy": "pair-named",
  "rows": [
    {
      "name": "named-both-each-side-counts-the-workers-that-can-be-sent-to",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "1",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "down", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "down", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [2, 1]}, {"pair": [2, 1]}]}
    },
    {
      "name": "named-both-a-prefill-over-its-thresholds-is-not-counted",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 0, "state": "over", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 0]}]}
    },
    {
      "name": "named-both-a-number-past-the-prefills-is-refused",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "2",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "declined"}}, {"none": {"role": "prefill", "why": "declined"}}]}
    },
    {
      "name": "named-both-a-number-past-the-partners-is-refused",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "2",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "declined"}}, {"none": {"role": "decode", "why": "declined"}}]}
    },
    {
      "name": "named-both-a-text-that-is-no-number-is-refused",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "first",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "declined"}}, {"none": {"role": "prefill", "why": "declined"}}]}
    },
    {
      "name": "named-both-the-named-prefill-at-its-bound-keeps-the-request",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "prefill_bound": 4,
      "target": "0",
      "prefills": [
        {"in_flight": 4, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "at_capacity"}}, {"none": {"role": "prefill", "why": "at_capacity"}}]}
    },
    {
      "name": "named-both-the-named-decode-with-no-free-room-is-taken",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 2}},
        {"in_flight": 0, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 0}}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 0]}, {"pair": [0, 0]}]}
    },
    {
      "name": "named-both-every-decode-over-goes-by-load",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "over", "transport": null},
        {"in_flight": 1, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 1]}, {"pair": [0, 1]}]}
    },
    {
      "name": "named-both-every-decode-over-under-shedding",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": true,
      "target": "0",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "over", "transport": null},
        {"in_flight": 1, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "overloaded"}}, {"none": {"role": "decode", "why": "overloaded"}}]}
    },
    {
      "name": "named-both-every-prefill-over-pairs-by-load",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 5, "state": "over", "transport": null},
        {"in_flight": 0, "state": "over", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "named-both-a-retry-keeps-the-named-pair",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "1",
      "tried": [1, 1],
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "named-prefill-the-decode-is-its-own-policys-pick",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "target": "2",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [2, 1]}, {"pair": [2, 1]}]}
    },
    {
      "name": "named-prefill-a-number-past-the-prefills-is-refused",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "target": "3",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "declined"}}, {"none": {"role": "prefill", "why": "declined"}}]}
    },
    {
      "name": "named-prefill-the-decode-side-keeps-the-decodes-with-a-room",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "target": "1",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 0}},
        {"in_flight": 1, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 2}},
        {"in_flight": 2, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 0}}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [1, 2]}]}
    },
    {
      "name": "named-prefill-a-retry-keeps-the-prefill-and-leaves-the-decode",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "target": "0",
      "tried": [0, 1],
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 2]}, {"pair": [0, 2]}]}
    },
    {
      "name": "named-decode-is-counted-behind-the-prefill-its-policy-picks",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "1",
      "prefills": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "down", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [1, 2]}]}
    },
    {
      "name": "named-decode-is-counted-among-the-picks-partners",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "1",
      "prefills": [
        {"in_flight": 2, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": "a"}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [1, 2]}]}
    },
    {
      "name": "named-decode-a-number-past-the-picks-partners-is-refused",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "2",
      "prefills": [
        {"in_flight": 2, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": "a"}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "declined"}}, {"none": {"role": "decode", "why": "declined"}}]}
    },
    {
      "name": "named-decode-every-decode-over-goes-by-load",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 3, "state": "over", "transport": null},
        {"in_flight": 1, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "named-decode-the-named-decode-with-no-free-room-is-taken",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 2}},
        {"in_flight": 0, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 0}}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 0]}]}
    },
    {
      "name": "named-decode-is-counted-among-the-first-partners",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "wire": "grpc",
      "target": "1",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 3, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null, "runtime": "unknown"},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [1, 2]}]}
    },
    {
      "name": "named-decode-a-retry-leaves-the-prefill-and-keeps-the-decode",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "tried": [0, 0],
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 0]}]}
    },
    {
      "name": "named-decode-no-open-prefill-pairs-by-load",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "shed": false,
      "target": "0",
      "prefills": [
        {"in_flight": 0, "state": "over", "transport": "a"},
        {"in_flight": 0, "state": "down", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "over", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 0]}, {"pair": [0, 0]}]}
    }
  ]
}"#;
