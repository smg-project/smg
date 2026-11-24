//! The classic rows of a pair's placement: a prefill pool and a decode
//! pool, the policy each side runs under, whether shedding is on, and a
//! number of requests. Each worker has the requests a router holds on it,
//! its state as the all-vetoed rows name it, and the transport its
//! registration names: two workers pair when their transports are equal or
//! one of them names none. A decode may report a running window with the
//! rooms of it the decode admission has claimed, and a row may run behind
//! the prefill admission, which lets one prefill hold a bound of requests.
//! A worker that is not on the sheet's one runtime names its own, a second
//! known runtime or none: two workers of two known runtimes never pair. A
//! row placed on the wire that carries one runtime's request (`grpc`) pairs
//! a prefill of a known runtime first with its partners on that runtime
//! ([`FirstPartners`]).
//!
//! A row's requests are placed one after another and none changes what the
//! router holds. An answer is the pair, the prefill's place in its list and
//! then the decode's, or no pair with the role the gateway named and why.
//! A drawn row counts how often each pair took its one request.
//!
//! The replay asks the pair's decision of the engine a document builds:
//! each side under its preset, the binding's word for a pool vetoed on
//! every worker, and each worker's bits as a host writes them. A host
//! answers two lists of which one is empty over no worker, and so does the
//! replay. Of a refusal the replay holds why, and not the role: a shed
//! names the stage that left nobody, and the gateway the pool its search
//! failed on.
//!
//! [`APART`] names each recorded row this crate decides apart from the
//! gateway, with its answer here.

use serde::Deserialize;
use serde_json::Value;
use smg_policy_controller::{
    document::{OverloadSpec, PolicySpec},
    engine::{
        Candidate, Constraints, Decision, Engine, FirstPartners, Health, PairPools, PairTable,
        PoolFacts, RequestFacts, Role, Scratch, ShedReason, WorkerId,
    },
    policies::power_of_two::PowerOfTwoMetric,
};

use crate::{same_shares, shares};

/// What a sheet's head names these rows.
pub const POLICY: &str = "pair-select";

/// The policy that takes the fewest requests in flight, with no draw and no
/// state, in the gateway's spelling: a row that must replay pick for pick
/// runs a side under it.
const FEWEST: &str = "power_of_two:least_requests";

/// What the router holds of a worker besides its requests.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Ready, under its thresholds, doubted by the liveness tracker.
    Stalled,
    /// Not ready.
    Down,
}

impl State {
    fn health(self) -> Health {
        Health {
            healthy: self != Self::Down,
            overloaded: self == Self::Over,
            stalled: self == Self::Stalled,
            ..Health::default()
        }
    }
}

/// A decode's running window and the rooms of it the admission has claimed.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Rooms {
    window: u32,
    admitted: u32,
}

/// One worker of a row.
#[derive(Clone, Debug, Deserialize)]
struct Worker {
    in_flight: u32,
    state: State,
    transport: Option<String>,
    #[serde(default)]
    rooms: Option<Rooms>,
    /// `other` or `unknown`, where it is not on the sheet's one runtime.
    #[serde(default)]
    runtime: Option<String>,
}

impl Worker {
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
/// then the decode's, or no pair.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Answer {
    Pair([usize; 2]),
    None { why: String },
}

/// What a row recorded: each request's answer in order, or how often each
/// pair took a drawn row's one request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Sequence(Vec<Answer>),
    Counts {
        draws: u32,
        pairs: Vec<(usize, usize, u32)>,
    },
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
    prefills: Vec<Worker>,
    decodes: Vec<Worker>,
    requests: usize,
    outcome: Outcome,
}

#[derive(Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// The preset of a side's policy, and whether the side runs it in the form
/// that takes the fewest requests.
fn preset(policy: &str) -> Result<(&str, bool), String> {
    match policy {
        FEWEST => Ok(("power_of_two", true)),
        "random" | "cache_aware" => Ok((policy, false)),
        other => Err(format!("a policy this replay does not build: {other}")),
    }
}

/// The engine of a row: its two presets under the pair plan, a side that
/// takes the fewest requests in its metric's form, and the row's word for a
/// pool vetoed on every worker.
fn engine(row: &Row) -> Result<Engine, String> {
    let (prefill, prefill_fewest) = preset(&row.prefill_policy)?;
    let (decode, decode_fewest) = preset(&row.decode_policy)?;
    let mut spec = PolicySpec::pair_presets(prefill, decode).map_err(|err| err.to_string())?;
    let fewest = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
        .map_err(|err| err.to_string())?
        .profiles
        .remove("power_of_two")
        .ok_or("the preset in its metric's form has no profile")?;
    for (side, takes_fewest) in [("prefill", prefill_fewest), ("decode", decode_fewest)] {
        if takes_fewest {
            let profile = spec.profiles.get_mut(side).ok_or("a side has no profile")?;
            (profile.score, profile.pick) = (fewest.score.clone(), fewest.pick.clone());
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
    spec.validate().map_err(|err| err.to_string())
}

/// One worker's row in a pool of the replay, every admission bit set.
fn candidate(at: usize, first: u64, role: Role, worker: &Worker) -> Candidate {
    let id = first + at as u64;
    Candidate {
        position: at as u32,
        id: WorkerId(id),
        role,
        health: worker.state.health(),
        in_flight: worker.in_flight,
        has_capacity: true,
        has_room: true,
        key: id as u32,
        ..Candidate::default()
    }
}

/// The pair classes of a row: each prefill is a class of its own, each
/// decode a class numbered as the prefills' lists first name it (`0` for a
/// decode no prefill pairs with), and the table's two slices follow. The
/// last two slices are each prefill's first partners, laid out as the
/// table's: its partners on its own known runtime, on the wire that carries
/// one runtime's request. They are empty where no prefill has one.
struct Pairing {
    decode_classes: Vec<u32>,
    ends: Vec<u32>,
    listed: Vec<u32>,
    first_ends: Vec<u32>,
    first_listed: Vec<u32>,
}

fn pairing(row: &Row) -> Pairing {
    let mut decode_classes = vec![0u32; row.decodes.len()];
    let mut named = 0u32;
    let (mut ends, mut listed) = (Vec::new(), Vec::new());
    let (mut first_ends, mut first_listed) = (Vec::new(), Vec::new());
    let apart = row.wire.as_deref() == Some("grpc");
    for prefill in &row.prefills {
        let pairs = |decode: &Worker| {
            let known = |one: Option<&str>, other: Option<&str>| match (one, other) {
                (Some(ours), Some(theirs)) => ours == theirs,
                _ => true,
            };
            known(prefill.transport.as_deref(), decode.transport.as_deref())
                && known(prefill.runtime(), decode.runtime())
        };
        let own = prefill.runtime().filter(|_| apart);
        let mut list: Vec<u32> = (row.decodes.iter().zip(&mut decode_classes))
            .filter(|(decode, _)| pairs(decode))
            .map(|(_, class)| {
                if *class == 0 {
                    named += 1;
                    *class = named;
                }
                *class
            })
            .collect();
        list.sort_unstable();
        listed.extend(list);
        ends.push(listed.len() as u32);
        let mut first: Vec<u32> = (row.decodes.iter().zip(&decode_classes))
            .filter(|(decode, _)| own.is_some() && decode.runtime() == own && pairs(decode))
            .map(|(_, class)| *class)
            .collect();
        first.sort_unstable();
        first_listed.extend(first);
        first_ends.push(first_listed.len() as u32);
    }
    if first_listed.is_empty() {
        first_ends.clear();
    }
    Pairing {
        decode_classes,
        ends,
        listed,
        first_ends,
        first_listed,
    }
}

/// Why the gateway answers no pair, read from a shed: an overload shed,
/// the prefill admission's bound, a role with no worker at all, and
/// `unavailable` for every other.
fn why(reason: ShedReason, stage: &str) -> &'static str {
    match (reason, stage) {
        (ShedReason::Saturated, _) => "overloaded",
        (_, "has-capacity") => "at_capacity",
        (_, "none") => "no_candidates",
        _ => "unavailable",
    }
}

/// What this crate answers one request of a row: the pair, or why none and
/// the stage its shed names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Here {
    Pair([usize; 2]),
    None(&'static str, &'static str),
}

/// The row's requests decided one after another, with `draw` separating the
/// seeds of a drawn row's runs.
fn run(row: &Row, engine: &Engine, draw: u64) -> Result<Vec<Here>, String> {
    let paired = pairing(row);
    let prefills: Vec<Candidate> = (row.prefills.iter().enumerate())
        .map(|(at, worker)| Candidate {
            pair_class: at as u32 + 1,
            has_capacity: (row.prefill_bound).is_none_or(|bound| worker.in_flight < bound),
            ..candidate(at, 1_000, Role::Prefill, worker)
        })
        .collect();
    let decodes: Vec<Candidate> = (row.decodes.iter().enumerate())
        .map(|(at, worker)| Candidate {
            pair_class: paired.decode_classes[at],
            has_room: (worker.rooms).is_none_or(|rooms| rooms.admitted < rooms.window),
            ..candidate(at, 2_000, Role::Decode, worker)
        })
        .collect();
    let table = PairTable::new(&paired.ends, &paired.listed);
    table
        .validate()
        .map_err(|clause| format!("the row's pair table: {clause}"))?;
    let first = FirstPartners::new(&paired.first_ends, &paired.first_listed);
    first
        .within(&table)
        .map_err(|clause| format!("the row's first partners: {clause}"))?;
    // One side alone is no pair: a host answers such lists over no worker.
    let (pools, pool_size) = if prefills.is_empty() || decodes.is_empty() {
        (PairPools::default(), 0)
    } else {
        let pools = PairPools {
            prefills: &prefills,
            decodes: &decodes,
            table,
            first,
            ..PairPools::default()
        };
        (pools, prefills.len() as u32)
    };
    let pool_facts = PoolFacts {
        pool_size,
        generation: 1,
        ..PoolFacts::default()
    };
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    Ok((0..row.requests)
        .map(|at| {
            let facts = RequestFacts {
                model: "m",
                seed: draw * 1_000 + at as u64 + 1,
                ..RequestFacts::default()
            };
            match engine.decide_pair(&facts, &pool_facts, pools, &cx, &cx, &mut scratch) {
                Decision::Pick(pick) => {
                    let target = pick.targets.first().expect("a target");
                    let decode = target.pair.expect("a pair").decode_position;
                    Here::Pair([target.position as usize, decode as usize])
                }
                Decision::Shed(shed) => {
                    Here::None(why(shed.reason, shed.emptied_by), shed.emptied_by)
                }
                Decision::Queue(_) => Here::None("queued", "queue"),
            }
        })
        .collect())
}

/// What this crate answers a row: each request's answer in order, or how
/// often each answer was given to a drawn row's one request.
#[derive(Debug)]
enum Replayed {
    Sequence(Vec<Here>),
    Counts(Vec<(Here, u32)>),
}

/// The row replayed in the form its outcome was recorded in, or the words
/// this crate refuses its form in.
fn replayed(row: &Row) -> Result<Replayed, String> {
    let engine = engine(row)?;
    let Outcome::Counts { draws, .. } = &row.outcome else {
        return run(row, &engine, 0).map(Replayed::Sequence);
    };
    let mut counts: Vec<(Here, u32)> = Vec::new();
    for draw in 0..u64::from(*draws) {
        let [here] = run(row, &engine, draw)?[..] else {
            return Err("a drawn row places one request".to_string());
        };
        match counts.iter_mut().find(|(answer, _)| *answer == here) {
            Some((_, count)) => *count += 1,
            None => counts.push((here, 1)),
        }
    }
    Ok(Replayed::Counts(counts))
}

/// Whether an answer of this crate is the recorded one: the pair, or no
/// pair for the recorded reason.
fn same(recorded: &Answer, here: &Here) -> bool {
    match (recorded, here) {
        (Answer::Pair(pair), Here::Pair(ours)) => pair == ours,
        (Answer::None { why }, Here::None(ours, _)) => why == ours,
        _ => false,
    }
}

/// What separates this crate's answers from the recorded ones: a sequence
/// answer for answer, a drawn row pair for pair and share for share.
fn differences(recorded: &Outcome, here: &Replayed) -> Vec<String> {
    match (recorded, here) {
        (Outcome::Sequence(recorded), Replayed::Sequence(here)) if recorded.len() == here.len() => {
            (recorded.iter().zip(here).enumerate())
                .filter(|(_, (recorded, here))| !same(recorded, here))
                .map(|(at, (recorded, here))| {
                    format!("request {at}: recorded {recorded:?}, replayed {here:?}")
                })
                .collect()
        }
        (Outcome::Counts { draws, pairs }, Replayed::Counts(here)) => {
            let listed = |answer: &Here| {
                pairs
                    .iter()
                    .any(|&(prefill, decode, _)| *answer == Here::Pair([prefill, decode]))
            };
            let mut found: Vec<String> = (here.iter())
                .filter(|(answer, _)| !listed(answer))
                .map(|(answer, count)| {
                    format!("{count} of {draws} draws replayed {answer:?}, which no draw recorded")
                })
                .collect();
            let count_of = |prefill, decode| {
                let pair = Here::Pair([prefill, decode]);
                let counted = here.iter().find(|(answer, _)| *answer == pair);
                counted.map_or(0, |&(_, count)| count)
            };
            let (recorded, ours): (Vec<u32>, Vec<u32>) = pairs
                .iter()
                .map(|&(prefill, decode, count)| (count, count_of(prefill, decode)))
                .unzip();
            if !same_shares(&shares(&ours, *draws), &shares(&recorded, *draws), *draws) {
                found.push(format!(
                    "recorded {pairs:?} of {draws} draws, replayed those pairs {ours:?} times"
                ));
            }
            found
        }
        (recorded, here) => vec![format!("recorded {recorded:?}, replayed {here:?}")],
    }
}

/// What an entry of [`APART`] holds its row to: every request of the row,
/// in every draw, answered no pair, and why.
#[derive(Clone, Copy, Debug)]
struct Held(&'static str);

impl Held {
    fn is(self, here: &Replayed) -> bool {
        let Self(why) = self;
        let holds = |answer: &Here| matches!(answer, Here::None(ours, _) if why == *ours);
        match here {
            Replayed::Sequence(answers) => answers.iter().all(holds),
            Replayed::Counts(counts) => counts.iter().all(|(answer, _)| holds(answer)),
        }
    }
}

/// A recorded row this crate decides apart from the gateway, by a rule of
/// its own or for want of one of the gateway's: what it answers every
/// request of the row, and the rule.
struct Apart {
    row: &'static str,
    here: Held,
    rule: &'static str,
}

/// The row decided apart: one refusal's reason. Where the only ready
/// prefill's partners are all over their thresholds beside a free decode of
/// another transport, the gateway reads the whole decode pool, finds it not
/// vetoed on every worker and answers unavailable. The pair's decision
/// reads the partners of the prefills it could use and names the overload
/// veto.
///
/// The four rows of a prefill policy that weighs load against cache
/// affinity itself replay as recorded: the gateway lifts the overload veto
/// for such a policy while shedding is off, and a prefill profile that
/// reads affinity runs with that veto lifted under a binding that steers.
const APART: &[Apart] = &[Apart {
    row: "pair-the-only-ready-prefills-partners-over-beside-a-free-decode",
    here: Held("overloaded"),
    rule: "a refusal names the overload veto where the usable prefills' partners are all over",
}];

/// What of one row's outcome this crate does not replay: nothing when its
/// answers are the recorded ones, or when [`APART`] names the row and its
/// answers are its entry's. A row decided apart that replays as recorded is
/// a defect too: its rule no longer decides it.
fn replay(row: &Row) -> Vec<String> {
    let here = match replayed(row) {
        Ok(here) => here,
        Err(refused) => return vec![format!("refused here: {refused}")],
    };
    let found = differences(&row.outcome, &here);
    match APART.iter().find(|apart| apart.row == row.name) {
        None => found,
        Some(apart) if found.is_empty() => vec![format!(
            "it replays as recorded now ({}): take the row out of APART",
            apart.rule
        )],
        Some(apart) if apart.here.is(&here) => Vec::new(),
        Some(apart) => vec![format!(
            "apart by its entry {:?}, replayed {here:?}",
            apart.here
        )],
    }
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

/// Every row decided apart is an inline row, so a run that reads no sheet
/// still checks that its rule decides it, and how: the row passes on its
/// entry's answer, and the same row with that answer recorded is a defect
/// that names the entry's rule.
#[test]
fn every_row_decided_apart_is_an_inline_row_held_to_its_entry() {
    let inline: Value = serde_json::from_str(INLINE).expect("the inline rows");
    let rows = inline["rows"].as_array().expect("rows");
    let alone = |row: &Value| {
        let text = serde_json::json!({ "rows": [row] }).to_string();
        replay_rows(&text, "inline").1
    };
    for apart in APART {
        let row = rows.iter().find(|row| row["name"] == apart.row);
        let row = row.unwrap_or_else(|| panic!("{} is no inline row", apart.row));
        assert_eq!(alone(row), Vec::<String>::new(), "{}", apart.row);
        let requests = row["requests"].as_u64().expect("a number of requests");
        let mut as_here = row.clone();
        let Held(why) = apart.here;
        as_here["outcome"] = serde_json::json!({
            "sequence": vec![serde_json::json!({ "none": { "role": "prefill", "why": why } }); requests as usize]
        });
        let found = alone(&as_here);
        assert!(
            matches!(&found[..], [one] if one.contains("take the row out of APART") && one.contains(apart.rule)),
            "{}: {found:?}",
            apart.row
        );
    }
}

/// The inline rows replay by themselves, so a run that reads no sheet
/// still replays a row of each form.
#[test]
fn the_inline_rows_replay() {
    let (rows, found) = replay_rows(INLINE, "inline");
    assert!(rows >= APART.len(), "{rows} inline rows");
    assert_eq!(found, Vec::<String>::new());
}

/// A sheet of another shape is one defect and replays no row.
#[test]
fn a_text_that_is_no_pair_sheet_is_a_defect() {
    let (rows, found) = replay_rows(r#"{"rows": [{"name": "a"}]}"#, "elsewhere");
    assert_eq!(rows, 0);
    assert!(
        matches!(&found[..], [one] if one.starts_with("elsewhere is not a sheet of pair-select rows")),
        "{found:?}"
    );
}

/// Rows copied from the recorded sheet, whole: every row decided apart, the
/// four of a prefill policy that weighs load against affinity, and one or
/// more of each form the replay reads.
pub const INLINE: &str = r#"{
  "recorded_from": "4ba2a6431357be5411a2383cdc4916b41471d722",
  "policy": "pair-select",
  "rows": [
    {
      "name": "pair-every-worker-open",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [1, 2]}]}
    },
    {
      "name": "pair-the-decode-policy-draws-among-the-chosen-prefills-partners",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "random",
      "shed": false,
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[1, 1, 2958], [1, 2, 3042]]}}
    },
    {
      "name": "pair-one-prefill-over-under-a-weighing-policy",
      "prefill_policy": "cache_aware",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 0, "state": "over", "transport": "a"},
        {"in_flight": 5, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [0, 2]}, {"pair": [0, 2]}]}
    },
    {
      "name": "pair-every-prefill-over-pairs-by-load",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "random",
      "shed": false,
      "prefills": [
        {"in_flight": 4, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 5, "state": "free", "transport": null}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[1, 1, 6000]]}}
    },
    {
      "name": "pair-every-prefill-over-under-a-weighing-policy",
      "prefill_policy": "cache_aware",
      "decode_policy": "random",
      "shed": false,
      "prefills": [
        {"in_flight": 4, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 5, "state": "free", "transport": null}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[1, 1, 3003], [1, 2, 2997]]}}
    },
    {
      "name": "pair-every-prefill-over-is-shed-under-shedding",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": true,
      "prefills": [
        {"in_flight": 4, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 5, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "overloaded"}}, {"none": {"role": "prefill", "why": "overloaded"}}]}
    },
    {
      "name": "pair-every-prefill-stalled-under-shedding",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": true,
      "prefills": [
        {"in_flight": 4, "state": "stalled", "transport": "a"},
        {"in_flight": 2, "state": "stalled", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 5, "state": "free", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "pair-every-decode-stalled-pairs-by-load",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 2, "state": "stalled", "transport": "a"},
        {"in_flight": 4, "state": "stalled", "transport": "b"},
        {"in_flight": 0, "state": "stalled", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [1, 2]}]}
    },
    {
      "name": "pair-one-prefill-over-and-one-stalled-under-a-weighing-policy",
      "prefill_policy": "cache_aware",
      "decode_policy": "random",
      "shed": false,
      "prefills": [
        {"in_flight": 5, "state": "over", "transport": "a"},
        {"in_flight": 1, "state": "stalled", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 5, "state": "free", "transport": null}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[0, 0, 2989], [0, 2, 3011]]}}
    },
    {
      "name": "pair-the-default-draws-a-prefill-then-its-tied-partners",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 2, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 5, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[0, 2, 2981], [1, 1, 1489], [1, 2, 1530]]}}
    },
    {
      "name": "pair-the-default-reads-no-room",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "random",
      "shed": false,
      "prefills": [
        {"in_flight": 4, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": "a", "rooms": {"window": 2, "admitted": 0}},
        {"in_flight": 0, "state": "free", "transport": "b", "rooms": {"window": 2, "admitted": 2}},
        {"in_flight": 5, "state": "free", "transport": null, "rooms": {"window": 2, "admitted": 0}}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[1, 1, 6000]]}}
    },
    {
      "name": "pair-the-default-sheds-when-its-prefill-is-at-capacity",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefill_bound": 4,
      "prefills": [
        {"in_flight": 4, "state": "over", "transport": "a"},
        {"in_flight": 6, "state": "over", "transport": "b"}
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
      "name": "pair-the-only-ready-prefills-partners-over-beside-a-free-decode",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "down", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "over", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 0, "state": "over", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "unavailable"}}, {"none": {"role": "decode", "why": "unavailable"}}]}
    },
    {
      "name": "pair-the-default-weighs-the-partners-load-between-tied-prefills",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 2, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 3, "state": "free", "transport": "a"},
        {"in_flight": 0, "state": "free", "transport": "b"},
        {"in_flight": 9, "state": "free", "transport": null}
      ],
      "requests": 1,
      "outcome": {"counts": {"draws": 6000, "pairs": [[1, 1, 6000]]}}
    },
    {
      "name": "pair-every-prefill-over-under-a-weighing-policy-and-every-decode-down",
      "prefill_policy": "cache_aware",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 4, "state": "over", "transport": "a"},
        {"in_flight": 2, "state": "over", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "down", "transport": "a"},
        {"in_flight": 0, "state": "down", "transport": "b"},
        {"in_flight": 0, "state": "down", "transport": null}
      ],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "prefill", "why": "unavailable"}}, {"none": {"role": "prefill", "why": "unavailable"}}]}
    },
    {
      "name": "pair-no-decode-of-the-model",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "shed": false,
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": "a"}
      ],
      "decodes": [],
      "requests": 2,
      "outcome": {"sequence": [{"none": {"role": "decode", "why": "no_candidates"}}, {"none": {"role": "decode", "why": "no_candidates"}}]}
    }
  ]
}"#;
