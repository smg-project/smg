//! The classic rows of a pair under the sticky routing-key override: a
//! prefill pool and a decode pool, the policy each side runs under, the
//! sticky map's assignment, and a list of steps. A step is a request, a
//! worker of a side that stops or starts being routable, the requests a
//! side's workers hold, or the requests of one key in flight on them. A
//! request states the key its header carries, the key of its own id, its
//! model and, where it is a retry, the pair its failed attempt was sent to.
//! Two workers pair when their transports are equal or one of them names
//! none. Each request is answered by the pair it was placed on, and a drawn
//! row counts how often each pair took its last request: the answers and
//! their comparison are the pair placement's
//! ([`pair_select`](super::pair_select)).
//!
//! The gateway asks its registry once a role: the prefill over the prefills
//! with an open partner, then the decode over the picked prefill's open
//! partners (`pair_pass` in
//! `model_gateway/src/routers/common/placement.rs`). Its sticky map places
//! a role whose policy does not keep the key's handling to itself, under
//! the key tagged by its model and its role (`sticky_for` and
//! `select_sticky` in `model_gateway/src/policies/registry.rs`).
//!
//! The replay places each request through the one call of a host in
//! process ([`InProcess::place`]), under the engine of the document a
//! gateway's routing settings give for the row
//! ([`RoutingSettings::policy_document`]): the pair mode with the row's two
//! side policies, and the override on with the row's assignment. A row's
//! workers are worker objects, and a step writes the object itself: its
//! status, its requests, and its requests of one key under the key's text,
//! which is the count a host's own guard keeps. A request's key travels as
//! its header and as its id, and the engine derives it from the sources
//! the document names. A placement is told sent on both sides and then
//! ended, as its host tells it: a side's pin is written at its dispatch.
//!
//! A pair with one side under `consistent_hashing` replays too. That policy
//! keeps the key's handling to itself, and the settings' rule names the
//! other side (`sessions.sides`). So the engine pins that side alone and
//! the hash side stays its own policy's.
//! A gateway lends the engine its own ring, which no sheet records: such a
//! row is lent one built from its recorded answers ([`ByAnswers`]).

use std::sync::Arc;

use serde::Deserialize;
use smg_grand_central::{
    document::{
        AppliedSpec, Assignment, LoadMetricSetting, ModeSettings, PolicySettings, PolicySpec,
        RoutingKeyOverrideSettings, RoutingSettings,
    },
    engine::{
        HeaderRead, InProcess, LentList, ListCell, PairAdmission, PairLists, PartnerClasses,
        PlaceRequest, Unplaced, WorkerId, WorkerLists,
    },
    policies::consistent_hashing::LentRing,
    tracking::{Outcome as Ended, Side as Sent, TransportClass},
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus, WorkerType},
};

use super::pair_select::{differences, why, Answer, Here, Outcome, Replayed};

/// What a sheet's head names these rows.
pub const POLICY: &str = "pair-sticky";

/// The model every worker of a row serves. A request names its own.
const MODEL: &str = "m";

/// The header a request's routing key travels in.
const HEADER: &str = "x-smg-routing-key";

/// The draws a replay makes of a drawn row: not the sheet's `draws`, which
/// counts the recording's and is a hundred times this ([`replayed`]).
const REPLAY_DRAWS: u32 = 6_000;

/// The policy that takes the fewest requests in flight, with no draw and no
/// state, in the gateway's spelling.
const FEWEST: &str = "power_of_two:least_requests";

/// The gateway's hash policy: it reads the routing key itself.
const RING: &str = "consistent_hashing";

/// What the router holds of a worker besides its requests, at a row's
/// start.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Not ready.
    Down,
}

/// One worker of a row.
#[derive(Clone, Debug, Deserialize)]
struct Member {
    in_flight: u32,
    state: State,
    transport: Option<String>,
}

/// A side of the pair.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Side {
    Prefill,
    Decode,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: the routing key its header carries, the key of its own
    /// id, its model, and the pair a failed attempt of it was sent to.
    Pick {
        key: Option<String>,
        #[serde(default)]
        id: Option<String>,
        model: String,
        #[serde(default)]
        tried: Option<[usize; 2]>,
    },
    /// A worker of a side stops being routable.
    Down(Side, usize),
    /// The worker is routable again.
    Up(Side, usize),
    /// The requests the router holds on each worker of a side, from here
    /// on.
    Loads(Side, Vec<u32>),
    /// The requests of one routing key in flight on each worker of a side,
    /// from here on: the count the rule's cap reads.
    Held {
        key: String,
        side: Side,
        on: Vec<u32>,
    },
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    prefill_policy: String,
    decode_policy: String,
    /// The sticky map's assignment, in the spelling the gateway's
    /// configuration and this crate's document share.
    assignment: Assignment,
    prefills: Vec<Member>,
    decodes: Vec<Member>,
    steps: Vec<Step>,
    outcome: Outcome,
}

#[derive(Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// The document a gateway's routing settings give for the row: the pair
/// mode with each side's policy, the override on under the row's
/// assignment, and every other setting at its default. The main policy is
/// the recorder's, which no pair reads. Or the settings' refusal, in its
/// own words.
fn spec(row: &Row) -> Result<PolicySpec, String> {
    let side = |policy: &str| match policy {
        FEWEST => Ok(PolicySettings::PowerOfTwo {
            load_metric: LoadMetricSetting::LeastRequests,
        }),
        "random" => Ok(PolicySettings::Random),
        RING => Ok(PolicySettings::ConsistentHashing),
        other => Err(format!("a policy this replay does not build: {other}")),
    };
    let settings = RoutingSettings {
        policy: PolicySettings::Random,
        mode: ModeSettings::PrefillDecode {
            prefill_policy: Some(side(&row.prefill_policy)?),
            decode_policy: Some(side(&row.decode_policy)?),
        },
        routing_key_override: RoutingKeyOverrideSettings {
            enabled: true,
            headers: vec![HEADER.into()],
            eviction_interval_secs: 60,
            max_idle_secs: 14_400,
            assignment_mode: row.assignment,
        },
        ..RoutingSettings::default()
    };
    settings.policy_document().map_err(|err| err.to_string())
}

/// A host with no admission of its own: every prefill has capacity and
/// every decode a room.
#[derive(Debug)]
struct Open;

impl PairAdmission for Open {
    fn has_capacity(&self, _prefill: usize) -> bool {
        true
    }

    fn has_room(&self, _decode: usize) -> bool {
        true
    }
}

/// A row's two lists of worker objects in the row's starting state, with
/// the pair classes of their transports, under the row's engine.
struct Bed<'r> {
    engine: &'static InProcess,
    prefills: Arc<[Arc<dyn Worker>]>,
    decodes: Arc<[Arc<dyn Worker>]>,
    classes: PartnerClasses,
    cells: [ListCell; 2],
    /// What separates this run's keys from another run's under the one
    /// engine: nothing for a sequence row, the draw's number for a drawn
    /// one.
    run: &'r str,
    /// The ring the row's host lends, where a side runs the hash policy.
    ring: Option<ByAnswers>,
}

/// The address of a row's worker: its side and its place in the side's
/// list.
fn address(side: &str, at: usize) -> String {
    format!("http://{side}-{at}:8000")
}

/// A caller's ring built from a row's recorded answers, for a row with a
/// side under the hash policy. A gateway lends the engine its own ring
/// ([`PlaceRequest::ring`]), which hashes a key's text and its workers'
/// addresses, and no row records where a key stands on it. Here a key
/// stands at its place among the row's keys.
///
/// Its walk first reaches the hash side's workers that answered its
/// requests, in the order the row first names them. That side's other
/// workers follow in the list's order. The order behind the answers is
/// this replay's own.
#[derive(Debug)]
struct ByAnswers {
    /// Each key of the row, as its run states it, and the addresses its
    /// walk reaches, in order.
    walks: Vec<(String, Vec<String>)>,
}

impl ByAnswers {
    /// The ring of `row` for the keys as `run` states them. `None` for a
    /// row neither side of which runs the hash policy, and for one whose
    /// outcome is no sequence of answers.
    fn of(row: &Row, run: &str) -> Option<Self> {
        let (side, name, listed) = match (row.prefill_policy.as_str(), row.decode_policy.as_str()) {
            (RING, _) => (0, "prefill", row.prefills.len()),
            (_, RING) => (1, "decode", row.decodes.len()),
            _ => return None,
        };
        let Outcome::Sequence(answers) = &row.outcome else {
            return None;
        };
        // A header's key goes ahead of the key of the request's id.
        let keys = row.steps.iter().filter_map(|step| match step {
            Step::Pick { key, id, .. } => Some(key.as_deref().or(id.as_deref())),
            _ => None,
        });
        let mut reached: Vec<(String, Vec<usize>)> = Vec::new();
        for (key, answer) in keys.zip(answers) {
            let (Some(key), Answer::Pair(pair)) = (key, answer) else {
                continue;
            };
            let key = format!("{key}{run}");
            let known = reached.iter().position(|(placed, _)| *placed == key);
            let at = known.unwrap_or_else(|| {
                reached.push((key, Vec::new()));
                reached.len() - 1
            });
            if !reached[at].1.contains(&pair[side]) {
                reached[at].1.push(pair[side]);
            }
        }
        let walks = reached.into_iter().map(|(key, first)| {
            let rest = (0..listed).filter(|at| !first.contains(at));
            let walk = first.iter().copied().chain(rest);
            (key, walk.map(|at| address(name, at)).collect())
        });
        Some(Self {
            walks: walks.collect(),
        })
    }
}

impl LentRing for ByAnswers {
    fn position(&self, key: &str) -> u64 {
        let at = self.walks.iter().position(|(placed, _)| placed == key);
        at.unwrap_or(self.walks.len()) as u64
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        if let Some((_, walk)) = self.walks.get(position as usize) {
            let _ = walk.iter().any(|address| take(address));
        }
    }
}

/// One worker object of a row's list, in the row's state, with the requests
/// the router holds on it as its object's own count.
fn object(side: &str, at: usize, role: WorkerType, member: &Member) -> Arc<dyn Worker> {
    let object = <BasicWorkerBuilder>::new(address(side, at))
        .worker_type(role)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        })
        .build();
    object.set_status(match member.state {
        State::Down => WorkerStatus::NotReady,
        State::Free => WorkerStatus::Ready,
    });
    for _ in 0..member.in_flight {
        object.increment_load();
    }
    Arc::new(object)
}

impl<'r> Bed<'r> {
    fn new(row: &Row, engine: &'static InProcess, run: &'r str) -> Self {
        let list = |side: &str, role: WorkerType, members: &[Member]| -> Arc<[Arc<dyn Worker>]> {
            (members.iter().enumerate())
                .map(|(at, member)| object(side, at, role, member))
                .collect()
        };
        // Each prefill's partners as places of the decode list, ascending.
        let pairs =
            |prefill: &Member, decode: &Member| match (&prefill.transport, &decode.transport) {
                (Some(ours), Some(theirs)) => ours == theirs,
                _ => true,
            };
        let partners: Vec<Vec<u32>> = (row.prefills.iter())
            .map(|prefill| {
                (0u32..)
                    .zip(&row.decodes)
                    .filter(|(_, decode)| pairs(prefill, decode))
                    .map(|(place, _)| place)
                    .collect()
            })
            .collect();
        Self {
            engine,
            prefills: list("prefill", WorkerType::Prefill, &row.prefills),
            decodes: list("decode", WorkerType::Decode, &row.decodes),
            classes: PartnerClasses::number(row.decodes.len(), partners.iter().map(Vec::as_slice)),
            cells: [ListCell::default(), ListCell::default()],
            run,
            ring: ByAnswers::of(row, run),
        }
    }

    /// A key of the row as this run states it.
    fn key(&self, key: &str) -> String {
        format!("{key}{}", self.run)
    }

    /// One request placed through the one call and sent on both sides, as
    /// a host commits it: the pair it was placed on, or why none.
    fn pick(
        &self,
        key: Option<&str>,
        id: Option<&str>,
        model: &str,
        tried: Option<[usize; 2]>,
    ) -> Result<Here, &'static str> {
        let [prefill_cell, decode_cell] = &self.cells;
        let lists = PairLists {
            prefill: LentList {
                workers: &self.prefills,
                cell: prefill_cell,
                facts: None,
            },
            decode: LentList {
                workers: &self.decodes,
                cell: decode_cell,
                facts: None,
            },
            classes: &self.classes,
            admission: &Open,
            prefill_places: None,
            decode_places: None,
            prefill_stalled: &[],
            prefill_held: &[],
            decode_stalled: &[],
            decode_held: &[],
        };
        let (key, id) = (key.map(|key| self.key(key)), id.map(|id| self.key(id)));
        let keyed;
        let headers: &dyn HeaderRead = match &key {
            Some(key) => {
                keyed = [(HEADER, key.as_str())];
                &keyed
            }
            None => &(),
        };
        let tried: Vec<WorkerId> = tried.map_or_else(Vec::new, |[prefill, decode]| {
            [&self.prefills[prefill], &self.decodes[decode]]
                .map(|worker| WorkerId::from_address(worker.url()))
                .to_vec()
        });
        let request = PlaceRequest {
            headers,
            request_id: id.as_deref(),
            attempt: u8::from(!tried.is_empty()),
            tried: &tried,
            ring: self.ring.as_ref().map(|ring| ring as &dyn LentRing),
            ..PlaceRequest::of(model, WorkerLists::Pair(lists))
        };
        let mut placement = match self.engine.place(&request) {
            Ok(placement) => placement,
            // A shed by its reason, its stage and its side, as the pair
            // placement's replay reads one. No row runs behind an admission,
            // so no request waits at one.
            Err(Unplaced::Shed {
                reason,
                emptied_by,
                side,
                ..
            }) => return Ok(Here::None(why(reason, emptied_by), emptied_by, side)),
            Err(_) => return Ok(Here::None("unavailable", "place", None)),
        };
        let at = |list: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>| {
            let place = list.iter().position(|listed| Arc::ptr_eq(listed, worker));
            place.ok_or("a placement names a worker of no list")
        };
        let decode = placement.decode.as_ref().ok_or("a pair with no decode")?;
        let here = Here::Pair([
            at(&self.prefills, &placement.worker)?,
            at(&self.decodes, decode)?,
        ]);
        for side in [Sent::Prefill, Sent::Decode] {
            placement.sent(side);
        }
        for side in [Sent::Prefill, Sent::Decode] {
            placement.ended(side, Ended::reported(200, TransportClass::Ok));
        }
        Ok(here)
    }

    /// One step of the row; a request answers where it was placed.
    fn step(&self, step: &Step) -> Result<Option<Here>, &'static str> {
        let list = |side: Side| match side {
            Side::Prefill => &self.prefills,
            Side::Decode => &self.decodes,
        };
        match step {
            Step::Pick {
                key,
                id,
                model,
                tried,
            } => {
                return self
                    .pick(key.as_deref(), id.as_deref(), model, *tried)
                    .map(Some)
            }
            Step::Down(side, at) => list(*side)[*at].set_status(WorkerStatus::NotReady),
            Step::Up(side, at) => list(*side)[*at].set_status(WorkerStatus::Ready),
            Step::Loads(side, loads) => {
                for (worker, &want) in list(*side).iter().zip(loads) {
                    while worker.load() < want as usize {
                        worker.increment_load();
                    }
                    while worker.load() > want as usize {
                        worker.decrement_load();
                    }
                }
            }
            Step::Held { key, side, on } => {
                let key = self.key(key);
                for (worker, &want) in list(*side).iter().zip(on) {
                    while worker.routing_key_inflight(&key) < want as usize {
                        worker.increment_routing_key_load(&key);
                    }
                    while worker.routing_key_inflight(&key) > want as usize {
                        worker.decrement_routing_key_load(&key);
                    }
                }
            }
        }
        Ok(None)
    }
}

/// The row's steps run once over fresh lists: each request's answer, in
/// order. `run` separates the keys of a drawn row's runs.
fn run(row: &Row, engine: &'static InProcess, run: &str) -> Result<Vec<Here>, &'static str> {
    let bed = Bed::new(row, engine, run);
    let mut answers = Vec::new();
    for step in &row.steps {
        answers.extend(bed.step(step)?);
    }
    Ok(answers)
}

/// The row replayed in the form its outcome was recorded in, under the
/// engine of `spec` as a host holds it: for the life of the process.
///
/// The gateway records each draw from a fresh sticky map. The engine keeps
/// its session table for its life, so every draw runs under the row's one
/// engine with keys of its own, the row's with the draw's number behind
/// them: no draw meets a pin of another.
///
/// A drawn row is recorded over a hundred times the draws its replay
/// makes, so one deviation of the recording is a tenth of a replay's. A
/// row recorded over fewer is refused. The replay's shares are held within
/// four deviations of the difference of two samples of the replay's size.
/// At a ninth that is 137 counts of 6,000, about 5.7 deviations of the
/// replay's own share.
///
/// Each draw goes through the host's one call, whose seed is the calling
/// thread's own. So a replay draws anew at every run.
fn replayed(row: &Row, spec: &PolicySpec) -> Result<Replayed, String> {
    let applied = AppliedSpec::validate(spec).map_err(|err| err.to_string())?;
    let engine: &'static InProcess = Box::leak(Box::new(InProcess::new(applied)));
    let Outcome::Counts { draws, .. } = &row.outcome else {
        return Ok(Replayed::Sequence(run(row, engine, "")?));
    };
    if *draws / 100 < REPLAY_DRAWS {
        return Err(format!(
            "a drawn row recorded over {draws} draws, fewer than 100 replays of {REPLAY_DRAWS}"
        ));
    }
    let mut counts: Vec<(Here, u32)> = Vec::new();
    for draw in 0..REPLAY_DRAWS {
        let answers = run(row, engine, &format!("-{draw}"))?;
        let here = *answers.last().ok_or("a drawn row places a request")?;
        match counts.iter_mut().find(|(answer, _)| *answer == here) {
            Some((_, count)) => *count += 1,
            None => counts.push((here, 1)),
        }
    }
    Ok(Replayed::Counts(counts))
}

/// What of one row's outcome this crate does not replay: nothing when its
/// answers are the recorded ones.
fn replay(row: &Row) -> Vec<String> {
    let spec = match spec(row) {
        Ok(spec) => spec,
        Err(refusal) => return vec![format!("refused here: {refusal}")],
    };
    match replayed(row, &spec) {
        Ok(here) => differences(&row.outcome, &here),
        Err(refused) => vec![format!("refused here: {refused}")],
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

/// The inline rows replay by themselves, so a run that reads no sheet
/// still replays a row of each form: a key's pair kept past the loads and
/// apart under two models, a key from the request's id, a key's turns and
/// retries, the map's own assignments, a cap, a worker down and up, a
/// decode that is no partner of the new prefill, a retry, and a hash
/// prefill beside a pinned decode.
#[test]
fn the_sticky_pair_rows_replay_answer_for_answer() {
    let (rows, found) = replay_rows(INLINE, "inline");
    assert_eq!(rows, 12);
    assert_eq!(found, Vec::<String>::new());
}

/// A drawn row's replay makes [`REPLAY_DRAWS`] draws, whatever the sheet
/// recorded, and a row recorded over fewer than a hundred times as many is
/// a defect. A recording of the replay's own size is one sample too: one
/// such stood far enough from its mean that a replay in 461 left the bound
/// by chance.
#[test]
fn a_drawn_row_is_replayed_over_a_hundredth_of_its_recorded_draws() {
    let sheet: Sheet = serde_json::from_str(INLINE).expect("the inline rows");
    let drawn = sheet.rows.iter().find_map(|row| match &row.outcome {
        Outcome::Counts { draws, .. } => Some((row, *draws)),
        Outcome::Sequence(_) => None,
    });
    let (row, recorded) = drawn.expect("a drawn row");
    assert_eq!(recorded, 100 * REPLAY_DRAWS);
    let spec = spec(row).expect("the row's document");
    let made: Option<u32> = match replayed(row, &spec) {
        Ok(Replayed::Counts(counts)) => Some(counts.iter().map(|&(_, count)| count).sum()),
        _ => None,
    };
    assert_eq!(made, Some(REPLAY_DRAWS));
    // The same row under a recording one draw short of a hundred replays.
    let stated = format!("\"draws\": {recorded}");
    let thin = INLINE.replace(&stated, &format!("\"draws\": {}", recorded - 1));
    assert_ne!(thin, INLINE);
    let (_, found) = replay_rows(&thin, "inline");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("recorded over 599999 draws"), "{found:?}");
}

/// Two rows of this crate's own, which no gateway recorded: a worker of a
/// hash side that stops being routable and starts again, a row for each
/// side. The key goes to the next worker of its ring's walk and comes back
/// to its own, as a ring places it.
///
/// A pin does not: under the sticky map's `delegate` a pinned side stays
/// with the worker placed last, as the row of a prefill down says
/// ([`INLINE`]). The other side keeps its pin all along. The gateway's
/// sticky map leaves a role under the hash policy to that policy
/// (`sticky_for` in `model_gateway/src/policies/registry.rs`).
#[test]
fn a_hash_sides_key_follows_its_ring_past_a_worker_down_and_back() {
    let (rows, found) = replay_rows(OWN, "own");
    assert_eq!(rows, 2);
    assert_eq!(found, Vec::<String>::new());
}

/// Rows in a sheet's form that state a ring's walk by their answers
/// ([`ByAnswers`]). A walk is the first worker a key's requests name, then
/// the one they name while that worker is down.
const OWN: &str = r#"{
  "rows": [
    {
      "name": "a-hash-prefill-down-is-the-rings-next-and-up-its-own-again",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": ["prefill", 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": ["prefill", 2]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [2, 1]}, {"pair": [0, 1]}, {"pair": [2, 1]}]}
    },
    {
      "name": "a-hash-decode-down-is-the-rings-next-and-up-its-own-again",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "consistent_hashing",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": ["decode", 0]},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": ["decode", 0]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 0]}, {"pair": [1, 2]}, {"pair": [1, 0]}]}
    }
  ]
}"#;

/// Rows copied from the recorded sheet, whole.
pub const INLINE: &str = r#"{
  "recorded_from": "0a3b3bfdf5b4a5789fc5b4b7e4ee216f84996318",
  "policy": "pair-sticky",
  "rows": [
    {
      "name": "sticky-pair-delegate-a-key-keeps-its-pair-past-the-loads",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"loads": ["prefill", [0, 5, 2]]},
        {"loads": ["decode", [0, 5, 1]]},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}, {"pair": [0, 0]}, {"pair": [1, 1]}]}
    },
    {
      "name": "sticky-pair-one-key-pins-apart-under-two-models",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m1"}},
        {"loads": ["prefill", [0, 5, 2]]},
        {"loads": ["decode", [0, 5, 1]]},
        {"pick": {"key": "k1", "model": "m2"}},
        {"loads": ["prefill", [9, 9, 0]]},
        {"loads": ["decode", [9, 9, 0]]},
        {"pick": {"key": "k1", "model": "m1"}},
        {"pick": {"key": "k1", "model": "m2"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [0, 0]}, {"pair": [1, 1]}, {"pair": [0, 0]}]}
    },
    {
      "name": "sticky-pair-a-key-from-the-requests-id-pins-as-a-headers-does-and-behind-it",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"loads": ["prefill", [0, 5, 2]]},
        {"loads": ["decode", [0, 5, 1]]},
        {"pick": {"key": null, "id": "k2", "model": "m"}},
        {"loads": ["prefill", [9, 9, 0]]},
        {"loads": ["decode", [9, 9, 0]]},
        {"pick": {"key": null, "id": "k1", "model": "m"}},
        {"pick": {"key": "k1", "id": "k2", "model": "m"}},
        {"pick": {"key": "k3", "id": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [0, 0]}, {"pair": [1, 1]}, {"pair": [1, 1]}, {"pair": [2, 2]}]}
    },
    {
      "name": "sticky-pair-a-keys-turns-and-retries-share-its-pair-and-its-cap",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "conv_t1", "model": "m"}},
        {"loads": ["prefill", [0, 5, 2]]},
        {"loads": ["decode", [0, 5, 1]]},
        {"pick": {"key": "conv_t2_r1", "model": "m"}},
        {"held": {"key": "conv", "side": "prefill", "on": [0, 2, 0]}},
        {"pick": {"key": "conv_t3", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 1]}, {"pair": [0, 1]}]}
    },
    {
      "name": "sticky-pair-min-group-a-new-key-takes-each-sides-fewest-keys",
      "prefill_policy": "random",
      "decode_policy": "random",
      "assignment": "min_group",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 5, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 5, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null}
      ],
      "steps": [
        {"held": {"key": "x1", "side": "prefill", "on": [1, 0, 1]}},
        {"held": {"key": "x2", "side": "prefill", "on": [1, 0, 0]}},
        {"held": {"key": "x1", "side": "decode", "on": [1, 1, 0]}},
        {"held": {"key": "x2", "side": "decode", "on": [0, 1, 0]}},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "side": "prefill", "on": [0, 1, 0]}},
        {"held": {"key": "x3", "side": "prefill", "on": [0, 1, 0]}},
        {"held": {"key": "x4", "side": "prefill", "on": [0, 1, 0]}},
        {"held": {"key": "k1", "side": "decode", "on": [0, 0, 1]}},
        {"held": {"key": "x3", "side": "decode", "on": [0, 0, 1]}},
        {"held": {"key": "x4", "side": "decode", "on": [0, 0, 1]}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 2]}, {"pair": [2, 0]}, {"pair": [1, 2]}]}
    },
    {
      "name": "sticky-pair-random-a-new-key-is-drawn-on-each-side",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "random",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"counts": {"draws": 600000, "pairs": [[0, 0, 66498], [0, 1, 66709], [0, 2, 66506], [1, 0, 66589], [1, 1, 66669], [1, 2, 66933], [2, 0, 66891], [2, 1, 67053], [2, 2, 66152]]}}
    },
    {
      "name": "sticky-pair-the-prefills-cap-moves-the-prefill-alone",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "side": "prefill", "on": [0, 2, 0]}},
        {"loads": ["prefill", [0, 3, 2]]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "side": "prefill", "on": [0, 0, 0]}},
        {"loads": ["prefill", [9, 0, 9]]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [0, 1]}, {"pair": [0, 1]}]}
    },
    {
      "name": "sticky-pair-delegate-a-prefill-down-again-goes-back-to-the-one-before",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": ["prefill", 1]},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": ["prefill", 1]},
        {"loads": ["prefill", [1, 5, 0]]},
        {"down": ["prefill", 0]},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": ["prefill", 0]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [0, 1]}, {"pair": [1, 1]}, {"pair": [0, 1]}]}
    },
    {
      "name": "sticky-pair-min-load-a-decode-down-lends-the-key-another-until-it-is-back",
      "prefill_policy": "random",
      "decode_policy": "random",
      "assignment": "min_load",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": ["decode", 1]},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": ["decode", 1]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [1, 0]}, {"pair": [1, 1]}]}
    },
    {
      "name": "sticky-pair-a-pinned-decode-no-partner-of-the-new-prefill-is-placed-again",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"}
      ],
      "decodes": [
        {"in_flight": 0, "state": "free", "transport": "a"},
        {"in_flight": 1, "state": "free", "transport": "b"},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": ["prefill", 0]},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": ["prefill", 0]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [0, 0]}, {"pair": [1, 1]}, {"pair": [1, 1]}]}
    },
    {
      "name": "sticky-pair-delegate-a-retry-steps-off-its-pair-and-moves-both-pins",
      "prefill_policy": "power_of_two:least_requests",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k1", "model": "m", "tried": [1, 1]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [1, 1]}, {"pair": [0, 0]}, {"pair": [0, 0]}]}
    },
    {
      "name": "sticky-pair-a-hash-prefill-is-the-rings-beside-a-pinned-decode",
      "prefill_policy": "consistent_hashing",
      "decode_policy": "power_of_two:least_requests",
      "assignment": "delegate",
      "prefills": [
        {"in_flight": 3, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null},
        {"in_flight": 2, "state": "free", "transport": null}
      ],
      "decodes": [
        {"in_flight": 2, "state": "free", "transport": null},
        {"in_flight": 0, "state": "free", "transport": null},
        {"in_flight": 1, "state": "free", "transport": null}
      ],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"loads": ["prefill", [0, 5, 2]]},
        {"loads": ["decode", [0, 5, 1]]},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [{"pair": [2, 1]}, {"pair": [2, 1]}, {"pair": [0, 0]}, {"pair": [2, 1]}]}
    }
  ]
}"#;
