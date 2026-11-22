//! The bed the hard-demand rows replay on. A row's workers are registered
//! in a store, each with its wire, its state and the facts it states for a
//! request's demands, and each request is decided through the door a host
//! in process decides through ([`Door`]), under `round_robin`. A request is
//! decided sixteen times the pool's size, as its recorder placed it, and
//! `placed` is the set of workers those decisions named.
//!
//! A worker states its runtime as a number: the numbers are the replay's
//! own, as they are a registering host's, one a spelling.
//!
//! A refusal replays by its status and its code. The crate sheds under a
//! reason and a stage's name, and the gateway keeps its own words for each:
//! a sheet's module holds that table ([`pool_refusal`] where no demand of
//! the sheet names the shed). The message and the retry hint are no part of
//! a decision.

use std::{collections::BTreeSet, sync::Arc, time::Instant};

use serde::{de::DeserializeOwned, Deserialize};
use smg_policy_controller::{
    document::{AppliedSpec, BindingSpec, OverloadSpec, PolicySpec},
    engine::{
        Constraints, Decision, DemandFacts, Door, LoadRow, Request, RequestFacts, ShedReason,
        WireModes, WorkerId,
    },
    store::{
        GatewayId, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore,
    },
};

const MODEL: &str = "m";

/// A request is decided this many times the pool's size.
const ROUNDS: usize = 16;

/// The queue of a worker over its thresholds: above the default bound.
const QUEUED_OVER: u32 = 64;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Ready and under its thresholds.
    Free,
    /// Ready and over its thresholds.
    Over,
    /// Not ready.
    Down,
}

/// The wire a retry's retained plan was built for.
#[derive(Debug, Deserialize)]
pub struct Wire {
    pub runtime: String,
    pub transport: String,
}

/// A refusal as the replay holds it: the response's status and its code.
#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct Refusal {
    status: u16,
    code: String,
}

impl Refusal {
    pub fn of(status: u16, code: &str) -> Self {
        Self {
            status,
            code: code.to_string(),
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Answer {
    /// The places, in the row's list, of the workers a request may land on.
    Placed(BTreeSet<usize>),
    Refused(Refusal),
}

/// The wire a transport's name states.
pub fn wire_of(transport: &str) -> Result<WireModes, String> {
    match transport {
        "grpc" => Ok(WireModes::GRPC),
        "zmq" => Ok(WireModes::ZMQ),
        other => Err(format!("no wire is named {other}")),
    }
}

/// The numbers of a row's runtimes, as a registering host states them: one
/// a spelling, none of them zero, which states no runtime.
#[derive(Default)]
pub struct Runtimes(Vec<String>);

impl Runtimes {
    pub fn number(&mut self, spelling: &str) -> u32 {
        let known = self.0.iter().position(|known| known == spelling);
        let at = known.unwrap_or_else(|| {
            self.0.push(spelling.to_string());
            self.0.len() - 1
        });
        u32::try_from(at + 1).unwrap_or(u32::MAX)
    }

    /// What a retry on `wire` states: its transport as the wire it speaks
    /// and its runtime's number. A first attempt stays on no wire.
    pub fn stays_on(&mut self, wire: Option<&Wire>) -> Result<(WireModes, u32), String> {
        match wire {
            None => Ok((WireModes::NONE, 0)),
            Some(wire) => Ok((wire_of(&wire.transport)?, self.number(&wire.runtime))),
        }
    }
}

/// The gateway's words for a shed no demand names. A pool over its
/// thresholds on every worker the request may take is overloaded. A model
/// no worker serves is no model, and so is a pool whose wire left a retry
/// nobody. A pool left nobody by any other stage is unavailable.
pub fn pool_refusal(reason: ShedReason, emptied_by: &str) -> Refusal {
    match (reason, emptied_by) {
        (ShedReason::Saturated, _) => Refusal::of(503, "worker_overload_protection_shed"),
        (_, "none" | "wires") => Refusal::of(404, "model_not_found"),
        _ => Refusal::of(503, "no_available_workers"),
    }
}

/// One worker of a row as the bed registers it.
pub struct Pooled {
    pub wire_modes: WireModes,
    pub state: State,
    /// Whether it takes images: its row's own bit.
    pub vision: bool,
    pub facts: DemandFacts,
}

/// A row's pool behind the door, with each worker's id by its place.
pub struct Bed {
    door: Door,
    ids: Vec<WorkerId>,
    now: Instant,
    seed: u64,
}

impl Bed {
    /// The bed of `workers`, shedding a pool over its thresholds on every
    /// worker where `shed`; or the validator's words when this crate does
    /// not build the row's form.
    pub fn new(shed: bool, workers: &[Pooled]) -> Result<Self, String> {
        let mut document = PolicySpec::preset("round_robin").map_err(|err| err.to_string())?;
        document.generation = 1;
        let profile = document.profiles.keys().next().cloned();
        document.bindings = vec![BindingSpec {
            profile: Some(profile.ok_or("the preset's profile")?),
            overload: Some(OverloadSpec {
                shed_when_all_vetoed: shed,
                ..OverloadSpec::default()
            }),
            ..BindingSpec::default()
        }];
        let applied = AppliedSpec::validate(&document).map_err(|err| err.to_string())?;
        let store = Arc::new(MemoryStore::new());
        let now = Instant::now();
        let mut ids = Vec::with_capacity(workers.len());
        for (place, worker) in workers.iter().enumerate() {
            let url = format!("http://w{place}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: url.clone(),
                    engine: url,
                    incarnation: 1,
                    models: vec![MODEL.to_string()],
                    dp_size: 1,
                    wire_modes: worker.wire_modes,
                    vision: worker.vision,
                    demand_facts: worker.facts,
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(&*store, id).ok_or("the row")?.revision;
            let status = match worker.state {
                State::Down => WorkerStatus::NotReady,
                State::Free | State::Over => WorkerStatus::Ready,
            };
            store
                .set_status(id, status, revision)
                .ok_or("the row's status")?;
            ids.push(id);
        }
        // The bed's host runs no loop: it asks the build its registrations
        // owe, so the rows stand before the first record and request.
        store.publish_now();
        store.apply(applied).map_err(|err| format!("{err:?}"))?;
        for (id, worker) in ids.iter().zip(workers) {
            if worker.state == State::Over {
                let over = LoadRow {
                    queued: QUEUED_OVER,
                    ..LoadRow::default()
                };
                store.record(*id, 0, over, now, LoadSource::Stream);
                // A record inside the window of the one before waits for
                // the window's end: the bed shows it there.
                if let Some(due) = store.publish_due() {
                    store.publish(due);
                }
            }
        }
        Ok(Self {
            door: Door::new(store),
            ids,
            now,
            seed: 0,
        })
    }

    /// The answer to one request under `constraints`: the workers its
    /// decisions named, or the refusal of a request no decision placed, in
    /// the words `refusal` keeps for a shed's reason and stage.
    pub fn answer(
        &mut self,
        constraints: Constraints<'_>,
        refusal: impl Fn(ShedReason, &str) -> Refusal,
    ) -> Result<Answer, String> {
        let mut placed = BTreeSet::new();
        for _ in 0..ROUNDS * self.ids.len().max(1) {
            self.seed += 1;
            let id = self.seed.to_le_bytes();
            let request = Request {
                id: &id,
                facts: RequestFacts {
                    model: MODEL,
                    seed: self.seed,
                    ..RequestFacts::default()
                },
                constraints,
            };
            let decided = self.door.decide(GatewayId::LOCAL, &request, self.now);
            let decided = decided.map_err(|refused| format!("no decision: {refused:?}"))?;
            match decided.decision {
                Decision::Pick(pick) => {
                    let target = pick.targets.first().map(|target| target.id);
                    let target = target.ok_or("a pick that names no worker")?;
                    let place = self.ids.iter().position(|id| *id == target);
                    placed.insert(place.ok_or("a worker of no place")?);
                }
                Decision::Shed(shed) if placed.is_empty() => {
                    return Ok(Answer::Refused(refusal(shed.reason, shed.emptied_by)));
                }
                Decision::Shed(shed) => {
                    return Err(format!(
                        "shed under {} after its workers took it",
                        shed.emptied_by
                    ));
                }
                Decision::Queue(_) => return Err("a queue ticket".to_string()),
            }
        }
        Ok(Answer::Placed(placed))
    }
}

/// A row of a hard-demand sheet: its name, the answers it recorded and the
/// answers this crate gives its requests, in their order.
pub trait Replayed {
    fn name(&self) -> &str;
    fn recorded(&self) -> &[Answer];
    fn replay(&self) -> Result<Vec<Answer>, String>;
}

#[derive(Deserialize)]
struct Sheet<R> {
    rows: Vec<R>,
}

/// The rows of a sheet of `policy` replayed: how many, and what did not
/// replay.
pub fn replay_rows<R: Replayed + DeserializeOwned>(
    text: &str,
    source: &str,
    policy: &str,
) -> (usize, Vec<String>) {
    let sheet: Sheet<R> = match serde_json::from_str(text) {
        Ok(sheet) => sheet,
        Err(err) => {
            return (
                0,
                vec![format!("{source} is not a sheet of {policy} rows: {err}")],
            )
        }
    };
    let defects = sheet
        .rows
        .iter()
        .filter_map(|row| {
            let why = match row.replay() {
                Ok(here) if here == row.recorded() => None,
                Ok(here) => Some(format!("recorded {:?}, replayed {here:?}", row.recorded())),
                Err(why) => Some(format!("refused here: {why}")),
            };
            why.map(|why| format!("{source}: {}: {why}", row.name()))
        })
        .collect();
    (sheet.rows.len(), defects)
}
