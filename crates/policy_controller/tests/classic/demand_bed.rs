//! The bed the hard-demand rows replay on. A row's workers are worker
//! objects in one list, each with its wire and its state, and the facts
//! each states for a request's demands stand beside the list
//! ([`FixedFacts`]). Each request is decided by the engine over worker
//! lists ([`PolicyEngine`]), whole, under `round_robin`. A request is
//! decided sixteen times the pool's size, as its recorder placed it, and
//! `placed` is the set of workers those decisions named.
//!
//! A worker states its runtime as a number: the numbers are the replay's
//! own, as they are a registering host's, one a spelling. The bed gives one
//! number to each runtime and transport, as a host of this engine does: a
//! retry stays on its wire by its demand alone, and its list is handed
//! whole, whatever wires its workers dial.
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
        Constraints, Decision, DecisionEngine, DemandFacts, FixedFacts, LentList, ListCell,
        PolicyEngine, Request, RequestFacts, RequestMaker, ShedReason, WireModes, WorkerId,
        WorkerLists,
    },
    store::GatewayId,
};
use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{ConnectionMode, HealthCheckConfig, WorkerStatus},
};

const MODEL: &str = "m";

/// A request is decided this many times the pool's size.
const ROUNDS: usize = 16;

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

/// One worker of a row as the bed makes it.
pub struct Pooled {
    pub wire_modes: WireModes,
    pub state: State,
    /// Whether it takes images: its record's own bit.
    pub vision: bool,
    pub facts: DemandFacts,
}

/// A row's pool: the engine, the row's one list, what its workers state
/// beside their objects and the cell kept beside the list, as a host of
/// this engine keeps the three.
pub struct Bed {
    engine: PolicyEngine,
    list: Arc<[Arc<dyn Worker>]>,
    facts: Vec<FixedFacts>,
    cell: ListCell,
    now: Instant,
    seed: u64,
}

/// One request of a bed, over the bed's list as it is lent.
struct Asked<'a> {
    request: Request<'a>,
    lent: LentList<'a>,
}

impl RequestMaker for Asked<'_> {
    fn model(&self) -> &str {
        self.request.facts.model
    }

    fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
        self.request
    }

    fn workers(&self) -> WorkerLists<'_> {
        WorkerLists::Single(self.lent)
    }
}

/// The one wire `wires` names, as a worker object dials one.
fn dialed(wires: WireModes) -> Result<ConnectionMode, String> {
    match wires {
        WireModes::HTTP => Ok(ConnectionMode::Http),
        WireModes::GRPC => Ok(ConnectionMode::Grpc),
        WireModes::ZMQ => Ok(ConnectionMode::Zmq),
        other => Err(format!("one wire is dialed, not {other:?}")),
    }
}

/// The number of `runtime` on `wire`, as a host of this engine states it:
/// one number to each runtime and transport it dials, written as a worker's
/// fact and as a retry's demand alike. No runtime states none.
fn on_wire(runtime: u32, wire: ConnectionMode) -> u32 {
    let transport = match wire {
        ConnectionMode::Http => 0,
        ConnectionMode::Grpc => 1,
        ConnectionMode::Zmq => 2,
    };
    if runtime == 0 {
        0
    } else {
        runtime.saturating_mul(3).saturating_add(transport)
    }
}

/// The worker object of `worker`, at `place` of its row: its wire is the
/// one its object dials, and its state is its object's own.
fn object(place: usize, worker: &Pooled) -> Result<Arc<dyn Worker>, String> {
    let built = <BasicWorkerBuilder>::new(format!("http://w{place}:8000"))
        .connection_mode(dialed(worker.wire_modes)?)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        });
    let object = built.build();
    object.set_status(match worker.state {
        State::Down => WorkerStatus::NotReady,
        State::Free | State::Over => WorkerStatus::Ready,
    });
    if worker.state == State::Over {
        object.set_overloaded(true);
    }
    Ok(Arc::new(object))
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
        let engine = PolicyEngine::new(applied);
        let objects: Result<Vec<_>, String> = (0..)
            .zip(workers)
            .map(|(place, worker)| object(place, worker))
            .collect();
        let list: Arc<[Arc<dyn Worker>]> = objects?.into();
        // One fact a worker, in the list's order. A worker states its
        // runtime on the wire its object dials.
        let stated = |(worker, object): (&Pooled, &Arc<dyn Worker>)| {
            let mut facts = FixedFacts::default();
            facts.vision = worker.vision;
            facts.demands = worker.facts;
            facts.demands.runtime = on_wire(worker.facts.runtime, *object.connection_mode());
            facts
        };
        let facts = workers.iter().zip(list.iter()).map(stated).collect();
        Ok(Self {
            engine,
            list,
            facts,
            cell: ListCell::default(),
            now: Instant::now(),
            seed: 0,
        })
    }

    /// The answer to one request under `constraints`: the workers its
    /// decisions named, or the refusal of a request no decision placed, in
    /// the words `refusal` keeps for a shed's reason and stage.
    pub fn answer(
        &mut self,
        mut constraints: Constraints<'_>,
        refusal: impl Fn(ShedReason, &str) -> Refusal,
    ) -> Result<Answer, String> {
        // A host of this engine holds a retry to its wire by its demand
        // alone, the number of its runtime on that wire, and states no wire
        // modes for a list it dials whole. A retry states the two together:
        // one without the other has no number here.
        match (
            constraints.demands.runtime,
            constraints.wire_modes.is_empty(),
        ) {
            (0, true) => {}
            (0, false) => return Err("a transport stated with no runtime".to_string()),
            (_, true) => return Err("a runtime stated with no transport".to_string()),
            (runtime, false) => {
                let wire = dialed(constraints.wire_modes)?;
                constraints.demands.runtime = on_wire(runtime, wire);
                constraints.wire_modes = WireModes::NONE;
            }
        }
        let mut placed = BTreeSet::new();
        for _ in 0..ROUNDS * self.list.len().max(1) {
            self.seed += 1;
            let id = self.seed.to_le_bytes();
            let asked = Asked {
                request: Request {
                    id: &id,
                    facts: RequestFacts {
                        model: MODEL,
                        seed: self.seed,
                        ..RequestFacts::default()
                    },
                    constraints,
                },
                lent: LentList {
                    workers: &self.list,
                    cell: &self.cell,
                    facts: Some(&self.facts),
                },
            };
            // A maker that says no request plain is answered whole.
            let decided = self.engine.decide_under(GatewayId::LOCAL, &asked, self.now);
            let decided = decided.map_err(|refused| format!("no decision: {refused:?}"))?;
            let (decided, _) = decided.whole().ok_or("a place alone")?;
            match decided.decision {
                Decision::Pick(pick) => {
                    let target = pick.targets.first().ok_or("a pick that names no worker")?;
                    let place = target.position as usize;
                    // A target names its worker twice: by its place in the
                    // list and by the id of the object there.
                    let object = self.list.get(place).ok_or("a worker of no place")?;
                    if target.id != WorkerId::from_address(object.url()) {
                        return Err(format!("place {place} under another worker's id"));
                    }
                    placed.insert(place);
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
