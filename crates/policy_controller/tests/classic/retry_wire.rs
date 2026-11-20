//! The classic retry-wire rows: a pool of regular workers, each with the
//! runtime and the transport it registered and its state, and a list of
//! requests, a retry among them stating the wire its retained plan was
//! built for. A row holds the gateway's answer to each request: the workers
//! it may be placed on, or its refusal.
//!
//! The replay registers the row's workers in a store. A worker states its
//! transport as its wire and its runtime as a number: the numbers are the
//! replay's own, as they are a registering host's, one a spelling. Each
//! request is decided through the door a host in process decides through
//! ([`Door`]), under `round_robin`: a retry states its transport as the
//! wire it speaks and its runtime as a hard demand
//! ([`Demands`]). A request is decided sixteen times the pool's size, as
//! its recorder placed it, and `placed` is the set of workers those
//! decisions named.
//!
//! A refusal replays by its status and its code. The crate sheds under a
//! reason and a stage's name, and the gateway keeps its own words for each
//! ([`refusal`]): the message and the retry hint are no part of a decision.

use std::{collections::BTreeSet, sync::Arc, time::Instant};

use serde::Deserialize;
use smg_policy_controller::{
    door::{Door, Request},
    spec::{BindingSpec, OverloadSpec},
    store::{
        GatewayId, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore,
    },
    AppliedSpec, Constraints, Decision, DemandFacts, Demands, LoadRow, PolicySpec, RequestFacts,
    ShedReason, WireModes, WorkerId,
};

/// What these rows record, as a sheet's head names it: the wire a retry
/// stays on.
pub const POLICY: &str = "retry-wire";

const MODEL: &str = "m";

/// A request is decided this many times the pool's size.
const ROUNDS: usize = 16;

/// The queue of a worker over its thresholds: above the default bound.
const QUEUED_OVER: u32 = 64;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum State {
    /// Ready and under its thresholds.
    Free,
    /// Ready and over its thresholds.
    Over,
    /// Not ready.
    Down,
}

#[derive(Debug, Deserialize)]
struct Worker {
    runtime: String,
    transport: String,
    state: State,
}

/// The wire a retry's retained plan was built for.
#[derive(Debug, Deserialize)]
struct Wire {
    runtime: String,
    transport: String,
}

#[derive(Debug, Deserialize)]
struct Asked {
    /// Absent on a first attempt, which stays on no wire.
    wire: Option<Wire>,
}

/// A refusal as the replay holds it: the response's status and its code.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Refusal {
    status: u16,
    code: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Answer {
    /// The places, in the row's list, of the workers a request may land on.
    Placed(BTreeSet<usize>),
    Refused(Refusal),
}

#[derive(Debug, Deserialize)]
struct Row {
    name: String,
    /// Whether a pool over its thresholds on every worker is shed.
    shed: bool,
    workers: Vec<Worker>,
    requests: Vec<Asked>,
    answers: Vec<Answer>,
}

#[derive(Debug, Deserialize)]
struct Sheet {
    rows: Vec<Row>,
}

/// The wire a transport's name states.
fn wire_of(transport: &str) -> Result<WireModes, String> {
    match transport {
        "grpc" => Ok(WireModes::GRPC),
        "zmq" => Ok(WireModes::ZMQ),
        other => Err(format!("no wire is named {other}")),
    }
}

/// The numbers of a row's runtimes, as a registering host states them: one
/// a spelling, none of them zero, which states no runtime.
#[derive(Default)]
struct Runtimes(Vec<String>);

impl Runtimes {
    fn number(&mut self, spelling: &str) -> u32 {
        let known = self.0.iter().position(|known| known == spelling);
        let at = known.unwrap_or_else(|| {
            self.0.push(spelling.to_string());
            self.0.len() - 1
        });
        u32::try_from(at + 1).unwrap_or(u32::MAX)
    }
}

/// The gateway's words for a shed. A pool over its thresholds on every
/// worker the request may take is overloaded. A model no worker serves is
/// no model, and so is a pool whose wire or whose demands left a retry
/// nobody. A pool left nobody by any other stage is unavailable.
fn refusal(reason: ShedReason, emptied_by: &str) -> Refusal {
    let (status, code) = match (reason, emptied_by) {
        (ShedReason::Saturated, _) => (503, "worker_overload_protection_shed"),
        (_, "none" | "wires") => (404, "model_not_found"),
        _ => (503, "no_available_workers"),
    };
    Refusal {
        status,
        code: code.to_string(),
    }
}

/// A row's pool behind the door, with each worker's id by its place.
struct Bed {
    door: Door,
    ids: Vec<WorkerId>,
    runtimes: Runtimes,
    now: Instant,
    seed: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form.
    fn new(row: &Row) -> Result<Self, String> {
        let mut document = PolicySpec::preset("round_robin").map_err(|err| err.to_string())?;
        document.generation = 1;
        let profile = document.profiles.keys().next().cloned();
        document.bindings = vec![BindingSpec {
            profile: Some(profile.ok_or("the preset's profile")?),
            overload: Some(OverloadSpec {
                shed_when_all_vetoed: row.shed,
                ..OverloadSpec::default()
            }),
            ..BindingSpec::default()
        }];
        let applied = AppliedSpec::validate(&document).map_err(|err| err.to_string())?;
        let store = Arc::new(MemoryStore::new());
        let now = Instant::now();
        let mut runtimes = Runtimes::default();
        let mut ids = Vec::with_capacity(row.workers.len());
        for (place, worker) in row.workers.iter().enumerate() {
            let url = format!("http://w{place}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: url.clone(),
                    engine: url,
                    incarnation: 1,
                    models: vec![MODEL.to_string()],
                    dp_size: 1,
                    wire_modes: wire_of(&worker.transport)?,
                    demand_facts: DemandFacts {
                        runtime: runtimes.number(&worker.runtime),
                    },
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
        for (id, worker) in ids.iter().zip(&row.workers) {
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
            runtimes,
            now,
            seed: 0,
        })
    }

    /// The answer to one request: the workers its decisions named, or the
    /// refusal of a request no decision placed.
    fn answer(&mut self, asked: &Asked) -> Result<Answer, String> {
        let constraints = match &asked.wire {
            None => Constraints::default(),
            Some(wire) => Constraints {
                wire_modes: wire_of(&wire.transport)?,
                demands: Demands {
                    runtime: self.runtimes.number(&wire.runtime),
                },
                ..Constraints::default()
            },
        };
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

/// What a row's requests are answered here, in their order.
fn replay(row: &Row) -> Result<Vec<Answer>, String> {
    let mut bed = Bed::new(row)?;
    row.requests.iter().map(|asked| bed.answer(asked)).collect()
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
        .filter_map(|row| {
            let why = match replay(row) {
                Ok(here) if here == row.answers => None,
                Ok(here) => Some(format!("recorded {:?}, replayed {here:?}", row.answers)),
                Err(why) => Some(format!("refused here: {why}")),
            };
            why.map(|why| format!("{source}: {}: {why}", row.name))
        })
        .collect();
    (sheet.rows.len(), defects)
}

/// The recorded sheet, whole: its rows are few. Each runtime's spelling is
/// replaced by a token: the replay compares the spellings for equality and
/// reads none, and the tokens keep the spellings' order.
pub const INLINE: &str = r#"{
  "recorded_from": "a4935a5acd15151641b1988e234016a95fd172da",
  "policy": "retry-wire",
  "rows": [
    {
      "name": "a-retry-stays-on-its-runtime-and-its-transport",
      "shed": false,
      "workers": [
        {"runtime": "sigma", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "zmq", "state": "free"},
        {"runtime": "tau", "transport": "grpc", "state": "free"}
      ],
      "requests": [
        {},
        {"wire": {"runtime": "zeta", "transport": "grpc"}},
        {"wire": {"runtime": "zeta", "transport": "zmq"}},
        {"wire": {"runtime": "sigma", "transport": "grpc"}},
        {"wire": {"runtime": "tau", "transport": "grpc"}},
        {"wire": {"runtime": "sigma", "transport": "zmq"}},
        {"wire": {"runtime": "tau", "transport": "zmq"}}
      ],
      "answers": [
        {"placed": [0, 1, 2, 3]},
        {"placed": [1]},
        {"placed": [2]},
        {"placed": [0]},
        {"placed": [3]},
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}},
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}}
      ]
    },
    {
      "name": "the-workers-of-a-wire-share-its-retries",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free"},
        {"runtime": "sigma", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "zmq", "state": "free"}
      ],
      "requests": [
        {"wire": {"runtime": "zeta", "transport": "grpc"}},
        {"wire": {"runtime": "sigma", "transport": "grpc"}}
      ],
      "answers": [
        {"placed": [0, 2]},
        {"placed": [1]}
      ]
    },
    {
      "name": "a-wire-whose-worker-is-down-is-not-left",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "down"},
        {"runtime": "sigma", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "zmq", "state": "free"}
      ],
      "requests": [
        {"wire": {"runtime": "zeta", "transport": "grpc"}}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers for model 'm' are unavailable (unhealthy or circuit breaker open)"}}
      ]
    },
    {
      "name": "a-worker-of-the-wire-over-its-thresholds-yields-to-one-under-them",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "over"},
        {"runtime": "zeta", "transport": "grpc", "state": "free"},
        {"runtime": "sigma", "transport": "grpc", "state": "free"}
      ],
      "requests": [
        {"wire": {"runtime": "zeta", "transport": "grpc"}}
      ],
      "answers": [
        {"placed": [1]}
      ]
    },
    {
      "name": "a-wire-over-on-every-worker-is-steered",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "over"},
        {"runtime": "zeta", "transport": "grpc", "state": "over"},
        {"runtime": "sigma", "transport": "grpc", "state": "free"}
      ],
      "requests": [
        {"wire": {"runtime": "zeta", "transport": "grpc"}}
      ],
      "answers": [
        {"placed": [0, 1]}
      ]
    },
    {
      "name": "a-wire-over-on-every-worker-is-shed-where-shedding-is-on",
      "shed": true,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "over"},
        {"runtime": "zeta", "transport": "grpc", "state": "over"},
        {"runtime": "sigma", "transport": "grpc", "state": "free"}
      ],
      "requests": [
        {"wire": {"runtime": "zeta", "transport": "grpc"}}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}}
      ]
    },
    {
      "name": "a-wire-no-worker-of-the-model-is-on",
      "shed": false,
      "workers": [
        {"runtime": "sigma", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "zmq", "state": "free"}
      ],
      "requests": [
        {"wire": {"runtime": "zeta", "transport": "grpc"}},
        {"wire": {"runtime": "tau", "transport": "grpc"}}
      ],
      "answers": [
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}},
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}}
      ]
    },
    {
      "name": "a-model-no-worker-serves",
      "shed": false,
      "workers": [],
      "requests": [
        {}
      ],
      "answers": [
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}}
      ]
    }
  ]
}"#;
