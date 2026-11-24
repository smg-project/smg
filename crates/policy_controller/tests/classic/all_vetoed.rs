//! The classic rows of a pool whose every usable worker is over its overload
//! thresholds: a pool of workers, the requests a router holds on each and
//! its state, the policy the pool runs under, whether shedding is on, and a
//! list of requests. The gateway answers such a pool by one rule under every
//! policy, so one sheet holds it and a row names its policy.
//!
//! A ring row (`ring_key`) is a request placed by its routing key on the
//! hash ring. Two rings put one key on different workers, so such a row
//! lists its workers in the key's ring order and every number in it counts
//! in that order: `0` is the key's own worker and `1` the ring's next
//! member, on whatever ring replays the row. Such a row is replayed twice:
//! on this crate's own ring, and on a ring the request's caller lends,
//! built in the row's order ([`replay_on_a_lent_ring`]).

use serde::Deserialize;
use smg_policy_controller::{
    document::{Assignment, OverloadSpec, PolicySpec},
    engine::{
        Candidate, Constraints, Decision, Engine, Health, Identity, PoolFacts, RequestFacts,
        Scratch, ShedReason, WorkerId,
    },
    policies::consistent_hashing::LentRing,
    stages::Lifecycle,
    tracking::{Lease, Side},
};

use crate::{verdict, Outcome, Recorded, Replayed, Rows};

/// What a sheet's head names these rows: the rule they record, under every
/// policy.
pub const POLICY: &str = "all_vetoed";

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
    /// Ready, over its thresholds and doubted by the liveness tracker.
    StalledOver,
    /// Ready and under its thresholds, behind an open circuit breaker.
    Breaker,
    /// Not ready.
    Down,
}

impl State {
    fn health(self) -> Health {
        Health {
            healthy: self != Self::Down,
            breaker_open: self == Self::Breaker,
            overloaded: matches!(self, Self::Over | Self::StalledOver),
            stalled: matches!(self, Self::Stalled | Self::StalledOver),
            ..Health::default()
        }
    }
}

/// One worker of a row: the requests the host holds on it and its state.
#[derive(Clone, Copy, Debug, Deserialize)]
struct WorkerState {
    in_flight: u32,
    state: State,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and the workers an
    /// earlier attempt of it already tried.
    Pick {
        key: Option<String>,
        tried: Vec<usize>,
    },
    /// The requests the host holds on each worker, from here on.
    Loads(Vec<u32>),
    /// Each worker's state, from here on.
    States(Vec<State>),
}

#[derive(Debug, Deserialize)]
pub struct Row {
    name: String,
    /// The policy, in the spelling the gateway's configuration and this
    /// crate's presets share.
    policy: String,
    /// Whether a pool over on every worker is shed: the binding's
    /// `overload.shed_when_all_vetoed`.
    shed: bool,
    /// A ring row's routing key: the row lists its workers in this key's
    /// ring order, the key's own worker first and then each next member,
    /// and its loads, states, tried lists and answers count in that order.
    #[serde(default)]
    ring_key: Option<String>,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
    outcome: Outcome,
}

impl Row {
    /// The row's preset under the row's shed word. The recorder runs the
    /// manual policy under its `min_load` assignment, so that no row starts
    /// with a draw.
    fn spec(&self) -> Result<PolicySpec, String> {
        let mut spec = PolicySpec::preset(&self.policy).map_err(|err| err.to_string())?;
        let binding = spec.bindings.first_mut().expect("the preset's binding");
        binding.overload = Some(OverloadSpec {
            shed_when_all_vetoed: self.shed,
            ..OverloadSpec::default()
        });
        if let Some(sessions) = binding.sessions.as_mut() {
            sessions.assignment = Assignment::MinLoad;
        }
        Ok(spec)
    }
}

/// The worker the row lists at `i`, as the view names it.
fn id(i: usize) -> WorkerId {
    WorkerId(1_000 + i as u64)
}

/// The address of the worker a ring row lists at `i`, under a lent ring.
fn address(i: usize) -> String {
    format!("worker-{i}")
}

/// A caller's hash ring over a ring row's workers, as many as it counts:
/// it reaches them in the row's own order from wherever a key stands, the
/// order the recorder found for the row's key on the gateway's ring.
#[derive(Debug)]
struct InRowOrder(usize);

impl LentRing for InRowOrder {
    fn position(&self, _: &str) -> u64 {
        1
    }

    fn walk(&self, _: u64, take: &mut dyn FnMut(&str) -> bool) {
        let _ = (0..self.0).any(|i| take(&address(i)));
    }
}

/// The engine of the row's policy over the row's pool in its starting state,
/// with the host's side of a dispatch.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
    /// Where each worker of the row stands in the pool: the row's own
    /// order, or the key's ring order on this crate's ring for a ring row.
    places: Vec<usize>,
    /// The ring the row's requests are lent, built in the row's order:
    /// the pool's rows then carry their addresses' ids.
    lent: Option<InRowOrder>,
    scratch: Scratch,
    /// The host's clock, microseconds: each request has its own instant.
    clock: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's policy. Under `lent` its requests are lent a ring built in the
    /// row's order.
    fn new(row: &Row, lent: bool) -> Result<Self, String> {
        let engine = row.spec()?.validate().map_err(|err| err.to_string())?;
        let places = match &row.ring_key {
            Some(key) if !lent => ring_order(row, key)?,
            _ => (0..row.workers.len()).collect(),
        };
        let mut pool: Vec<Candidate> = (0..row.workers.len())
            .map(|place| Candidate {
                position: place as u32,
                id: if lent {
                    WorkerId::from_address(&address(place))
                } else {
                    id(place)
                },
                key: 1_000 + place as u32,
                ..Candidate::default()
            })
            .collect();
        for (&place, worker) in places.iter().zip(&row.workers) {
            pool[place].health = worker.state.health();
            pool[place].in_flight = worker.in_flight;
        }
        Ok(Self {
            engine,
            pool,
            places,
            lent: lent.then_some(InRowOrder(row.workers.len())),
            scratch: Scratch::new(),
            clock: 1_000,
        })
    }

    /// One request decided and, when it is placed, dispatched as a host
    /// commits it. The answer is the worker's place in the row, or a number
    /// past the last worker for a request placed nowhere: the count of the
    /// workers for a shed as saturated, the count plus one for any other.
    fn pick(&mut self, key: Option<&str>, tried: &[usize], seed: u64) -> usize {
        self.clock += 1_000;
        let tried: Vec<WorkerId> = tried
            .iter()
            .map(|&i| self.pool[self.places[i]].id)
            .collect();
        let ring = self.lent.as_ref().map(|ring| ring as &dyn LentRing);
        let at = ring.zip(key).map(|(ring, key)| ring.position(key));
        let facts = RequestFacts {
            model: "m",
            session_key: key.map_or(Identity::NONE, Identity::plain),
            attempt: u8::from(!tried.is_empty()),
            tried: &tried,
            seed,
            now_micros: self.clock,
            ring,
            ring_positions: at.as_slice(),
            ..RequestFacts::default()
        };
        let pool_facts = PoolFacts {
            pool_size: self.pool.len() as u32,
            generation: 1,
            ..PoolFacts::default()
        };
        let decision = self.engine.decide(
            &facts,
            &pool_facts,
            &self.pool,
            &Constraints::default(),
            &mut self.scratch,
        );
        let pick = match &decision {
            Decision::Pick(pick) => pick,
            Decision::Shed(shed) if shed.reason == ShedReason::Saturated => return self.pool.len(),
            Decision::Shed(_) | Decision::Queue(_) => return self.pool.len() + 1,
        };
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            issued_micros: self.clock,
            session: facts.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        self.engine.on_dispatched(&lease, &target, Side::Single, 0);
        let place = target.position as usize;
        self.places
            .iter()
            .position(|&p| p == place)
            .expect("a worker of the row")
    }

    /// One step of the row; a request answers where it was placed.
    fn step(&mut self, step: &Step, seed: u64) -> Option<usize> {
        match step {
            Step::Pick { key, tried } => return Some(self.pick(key.as_deref(), tried, seed)),
            Step::Loads(loads) => {
                assert_eq!(loads.len(), self.pool.len(), "a load of every worker");
                for (&place, &load) in self.places.iter().zip(loads) {
                    self.pool[place].in_flight = load;
                }
            }
            Step::States(states) => {
                assert_eq!(states.len(), self.pool.len(), "a state of every worker");
                for (&place, state) in self.places.iter().zip(states) {
                    self.pool[place].health = state.health();
                }
            }
        }
        None
    }
}

/// The ring order of `key` over the row's pool on this crate's ring: the
/// worker the key lands on with every worker free and idle, then with the
/// members found so far down, as the recorder found it on the gateway's
/// ring. Each probe runs on a fresh engine of the row's policy.
fn ring_order(row: &Row, key: &str) -> Result<Vec<usize>, String> {
    let workers = row.workers.len();
    let mut order = Vec::with_capacity(workers);
    while order.len() < workers {
        let probe = Row {
            name: String::new(),
            policy: row.policy.clone(),
            shed: false,
            ring_key: None,
            workers: (0..workers)
                .map(|place| WorkerState {
                    in_flight: 0,
                    state: if order.contains(&place) {
                        State::Down
                    } else {
                        State::Free
                    },
                })
                .collect(),
            steps: Vec::new(),
            outcome: Outcome::Sequence(Vec::new()),
        };
        let next = Bed::new(&probe, false)?.pick(Some(key), &[], 1);
        if next >= workers || order.contains(&next) {
            return Err(format!(
                "the ring's order of {key}: the next member is a worker not yet placed, got {next}"
            ));
        }
        order.push(next);
    }
    Ok(order)
}

/// The row run once from a fresh engine: where each request was placed, or
/// the validator's words. `draw` separates the seeds of a drawn row's runs;
/// a sequence row is run `0`.
fn run(row: &Row, draw: u64, lent: bool) -> Result<Vec<usize>, String> {
    let mut bed = Bed::new(row, lent)?;
    Ok(row
        .steps
        .iter()
        .enumerate()
        .filter_map(|(n, step)| bed.step(step, draw * 1_000 + n as u64 + 1))
        .collect())
}

/// This crate's outcome for the row, in the shape of the recorded one, or
/// its refusal of the row's policy.
fn replay(row: &Row, lent: bool) -> Replayed {
    match &row.outcome {
        Outcome::Sequence(_) => match run(row, 0, lent) {
            Ok(picked) => Replayed::Ran(Outcome::Sequence(picked)),
            Err(why) => Replayed::Refused(why),
        },
        Outcome::Counts { draws, .. } => {
            // Each draw runs the row from a fresh engine, as the recorder
            // ran it from a fresh policy. The row's last request is the one
            // counted; a drawn row places every request.
            let mut counts = vec![0u32; row.workers.len()];
            for draw in 1..=u64::from(*draws) {
                match run(row, draw, lent) {
                    Ok(picked) => {
                        let last = *picked.last().expect("a drawn row ends with a request");
                        match counts.get_mut(last) {
                            Some(count) => *count += 1,
                            None => {
                                return Replayed::Ran(Outcome::Sequence(picked));
                            }
                        }
                    }
                    Err(why) => return Replayed::Refused(why),
                }
            }
            Replayed::Ran(Outcome::Counts {
                draws: *draws,
                counts,
            })
        }
    }
}

impl Recorded for Row {
    const POLICY: &'static str = POLICY;

    fn name(&self) -> &str {
        &self.name
    }

    fn outcome(&self) -> &Outcome {
        &self.outcome
    }

    fn replay(&self) -> Replayed {
        replay(self, false)
    }
}

/// The ring rows of a sheet replayed on a ring their requests' caller
/// lends, built in each row's order: how many they are, and each one whose
/// answer is not the recorded one.
pub fn replay_on_a_lent_ring(text: &str) -> (usize, Vec<String>) {
    let sheet: Rows<Row> = serde_json::from_str(text).expect("a sheet of these rows");
    let ring_rows = sheet.rows.iter().filter(|row| row.ring_key.is_some());
    let verdicts: Vec<Option<String>> = ring_rows
        .map(|row| verdict(&row.outcome, &replay(row, true)))
        .collect();
    (verdicts.len(), verdicts.into_iter().flatten().collect())
}

/// Rows copied from the recorded sheet: the rule under the policies that
/// answer such a pool themselves and under the ones the placement filters
/// for, the shed, a tie, the other vetoes beside the overload one, a retry
/// and a key; and the ring rows, a routing key under consistent hashing.
pub const INLINE: &str = r#"{
  "recorded_from": "d9f8018881cc54c9b0f6e993849a59f8f7c54ab5",
  "policy": "all_vetoed",
  "rows": [
    {
      "name": "round-robin-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "least-load-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "least_load",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "cache-aware-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "cache_aware",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "manual-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "manual",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "consistent-hashing-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "consistent_hashing",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "prefix-hash-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "prefix_hash",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "bucket-a-pool-over-on-every-worker-goes-to-the-fewest-in-flight",
      "policy": "bucket",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "tried": []}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "round-robin-a-pool-over-on-every-worker-is-shed-under-shedding",
      "policy": "round_robin",
      "shed": true,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "prefix-hash-a-pool-over-on-every-worker-is-shed-under-shedding",
      "policy": "prefix_hash",
      "shed": true,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "bucket-a-pool-over-on-every-worker-is-shed-under-shedding",
      "policy": "bucket",
      "shed": true,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "round-robin-equal-loads-over-on-every-worker-are-drawn",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 2, "state": "over"}, {"in_flight": 2, "state": "over"}, {"in_flight": 5, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [2988, 3012, 0]}}
    },
    {
      "name": "round-robin-one-worker-under-its-thresholds-takes-every-request",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 0, "state": "over"}, {"in_flight": 9, "state": "free"}, {"in_flight": 0, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"pick": {"key": null, "tried": []}},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [1, 1, 1]}
    },
    {
      "name": "round-robin-a-worker-that-is-down-is-never-the-fewest-in-flight",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 0, "state": "down"}, {"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 1, 3]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [2, 1]}
    },
    {
      "name": "round-robin-a-doubted-worker-is-passed-over-for-one-over-its-thresholds",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 0, "state": "stalled"}, {"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": []}},
        {"loads": [0, 1, 3]},
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [2, 1]}
    },
    {
      "name": "round-robin-a-pool-with-nothing-ready-finds-no-worker",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 0, "state": "down"}, {"in_flight": 0, "state": "down"}],
      "steps": [
        {"pick": {"key": null, "tried": []}}
      ],
      "outcome": {"sequence": [3]}
    },
    {
      "name": "round-robin-a-tried-worker-is-taken-again-when-it-holds-the-fewest",
      "policy": "round_robin",
      "shed": false,
      "workers": [{"in_flight": 3, "state": "over"}, {"in_flight": 1, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": null, "tried": [1]}},
        {"pick": {"key": null, "tried": [1, 2]}}
      ],
      "outcome": {"sequence": [1, 1]}
    },
    {
      "name": "manual-a-key-goes-to-the-fewest-in-flight-and-returns-to-its-worker",
      "policy": "manual",
      "shed": false,
      "workers": [{"in_flight": 1, "state": "free"}, {"in_flight": 2, "state": "free"}, {"in_flight": 3, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}},
        {"states": ["over", "over", "over"]},
        {"loads": [9, 2, 3]},
        {"pick": {"key": "k1", "tried": []}},
        {"pick": {"key": "k1", "tried": []}},
        {"states": ["free", "free", "free"]},
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [0, 1, 1, 0]}
    },
    {
      "name": "manual-a-key-is-shed-with-every-worker-over-under-shedding",
      "policy": "manual",
      "shed": true,
      "workers": [{"in_flight": 1, "state": "free"}, {"in_flight": 2, "state": "free"}, {"in_flight": 3, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}},
        {"states": ["over", "over", "over"]},
        {"pick": {"key": "k1", "tried": []}},
        {"states": ["free", "free", "free"]},
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [0, 3, 0]}
    },
    {
      "name": "consistent-hashing-a-key-keeps-its-worker-over-its-thresholds-beside-free-workers",
      "policy": "consistent_hashing",
      "shed": false,
      "ring_key": "k1",
      "workers": [{"in_flight": 5, "state": "over"}, {"in_flight": 4, "state": "free"}, {"in_flight": 0, "state": "free"}, {"in_flight": 2, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [0]}
    },
    {
      "name": "consistent-hashing-a-key-keeps-its-worker-with-every-worker-over",
      "policy": "consistent_hashing",
      "shed": false,
      "ring_key": "k1",
      "workers": [{"in_flight": 5, "state": "over"}, {"in_flight": 4, "state": "over"}, {"in_flight": 0, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [0]}
    },
    {
      "name": "consistent-hashing-a-key-leaves-its-worker-over-its-thresholds-under-shedding",
      "policy": "consistent_hashing",
      "shed": true,
      "ring_key": "k1",
      "workers": [{"in_flight": 5, "state": "over"}, {"in_flight": 4, "state": "free"}, {"in_flight": 0, "state": "free"}, {"in_flight": 2, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "consistent-hashing-a-key-is-shed-with-every-worker-over-under-shedding",
      "policy": "consistent_hashing",
      "shed": true,
      "ring_key": "k1",
      "workers": [{"in_flight": 5, "state": "over"}, {"in_flight": 4, "state": "over"}, {"in_flight": 0, "state": "over"}, {"in_flight": 2, "state": "over"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [4]}
    },
    {
      "name": "consistent-hashing-a-key-passes-to-the-next-member-when-its-worker-is-down",
      "policy": "consistent_hashing",
      "shed": false,
      "ring_key": "k1",
      "workers": [{"in_flight": 0, "state": "down"}, {"in_flight": 4, "state": "free"}, {"in_flight": 0, "state": "free"}, {"in_flight": 2, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "consistent-hashing-a-retry-that-tried-the-keys-worker-passes-the-key-on",
      "policy": "consistent_hashing",
      "shed": false,
      "ring_key": "k1",
      "workers": [{"in_flight": 5, "state": "over"}, {"in_flight": 4, "state": "free"}, {"in_flight": 0, "state": "free"}, {"in_flight": 2, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": [0]}}
      ],
      "outcome": {"sequence": [1]}
    },
    {
      "name": "consistent-hashing-a-key-keeps-its-worker-when-the-next-member-is-over-too",
      "policy": "consistent_hashing",
      "shed": false,
      "ring_key": "k1",
      "workers": [{"in_flight": 5, "state": "over"}, {"in_flight": 4, "state": "over"}, {"in_flight": 0, "state": "free"}, {"in_flight": 2, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [0]}
    },
    {
      "name": "consistent-hashing-a-key-passes-to-a-next-member-over-its-thresholds-when-its-worker-is-down",
      "policy": "consistent_hashing",
      "shed": false,
      "ring_key": "k1",
      "workers": [{"in_flight": 0, "state": "down"}, {"in_flight": 4, "state": "over"}, {"in_flight": 0, "state": "free"}, {"in_flight": 2, "state": "free"}],
      "steps": [
        {"pick": {"key": "k1", "tried": []}}
      ],
      "outcome": {"sequence": [1]}
    }
  ]
}"#;
