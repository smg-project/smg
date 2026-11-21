//! The classic routing-key override rows: a pool of workers, what a router
//! holds on them (the requests, the routing keys, the requests of one key),
//! and a list of requests, keyed or not, under a policy and one of the
//! sticky map's four assignments. The gateway places a keyed request of a
//! policy that reads no key itself by its sticky map, and puts a key with
//! two requests on its worker to the policy or the map's mode again.

use std::collections::BTreeMap;

use serde::Deserialize;
use smg_policy_controller::{
    document::{PolicySpec, SessionsSpec},
    engine::{
        Candidate, Constraints, Engine, Health, Identity, PoolFacts, RequestFacts, Scratch,
        WorkerId,
    },
    sessions::{Assignment, Recovery},
    tracking::{Lease, Lifecycle, Side},
};

use crate::{Outcome, Recorded, Replayed};

/// The rule these rows record, as a sheet's head names it.
pub const POLICY: &str = "sticky_override";

/// The header a document names for the key. A row's request states its key
/// as a fact, so the replay reads no header.
const HEADER: &str = "x-smg-routing-key";

/// One worker of a row: the requests the host holds on it and the routing
/// keys that have a request in flight on it.
#[derive(Clone, Copy, Debug, Deserialize)]
struct WorkerState {
    in_flight: u32,
    keys: u16,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and its model.
    Pick { key: Option<String>, model: String },
    /// A worker stops being routable.
    Down(usize),
    /// The worker is routable again.
    Up(usize),
    /// The requests the host holds on each worker, from here on.
    Loads(Vec<u32>),
    /// The requests of one routing key in flight on each worker, from here
    /// on: the count a rule's cap reads.
    Held { key: String, on: Vec<u16> },
}

#[derive(Debug, Deserialize)]
pub struct Row {
    name: String,
    /// The policy under the override: the preset of the same name here.
    policy: String,
    /// The assignment, in the spelling the gateway's configuration and this
    /// crate's document share.
    assignment: Assignment,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
    outcome: Outcome,
}

impl Row {
    /// The preset of the row's policy with a sessions rule on its binding,
    /// as a document states the override: the row's assignment, the cap a
    /// rule takes when its document names none, and the pin order of the
    /// assignment. The latest worker leads under `delegate`, the first
    /// owner under the map's own modes.
    fn spec(&self) -> PolicySpec {
        let recovery = if self.assignment == Assignment::Delegate {
            Recovery::Latest
        } else {
            Recovery::Original
        };
        let sessions: SessionsSpec = serde_json::from_value(serde_json::json!({
            "sticky_header": HEADER,
            "assignment": self.assignment,
            "recovery": recovery,
        }))
        .expect("the document's form of a sessions rule");
        let mut spec = PolicySpec::preset(&self.policy).expect("the preset's form");
        spec.view.headers = vec![HEADER.into()];
        spec.bindings
            .first_mut()
            .expect("the preset's binding")
            .sessions = Some(sessions);
        spec
    }
}

/// The engine of the row over the row's pool in its starting state, with
/// the host's side of a dispatch: the lease carries the session and what
/// the decision did with its key.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
    /// The routing keys each worker held at the row's start.
    keys: Vec<u16>,
    /// The requests of each key the row's steps hold on each worker.
    held: BTreeMap<String, Vec<u16>>,
    scratch: Scratch,
    /// The host's clock, microseconds: each request has its own instant.
    clock: u64,
}

impl Bed {
    /// The bed, or the validator's words when this crate does not build the
    /// row's form.
    fn new(row: &Row) -> Result<Self, String> {
        let engine = row.spec().validate().map_err(|err| err.to_string())?;
        let pool = row
            .workers
            .iter()
            .enumerate()
            .map(|(i, worker)| Candidate {
                position: i as u32,
                id: WorkerId(1_000 + i as u64),
                key: 1_000 + i as u32,
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                in_flight: worker.in_flight,
                ..Candidate::default()
            })
            .collect();
        Ok(Self {
            engine,
            pool,
            keys: row.workers.iter().map(|worker| worker.keys).collect(),
            held: BTreeMap::new(),
            scratch: Scratch::new(),
            clock: 1_000,
        })
    }

    /// One request decided and dispatched, as a host commits it: the pins
    /// move by the lease's session and verdict. Each row of the pool states
    /// what the host counts for this request: the requests of its own key on
    /// the worker, and the keys the worker holds. The counts move by the
    /// row's own steps, as they did under the recorder.
    fn pick(&mut self, key: Option<&str>, model: &str, seed: u64) -> usize {
        self.clock += 1_000;
        let own = key.and_then(|key| self.held.get(key));
        for (i, candidate) in self.pool.iter_mut().enumerate() {
            candidate.session_in_flight = own.map_or(0, |on| on[i]);
            let held = self.held.values().filter(|on| on[i] > 0).count();
            candidate.active_sessions = self.keys[i] + held as u16;
        }
        let facts = RequestFacts {
            model,
            session_key: key.map_or(Identity::NONE, Identity::plain),
            session_counted: true,
            seed,
            now_micros: self.clock,
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
        let pick = decision.pick().expect("a pick");
        let target = *pick.targets.first().expect("a target");
        let lease = Lease {
            issued_micros: self.clock,
            session: facts.session(),
            session_verdict: pick.session_verdict,
            ..Lease::default()
        };
        self.engine.on_dispatched(&lease, &target, Side::Single, 0);
        target.position as usize
    }

    /// One step of the row; a request answers the worker it was placed on.
    fn step(&mut self, step: &Step, seed: u64) -> Option<usize> {
        match step {
            Step::Pick { key, model } => return Some(self.pick(key.as_deref(), model, seed)),
            Step::Down(i) => self.pool[*i].health.healthy = false,
            Step::Up(i) => self.pool[*i].health.healthy = true,
            Step::Loads(loads) => {
                assert_eq!(loads.len(), self.pool.len(), "a load of every worker");
                for (candidate, &load) in self.pool.iter_mut().zip(loads) {
                    candidate.in_flight = load;
                }
            }
            Step::Held { key, on } => {
                assert_eq!(on.len(), self.pool.len(), "a count of every worker");
                self.held.insert(key.clone(), on.clone());
            }
        }
        None
    }
}

/// The row run once from a fresh engine: the worker each request went to,
/// or the validator's words. `draw` separates the seeds of a drawn row's
/// runs; a sequence row is run `0`.
fn run(row: &Row, draw: u64) -> Result<Vec<usize>, String> {
    let mut bed = Bed::new(row)?;
    Ok(row
        .steps
        .iter()
        .enumerate()
        .filter_map(|(n, step)| bed.step(step, draw * 1_000 + n as u64 + 1))
        .collect())
}

/// This crate's outcome for the row, in the shape of the recorded one, or
/// its refusal of the row's form.
fn replay(row: &Row) -> Replayed {
    match &row.outcome {
        Outcome::Sequence(_) => match run(row, 0) {
            Ok(picked) => Replayed::Ran(Outcome::Sequence(picked)),
            Err(why) => Replayed::Refused(why),
        },
        Outcome::Counts { draws, .. } => {
            // Each draw runs the row from a fresh engine, as the recorder
            // ran it from a fresh registry: a key pinned in one draw would
            // decide the next. The row's last request is the one counted.
            let mut counts = vec![0u32; row.workers.len()];
            for draw in 1..=u64::from(*draws) {
                match run(row, draw) {
                    Ok(picked) => {
                        counts[*picked.last().expect("a drawn row ends with a request")] += 1;
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
        replay(self)
    }
}

/// Rows copied from the recorded sheet: a pin under the cap and under two
/// models, a key seen first under three assignments, each pin order after a
/// failover, and at the cap a pin that follows, a pick on the pinned worker
/// again, a pick on a worker at the cap, every worker at the cap and the
/// policy's draw.
pub const INLINE: &str = r#"{
  "recorded_from": "26f90083091e4dd791420febd88beda3bef3faac",
  "policy": "sticky_override",
  "rows": [
    {
      "name": "sticky-min-load-a-key-keeps-its-worker-under-the-cap-past-the-load",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 1, 0]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 1, 0, 1]}
    },
    {
      "name": "sticky-one-key-pins-apart-under-two-models",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m1"}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m2"}},
        {"loads": [9, 9, 0]},
        {"pick": {"key": "k1", "model": "m1"}},
        {"pick": {"key": "k1", "model": "m2"}}
      ],
      "outcome": {"sequence": [1, 0, 1, 0]}
    },
    {
      "name": "sticky-min-group-a-new-key-takes-the-fewest-keys-whatever-the-load",
      "policy": "random",
      "assignment": "min_group",
      "workers": [{"in_flight": 0, "keys": 2}, {"in_flight": 5, "keys": 0}, {"in_flight": 0, "keys": 1}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 1, 0]}},
        {"held": {"key": "x1", "on": [0, 1, 0]}},
        {"held": {"key": "x2", "on": [0, 1, 0]}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 2, 1]}
    },
    {
      "name": "sticky-random-a-new-key-is-drawn-over-every-worker",
      "policy": "random",
      "assignment": "random",
      "workers": [{"in_flight": 9, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [2020, 1989, 1991]}}
    },
    {
      "name": "sticky-min-load-a-key-moves-off-a-worker-that-is-down-and-the-first-owner-leads-again",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": 0},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 0},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [0, 1, 1, 0]}
    },
    {
      "name": "sticky-delegate-a-new-key-is-the-policy-s-pick-and-stays-past-the-load",
      "policy": "power_of_two",
      "assignment": "delegate",
      "workers": [{"in_flight": 5, "keys": 0}, {"in_flight": 0, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"loads": [0, 5]},
        {"pick": {"key": "k1", "model": "m"}},
        {"pick": {"key": null, "model": "m"}},
        {"pick": {"key": "k2", "model": "m"}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 1, 0, 0, 1]}
    },
    {
      "name": "sticky-delegate-the-latest-worker-leads-after-a-failover",
      "policy": "power_of_two",
      "assignment": "delegate",
      "workers": [{"in_flight": 0, "keys": 0}, {"in_flight": 5, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"down": 0},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 0},
        {"pick": {"key": "k1", "model": "m"}},
        {"down": 1},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 1},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [0, 1, 1, 0, 1]}
    },
    {
      "name": "sticky-min-load-at-the-cap-the-request-goes-to-the-least-loaded-and-the-pin-follows",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 2, 0]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [1, 2, 0]}},
        {"loads": [5, 0, 2]},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 0, 0]}
    },
    {
      "name": "sticky-min-load-at-the-cap-the-least-loaded-is-the-pinned-worker-again",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 2, 0]}},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [0, 3, 0]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 1, 1]}
    },
    {
      "name": "sticky-min-load-at-the-cap-the-pin-does-not-move-to-a-worker-at-the-cap",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 2, 0]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 1, 0]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 0, 1]}
    },
    {
      "name": "sticky-min-load-every-worker-at-the-cap",
      "policy": "random",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 2, 2]}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "model": "m"}},
        {"held": {"key": "k1", "on": [2, 1, 2]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"sequence": [1, 0, 1]}
    },
    {
      "name": "sticky-delegate-at-the-cap-the-policy-draws-over-every-worker",
      "policy": "random",
      "assignment": "delegate",
      "workers": [{"in_flight": 0, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 0, "keys": 0}],
      "steps": [
        {"down": 1},
        {"down": 2},
        {"pick": {"key": "k1", "model": "m"}},
        {"up": 1},
        {"up": 2},
        {"held": {"key": "k1", "on": [2, 0, 0]}},
        {"pick": {"key": "k1", "model": "m"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [1959, 2013, 2028]}}
    }
  ]
}"#;
