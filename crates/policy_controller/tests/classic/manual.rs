//! The classic manual rows: a pool of workers, the requests and the routing
//! keys a router holds on them, and a list of requests, keyed or not, under
//! one of the policy's four assignments.

use serde::Deserialize;
use smg_policy_controller::{
    Assignment, Candidate, Constraints, Engine, Health, Identity, Lease, Lifecycle, PolicySpec,
    PoolFacts, RequestFacts, Scratch, Side, WorkerId,
};

use crate::{Outcome, Recorded, Replayed};

/// The classic policy these rows record, as a sheet's head names it, and
/// the preset that replays it.
pub const POLICY: &str = "manual";

/// One worker of a row: the requests the host holds on it and the routing
/// keys that have a request in flight on it.
#[derive(Clone, Copy, Debug, Deserialize)]
struct WorkerState {
    in_flight: u32,
    keys: u16,
}

/// The role a request is placed under: the single-worker path, or one side
/// of a prefill and decode pair.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Role {
    Single,
    Prefill,
    Decode,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Step {
    /// One request: its routing key, if it carries one, and its role.
    Pick { key: Option<String>, role: Role },
    /// A worker stops being routable.
    Down(usize),
    /// The worker is routable again.
    Up(usize),
    /// The requests the host holds on each worker, from here on.
    Loads(Vec<u32>),
    /// The routing keys with a request in flight on each worker, from here
    /// on.
    Keys(Vec<u16>),
}

#[derive(Debug, Deserialize)]
pub struct Row {
    name: String,
    /// The assignment, in the spelling the gateway's configuration and this
    /// crate's document share.
    assignment: Assignment,
    workers: Vec<WorkerState>,
    steps: Vec<Step>,
    outcome: Outcome,
}

impl Row {
    /// Whether a request of the row is placed on a side of a pair.
    fn pairs(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, Step::Pick { role, .. } if *role != Role::Single))
    }

    /// The manual preset under the row's assignment. A row with a request
    /// on a side of a pair runs the preset on both sides, under the same
    /// rule.
    fn spec(&self) -> PolicySpec {
        let mut spec = if self.pairs() {
            PolicySpec::pair_presets(POLICY, POLICY)
        } else {
            PolicySpec::preset(POLICY)
        }
        .expect("the preset's form");
        let mut sessions = PolicySpec::preset(POLICY)
            .expect("the preset's form")
            .bindings
            .pop()
            .and_then(|binding| binding.sessions)
            .expect("the preset's sessions rule");
        sessions.assignment = self.assignment;
        spec.bindings
            .first_mut()
            .expect("the preset's binding")
            .sessions = Some(sessions);
        spec
    }
}

/// The engine of the row's assignment over the row's pool in its starting
/// state, with the host's side of a dispatch: the lease carries the session
/// and what the decision did with its key.
struct Bed {
    engine: Engine,
    pool: Vec<Candidate>,
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
                active_sessions: worker.keys,
                ..Candidate::default()
            })
            .collect();
        Ok(Self {
            engine,
            pool,
            scratch: Scratch::new(),
            clock: 1_000,
        })
    }

    /// One request decided and dispatched, as a host commits it: the pins
    /// move by the lease's session and verdict. The counts a host holds
    /// move by the row's own steps, as they did under the recorder.
    fn pick(&mut self, key: Option<&str>, seed: u64) -> usize {
        self.clock += 1_000;
        let facts = RequestFacts {
            model: "m",
            session_key: key.map_or(Identity::NONE, Identity::plain),
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
            Step::Pick { key, .. } => return Some(self.pick(key.as_deref(), seed)),
            Step::Down(i) => self.pool[*i].health.healthy = false,
            Step::Up(i) => self.pool[*i].health.healthy = true,
            Step::Loads(loads) => {
                assert_eq!(loads.len(), self.pool.len(), "a load of every worker");
                for (candidate, &load) in self.pool.iter_mut().zip(loads) {
                    candidate.in_flight = load;
                }
            }
            Step::Keys(keys) => {
                assert_eq!(keys.len(), self.pool.len(), "a count of every worker");
                for (candidate, &count) in self.pool.iter_mut().zip(keys) {
                    candidate.active_sessions = count;
                }
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
            // ran it from a fresh policy: a key claimed in one draw would
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

/// Rows copied from the recorded sheet, one or more of each assignment.
pub const INLINE: &str = r#"{
  "recorded_from": "1b141d03b5ca9097f54f0ebc4b95d72e810a62c3",
  "policy": "manual",
  "rows": [
    {
      "name": "manual-min-load-a-key-takes-the-fewest-in-flight-and-stays-past-the-load",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": "k1", "role": "single"}},
        {"pick": {"key": "k2", "role": "single"}},
        {"pick": {"key": "k1", "role": "single"}},
        {"pick": {"key": "k2", "role": "single"}}
      ],
      "outcome": {"sequence": [1, 1, 0, 1, 0]}
    },
    {
      "name": "manual-min-group-a-key-takes-the-fewest-keys-whatever-the-load",
      "assignment": "min_group",
      "workers": [{"in_flight": 0, "keys": 3}, {"in_flight": 5, "keys": 0}, {"in_flight": 0, "keys": 2}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}},
        {"keys": [3, 1, 2]},
        {"pick": {"key": "k2", "role": "single"}},
        {"keys": [3, 3, 2]},
        {"pick": {"key": "k3", "role": "single"}},
        {"pick": {"key": "k1", "role": "single"}}
      ],
      "outcome": {"sequence": [1, 1, 2, 1]}
    },
    {
      "name": "manual-delegate-with-no-policy-under-it-takes-the-fewest-in-flight",
      "assignment": "delegate",
      "workers": [{"in_flight": 4, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}},
        {"pick": {"key": null, "role": "single"}}
      ],
      "outcome": {"sequence": [1, 1]}
    },
    {
      "name": "manual-random-a-new-key-is-drawn-over-every-worker",
      "assignment": "random",
      "workers": [{"in_flight": 9, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [1991, 1978, 2031]}}
    },
    {
      "name": "manual-min-load-equal-loads-are-drawn",
      "assignment": "min_load",
      "workers": [{"in_flight": 2, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 5, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [3017, 2983, 0]}}
    },
    {
      "name": "manual-min-load-a-request-with-no-key-follows-the-load-and-pins-nothing",
      "assignment": "min_load",
      "workers": [{"in_flight": 3, "keys": 0}, {"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"pick": {"key": null, "role": "single"}},
        {"loads": [0, 5, 2]},
        {"pick": {"key": null, "role": "single"}},
        {"loads": [4, 5, 2]},
        {"pick": {"key": null, "role": "single"}}
      ],
      "outcome": {"sequence": [1, 0, 2]}
    },
    {
      "name": "manual-min-load-a-request-with-no-key-passes-over-a-worker-that-is-down",
      "assignment": "min_load",
      "workers": [{"in_flight": 0, "keys": 0}, {"in_flight": 3, "keys": 0}, {"in_flight": 2, "keys": 0}],
      "steps": [
        {"down": 0},
        {"pick": {"key": null, "role": "single"}},
        {"up": 0},
        {"pick": {"key": null, "role": "single"}}
      ],
      "outcome": {"sequence": [2, 0]}
    },
    {
      "name": "manual-random-a-request-with-no-key-is-never-drawn-on-a-worker-that-is-down",
      "assignment": "random",
      "workers": [{"in_flight": 9, "keys": 0}, {"in_flight": 0, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"down": 1},
        {"pick": {"key": null, "role": "single"}}
      ],
      "outcome": {"counts": {"draws": 6000, "counts": [3003, 0, 2997]}}
    },
    {
      "name": "manual-a-key-moves-off-a-worker-that-is-down-and-the-first-owner-leads-again",
      "assignment": "min_load",
      "workers": [{"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}},
        {"down": 0},
        {"pick": {"key": "k1", "role": "single"}},
        {"pick": {"key": "k1", "role": "single"}},
        {"up": 0},
        {"pick": {"key": "k1", "role": "single"}}
      ],
      "outcome": {"sequence": [0, 1, 1, 0]}
    },
    {
      "name": "manual-a-key-keeps-two-workers-and-drops-the-oldest-at-the-third",
      "assignment": "min_load",
      "workers": [{"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 3, "keys": 0}, {"in_flight": 4, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "single"}},
        {"down": 0},
        {"pick": {"key": "k1", "role": "single"}},
        {"down": 1},
        {"pick": {"key": "k1", "role": "single"}},
        {"up": 0},
        {"up": 1},
        {"pick": {"key": "k1", "role": "single"}},
        {"down": 1},
        {"pick": {"key": "k1", "role": "single"}}
      ],
      "outcome": {"sequence": [0, 1, 2, 1, 2]}
    },
    {
      "name": "manual-two-keys-pin-apart",
      "assignment": "min_load",
      "workers": [{"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "ka", "role": "single"}},
        {"loads": [5, 2, 3]},
        {"pick": {"key": "kb", "role": "single"}},
        {"loads": [9, 9, 0]},
        {"pick": {"key": "ka", "role": "single"}},
        {"pick": {"key": "kb", "role": "single"}}
      ],
      "outcome": {"sequence": [0, 1, 0, 1]}
    },
    {
      "name": "manual-one-key-pins-apart-under-the-prefill-and-the-decode-role",
      "assignment": "min_load",
      "workers": [{"in_flight": 1, "keys": 0}, {"in_flight": 2, "keys": 0}, {"in_flight": 3, "keys": 0}],
      "steps": [
        {"pick": {"key": "k1", "role": "prefill"}},
        {"loads": [5, 2, 3]},
        {"pick": {"key": "k1", "role": "decode"}},
        {"loads": [9, 9, 0]},
        {"pick": {"key": "k1", "role": "prefill"}},
        {"pick": {"key": "k1", "role": "decode"}},
        {"pick": {"key": "k1", "role": "single"}}
      ],
      "outcome": {"sequence": [0, 1, 0, 1, 2]}
    }
  ]
}"#;
