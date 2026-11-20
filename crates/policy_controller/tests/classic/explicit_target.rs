//! The classic explicit-target rows: a pool of regular workers, each with
//! its state, and a list of requests that each name a worker by the
//! gateway's target header. A row holds the gateway's answer to each
//! request: the worker it is placed on, or its refusal.
//!
//! The header's value is an index into the workers the gateway reads as
//! available when the request arrives, in the pool's order. A row records
//! that reading beside each worker's state (`available`), and the replay
//! resolves the index over it as a host does. It states the worker the
//! index names as a hard demand ([`Demands::target`]). An index that names
//! nobody has no worker to state: the replay states a worker no row has,
//! and the crate leaves the request nobody. A pool with no available worker
//! has no list to index: the request states no target, and the pool's own
//! rule answers.
//!
//! The rows replay on the hard demands' bed ([`Bed`]). A refusal replays in
//! the gateway's words for a pool ([`pool_refusal`]): the demand has none of
//! its own, and a request whose index names nobody is answered as a pool
//! with nobody available.

use serde::Deserialize;
use smg_policy_controller::{Constraints, DemandFacts, Demands, WireModes, WorkerId};

use super::demand_bed::{self, pool_refusal, Answer, Bed, Pooled, Refusal, Replayed, State};

/// What these rows record, as a sheet's head names it: the worker a request
/// names.
pub const POLICY: &str = "explicit-target";

#[derive(Debug, Deserialize)]
struct Worker {
    state: State,
    /// Whether the gateway reads the worker as available: the header
    /// indexes those workers alone.
    available: bool,
}

#[derive(Debug, Deserialize)]
struct Asked {
    /// The header's value.
    target: String,
}

#[derive(Debug, Deserialize)]
pub struct Row {
    name: String,
    /// Whether a pool over its thresholds on every worker is shed.
    shed: bool,
    workers: Vec<Worker>,
    requests: Vec<Asked>,
    answers: Vec<Answer>,
}

impl Replayed for Row {
    fn name(&self) -> &str {
        &self.name
    }

    fn recorded(&self) -> &[Answer] {
        &self.answers
    }

    fn replay(&self) -> Result<Vec<Answer>, String> {
        let pooled: Vec<Pooled> = self
            .workers
            .iter()
            .map(|worker| Pooled {
                wire_modes: WireModes::HTTP,
                state: worker.state,
                vision: false,
                facts: DemandFacts::default(),
            })
            .collect();
        let mut bed = Bed::new(self.shed, &pooled)?;
        // The list the header indexes: the places of the available workers.
        let listed: Vec<usize> = (0..self.workers.len())
            .filter(|place| self.workers[*place].available)
            .collect();
        // A worker no row has, for an index that names nobody.
        let nobody = (0..).map(WorkerId).find(|id| !bed.ids().contains(id));
        let nobody = nobody.ok_or("an id no row has")?;
        let mut answers = Vec::with_capacity(self.requests.len());
        for asked in &self.requests {
            let named = asked.target.parse::<usize>().ok();
            let named = named.and_then(|index| listed.get(index));
            // No available worker, no list: the request states no target.
            let target =
                (!listed.is_empty()).then(|| named.map_or(nobody, |place| bed.ids()[*place]));
            let constraints = Constraints {
                demands: Demands {
                    target,
                    ..Demands::default()
                },
                ..Constraints::default()
            };
            answers.push(bed.answer(constraints, pool_refusal)?);
        }
        Ok(answers)
    }
}

/// The rows of a sheet replayed: how many, and what did not replay.
pub fn replay_rows(text: &str, source: &str) -> (usize, Vec<String>) {
    demand_bed::replay_rows::<Row>(text, source, POLICY)
}

/// No sheet names a worker that is over its thresholds or down: the
/// gateway's list holds neither. Named to the crate, such a worker is the
/// one row its demand leaves, so the binding's rule for a pool the veto
/// empties answers: the worker takes the request where the binding steers,
/// and the pool is shed where it sheds. A named worker that is down leaves
/// nobody.
#[test]
fn a_named_worker_the_veto_refuses_is_answered_by_the_bindings_rule() {
    let pool = |named| {
        [named, State::Free].map(|state| Pooled {
            wire_modes: WireModes::HTTP,
            state,
            vision: false,
            facts: DemandFacts::default(),
        })
    };
    for (shed, named, answer) in [
        (false, State::Over, Answer::Placed([0].into())),
        (
            true,
            State::Over,
            Answer::Refused(Refusal::of(503, "worker_overload_protection_shed")),
        ),
        (
            false,
            State::Down,
            Answer::Refused(Refusal::of(503, "no_available_workers")),
        ),
    ] {
        let mut bed = Bed::new(shed, &pool(named)).expect("the bed");
        let demands = Demands {
            target: Some(bed.ids()[0]),
            ..Demands::default()
        };
        let constraints = Constraints {
            demands,
            ..Constraints::default()
        };
        assert_eq!(
            bed.answer(constraints, pool_refusal),
            Ok(answer),
            "{named:?}, shed {shed}"
        );
    }
}

/// The recorded sheet, whole: its rows are few.
pub const INLINE: &str = r#"{
  "recorded_from": "d6be6744234a29f6e8f759b9a5e3667620b8c1ad",
  "policy": "explicit-target",
  "rows": [
    {
      "name": "a-named-worker-takes-the-request",
      "shed": false,
      "workers": [
        {"state": "free", "available": true},
        {"state": "free", "available": true},
        {"state": "free", "available": true}
      ],
      "requests": [
        {"target": "0"},
        {"target": "1"},
        {"target": "2"}
      ],
      "answers": [
        {"placed": [0]},
        {"placed": [1]},
        {"placed": [2]}
      ]
    },
    {
      "name": "an-index-past-the-list-names-nobody",
      "shed": false,
      "workers": [
        {"state": "free", "available": true},
        {"state": "free", "available": true}
      ],
      "requests": [
        {"target": "2"},
        {"target": "7"}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}},
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}}
      ]
    },
    {
      "name": "a-value-that-is-no-index-names-nobody",
      "shed": false,
      "workers": [
        {"state": "free", "available": true},
        {"state": "free", "available": true}
      ],
      "requests": [
        {"target": "x"},
        {"target": "-1"}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}},
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}}
      ]
    },
    {
      "name": "a-worker-that-is-out-moves-the-indexes-behind-it",
      "shed": false,
      "workers": [
        {"state": "free", "available": true},
        {"state": "down", "available": false},
        {"state": "free", "available": true}
      ],
      "requests": [
        {"target": "0"},
        {"target": "1"},
        {"target": "2"}
      ],
      "answers": [
        {"placed": [0]},
        {"placed": [2]},
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}}
      ]
    },
    {
      "name": "the-one-worker-is-down",
      "shed": false,
      "workers": [
        {"state": "down", "available": false}
      ],
      "requests": [
        {"target": "0"}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}}
      ]
    },
    {
      "name": "a-model-no-worker-serves",
      "shed": false,
      "workers": [],
      "requests": [
        {"target": "0"}
      ],
      "answers": [
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}}
      ]
    },
    {
      "name": "a-named-worker-over-its-thresholds-beside-one-under-them",
      "shed": false,
      "workers": [
        {"state": "over", "available": false},
        {"state": "free", "available": true}
      ],
      "requests": [
        {"target": "0"},
        {"target": "1"}
      ],
      "answers": [
        {"placed": [1]},
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}}
      ]
    },
    {
      "name": "every-worker-is-over-its-thresholds",
      "shed": false,
      "workers": [
        {"state": "over", "available": false},
        {"state": "over", "available": false}
      ],
      "requests": [
        {"target": "0"},
        {"target": "1"},
        {"target": "2"}
      ],
      "answers": [
        {"placed": [0, 1]},
        {"placed": [0, 1]},
        {"placed": [0, 1]}
      ]
    },
    {
      "name": "every-worker-is-over-and-shedding-is-on",
      "shed": true,
      "workers": [
        {"state": "over", "available": false},
        {"state": "over", "available": false}
      ],
      "requests": [
        {"target": "0"},
        {"target": "1"}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}},
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}}
      ]
    }
  ]
}"#;
