//! The classic retry-wire rows: a pool of regular workers, each with the
//! runtime and the transport it registered and its state, and a list of
//! requests, a retry among them stating the wire its retained plan was
//! built for. A row holds the gateway's answer to each request: the workers
//! it may be placed on, or its refusal.
//!
//! The rows replay on the hard demands' bed ([`Bed`]). A worker states its
//! transport as its wire and its runtime as a number. A retry states its
//! transport as the wire it speaks and its runtime as a hard demand
//! ([`Demands`]). A refusal replays in the gateway's words for a pool
//! ([`pool_refusal`]): no demand of this sheet has words of its own.

use serde::Deserialize;
use smg_policy_controller::engine::{Constraints, DemandFacts, Demands};

use super::demand_bed::{
    self, pool_refusal, wire_of, Answer, Bed, Pooled, Replayed, Runtimes, State, Wire,
};

/// What these rows record, as a sheet's head names it: the wire a retry
/// stays on.
pub const POLICY: &str = "retry-wire";

#[derive(Debug, Deserialize)]
struct Worker {
    runtime: String,
    transport: String,
    state: State,
}

#[derive(Debug, Deserialize)]
struct Asked {
    /// Absent on a first attempt, which stays on no wire.
    wire: Option<Wire>,
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
        let mut runtimes = Runtimes::default();
        let mut pooled = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            pooled.push(Pooled {
                wire_modes: wire_of(&worker.transport)?,
                state: worker.state,
                vision: false,
                facts: DemandFacts {
                    runtime: runtimes.number(&worker.runtime),
                    ..DemandFacts::default()
                },
            });
        }
        let mut bed = Bed::new(self.shed, &pooled)?;
        let mut answers = Vec::with_capacity(self.requests.len());
        for asked in &self.requests {
            let (wire_modes, runtime) = runtimes.stays_on(asked.wire.as_ref())?;
            let constraints = Constraints {
                wire_modes,
                demands: Demands {
                    runtime,
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
