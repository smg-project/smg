//! The classic context-window rows: a pool of regular workers, each with
//! the context window it registered where it registered one, and a list of
//! requests of an input and a completion budget, a request that scores a
//! position and a retry on its wire among them. A row holds the gateway's
//! answer to each request: the workers it may be placed on, or its refusal.
//!
//! The rows replay on the hard demands' bed ([`Bed`]). A worker states its
//! window where the gateway holds a request to one as it selects: on the
//! gRPC wire. A worker on another wire states none there, and takes a
//! request of any size. A request states its two bounds as hard demands
//! ([`Demands`]): what its input needs, one token more than its prompt
//! where it scores a position, and its completion budget.
//!
//! A refusal replays in the gateway's words ([`refusal`]): a pool whose
//! windows all hold less than the request has words of its own.

use serde::Deserialize;
use smg_policy_controller::engine::{Constraints, DemandFacts, Demands, ShedReason};

use super::demand_bed::{
    self, pool_refusal, wire_of, Answer, Bed, Pooled, Refusal, Replayed, Runtimes, State, Wire,
};

/// What these rows record, as a sheet's head names it: the window that
/// holds a request.
pub const POLICY: &str = "context-window";

/// The transport on which the gateway holds a request to a worker's window
/// as it selects.
const BOUND: &str = "grpc";

#[derive(Debug, Deserialize)]
struct Worker {
    runtime: String,
    transport: String,
    state: State,
    /// Absent where the worker registered no window.
    window: Option<u32>,
}

impl Worker {
    /// The window the worker states for a request's demands: its own on
    /// the bound wire, and none on another.
    fn window(&self) -> u32 {
        let bound = self.transport == BOUND;
        self.window.filter(|_| bound).unwrap_or(0)
    }
}

#[derive(Debug, Deserialize)]
struct Asked {
    /// Absent on a first attempt, which stays on no wire.
    wire: Option<Wire>,
    /// The tokens of the request's prompt.
    #[serde(default)]
    input_tokens: u32,
    /// Whether the request scores the position behind its prompt.
    #[serde(default)]
    scoring_position: bool,
    /// Absent where the request names no completion budget.
    output_tokens: Option<u32>,
}

impl Asked {
    /// The window the request's input needs: its prompt, and one token more
    /// where it scores the position behind it.
    fn input_window(&self) -> u32 {
        let scored = u32::from(self.scoring_position);
        self.input_tokens.saturating_add(scored)
    }
}

/// The gateway's words for a shed under `emptied_by`. Windows that all hold
/// less than the request: its context is too long for the model. Any other
/// shed is the pool's ([`pool_refusal`]).
fn refusal(reason: ShedReason, emptied_by: &str) -> Refusal {
    if emptied_by == "context-window" {
        Refusal::of(400, "context_length_exceeded")
    } else {
        pool_refusal(reason, emptied_by)
    }
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
                    window: worker.window(),
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
                    input_window: asked.input_window(),
                    output_window: asked.output_tokens.unwrap_or(0),
                    ..Demands::default()
                },
                ..Constraints::default()
            };
            answers.push(bed.answer(constraints, refusal)?);
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
  "policy": "context-window",
  "rows": [
    {
      "name": "each-bound-is-held-against-the-window-alone",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 200},
        {"runtime": "zeta", "transport": "grpc", "state": "free"},
        {"runtime": "zeta", "transport": "zmq", "state": "free", "window": 50}
      ],
      "requests": [
        {},
        {"input_tokens": 100},
        {"input_tokens": 101},
        {"input_tokens": 100, "scoring_position": true},
        {"input_tokens": 99, "scoring_position": true},
        {"input_tokens": 0, "output_tokens": 100},
        {"input_tokens": 0, "output_tokens": 101},
        {"input_tokens": 60, "output_tokens": 60},
        {"input_tokens": 100, "output_tokens": 100},
        {"input_tokens": 150, "output_tokens": 150},
        {"input_tokens": 200},
        {"input_tokens": 201},
        {"input_tokens": 200, "scoring_position": true}
      ],
      "answers": [
        {"placed": [0, 1, 2, 3]},
        {"placed": [0, 1, 2, 3]},
        {"placed": [1, 2, 3]},
        {"placed": [1, 2, 3]},
        {"placed": [0, 1, 2, 3]},
        {"placed": [0, 1, 2, 3]},
        {"placed": [1, 2, 3]},
        {"placed": [0, 1, 2, 3]},
        {"placed": [0, 1, 2, 3]},
        {"placed": [1, 2, 3]},
        {"placed": [1, 2, 3]},
        {"placed": [2, 3]},
        {"placed": [2, 3]}
      ]
    },
    {
      "name": "no-window-holds-the-input",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 200}
      ],
      "requests": [
        {"input_tokens": 201},
        {"input_tokens": 200, "scoring_position": true},
        {"input_tokens": 1000}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 200 tokens. However, your request has 201 input tokens. Please reduce the length of the input."}},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "Decision prompt has 200 tokens and cannot fit the 200-token context window with its scoring position intact"}},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 200 tokens. However, your request has 1000 input tokens. Please reduce the length of the input."}}
      ]
    },
    {
      "name": "no-window-holds-the-budget",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 200}
      ],
      "requests": [
        {"input_tokens": 0, "output_tokens": 201},
        {"input_tokens": 10, "output_tokens": 201},
        {"input_tokens": 201, "output_tokens": 201}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 200 tokens. However, you requested 201 tokens for the completion. Please reduce max_tokens."}},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 200 tokens. However, you requested 201 tokens for the completion. Please reduce max_tokens."}},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 200 tokens. However, your request has 201 input tokens. Please reduce the length of the input."}}
      ]
    },
    {
      "name": "the-widest-window-is-read-on-the-retry-s-wire",
      "shed": false,
      "workers": [
        {"runtime": "sigma", "transport": "grpc", "state": "free", "window": 400},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 120}
      ],
      "requests": [
        {"input_tokens": 150},
        {"wire": {"runtime": "zeta", "transport": "grpc"}, "input_tokens": 150},
        {"wire": {"runtime": "zeta", "transport": "grpc"}, "input_tokens": 110},
        {"input_tokens": 500}
      ],
      "answers": [
        {"placed": [0]},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 120 tokens. However, your request has 150 input tokens. Please reduce the length of the input."}},
        {"placed": [2]},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 400 tokens. However, your request has 500 input tokens. Please reduce the length of the input."}}
      ]
    },
    {
      "name": "the-worker-whose-window-holds-the-request-is-down",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "down", "window": 200}
      ],
      "requests": [
        {"input_tokens": 150},
        {"input_tokens": 50},
        {"input_tokens": 250}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers for model 'm' are unavailable (unhealthy or circuit breaker open)"}},
        {"placed": [0]},
        {"refused": {"status": 400, "code": "context_length_exceeded", "retry_after": false, "message": "This model's maximum context length is 200 tokens. However, your request has 250 input tokens. Please reduce the length of the input."}}
      ]
    },
    {
      "name": "the-worker-whose-window-holds-the-request-is-over-its-thresholds",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "over", "window": 200}
      ],
      "requests": [
        {"input_tokens": 150},
        {"input_tokens": 50}
      ],
      "answers": [
        {"placed": [1]},
        {"placed": [0]}
      ]
    },
    {
      "name": "the-worker-whose-window-holds-the-request-is-over-and-shedding-is-on",
      "shed": true,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "window": 100},
        {"runtime": "zeta", "transport": "grpc", "state": "over", "window": 200}
      ],
      "requests": [
        {"input_tokens": 150},
        {"input_tokens": 50}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}},
        {"placed": [0]}
      ]
    },
    {
      "name": "a-pool-of-unknown-windows-takes-any-request",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free"},
        {"runtime": "sigma", "transport": "grpc", "state": "free"}
      ],
      "requests": [
        {"input_tokens": 1000000},
        {"input_tokens": 1000000, "output_tokens": 1000000}
      ],
      "answers": [
        {"placed": [0, 1]},
        {"placed": [0, 1]}
      ]
    }
  ]
}"#;
