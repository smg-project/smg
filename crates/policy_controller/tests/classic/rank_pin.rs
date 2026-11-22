//! The classic rank rows: a pool of regular workers, each with the
//! data-parallel rank and the group size it registered with where it named
//! a rank, and requests of a route that can pin no rank. A row holds the
//! gateway's answer to each request: the workers it may be placed on, or
//! its refusal.
//!
//! The rows replay on the hard demands' bed ([`Bed`]). A worker states
//! whether it registered a rank as the row's recorder read it off the
//! worker's own object (`rank_registered`): a group of one registered a
//! rank too. Each request states that no rank can be pinned for it, as a
//! hard demand ([`Demands::no_rank`]), and carries nothing else: a row
//! holds one answer a request.
//!
//! A refusal replays in the gateway's words ([`refusal`]): a pool whose
//! every worker registered a rank has words of its own, whatever the
//! health of those workers.

use serde::Deserialize;
use smg_policy_controller::engine::{Constraints, DemandFacts, Demands, ShedReason, WireModes};

use super::demand_bed::{self, pool_refusal, Answer, Bed, Pooled, Refusal, Replayed, State};

/// What these rows record, as a sheet's head names it: the workers a
/// request no rank can be pinned for may take.
pub const POLICY: &str = "rank-pin";

#[derive(Debug, Deserialize)]
struct Worker {
    state: State,
    /// Whether the worker registered a rank, as the gateway reads it.
    rank_registered: bool,
}

/// The gateway's words for a shed under `emptied_by`. A pool whose every
/// worker registered a rank: the route serves no such worker. Any other
/// shed is the pool's ([`pool_refusal`]).
fn refusal(reason: ShedReason, emptied_by: &str) -> Refusal {
    if emptied_by == "rank" {
        Refusal::of(400, "dp_aware_not_supported")
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
        let pooled = |worker: &Worker| Pooled {
            wire_modes: WireModes::HTTP,
            state: worker.state,
            vision: false,
            facts: DemandFacts {
                ranked: worker.rank_registered,
                ..DemandFacts::default()
            },
        };
        let pooled: Vec<Pooled> = self.workers.iter().map(pooled).collect();
        let mut bed = Bed::new(self.shed, &pooled)?;
        let constraints = Constraints {
            demands: Demands {
                no_rank: true,
                ..Demands::default()
            },
            ..Constraints::default()
        };
        let answer = |_: &Answer| bed.answer(constraints, refusal);
        self.answers.iter().map(answer).collect()
    }
}

/// The rows of a sheet replayed: how many, and what did not replay.
pub fn replay_rows(text: &str, source: &str) -> (usize, Vec<String>) {
    demand_bed::replay_rows::<Row>(text, source, POLICY)
}

/// The recorded sheet, whole: its rows are few, and they name no runtime.
pub const INLINE: &str = r#"{
  "recorded_from": "a4935a5acd15151641b1988e234016a95fd172da",
  "policy": "rank-pin",
  "rows": [
    {
      "name": "a-request-no-rank-can-be-pinned-for-leaves-the-workers-that-registered-one",
      "shed": false,
      "workers": [
        {"state": "free", "rank_registered": false},
        {"state": "free", "dp_rank": 0, "dp_size": 2, "rank_registered": true},
        {"state": "free", "dp_rank": 1, "dp_size": 2, "rank_registered": true},
        {"state": "free", "rank_registered": false}
      ],
      "answers": [
        {"placed": [0, 3]}
      ]
    },
    {
      "name": "a-group-of-one-still-registered-a-rank",
      "shed": false,
      "workers": [
        {"state": "free", "dp_rank": 0, "dp_size": 1, "rank_registered": true},
        {"state": "free", "rank_registered": false}
      ],
      "answers": [
        {"placed": [1]}
      ]
    },
    {
      "name": "a-pool-with-no-rank-takes-the-request-whole",
      "shed": false,
      "workers": [
        {"state": "free", "rank_registered": false},
        {"state": "free", "rank_registered": false}
      ],
      "answers": [
        {"placed": [0, 1]}
      ]
    },
    {
      "name": "every-worker-registered-a-rank",
      "shed": false,
      "workers": [
        {"state": "free", "dp_rank": 0, "dp_size": 2, "rank_registered": true},
        {"state": "free", "dp_rank": 1, "dp_size": 2, "rank_registered": true}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "dp_aware_not_supported", "retry_after": false, "message": "/v1/audio/transcriptions does not yet support DP-aware workers"}}
      ]
    },
    {
      "name": "one-worker-and-it-registered-a-rank",
      "shed": false,
      "workers": [
        {"state": "free", "dp_rank": 0, "dp_size": 1, "rank_registered": true}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "dp_aware_not_supported", "retry_after": false, "message": "/v1/audio/transcriptions does not yet support DP-aware workers"}}
      ]
    },
    {
      "name": "a-transcription-of-a-model-no-worker-serves",
      "shed": false,
      "workers": [],
      "answers": [
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}}
      ]
    },
    {
      "name": "the-worker-with-no-rank-is-down",
      "shed": false,
      "workers": [
        {"state": "down", "rank_registered": false},
        {"state": "free", "dp_rank": 0, "dp_size": 2, "rank_registered": true},
        {"state": "free", "dp_rank": 1, "dp_size": 2, "rank_registered": true}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers are unavailable (circuit breaker open or unhealthy)"}}
      ]
    },
    {
      "name": "a-worker-with-no-rank-over-its-thresholds-yields-to-one-under-them",
      "shed": false,
      "workers": [
        {"state": "over", "rank_registered": false},
        {"state": "free", "rank_registered": false},
        {"state": "free", "dp_rank": 0, "dp_size": 1, "rank_registered": true}
      ],
      "answers": [
        {"placed": [1]}
      ]
    },
    {
      "name": "the-workers-with-no-rank-are-over-their-thresholds",
      "shed": false,
      "workers": [
        {"state": "over", "rank_registered": false},
        {"state": "over", "rank_registered": false},
        {"state": "free", "dp_rank": 0, "dp_size": 1, "rank_registered": true}
      ],
      "answers": [
        {"placed": [0, 1]}
      ]
    },
    {
      "name": "the-workers-with-no-rank-are-over-and-shedding-is-on",
      "shed": true,
      "workers": [
        {"state": "over", "rank_registered": false},
        {"state": "over", "rank_registered": false},
        {"state": "free", "dp_rank": 0, "dp_size": 1, "rank_registered": true}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}}
      ]
    },
    {
      "name": "a-worker-with-a-rank-that-is-down-still-counts-as-one",
      "shed": false,
      "workers": [
        {"state": "down", "dp_rank": 0, "dp_size": 2, "rank_registered": true},
        {"state": "free", "dp_rank": 1, "dp_size": 2, "rank_registered": true}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "dp_aware_not_supported", "retry_after": false, "message": "/v1/audio/transcriptions does not yet support DP-aware workers"}}
      ]
    }
  ]
}"#;
