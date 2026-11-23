//! The classic media rows: a pool of regular workers, each with what it
//! registered of the media it takes, and a list of requests that carry
//! media: pixel media, media by reference under the pipeline pin of its
//! plan, or both, a retry on its wire among them. A row holds the gateway's
//! answer to each request: the workers it may be placed on, or its refusal.
//!
//! The rows replay on the hard demands' bed ([`Bed`]). A worker states the
//! three facts its row records: whether it takes images, which is its own
//! vision bit, whether it processes media by reference, and whether by the
//! gateway's own pipeline.
//! The labels a row also records are the gateway's source of those facts,
//! and the replay reads none. A request states its media as hard demands
//! ([`Demands`]): `any` worker that processes references, `only` one that
//! runs the gateway's pipeline, or `never` one.
//!
//! A refusal replays in the gateway's words ([`refusal`]): each media
//! demand that leaves a pool nobody has words of its own.

use serde::Deserialize;
use smg_policy_controller::engine::{Constraints, DemandFacts, Demands, References, ShedReason};

use super::demand_bed::{
    self, pool_refusal, wire_of, Answer, Bed, Pooled, Refusal, Replayed, Runtimes, State, Wire,
};

/// What these rows record, as a sheet's head names it: where a request's
/// media may go.
pub const POLICY: &str = "media-refs";

#[derive(Debug, Deserialize)]
struct Worker {
    runtime: String,
    transport: String,
    state: State,
    processes_references: bool,
    gateway_pipeline: bool,
    vision: bool,
}

/// The pipeline pin of a request's plan.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Pin {
    Any,
    Only,
    Never,
}

#[derive(Debug, Deserialize)]
struct Asked {
    /// Absent on a first attempt, which stays on no wire.
    wire: Option<Wire>,
    /// Absent where the request carries no media by reference.
    references: Option<Pin>,
    /// Whether the request carries pixel media: it needs vision.
    #[serde(default)]
    pixels: bool,
}

impl Asked {
    fn references(&self) -> References {
        match self.references {
            None => References::None,
            Some(Pin::Any) => References::Any,
            Some(Pin::Only) => References::Caller,
            Some(Pin::Never) => References::Worker,
        }
    }
}

/// The gateway's words for a shed under `demands`. No worker that takes
/// pixel media: the model supports no such input. No worker that processes
/// references: none is capable. Workers that do, all on the other side of
/// the pin: a plan that takes only the gateway's pipeline finds the model
/// unsupported there, and one that never takes it finds the input
/// unsupported. Any other shed is the pool's ([`pool_refusal`]).
fn refusal(demands: Demands, reason: ShedReason, emptied_by: &str) -> Refusal {
    match (emptied_by, demands.references) {
        ("vision", _) | ("pipeline", References::Worker) => {
            Refusal::of(400, "multimodal_not_supported")
        }
        ("references", _) => Refusal::of(503, "no_media_ref_capable_worker"),
        ("pipeline", _) => Refusal::of(400, "multimodal_worker_processing_unsupported_model"),
        _ => pool_refusal(reason, emptied_by),
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
                vision: worker.vision,
                facts: DemandFacts {
                    runtime: runtimes.number(&worker.runtime),
                    references: worker.processes_references,
                    host_pipeline: worker.gateway_pipeline,
                    ..DemandFacts::default()
                },
            });
        }
        let mut bed = Bed::new(self.shed, &pooled)?;
        let mut answers = Vec::with_capacity(self.requests.len());
        for asked in &self.requests {
            let (wire_modes, runtime) = runtimes.stays_on(asked.wire.as_ref())?;
            let demands = Demands {
                runtime,
                vision: asked.pixels,
                references: asked.references(),
                ..Demands::default()
            };
            let constraints = Constraints {
                wire_modes,
                demands,
                ..Constraints::default()
            };
            let words = |reason, emptied_by: &str| refusal(demands, reason, emptied_by);
            answers.push(bed.answer(constraints, words)?);
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
  "policy": "media-refs",
  "rows": [
    {
      "name": "references-go-to-the-workers-that-process-them",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "mm_processor": "engine", "processes_references": true, "gateway_pipeline": false, "vision": true},
        {"runtime": "sigma", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "sigma", "transport": "grpc", "state": "free", "mm_processor": "engine", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "sigma", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "tau", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "zeta", "transport": "zmq", "state": "free", "mm_processor": "smg", "processes_references": false, "gateway_pipeline": false, "vision": true}
      ],
      "requests": [
        {},
        {"references": "any"},
        {"references": "only"},
        {"references": "never"},
        {"wire": {"runtime": "zeta", "transport": "grpc"}, "references": "any"},
        {"wire": {"runtime": "sigma", "transport": "grpc"}, "references": "only"},
        {"wire": {"runtime": "sigma", "transport": "grpc"}, "references": "never"}
      ],
      "answers": [
        {"placed": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"placed": [1, 2, 3]},
        {"placed": [1, 3]},
        {"placed": [2]},
        {"placed": [1, 2]},
        {"placed": [3]},
        {"refused": {"status": 400, "code": "multimodal_not_supported", "retry_after": false, "message": "the workers processing media references for model m all run the gateway's own pipeline (mm_processor=smg), which takes image and video references, not audio; a worker with --mm-processor inprocess or redis processes audio with the engine's own processor"}}
      ]
    },
    {
      "name": "pixel-media-goes-to-a-worker-with-a-vision-encoder",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "supports_vision": "false", "processes_references": false, "gateway_pipeline": false, "vision": false},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "mm_processor": "smg", "supports_vision": "false", "processes_references": true, "gateway_pipeline": true, "vision": false}
      ],
      "requests": [
        {"pixels": true},
        {"references": "any"},
        {"references": "any", "pixels": true},
        {"references": "only", "pixels": true}
      ],
      "answers": [
        {"placed": [0, 2]},
        {"placed": [2, 3]},
        {"placed": [2]},
        {"placed": [2]}
      ]
    },
    {
      "name": "no-worker-has-a-vision-encoder",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "supports_vision": "false", "processes_references": false, "gateway_pipeline": false, "vision": false},
        {"runtime": "sigma", "transport": "grpc", "state": "free", "supports_vision": "false", "processes_references": false, "gateway_pipeline": false, "vision": false}
      ],
      "requests": [
        {"pixels": true},
        {"references": "any", "pixels": true}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "multimodal_not_supported", "retry_after": false, "message": "image or video input is not supported by model 'm': none of its workers accept vision input (the engine runs as a language model only, or its servicer does not carry media to it)"}},
        {"refused": {"status": 400, "code": "multimodal_not_supported", "retry_after": false, "message": "image or video input is not supported by model 'm': none of its workers accept vision input (the engine runs as a language model only, or its servicer does not carry media to it)"}}
      ]
    },
    {
      "name": "no-worker-processes-references",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "sigma", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true},
        {"runtime": "tau", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": false, "gateway_pipeline": false, "vision": true}
      ],
      "requests": [
        {"references": "any"},
        {"references": "only"},
        {"references": "never"}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_media_ref_capable_worker", "retry_after": false, "message": "no worker of model 'm' processes media references (none advertises an mm_processor), so a request whose media is forwarded to the worker cannot be served: start the workers with worker-side media processing, or run the router with --mm-processing auto or router"}},
        {"refused": {"status": 503, "code": "no_media_ref_capable_worker", "retry_after": false, "message": "no worker of model 'm' processes media references (none advertises an mm_processor), so a request whose media is forwarded to the worker cannot be served: start the workers with worker-side media processing, or run the router with --mm-processing auto or router"}},
        {"refused": {"status": 503, "code": "no_media_ref_capable_worker", "retry_after": false, "message": "no worker of model 'm' processes media references (none advertises an mm_processor), so a request whose media is forwarded to the worker cannot be served: start the workers with worker-side media processing, or run the router with --mm-processing auto or router"}}
      ]
    },
    {
      "name": "the-workers-that-process-references-stand-on-the-other-side-of-the-pin",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "mm_processor": "engine", "processes_references": true, "gateway_pipeline": false, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true}
      ],
      "requests": [
        {"references": "only"},
        {"references": "never"},
        {"references": "any"}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "multimodal_worker_processing_unsupported_model", "retry_after": false, "message": "model m has workers processing media references, but none runs the gateway's own pipeline (mm_processor=smg), the only processor that expands this model's placeholders; start the workers with --mm-processor smg or the router with --mm-processing router"}},
        {"placed": [0]},
        {"placed": [0]}
      ]
    },
    {
      "name": "the-workers-that-process-references-run-the-pipeline-a-plan-pins-away",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "sigma", "transport": "grpc", "state": "free", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true}
      ],
      "requests": [
        {"references": "never"},
        {"references": "only"}
      ],
      "answers": [
        {"refused": {"status": 400, "code": "multimodal_not_supported", "retry_after": false, "message": "the workers processing media references for model m all run the gateway's own pipeline (mm_processor=smg), which takes image and video references, not audio; a worker with --mm-processor inprocess or redis processes audio with the engine's own processor"}},
        {"placed": [0, 1]}
      ]
    },
    {
      "name": "the-one-worker-that-processes-references-is-down",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "down", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true}
      ],
      "requests": [
        {"references": "any"},
        {}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "no_available_workers", "retry_after": true, "message": "All workers for model 'm' are unavailable (unhealthy or circuit breaker open)"}},
        {"placed": [1]}
      ]
    },
    {
      "name": "the-workers-that-process-references-are-over-their-thresholds",
      "shed": false,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "over", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "over", "mm_processor": "engine", "processes_references": true, "gateway_pipeline": false, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true}
      ],
      "requests": [
        {"references": "any"},
        {"references": "only"}
      ],
      "answers": [
        {"placed": [0, 1]},
        {"placed": [0]}
      ]
    },
    {
      "name": "the-workers-that-process-references-are-over-and-shedding-is-on",
      "shed": true,
      "workers": [
        {"runtime": "zeta", "transport": "grpc", "state": "over", "mm_processor": "smg", "processes_references": true, "gateway_pipeline": true, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "over", "mm_processor": "engine", "processes_references": true, "gateway_pipeline": false, "vision": true},
        {"runtime": "zeta", "transport": "grpc", "state": "free", "processes_references": false, "gateway_pipeline": false, "vision": true}
      ],
      "requests": [
        {"references": "any"},
        {"references": "only"}
      ],
      "answers": [
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}},
        {"refused": {"status": 503, "code": "worker_overload_protection_shed", "retry_after": true, "message": "All workers for model 'm' are overloaded"}}
      ]
    },
    {
      "name": "references-and-no-worker-of-the-model",
      "shed": false,
      "workers": [],
      "requests": [
        {"references": "any"},
        {"pixels": true}
      ],
      "answers": [
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}},
        {"refused": {"status": 404, "code": "model_not_found", "retry_after": false, "message": "No worker available for model 'm'"}}
      ]
    }
  ]
}"#;
