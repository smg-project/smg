//! The recorder of the gateway's hard demands on a single worker: what a
//! request must have of the worker that takes it, kept before any policy
//! runs. Four sheets, one a demand:
//!
//! - `retry_wire`: a retry stays on the runtime and the transport its
//!   retained plan was built for.
//! - `media_refs`: forwarded media references go to a worker that processes
//!   them, on the side of the pipeline the plan pins, and pixel media to a
//!   worker with a vision encoder.
//! - `context_window`: the input and the completion budget are each held
//!   against the worker's context window, never their sum.
//! - `constraint_decider`: a retry stays on the workers that decide the
//!   request's constraint as the first attempt's worker did.
//!
//! Each row is a pool of regular workers, with what the gateway reads of
//! each from its registration and its state, whether an all-overloaded pool
//! is shed, and a list of requests. The row records what the gateway's own
//! selection answered at each request: the workers it placed the request on,
//! or its refusal. The policy controller crate replays the rows through its
//! own decision.
//!
//! A row runs the two functions a first attempt and a retry of the gRPC
//! pipeline both call for a single worker: the selection, and the refusal
//! where it names nobody. The pool runs under `round_robin`, and a request is
//! placed sixteen times the pool's size: a rotation walks every worker the
//! demands leave it, and a pool over on every such worker, steered by a draw
//! among the fewest in flight, reaches each of them. So `placed` is the whole
//! set a request may land on, by each worker's place in the row's list. A
//! refusal is the response's status, its code, whether it says when to come
//! back, and its message.
//!
//! A worker's facts are written twice where the gateway derives one from
//! another: the registration's own words (its runtime, its transport, its
//! window, its labels) and what the gateway's predicates read of them.
//!
//! This file is an ignored test of the gateway crate, kept beside the code
//! it runs. It records from the checkout it is built in and writes the
//! sheets under `CLASSIC_ROWS_OUT`, naming `CLASSIC_ROWS_COMMIT` as the
//! commit it recorded from. The sheets the policy controller crate replays
//! are in `crates/policy_controller/tests/classic/sheets`; to record these
//! again:
//!
//! ```text
//! CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
//!   cargo test -p smg --lib worker_selection::classic_hard_demand_rows -- --ignored --nocapture
//! ```
//!
//! Every row is recorded twenty times, each time from a fresh pool and a
//! fresh policy, and the recorder stops if two recordings differ.

use std::{collections::BTreeSet, fs, path::PathBuf, sync::Arc};

use axum::http::header::RETRY_AFTER;
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Map, Value};
use smg_worker::test_support::no_health_check;

use super::*;
use crate::{
    config::types::PolicyConfig,
    routers::grpc::{backend_client::BasicWorkerBuilder, utils::WorkerDecoding},
};

const MODEL: &str = "m";
/// A request is placed this many times the pool's size.
const ROUNDS: usize = 16;
/// The `mm_processor` label of a worker that runs the gateway's own pipeline.
const OWN: &str = multimodal::GATEWAY_PIPELINE_PROCESSOR;
/// The same label of a worker that runs its engine's processor.
const ENGINE: &str = "engine";

/// The sheet a row belongs to: the demand it records.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sheet {
    Wire,
    Media,
    Window,
    Decider,
}

impl Sheet {
    const ALL: [Self; 4] = [Self::Wire, Self::Media, Self::Window, Self::Decider];

    const fn name(self) -> &'static str {
        match self {
            Self::Wire => "retry_wire",
            Self::Media => "media_refs",
            Self::Window => "context_window",
            Self::Decider => "constraint_decider",
        }
    }
}

/// What the router holds of a worker.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Not ready.
    Down,
}

impl State {
    const fn name(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Over => "over",
            Self::Down => "down",
        }
    }
}

/// A worker's transport, as a wire names it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Transport {
    Grpc,
    Zmq,
}

impl Transport {
    const fn name(self) -> &'static str {
        match self {
            Self::Grpc => "grpc",
            Self::Zmq => "zmq",
        }
    }

    const fn mode(self) -> ConnectionMode {
        match self {
            Self::Grpc => ConnectionMode::Grpc,
            Self::Zmq => ConnectionMode::Zmq,
        }
    }
}

/// A runtime in a sheet's spelling.
fn runtime_name(runtime: RuntimeType) -> String {
    format!("{runtime:?}").to_lowercase()
}

/// One worker of a row, as it registers.
#[derive(Clone, Copy)]
struct Member {
    runtime: RuntimeType,
    transport: Transport,
    state: State,
    /// The context window its model card states.
    window: Option<u32>,
    /// Its `mm_processor` label.
    processor: Option<&'static str>,
    /// It registers `supports_vision=false`.
    language_only: bool,
    /// Its `constrained_decoding`, `constrained_decoding_start` and
    /// `structural_tag_vocabulary` labels.
    decoding: [Option<&'static str>; 3],
}

const fn on(runtime: RuntimeType) -> Member {
    Member {
        runtime,
        transport: Transport::Grpc,
        state: State::Free,
        window: None,
        processor: None,
        language_only: false,
        decoding: [None, None, None],
    }
}

impl Member {
    const fn zmq(self) -> Self {
        Self {
            transport: Transport::Zmq,
            ..self
        }
    }

    const fn state(self, state: State) -> Self {
        Self { state, ..self }
    }

    const fn window(self, window: u32) -> Self {
        Self {
            window: Some(window),
            ..self
        }
    }

    const fn processor(self, processor: &'static str) -> Self {
        Self {
            processor: Some(processor),
            ..self
        }
    }

    const fn language_only(self) -> Self {
        Self {
            language_only: true,
            ..self
        }
    }

    const fn decoding(
        self,
        policy: Option<&'static str>,
        start: Option<&'static str>,
        vocabulary: Option<&'static str>,
    ) -> Self {
        Self {
            decoding: [policy, start, vocabulary],
            ..self
        }
    }

    /// The labels of the registration, in the gateway's own names.
    fn labels(&self) -> Vec<(&'static str, &'static str)> {
        let mut labels = Vec::new();
        if let Some(processor) = self.processor {
            labels.push(("mm_processor", processor));
        }
        if self.language_only {
            labels.push(("supports_vision", "false"));
        }
        let names = [
            "constrained_decoding",
            "constrained_decoding_start",
            "structural_tag_vocabulary",
        ];
        for (name, value) in names.into_iter().zip(self.decoding) {
            if let Some(value) = value {
                labels.push((name, value));
            }
        }
        labels
    }
}

/// The pin of a plan that forwards references.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pin {
    Any,
    Only,
    Never,
}

impl Pin {
    const fn name(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Only => "only",
            Self::Never => "never",
        }
    }

    const fn of(self) -> PipelinePin {
        match self {
            Self::Any => PipelinePin::Any,
            Self::Only => PipelinePin::Only,
            Self::Never => PipelinePin::Never,
        }
    }
}

/// One request of a row: its demands, each absent where the request has none.
#[derive(Clone, Copy)]
struct Request {
    /// The wire of the retained plan on a retry; none on a first attempt.
    wire: Option<(RuntimeType, Transport)>,
    /// The request forwards media references, under this pin.
    refs: Option<Pin>,
    /// The request carries pixel media.
    vision: bool,
    /// The input's token count, whether the request reads a scoring
    /// position, and the completion budget.
    window: Option<(usize, bool, Option<u32>)>,
    /// The worker the first attempt's constraint was decided on, by place.
    decided_on: Option<usize>,
}

const ANY: Request = Request {
    wire: None,
    refs: None,
    vision: false,
    window: None,
    decided_on: None,
};

const fn wired(runtime: RuntimeType, transport: Transport) -> Request {
    Request {
        wire: Some((runtime, transport)),
        ..ANY
    }
}

const fn refs(pin: Pin) -> Request {
    Request {
        refs: Some(pin),
        ..ANY
    }
}

const fn pixels() -> Request {
    Request {
        vision: true,
        ..ANY
    }
}

const fn input(tokens: usize) -> Request {
    Request {
        window: Some((tokens, false, None)),
        ..ANY
    }
}

const fn scored(tokens: usize) -> Request {
    Request {
        window: Some((tokens, true, None)),
        ..ANY
    }
}

const fn budget(tokens: usize, budget: u32) -> Request {
    Request {
        window: Some((tokens, false, Some(budget))),
        ..ANY
    }
}

const fn decided(on: usize) -> Request {
    Request {
        decided_on: Some(on),
        ..ANY
    }
}

impl Request {
    const fn with_pixels(self) -> Self {
        Self {
            vision: true,
            ..self
        }
    }

    const fn on_wire(self, runtime: RuntimeType, transport: Transport) -> Self {
        Self {
            wire: Some((runtime, transport)),
            ..self
        }
    }

    fn json(&self) -> Value {
        let mut out = Map::new();
        if let Some((runtime, transport)) = self.wire {
            out.insert(
                "wire".into(),
                json!({"runtime": runtime_name(runtime), "transport": transport.name()}),
            );
        }
        if let Some(pin) = self.refs {
            out.insert("references".into(), json!(pin.name()));
        }
        if self.vision {
            out.insert("pixels".into(), json!(true));
        }
        if let Some((tokens, scoring, budget)) = self.window {
            out.insert("input_tokens".into(), json!(tokens));
            if scoring {
                out.insert("scoring_position".into(), json!(true));
            }
            if let Some(budget) = budget {
                out.insert("output_tokens".into(), json!(budget));
            }
        }
        if let Some(on) = self.decided_on {
            out.insert("decided_on".into(), json!(on));
        }
        Value::Object(out)
    }
}

/// One row as it is written down.
struct Row {
    sheet: Sheet,
    name: &'static str,
    /// Whether an all-overloaded pool is shed (`--worker-overload-shed`).
    shed: bool,
    workers: Vec<Member>,
    requests: Vec<Request>,
}

fn row(sheet: Sheet, name: &'static str, workers: &[Member], requests: &[Request]) -> Row {
    Row {
        sheet,
        name,
        shed: false,
        workers: workers.to_vec(),
        requests: requests.to_vec(),
    }
}

impl Row {
    fn shedding(self) -> Self {
        Self { shed: true, ..self }
    }
}

/// A fresh pool in the row's state, under a fresh policy.
struct Bed {
    stage: WorkerSelectionStage,
    workers: Vec<Arc<dyn Worker>>,
}

impl Bed {
    fn of(row: &Row) -> Self {
        let registry = Arc::new(WorkerRegistry::new());
        registry.set_overload_shed(row.shed);
        let mut workers = Vec::new();
        for (at, member) in row.workers.iter().enumerate() {
            let url = match member.transport {
                Transport::Grpc => format!("grpc://127.0.0.1:{}", 9400 + at),
                Transport::Zmq => format!("ipc:///tmp/smg-classic-hard-demand-{at}"),
            };
            let mut card = ModelCard::new(MODEL);
            if let Some(window) = member.window {
                card = card.with_context_length(window);
            }
            let mut builder = BasicWorkerBuilder::new(url)
                .model(card)
                .worker_type(WorkerType::Regular)
                .connection_mode(member.transport.mode())
                .runtime_type(member.runtime)
                .health_config(no_health_check());
            for (key, value) in member.labels() {
                builder = builder.label(key, value);
            }
            let worker: Arc<dyn Worker> = Arc::new(builder.build());
            worker.set_status(WorkerStatus::Ready);
            registry
                .register(Arc::clone(&worker))
                .expect("a worker of the row registers");
            worker.set_status(if member.state == State::Down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
            registry.set_worker_overloaded(&worker, member.state == State::Over);
            workers.push(worker);
        }
        let stage = WorkerSelectionStage::new(
            registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
            None,
        );
        Self { stage, workers }
    }

    /// A worker as the sheet writes it: its registration's words, and what
    /// the gateway's own predicates read of it for the sheet's demand.
    fn worker_json(&self, sheet: Sheet, at: usize, member: &Member) -> Value {
        let worker = &self.workers[at];
        let mut out = Map::new();
        out.insert("runtime".into(), json!(runtime_name(member.runtime)));
        out.insert("transport".into(), json!(member.transport.name()));
        out.insert("state".into(), json!(member.state.name()));
        match sheet {
            Sheet::Wire => {}
            Sheet::Media => {
                if let Some(processor) = member.processor {
                    out.insert("mm_processor".into(), json!(processor));
                }
                if member.language_only {
                    out.insert("supports_vision".into(), json!("false"));
                }
                out.insert(
                    "processes_references".into(),
                    json!(accepts_media_refs(worker.as_ref())),
                );
                out.insert(
                    "gateway_pipeline".into(),
                    json!(pipeline_admits(PipelinePin::Only, worker.as_ref())),
                );
                out.insert("vision".into(), json!(vision_capable(worker.as_ref())));
            }
            Sheet::Window => {
                if let Some(window) = member.window {
                    out.insert("window".into(), json!(window));
                }
            }
            Sheet::Decider => {
                for (name, value) in member.labels() {
                    out.insert(name.into(), json!(value));
                }
                let decoding = WorkerDecoding::of(Some(&WorkerSelection::Single {
                    worker: Arc::clone(worker).into(),
                }));
                out.insert(
                    "policy".into(),
                    json!(format!("{:?}", decoding.policy).to_lowercase()),
                );
                out.insert("whole_turn".into(), json!(decoding.takes_the_whole_turn()));
                out.insert(
                    "vocabulary".into(),
                    json!(format!("{:?}", decoding.vocabulary).to_lowercase()),
                );
            }
        }
        Value::Object(out)
    }

    /// The gateway's answer to one request: the workers its selection placed
    /// the request on over the rounds, or its refusal.
    async fn answer(&self, request: &Request) -> Value {
        let pin = request.refs.map_or(PipelinePin::Any, Pin::of);
        // A retry's wire carries the plan's media pins, as the dispatch
        // context hands them to the stage.
        let wire = request.wire.map(|(runtime, transport)| WireConstraint {
            runtime,
            connection: transport.mode(),
            requires_media_refs: request.refs.is_some(),
            pipeline: pin,
        });
        let media = MediaDemands {
            media_refs: request.refs.is_some(),
            vision: request.vision,
            pipeline: pin,
        };
        let context = request
            .window
            .map_or_else(ContextRequirements::default, |(tokens, scoring, budget)| {
                ContextRequirements::of_demand(tokens, scoring, budget)
            });
        let decided_on = request.decided_on.map(|at| {
            WorkerDecoding::of(Some(&WorkerSelection::Single {
                worker: Arc::clone(&self.workers[at]).into(),
            }))
        });
        let alike =
            |worker: &dyn Worker| decided_on.is_none_or(|decoding| decoding.decides_alike(worker));
        let admits: Option<CandidateFilter<'_>> = decided_on.is_some().then_some(&alike);
        let mut placed = BTreeSet::new();
        for _ in 0..ROUNDS * self.workers.len().max(1) {
            let Some(worker) = self.stage.select_regular(
                MODEL,
                context,
                media,
                admits,
                None,
                None,
                None,
                None,
                None,
                None,
                wire,
                &[],
            ) else {
                assert!(placed.is_empty(), "a request placed once is placed again");
                let refusal = self
                    .stage
                    .context_selection_failure(MODEL, wire, media, context);
                return refused(refusal).await;
            };
            let at = self
                .workers
                .iter()
                .position(|w| w.url() == worker.url())
                .expect("a worker of the row");
            placed.insert(at);
        }
        json!({"placed": placed.into_iter().collect::<Vec<_>>()})
    }
}

/// A refusal as a sheet writes it.
async fn refused(refusal: Response) -> Value {
    let status = refusal.status().as_u16();
    let code = error::extract_error_code_from_response(&refusal).to_string();
    let retry_after = refusal.headers().contains_key(RETRY_AFTER);
    let body = axum::body::to_bytes(refusal.into_body(), usize::MAX)
        .await
        .expect("a refusal's body is read");
    let message = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|body| body["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_default();
    json!({"refused": {"status": status, "code": code, "retry_after": retry_after, "message": message}})
}

/// One recording of a row: its workers as the gateway reads them, and the
/// answer to each request.
async fn recorded(row: &Row) -> Value {
    let bed = Bed::of(row);
    let workers: Vec<Value> = row
        .workers
        .iter()
        .enumerate()
        .map(|(at, member)| bed.worker_json(row.sheet, at, member))
        .collect();
    let mut answers = Vec::new();
    for request in &row.requests {
        answers.push(bed.answer(request).await);
    }
    json!({
        "name": row.name,
        "shed": row.shed,
        "workers": workers,
        "requests": row.requests.iter().map(Request::json).collect::<Vec<_>>(),
        "answers": answers,
    })
}

fn rows() -> Vec<Row> {
    use RuntimeType::{Sglang, TokenSpeed, Vllm};
    use Transport::{Grpc, Zmq};
    let mut rows = vec![
        // The wire of a retry.
        row(
            Sheet::Wire,
            "a-retry-stays-on-its-runtime-and-its-transport",
            &[on(Sglang), on(Vllm), on(Vllm).zmq(), on(TokenSpeed)],
            &[
                ANY,
                wired(Vllm, Grpc),
                wired(Vllm, Zmq),
                wired(Sglang, Grpc),
                wired(TokenSpeed, Grpc),
                wired(Sglang, Zmq),
                wired(TokenSpeed, Zmq),
            ],
        ),
        row(
            Sheet::Wire,
            "the-workers-of-a-wire-share-its-retries",
            &[on(Vllm), on(Sglang), on(Vllm), on(Vllm).zmq()],
            &[wired(Vllm, Grpc), wired(Sglang, Grpc)],
        ),
        row(
            Sheet::Wire,
            "a-wire-whose-worker-is-down-is-not-left",
            &[on(Vllm).state(State::Down), on(Sglang), on(Vllm).zmq()],
            &[wired(Vllm, Grpc)],
        ),
        row(
            Sheet::Wire,
            "a-worker-of-the-wire-over-its-thresholds-yields-to-one-under-them",
            &[on(Vllm).state(State::Over), on(Vllm), on(Sglang)],
            &[wired(Vllm, Grpc)],
        ),
        row(
            Sheet::Wire,
            "a-wire-over-on-every-worker-is-steered",
            &[
                on(Vllm).state(State::Over),
                on(Vllm).state(State::Over),
                on(Sglang),
            ],
            &[wired(Vllm, Grpc)],
        ),
        row(
            Sheet::Wire,
            "a-wire-over-on-every-worker-is-shed-where-shedding-is-on",
            &[
                on(Vllm).state(State::Over),
                on(Vllm).state(State::Over),
                on(Sglang),
            ],
            &[wired(Vllm, Grpc)],
        )
        .shedding(),
        row(
            Sheet::Wire,
            "a-wire-no-worker-of-the-model-is-on",
            &[on(Sglang), on(Vllm).zmq()],
            &[wired(Vllm, Grpc), wired(TokenSpeed, Grpc)],
        ),
        row(Sheet::Wire, "a-model-no-worker-serves", &[], &[ANY]),
        // Media by reference, and pixel media.
        row(
            Sheet::Media,
            "references-go-to-the-workers-that-process-them",
            &[
                on(Vllm),
                on(Vllm).processor(OWN),
                on(Vllm).processor(ENGINE),
                on(Sglang).processor(OWN),
                on(Sglang).processor(ENGINE),
                on(Sglang),
                on(TokenSpeed).processor(OWN),
                on(Vllm).processor(OWN).zmq(),
            ],
            &[
                ANY,
                refs(Pin::Any),
                refs(Pin::Only),
                refs(Pin::Never),
                refs(Pin::Any).on_wire(Vllm, Grpc),
                refs(Pin::Only).on_wire(Sglang, Grpc),
                refs(Pin::Never).on_wire(Sglang, Grpc),
            ],
        ),
        row(
            Sheet::Media,
            "pixel-media-goes-to-a-worker-with-a-vision-encoder",
            &[
                on(Vllm),
                on(Vllm).language_only(),
                on(Vllm).processor(OWN),
                on(Vllm).processor(OWN).language_only(),
            ],
            &[
                pixels(),
                refs(Pin::Any),
                refs(Pin::Any).with_pixels(),
                refs(Pin::Only).with_pixels(),
            ],
        ),
        row(
            Sheet::Media,
            "no-worker-has-a-vision-encoder",
            &[on(Vllm).language_only(), on(Sglang).language_only()],
            &[pixels(), refs(Pin::Any).with_pixels()],
        ),
        row(
            Sheet::Media,
            "no-worker-processes-references",
            &[on(Vllm), on(Sglang), on(TokenSpeed).processor(OWN)],
            &[refs(Pin::Any), refs(Pin::Only), refs(Pin::Never)],
        ),
        row(
            Sheet::Media,
            "the-workers-that-process-references-stand-on-the-other-side-of-the-pin",
            &[on(Vllm).processor(ENGINE), on(Vllm)],
            &[refs(Pin::Only), refs(Pin::Never), refs(Pin::Any)],
        ),
        row(
            Sheet::Media,
            "the-workers-that-process-references-run-the-pipeline-a-plan-pins-away",
            &[on(Vllm).processor(OWN), on(Sglang).processor(OWN)],
            &[refs(Pin::Never), refs(Pin::Only)],
        ),
        row(
            Sheet::Media,
            "the-one-worker-that-processes-references-is-down",
            &[on(Vllm).processor(OWN).state(State::Down), on(Vllm)],
            &[refs(Pin::Any), ANY],
        ),
        row(
            Sheet::Media,
            "the-workers-that-process-references-are-over-their-thresholds",
            &[
                on(Vllm).processor(OWN).state(State::Over),
                on(Vllm).processor(ENGINE).state(State::Over),
                on(Vllm),
            ],
            &[refs(Pin::Any), refs(Pin::Only)],
        ),
        row(
            Sheet::Media,
            "the-workers-that-process-references-are-over-and-shedding-is-on",
            &[
                on(Vllm).processor(OWN).state(State::Over),
                on(Vllm).processor(ENGINE).state(State::Over),
                on(Vllm),
            ],
            &[refs(Pin::Any), refs(Pin::Only)],
        )
        .shedding(),
        row(
            Sheet::Media,
            "references-and-no-worker-of-the-model",
            &[],
            &[refs(Pin::Any), pixels()],
        ),
        // The context window.
        row(
            Sheet::Window,
            "each-bound-is-held-against-the-window-alone",
            &[
                on(Vllm).window(100),
                on(Vllm).window(200),
                on(Vllm),
                on(Vllm).window(50).zmq(),
            ],
            &[
                ANY,
                input(100),
                input(101),
                scored(100),
                scored(99),
                budget(0, 100),
                budget(0, 101),
                budget(60, 60),
                budget(100, 100),
                budget(150, 150),
                input(200),
                input(201),
                scored(200),
            ],
        ),
        row(
            Sheet::Window,
            "no-window-holds-the-input",
            &[on(Vllm).window(100), on(Vllm).window(200)],
            &[input(201), scored(200), input(1000)],
        ),
        row(
            Sheet::Window,
            "no-window-holds-the-budget",
            &[on(Vllm).window(100), on(Vllm).window(200)],
            &[budget(0, 201), budget(10, 201), budget(201, 201)],
        ),
        row(
            Sheet::Window,
            "the-widest-window-is-read-on-the-retry-s-wire",
            &[
                on(Sglang).window(400),
                on(Vllm).window(100),
                on(Vllm).window(120),
            ],
            &[
                input(150),
                input(150).on_wire(Vllm, Grpc),
                input(110).on_wire(Vllm, Grpc),
                input(500),
            ],
        ),
        row(
            Sheet::Window,
            "the-worker-whose-window-holds-the-request-is-down",
            &[
                on(Vllm).window(100),
                on(Vllm).window(200).state(State::Down),
            ],
            &[input(150), input(50), input(250)],
        ),
        row(
            Sheet::Window,
            "the-worker-whose-window-holds-the-request-is-over-its-thresholds",
            &[
                on(Vllm).window(100),
                on(Vllm).window(200).state(State::Over),
            ],
            &[input(150), input(50)],
        ),
        row(
            Sheet::Window,
            "the-worker-whose-window-holds-the-request-is-over-and-shedding-is-on",
            &[
                on(Vllm).window(100),
                on(Vllm).window(200).state(State::Over),
            ],
            &[input(150), input(50)],
        )
        .shedding(),
        row(
            Sheet::Window,
            "a-pool-of-unknown-windows-takes-any-request",
            &[on(Vllm), on(Sglang)],
            &[input(1_000_000), budget(1_000_000, 1_000_000)],
        ),
    ];

    // The constraint's decider on a retry.
    let whole = on(TokenSpeed).decoding(Some("always"), Some("first_token"), None);
    let later = on(TokenSpeed).decoding(Some("always"), Some("after_reasoning"), None);
    let unsaid = on(TokenSpeed).decoding(Some("always"), None, None);
    let asked = on(TokenSpeed);
    let older = on(TokenSpeed).decoding(Some("always"), Some("first_token"), Some("older"));
    let older_later =
        on(TokenSpeed).decoding(Some("always"), Some("after_reasoning"), Some("older"));
    rows.push(row(
        Sheet::Decider,
        "a-retry-stays-on-the-workers-that-decide-the-constraint-alike",
        &[
            whole,
            asked,
            later,
            whole,
            older,
            unsaid,
            older_later,
            older,
        ],
        &[
            ANY,
            decided(0),
            decided(1),
            decided(2),
            decided(4),
            decided(5),
            decided(6),
        ],
    ));
    rows.push(row(
        Sheet::Decider,
        "the-one-worker-that-decides-alike-is-down",
        &[whole.state(State::Down), asked, later],
        &[decided(0), decided(1)],
    ));
    rows.push(row(
        Sheet::Decider,
        "the-workers-that-decide-alike-are-over-their-thresholds",
        &[whole.state(State::Over), whole.state(State::Over), asked],
        &[decided(0), decided(2)],
    ));
    rows.push(
        row(
            Sheet::Decider,
            "the-workers-that-decide-alike-are-over-and-shedding-is-on",
            &[whole.state(State::Over), whole.state(State::Over), asked],
            &[decided(0), decided(2)],
        )
        .shedding(),
    );
    rows.push(
        row(
            Sheet::Decider,
            "a-pool-over-on-every-worker-is-shed-whoever-decides-alike",
            &[whole.state(State::Over), asked.state(State::Over)],
            &[decided(0), decided(1), ANY],
        )
        .shedding(),
    );
    rows
}

#[tokio::test]
#[ignore = "records the classic rows into CLASSIC_ROWS_OUT"]
async fn record_the_classic_hard_demand_rows() {
    let out = PathBuf::from(
        std::env::var_os("CLASSIC_ROWS_OUT").expect("CLASSIC_ROWS_OUT names the output directory"),
    );
    let commit =
        std::env::var("CLASSIC_ROWS_COMMIT").expect("CLASSIC_ROWS_COMMIT names the commit");
    fs::create_dir_all(&out).expect("the output directory");
    let rows = rows();
    for sheet in Sheet::ALL {
        let mut written = Vec::new();
        for row in rows.iter().filter(|row| row.sheet == sheet) {
            let first = recorded(row).await;
            for _ in 1..20 {
                assert_eq!(
                    recorded(row).await,
                    first,
                    "{}: two recordings differ",
                    row.name
                );
            }
            println!("ROW {}: {}", row.name, first["answers"]);
            written.push(first);
        }
        let path = out.join(format!("{}.json", sheet.name()));
        let text = serde_json::to_string_pretty(&json!({
            "recorded_from": commit,
            "policy": sheet.name().replace('_', "-"),
            "rows": written,
        }))
        .expect("the sheet serializes");
        fs::write(&path, text + "\n").expect("the sheet is written");
        println!("{} rows -> {}", written.len(), path.display());
    }
}
