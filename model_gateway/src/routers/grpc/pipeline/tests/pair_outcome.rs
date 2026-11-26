//! A pair on this wire tells Grand Central each side's own end, in both
//! dispatch shapes. The decode's stream tells its end as a single request's
//! does, and the execution stage tells a decode whose dispatch failed. The
//! prefill is told through its load where its dispatch records it, and by
//! its reader where its stream fails under one. A side dropped for its
//! partner's failure is told by nobody.
//!
//! Each case runs the completion pipeline against stub engines, under the
//! stock pair document with a stage that names the side of each end it
//! hears. The stub serves both wires: the one whose pairs go out both at
//! once, and the one whose pairs go out one behind the other.

use openai_protocol::worker::WorkerStatus;
use vllm_proto::vllm_engine_server::{VllmEngine, VllmEngineServer};

use super::{outcome::complete, *};
use crate::routers::{
    common::placement::tests::{hearing_later_pair_sides_on, hearing_pair_sides_on, Heard},
    grpc::{
        backend_client::get_backend_client,
        common::stages::request_execution::tests::bears_untold_end,
    },
};

/// A deadline no case of this file reaches.
const NO_DEADLINE: Duration = Duration::from_secs(60);
/// A wait no case of this file sits out.
const NEVER: Duration = Duration::from_secs(3600);
/// How far behind its partner's answer a side's late refusal lands.
const LATE: Duration = Duration::from_millis(150);

/// Both sides of a pair ended well.
const BOTH_GOOD: [&str; 2] = ["completed Decode Ok 200", "completed Prefill Ok 200"];
/// A decode that was sent and dropped untold, and one that was never sent.
const DECODE_DROPPED: &str = "completed Decode Cancelled";
const DECODE_NEVER_SENT: &str = "released Decode Abandoned";
/// A prefill that was sent and dropped untold.
const PREFILL_DROPPED: &str = "completed Prefill Cancelled";

type VllmFrames = Pin<Box<dyn Stream<Item = Result<vllm_proto::GenerateResponse, Status>> + Send>>;

/// The stub's frame on the wire whose pairs go out one behind the other:
/// the same tokens and counts.
fn vllm_frame(frame: ts::GenerateResponse) -> vllm_proto::GenerateResponse {
    use ts::generate_response::Response as Own;
    use vllm_proto::generate_response::Response as Other;
    let response = frame.response.map(|response| match response {
        Own::Chunk(chunk) => Other::Chunk(vllm_proto::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            ..Default::default()
        }),
        Own::Complete(complete) => Other::Complete(vllm_proto::GenerateComplete {
            output_ids: complete.output_ids,
            finish_reason: complete.finish_reason,
            prompt_tokens: complete.prompt_tokens,
            completion_tokens: complete.completion_tokens,
            ..Default::default()
        }),
    });
    vllm_proto::GenerateResponse { response }
}

fn unused<T>() -> Result<TonicResponse<T>, Status> {
    Err(Status::unimplemented("the pair stub generates and no more"))
}

/// The stub engine on the wire whose pairs go out one behind the other.
/// Each call is answered as the stub answers its own wire's, frame for
/// frame, under the same knobs.
#[tonic::async_trait]
impl VllmEngine for GatedScheduler {
    type GenerateStream = VllmFrames;
    type GetTokenizerStream = TokenizerStream;
    type SubscribeKvEventsStream = KvEventStream;

    async fn generate(
        &self,
        request: TonicRequest<vllm_proto::GenerateRequest>,
    ) -> Result<TonicResponse<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let input_ids = match request.input {
            Some(vllm_proto::generate_request::Input::Tokenized(tokenized)) => tokenized.input_ids,
            _ => Vec::new(),
        };
        let own = ts::GenerateRequest {
            request_id: request.request_id,
            tokenized: Some(ts::TokenizedInput {
                input_ids,
                ..Default::default()
            }),
            ..Default::default()
        };
        let frames = TokenSpeedScheduler::generate(self, TonicRequest::new(own)).await?;
        let frames = frames.into_inner().map(|frame| frame.map(vllm_frame));
        Ok(TonicResponse::new(Box::pin(frames)))
    }

    async fn abort(
        &self,
        _request: TonicRequest<vllm_proto::AbortRequest>,
    ) -> Result<TonicResponse<vllm_proto::AbortResponse>, Status> {
        Ok(TonicResponse::new(vllm_proto::AbortResponse::default()))
    }

    async fn embed(
        &self,
        _request: TonicRequest<vllm_proto::EmbedRequest>,
    ) -> Result<TonicResponse<vllm_proto::EmbedResponse>, Status> {
        unused()
    }

    async fn health_check(
        &self,
        _request: TonicRequest<vllm_proto::HealthCheckRequest>,
    ) -> Result<TonicResponse<vllm_proto::HealthCheckResponse>, Status> {
        unused()
    }

    async fn flush_cache(
        &self,
        _request: TonicRequest<common::FlushCacheRequest>,
    ) -> Result<TonicResponse<common::FlushCacheResponse>, Status> {
        unused()
    }

    async fn start_profile(
        &self,
        _request: TonicRequest<common::StartProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        unused()
    }

    async fn stop_profile(
        &self,
        _request: TonicRequest<common::StopProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        unused()
    }

    async fn get_model_info(
        &self,
        _request: TonicRequest<vllm_proto::GetModelInfoRequest>,
    ) -> Result<TonicResponse<vllm_proto::GetModelInfoResponse>, Status> {
        unused()
    }

    async fn get_server_info(
        &self,
        _request: TonicRequest<vllm_proto::GetServerInfoRequest>,
    ) -> Result<TonicResponse<vllm_proto::GetServerInfoResponse>, Status> {
        unused()
    }

    async fn get_loads(
        &self,
        _request: TonicRequest<vllm_proto::GetLoadsRequest>,
    ) -> Result<TonicResponse<vllm_proto::GetLoadsResponse>, Status> {
        unused()
    }

    async fn get_tokenizer(
        &self,
        _request: TonicRequest<common::GetTokenizerRequest>,
    ) -> Result<TonicResponse<Self::GetTokenizerStream>, Status> {
        unused()
    }

    async fn subscribe_kv_events(
        &self,
        _request: TonicRequest<common::SubscribeKvEventsRequest>,
    ) -> Result<TonicResponse<Self::SubscribeKvEventsStream>, Status> {
        unused()
    }
}

/// How a pair's two sides go out. The runtime its workers are registered
/// under decides it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// Both at once.
    Parallel,
    /// One behind the other: the decode goes out once its prefill is
    /// drained.
    Sequential,
}

const SHAPES: [Shape; 2] = [Shape::Parallel, Shape::Sequential];

impl Shape {
    fn runtime(self) -> RuntimeType {
        match self {
            Self::Parallel => RuntimeType::TokenSpeed,
            Self::Sequential => RuntimeType::Vllm,
        }
    }

    /// `scheduler` served on a loopback port, on this shape's wire.
    async fn serve(self, scheduler: GatedScheduler) -> u16 {
        match self {
            Self::Parallel => spawn_stub(scheduler).await,
            Self::Sequential => {
                let router = Server::builder().add_service(VllmEngineServer::new(scheduler));
                serve_on_loopback(router).await.0.port()
            }
        }
    }
}

/// The pair pipeline of `shape` over one prefill stub and the decode stubs,
/// whose requests `policies` place, with the stubs' workers: the prefill's
/// first.
async fn placed_by(
    shape: Shape,
    policies: Arc<PolicyRegistry>,
    prefill: GatedScheduler,
    decodes: Vec<GatedScheduler>,
    request_timeout: Duration,
) -> (RequestPipeline, Arc<SharedComponents>, Vec<Arc<dyn Worker>>) {
    let registry = Arc::new(WorkerRegistry::new());
    let mut workers = Vec::new();
    let decodes = decodes
        .into_iter()
        .map(|decode| (decode, WorkerType::Decode));
    for (scheduler, side) in std::iter::once((prefill, WorkerType::Prefill)).chain(decodes) {
        let port = shape.serve(scheduler).await;
        let runtime = shape.runtime();
        workers.push(register_worker_of_runtime(
            &registry,
            port,
            side,
            runtime,
            &[],
        ));
    }
    let deps = PipelineDeps::pair(
        registry.clone(),
        policies,
        None,
        None,
        Duration::from_secs(60),
    )
    .with_request_timeout(request_timeout);
    let pipeline = RequestPipeline::build(Endpoint::Completion, Mode::PrefillDecode, &deps)
        .expect("the pair's pipeline");
    (pipeline, components(registry).await, workers)
}

/// The pair pipeline of `shape` over one prefill stub and one decode stub. A
/// Grand Central places its requests, under a stage that names the side of each
/// end it hears. Answers the pipeline, what the stage heard and the two
/// workers.
async fn hearing_within(
    shape: Shape,
    prefill: GatedScheduler,
    decode: GatedScheduler,
    request_timeout: Duration,
) -> (
    RequestPipeline,
    Arc<SharedComponents>,
    Heard,
    Vec<Arc<dyn Worker>>,
) {
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let heard = hearing_pair_sides_on(&policies);
    let (pipeline, components, workers) =
        placed_by(shape, policies, prefill, vec![decode], request_timeout).await;
    (pipeline, components, heard, workers)
}

/// [`hearing_within`] under a deadline no case reaches.
async fn hearing(
    shape: Shape,
    prefill: GatedScheduler,
    decode: GatedScheduler,
) -> (
    RequestPipeline,
    Arc<SharedComponents>,
    Heard,
    Vec<Arc<dyn Worker>>,
) {
    hearing_within(shape, prefill, decode, NO_DEADLINE).await
}

/// A completion request of `body`, under the cases' model.
fn request_of(mut body: serde_json::Value) -> Arc<CompletionRequest> {
    body["model"] = MODEL.into();
    Arc::new(serde_json::from_value(body).expect("a completion request"))
}

/// Read `response`'s body to its end, as its client does.
async fn read(response: Response) {
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body is read to its end");
}

/// What the stage heard of each side's end so far, sorted: a completion or
/// a release. The two sides of a pair end in an order the cases do not
/// hold.
fn ends(heard: &Heard) -> Vec<String> {
    let heard = heard.lock().expect("the stage heard");
    let ends = heard
        .iter()
        .filter(|word| word.starts_with("completed") || word.starts_with("released"));
    let mut ends: Vec<String> = ends.cloned().collect();
    ends.sort();
    ends
}

/// [`ends`] once the stage heard `count` of them: a reader tells its side's
/// end from a task of its own.
async fn ends_at(heard: &Heard, count: usize) -> Vec<String> {
    let deadline = Instant::now() + Duration::from_secs(10);
    while ends(heard).len() < count && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    ends(heard)
}

/// `words`, sorted as [`ends`] sorts what the stage heard.
fn sorted<const N: usize>(mut words: [&str; N]) -> [&str; N] {
    words.sort_unstable();
    words
}

/// A stub that refuses every call.
fn refuses() -> GatedScheduler {
    GatedScheduler {
        fail_always: true,
        ..Default::default()
    }
}

/// A stub that refuses every call behind its partner's answer.
fn refuses_late() -> GatedScheduler {
    GatedScheduler {
        fail_after: Some(LATE),
        ..Default::default()
    }
}

/// A stub whose call has not answered when its partner's fails.
fn answers_late() -> GatedScheduler {
    GatedScheduler {
        answer_after: Some(Duration::from_secs(2)),
        ..Default::default()
    }
}

/// A stub whose call never answers.
fn silent() -> GatedScheduler {
    GatedScheduler {
        answer_after: Some(NEVER),
        ..Default::default()
    }
}

/// A pair that ends well tells both sides the worker's 200, read whole and
/// streamed, in both shapes. A request of several samples and a batch of
/// prompts are one placement of each side. The engine hears one end of the
/// prefill and one of the decode, however many calls each side took.
#[tokio::test]
async fn the_engine_hears_both_ends_of_a_pair() {
    let bodies = [
        ("one sample", serde_json::json!({"prompt": "Hello world"})),
        (
            "three samples",
            serde_json::json!({"prompt": "Hello world", "n": 3}),
        ),
        (
            "two prompts",
            serde_json::json!({"prompt": ["Hello world", "Hello again"]}),
        ),
    ];
    for shape in SHAPES {
        for (what, body) in &bodies {
            // A request of several samples fans out where both sides go
            // out at once. The other wire's engine samples them itself.
            if shape == Shape::Sequential && *what == "three samples" {
                continue;
            }
            for stream in [false, true] {
                let row = format!("{shape:?}, {what}, stream {stream}");
                let (pipeline, components, heard, _workers) =
                    hearing(shape, GatedScheduler::default(), GatedScheduler::default()).await;
                let mut body = body.clone();
                body["stream"] = stream.into();

                let response = complete(&pipeline, &components, request_of(body), None).await;
                assert_eq!(response.status(), http::StatusCode::OK, "{row}");
                read(response).await;

                assert_eq!(ends_at(&heard, 2).await, BOTH_GOOD, "{row}");
            }
        }
    }
}

/// A streamed pair's decode is told where its stream ends. Nothing of the
/// decode is heard while the stream stands, and the worker's 200 once it was
/// read to its end.
#[tokio::test]
async fn the_engine_hears_a_pairs_decode_where_its_stream_ends() {
    for shape in SHAPES {
        let gate = Arc::new(AtomicBool::new(false));
        let decode = GatedScheduler {
            gate_after_first: Some(Arc::clone(&gate)),
            ..Default::default()
        };
        let (pipeline, components, heard, workers) =
            hearing(shape, GatedScheduler::default(), decode).await;

        let response = complete(&pipeline, &components, completion_request(true), None).await;
        assert_eq!(response.status(), http::StatusCode::OK, "{shape:?}");
        let client = read_body(response);
        first_response_read(&workers[1]).await;
        let so_far = ends(&heard);
        assert!(
            !so_far.iter().any(|end| end.contains("Decode")),
            "{shape:?}: no end of the decode while its stream stands: {so_far:?}"
        );

        gate.store(true, Ordering::SeqCst);
        client.await.expect("the client read the body to its end");
        assert_eq!(ends_at(&heard, 2).await, BOTH_GOOD, "{shape:?}");
    }
}

/// A prefill whose call fails is told its failure through its load, where
/// its dispatch records it, and the client's answer bears no end for the
/// decode. The decode is told nothing: sent one behind the other it never
/// went out, and sent beside its prefill it is dropped or retired.
#[tokio::test]
async fn a_pairs_failed_prefill_is_told_and_its_decode_is_told_nothing() {
    let rows = [
        (
            "one behind the other",
            Shape::Sequential,
            refuses(),
            GatedScheduler::default(),
            DECODE_NEVER_SENT,
        ),
        (
            "both at once, while its decode still dispatches",
            Shape::Parallel,
            refuses(),
            answers_late(),
            DECODE_DROPPED,
        ),
        (
            "both at once, behind its decode's answer",
            Shape::Parallel,
            refuses_late(),
            GatedScheduler::default(),
            DECODE_DROPPED,
        ),
    ];
    for (row, shape, prefill, decode, decode_end) in rows {
        let (pipeline, components, heard, _workers) = hearing(shape, prefill, decode).await;

        let response = complete(&pipeline, &components, completion_request(false), None).await;

        assert_eq!(
            response.status(),
            http::StatusCode::SERVICE_UNAVAILABLE,
            "{row}"
        );
        assert!(!bears_untold_end(&response), "{row}");
        assert_eq!(
            ends_at(&heard, 2).await,
            sorted([decode_end, "completed Prefill ServerError 503"]),
            "{row}"
        );
    }
}

/// A decode whose call fails opened no stream to tell its end: its answer
/// bears the end it owes, and the stage's plan takes it off and tells the
/// decode. The prefill keeps its own end. One behind the other it was
/// drained and told its good answer. Sent beside its decode, its first
/// response was not waited for, and nobody tells it.
#[tokio::test]
async fn a_pairs_failed_decode_is_told_by_the_plan_and_its_prefill_keeps_its_own_end() {
    let rows = [
        (
            "one behind the other",
            Shape::Sequential,
            GatedScheduler::default(),
            refuses(),
            "completed Prefill Ok 200",
        ),
        (
            "both at once, while its prefill still dispatches",
            Shape::Parallel,
            answers_late(),
            refuses(),
            PREFILL_DROPPED,
        ),
        (
            "both at once, behind its prefill's call",
            Shape::Parallel,
            GatedScheduler::default(),
            refuses_late(),
            PREFILL_DROPPED,
        ),
    ];
    for (row, shape, prefill, decode, prefill_end) in rows {
        let (pipeline, components, heard, _workers) = hearing(shape, prefill, decode).await;

        let response = complete(&pipeline, &components, completion_request(false), None).await;

        assert_eq!(
            response.status(),
            http::StatusCode::SERVICE_UNAVAILABLE,
            "{row}"
        );
        assert!(!bears_untold_end(&response), "{row}");
        assert_eq!(
            ends_at(&heard, 2).await,
            ["completed Decode ServerError 503", prefill_end],
            "{row}"
        );
    }
}

/// A prefill stream that fails behind its first frame, with the engine's
/// own report, is told as its reader samples it. One behind the other the
/// dispatch drains it: a streamed answer charges the report and tells it,
/// and a collected answer does neither. Sent beside its decode the prefill
/// told its good answer at its first response: a streamed answer's reader
/// tells the failure over it, and a collected answer's leaves it.
#[tokio::test]
async fn a_prefill_stream_that_fails_is_told_as_its_reader_samples_it() {
    let rows = [
        (Shape::Sequential, false, PREFILL_DROPPED),
        (Shape::Sequential, true, "completed Prefill ServerError 500"),
        (Shape::Parallel, false, "completed Prefill Ok 200"),
        (Shape::Parallel, true, "completed Prefill ServerError 500"),
    ];
    for (shape, stream, prefill_end) in rows {
        let prefill = GatedScheduler {
            fail_mid_stream: true,
            ..Default::default()
        };
        let (pipeline, components, heard, _workers) =
            hearing(shape, prefill, GatedScheduler::default()).await;

        let response = complete(&pipeline, &components, completion_request(stream), None).await;
        read(response).await;

        let all = ends_at(&heard, 2).await;
        let of_the_prefill: Vec<&String> =
            all.iter().filter(|end| end.contains("Prefill")).collect();
        assert_eq!(
            of_the_prefill,
            [prefill_end],
            "{shape:?}, stream {stream}: {all:?}"
        );
    }
}

/// A decode stream that fails is told once, by the status the circuit
/// breaker samples for it, to a collected answer and a streamed one alike.
/// The failure may be the stream's first item or lie behind its first
/// frame. The prefill had answered, and keeps its good end.
#[tokio::test]
async fn a_decode_stream_that_fails_is_told_once_by_its_status() {
    for shape in SHAPES {
        for stream in [false, true] {
            for at_once in [false, true] {
                let row = format!("{shape:?}, stream {stream}, at its first item {at_once}");
                let decode = GatedScheduler {
                    fail_mid_stream: !at_once,
                    fail_engine_report: at_once,
                    ..Default::default()
                };
                let (pipeline, components, heard, _workers) =
                    hearing(shape, GatedScheduler::default(), decode).await;

                let response =
                    complete(&pipeline, &components, completion_request(stream), None).await;
                read(response).await;

                assert_eq!(
                    ends_at(&heard, 2).await,
                    [
                        "completed Decode ServerError 500",
                        "completed Prefill Ok 200"
                    ],
                    "{row}"
                );
            }
        }
    }
}

/// The request's deadline is a timeout of the decode, as the circuit breaker
/// samples it, wherever it passes once the decode was sent. Before the decode's
/// call answered, the stage's plan tells it. Behind that answer, the decode's
/// stream ends with it and tells it. A prefill still in flight then is told
/// nothing. A deadline that passes while a pair sent one behind the other still
/// waits on its prefill tells nobody: the decode never went out. No answer
/// leaves the stage with an end it owes.
///
/// A row whose prefill must have answered runs under a deadline that a
/// loaded machine's prefill answers inside. The rows run side by side.
#[tokio::test]
async fn a_pairs_deadline_is_told_to_its_decode_once_the_decode_was_sent() {
    const TIMED_OUT: &str = "completed Decode Timeout 504";
    const AMPLE: Duration = Duration::from_secs(3);
    let slow_stream = || GatedScheduler {
        frame_gap: Some(NEVER),
        ..Default::default()
    };
    let rows = [
        (
            "both at once, the decode's call",
            Shape::Parallel,
            GatedScheduler::default(),
            silent(),
            REQUEST_TIMEOUT,
            [TIMED_OUT, PREFILL_DROPPED],
        ),
        (
            "both at once, the prefill's call",
            Shape::Parallel,
            silent(),
            GatedScheduler::default(),
            REQUEST_TIMEOUT,
            [TIMED_OUT, PREFILL_DROPPED],
        ),
        (
            "both at once, the decode's stream",
            Shape::Parallel,
            GatedScheduler::default(),
            slow_stream(),
            AMPLE,
            [TIMED_OUT, "completed Prefill Ok 200"],
        ),
        (
            "one behind the other, the decode's call",
            Shape::Sequential,
            GatedScheduler::default(),
            silent(),
            AMPLE,
            [TIMED_OUT, "completed Prefill Ok 200"],
        ),
        (
            "one behind the other, the decode's stream",
            Shape::Sequential,
            GatedScheduler::default(),
            slow_stream(),
            AMPLE,
            [TIMED_OUT, "completed Prefill Ok 200"],
        ),
        (
            "one behind the other, the prefill's call",
            Shape::Sequential,
            silent(),
            GatedScheduler::default(),
            REQUEST_TIMEOUT,
            [PREFILL_DROPPED, DECODE_NEVER_SENT],
        ),
    ];
    let rows = rows.map(
        |(row, shape, prefill, decode, deadline, expected)| async move {
            let (pipeline, components, heard, _workers) =
                hearing_within(shape, prefill, decode, deadline).await;

            let response = complete(&pipeline, &components, completion_request(false), None).await;

            assert_eq!(
                response.status(),
                http::StatusCode::GATEWAY_TIMEOUT,
                "{row}"
            );
            assert!(!bears_untold_end(&response), "{row}");
            assert_eq!(ends_at(&heard, 2).await, expected, "{row}");
        },
    );
    futures::future::join_all(rows).await;
}

/// A pair's decode that moves behind its prefill is told its end on the decode
/// the request went to. The document here picks the decode behind the prefill,
/// and the decode named first leaves the pool while the prefill is in flight.
/// The request goes to the other decode, whose stream tells the worker's 200
/// through the guard that moved with it. A request moves only to a decode whose
/// client the gateway holds, so both are connected first.
#[tokio::test]
async fn a_moved_decodes_end_is_told_on_the_decode_the_request_went_to() {
    let gate = Arc::new(AtomicBool::new(false));
    let prefill = GatedScheduler {
        call_gate: Some(Arc::clone(&gate)),
        ..Default::default()
    };
    let decodes = [GatedScheduler::default(), GatedScheduler::default()];
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let heard = hearing_later_pair_sides_on(&policies);
    let (pipeline, components, workers) = placed_by(
        Shape::Sequential,
        policies,
        prefill.clone(),
        decodes.to_vec(),
        NO_DEADLINE,
    )
    .await;
    for decode in &workers[1..] {
        let client = get_backend_client(decode.as_ref()).await;
        assert!(
            matches!(client, Ok(Some(_))),
            "the decode's client connects"
        );
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "the request runs beside the case, which moves its decode during the prefill"
    )]
    let request = tokio::spawn(async move {
        let request = completion_request(false);
        complete(&pipeline, &components, request, None).await
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while prefill.started.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "the prefill was never called");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // The decode named first is counted from its selection on.
    let first = workers[1..]
        .iter()
        .position(|decode| decode.load() == 1)
        .expect("the decode named first is counted ahead of its send");
    workers[1 + first].set_status(WorkerStatus::NotReady);
    gate.store(true, Ordering::SeqCst);

    let response = request.await.expect("the request ran");
    assert_eq!(response.status(), http::StatusCode::OK);
    read(response).await;

    let calls = |decode: &GatedScheduler| decode.started.load(Ordering::SeqCst);
    assert_eq!(
        (calls(&decodes[first]), calls(&decodes[1 - first])),
        (0, 1),
        "the request went to the other decode"
    );
    assert_eq!(ends_at(&heard, 2).await, BOTH_GOOD);
}

/// A prefill that answered keeps its good end where its client leaves behind
/// it. The prefill was told the worker's 200 inside its dispatch. The decode's
/// stream, dropped before its end, tells nothing.
#[tokio::test]
async fn a_prefill_that_answered_keeps_its_good_end_where_its_client_leaves() {
    for shape in SHAPES {
        let decode = GatedScheduler {
            hold_after_first: true,
            ..Default::default()
        };
        let (pipeline, components, heard, workers) =
            hearing(shape, GatedScheduler::default(), decode).await;

        let response = complete(&pipeline, &components, completion_request(true), None).await;
        assert_eq!(response.status(), http::StatusCode::OK, "{shape:?}");
        let client = read_body(response);
        first_response_read(&workers[1]).await;
        client.abort();

        assert_eq!(
            ends_at(&heard, 2).await,
            [DECODE_DROPPED, "completed Prefill Ok 200"],
            "{shape:?}"
        );
    }
}

/// Each attempt of a request is a pair of its own. The engine hears the failed
/// attempt's prefill by its status and its decode dropped, and then both good
/// ends of the retry.
#[tokio::test]
async fn a_pairs_retry_is_a_pair_of_its_own() {
    let prefill = GatedScheduler {
        fail_first: true,
        ..Default::default()
    };
    let (pipeline, components, heard, _workers) =
        hearing(Shape::Parallel, prefill, GatedScheduler::default()).await;
    let retry = fast_retry_config(2);

    let request = completion_request(false);
    let response = complete(&pipeline, &components, request, Some(&retry)).await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        ends_at(&heard, 4).await,
        [
            DECODE_DROPPED,
            "completed Decode Ok 200",
            "completed Prefill Ok 200",
            "completed Prefill ServerError 503",
        ]
    );
}

/// A request of several samples and a batch of prompts tell one end of
/// each side where a child fails, too. A fan-out child whose prefill is
/// refused ends the prefill side with that failure, against a sibling's
/// good answer, and tells the decode nothing. A batch's unit whose decode
/// is refused ends the decode side with that failure: the unit's answer
/// bears the end, and the plan tells it. A unit sent one behind the other
/// whose prefill stream fails behind its sibling's drain ends the prefill
/// side with the failure, over the sibling's good answer.
#[tokio::test]
async fn a_failed_child_of_a_fan_out_or_a_batch_ends_its_side_once() {
    let refuses_second = || GatedScheduler {
        fail_call: Some(1),
        ..Default::default()
    };
    let two_prompts = |stream: bool| {
        request_of(serde_json::json!({
            "prompt": ["Hello world", "Hello again"],
            "stream": stream,
        }))
    };

    let (pipeline, components, heard, _workers) =
        hearing(Shape::Parallel, refuses_second(), GatedScheduler::default()).await;
    let fan_out = request_of(serde_json::json!({"prompt": "Hello world", "n": 3}));
    let response = complete(&pipeline, &components, fan_out, None).await;
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        ends_at(&heard, 2).await,
        [DECODE_DROPPED, "completed Prefill ServerError 503"],
        "a fan-out child's refused prefill"
    );

    for shape in SHAPES {
        let (pipeline, components, heard, _workers) =
            hearing(shape, GatedScheduler::default(), refuses_second()).await;
        let response = complete(&pipeline, &components, two_prompts(false), None).await;
        assert_eq!(
            response.status(),
            http::StatusCode::SERVICE_UNAVAILABLE,
            "{shape:?}"
        );
        assert!(!bears_untold_end(&response), "{shape:?}");
        let all = ends_at(&heard, 2).await;
        let of_the_decode: Vec<&String> = all.iter().filter(|end| end.contains("Decode")).collect();
        assert_eq!(
            of_the_decode,
            ["completed Decode ServerError 503"],
            "a batch unit's refused decode, {shape:?}: {all:?}"
        );
    }

    let fails_late = GatedScheduler {
        fail_late: Some(1),
        ..Default::default()
    };
    let (pipeline, components, heard, _workers) =
        hearing(Shape::Sequential, fails_late, GatedScheduler::default()).await;
    let response = complete(&pipeline, &components, two_prompts(true), None).await;
    read(response).await;
    assert_eq!(
        ends_at(&heard, 2).await,
        [DECODE_DROPPED, "completed Prefill ServerError 500"],
        "a batch unit's prefill stream that failed behind its sibling's drain"
    );
}

/// A fan-out of more than thirty samples joins its children in their
/// order: a child's failure waits for every child ahead of it. Here the
/// last child's prefill is refused at once and the others answer behind
/// it. Their good answers do not write the failure over: the request
/// fails, and the engine hears its prefill side end with the refusal.
#[tokio::test]
async fn a_wide_fan_outs_early_failure_stands_against_late_good_answers() {
    const SAMPLES: usize = 31;
    let prefill = GatedScheduler {
        fail_call: Some(SAMPLES - 1),
        answer_after: Some(Duration::from_millis(200)),
        ..Default::default()
    };
    let (pipeline, components, heard, _workers) =
        hearing(Shape::Parallel, prefill, GatedScheduler::default()).await;
    let fan_out = request_of(serde_json::json!({"prompt": "Hello world", "n": SAMPLES}));

    let response = complete(&pipeline, &components, fan_out, None).await;

    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        ends_at(&heard, 2).await,
        [DECODE_DROPPED, "completed Prefill ServerError 503"]
    );
}

/// A prefill whose call answers and whose stream then says nothing is
/// given up on at its silence bound, three seconds for a worker that has
/// answered nothing yet. The verdict is a timeout of the prefill, told
/// through its load in both shapes. The decode is told nothing: sent one
/// behind the other it never went out, and sent beside its prefill it is
/// dropped.
#[tokio::test]
async fn a_prefill_that_falls_silent_is_told_its_timeout_through_its_load() {
    let silent_stream = || GatedScheduler {
        frame_gap: Some(NEVER),
        ..Default::default()
    };
    let one = |shape: Shape, decode_end: &'static str| async move {
        let (pipeline, components, heard, _workers) =
            hearing(shape, silent_stream(), GatedScheduler::default()).await;

        let response = complete(&pipeline, &components, completion_request(false), None).await;

        assert_eq!(
            response.status(),
            http::StatusCode::GATEWAY_TIMEOUT,
            "{shape:?}"
        );
        assert!(!bears_untold_end(&response), "{shape:?}");
        assert_eq!(
            ends_at(&heard, 2).await,
            sorted([decode_end, "completed Prefill Timeout 504"]),
            "{shape:?}"
        );
    };
    tokio::join!(
        one(Shape::Parallel, DECODE_DROPPED),
        one(Shape::Sequential, DECODE_NEVER_SENT)
    );
}

/// One request of `body` through a pair pipeline of `shape` that the
/// classic policies place: it is answered well, and both workers' loads are
/// released behind its body.
async fn classic_pair(shape: Shape, body: serde_json::Value) {
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let decodes = vec![GatedScheduler::default()];
    let (pipeline, components, workers) = placed_by(
        shape,
        policies,
        GatedScheduler::default(),
        decodes,
        NO_DEADLINE,
    )
    .await;

    let response = complete(&pipeline, &components, request_of(body), None).await;
    assert_eq!(response.status(), http::StatusCode::OK, "{shape:?}");
    assert!(!bears_untold_end(&response), "{shape:?}");
    read(response).await;

    let deadline = Instant::now() + Duration::from_secs(5);
    while workers.iter().any(|worker| worker.load() != 0) {
        assert!(Instant::now() < deadline, "{shape:?}: a load was kept");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// A pair the classic policies placed owes Grand Central nothing and is
// served as it was: read whole and streamed in both shapes, as a fan-out and
// as a batch. Each case below is one such request. What a classic pair pays
// for the telling is counted on them.

#[tokio::test]
async fn a_classic_pair_sent_at_once_is_read_whole() {
    classic_pair(
        Shape::Parallel,
        serde_json::json!({"prompt": "Hello world"}),
    )
    .await;
}

#[tokio::test]
async fn a_classic_pair_sent_at_once_is_streamed() {
    let body = serde_json::json!({"prompt": "Hello world", "stream": true});
    classic_pair(Shape::Parallel, body).await;
}

#[tokio::test]
async fn a_classic_pair_sent_one_behind_the_other_is_read_whole() {
    classic_pair(
        Shape::Sequential,
        serde_json::json!({"prompt": "Hello world"}),
    )
    .await;
}

#[tokio::test]
async fn a_classic_pair_sent_one_behind_the_other_is_streamed() {
    let body = serde_json::json!({"prompt": "Hello world", "stream": true});
    classic_pair(Shape::Sequential, body).await;
}

#[tokio::test]
async fn a_classic_fan_out_of_three_is_read_whole() {
    let body = serde_json::json!({"prompt": "Hello world", "n": 3});
    classic_pair(Shape::Parallel, body).await;
}

#[tokio::test]
async fn a_classic_batch_of_two_pairs_is_read_whole() {
    let body = serde_json::json!({"prompt": ["Hello world", "Hello again"]});
    classic_pair(Shape::Parallel, body).await;
}
