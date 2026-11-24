//! A single request on this wire tells Grand Central its outcome. The
//! request's stream tells its own end, a clean one as the worker's 200 and a
//! failed one by the status the circuit breaker samples for it. The
//! execution stage tells an attempt that opened no stream: a call that
//! failed, a deadline that passed before the call answered. A stream dropped
//! before its end tells nothing.
//!
//! Each case runs the completion pipeline against the stub engine, under a
//! policy document whose one stage notes what it hears of each lease.

use super::*;
use crate::{
    config::types::RetryConfig,
    routers::{
        common::placement::tests::{hearing_engine_on, Heard},
        grpc::common::stages::request_execution::tests::bears_untold_end,
    },
};

/// A deadline no case of this file reaches.
const NO_DEADLINE: Duration = Duration::from_secs(60);

/// The completion pipeline over one stub engine, placed by a Grand Central
/// whose stage notes what it hears, with the stub's worker.
async fn hearing(
    scheduler: GatedScheduler,
    request_timeout: Duration,
) -> (
    RequestPipeline,
    Heard,
    Arc<SharedComponents>,
    Arc<dyn Worker>,
) {
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let heard = hearing_engine_on(&policies);
    let (pipeline, components, worker) = placed_by(policies, scheduler, request_timeout).await;
    (pipeline, heard, components, worker)
}

/// The completion pipeline over one stub engine whose requests `policies`
/// place, with the stub's worker.
async fn placed_by(
    policies: Arc<PolicyRegistry>,
    scheduler: GatedScheduler,
    request_timeout: Duration,
) -> (RequestPipeline, Arc<SharedComponents>, Arc<dyn Worker>) {
    let port = spawn_stub(scheduler).await;
    let registry = Arc::new(WorkerRegistry::new());
    let worker = register_worker(&registry, port, WorkerType::Regular);
    let deps = PipelineDeps::pair(
        registry.clone(),
        policies,
        None,
        None,
        Duration::from_secs(60),
    )
    .with_request_timeout(request_timeout);
    let pipeline = RequestPipeline::build(Endpoint::Completion, Mode::Regular, &deps)
        .expect("completion pipeline");
    (pipeline, components(registry).await, worker)
}

/// One completion through `pipeline`.
async fn complete(
    pipeline: &RequestPipeline,
    components: &Arc<SharedComponents>,
    request: Arc<CompletionRequest>,
    retry: Option<&RetryConfig>,
) -> Response {
    let run = pipeline.execute_completion(
        request,
        None,
        MODEL.to_string(),
        Arc::clone(components),
        None,
        None,
        retry,
    );
    tokio::time::timeout(CASE_CAP, run)
        .await
        .expect("the pipeline answers within the case's cap")
}

/// What the stage has heard so far.
fn said(heard: &Heard) -> Vec<String> {
    heard.lock().expect("the stage heard").clone()
}

/// What the stage heard once it has heard `words` of them: a streamed
/// answer's end is heard where its body goes, on the reader's task.
async fn said_after(heard: &Heard, words: usize) -> Vec<String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let said = said(heard);
        if said.len() >= words || Instant::now() >= deadline {
            return said;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// One lease the stage heard to its end with `end`.
fn lease(end: &str) -> Vec<String> {
    ["issued", "dispatched Single", end]
        .map(str::to_owned)
        .to_vec()
}

/// An answer read whole is read through the request's stream, which tells
/// the worker's 200 at its end. The reader's completed mark reaches the
/// worker's stream through the layer that tells the end: nothing is aborted
/// on the engine.
#[tokio::test]
async fn the_engine_hears_an_answer_read_whole_by_its_status() {
    let aborted = Arc::new(Mutex::new(Vec::new()));
    let scheduler = GatedScheduler {
        aborted_request_ids: Arc::clone(&aborted),
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;

    let response = complete(&pipeline, &components, completion_request(false), None).await;

    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(said(&heard), lease("completed Ok 200"));
    // A stream dropped unmarked sends its abort off the reader's task.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let aborted = aborted.lock().unwrap_or_else(PoisonError::into_inner);
    assert!(
        aborted.is_empty(),
        "a completed answer was aborted: {aborted:?}"
    );
}

/// A streamed answer's end is told where its stream ends: nothing is heard
/// while the stream stands, and the worker's 200 once it was read to its
/// end.
#[tokio::test]
async fn the_engine_hears_a_streamed_answers_end_where_its_stream_ends() {
    let gate = Arc::new(AtomicBool::new(false));
    let scheduler = GatedScheduler {
        gate_after_first: Some(Arc::clone(&gate)),
        ..Default::default()
    };
    let (pipeline, heard, components, worker) = hearing(scheduler, NO_DEADLINE).await;

    let response = complete(&pipeline, &components, completion_request(true), None).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let client = read_body(response);
    first_response_read(&worker).await;
    assert_eq!(
        said(&heard),
        ["issued", "dispatched Single"],
        "no end while the stream stands"
    );

    gate.store(true, Ordering::SeqCst);
    client.await.expect("the client read the body to its end");
    assert_eq!(said_after(&heard, 3).await, lease("completed Ok 200"));
}

/// A call the worker fails opens no stream, so the execution stage tells
/// its status. The engine hears that end before the retry's backoff is
/// over: neither the selection the context keeps nor a stream holds it
/// back. The retry is a lease of its own.
#[tokio::test]
async fn the_engine_hears_a_failed_call_before_its_retrys_backoff() {
    const BACKOFF_MS: u64 = 600;
    const BACKOFF: Duration = Duration::from_millis(BACKOFF_MS);
    let scheduler = GatedScheduler {
        fail_first: true,
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;
    let retry = RetryConfig {
        max_retries: 2,
        initial_backoff_ms: BACKOFF_MS,
        max_backoff_ms: BACKOFF_MS,
        backoff_multiplier: 1.0,
        jitter_factor: 0.0,
    };

    let started = Instant::now();
    #[expect(
        clippy::disallowed_methods,
        reason = "the request runs beside the case, which reads what the stage heard between its attempts"
    )]
    let request = tokio::spawn(async move {
        complete(
            &pipeline,
            &components,
            completion_request(false),
            Some(&retry),
        )
        .await
    });
    let failed = said_after(&heard, 3).await;
    let heard_at = started.elapsed();
    assert_eq!(failed, lease("completed ServerError 503"));
    assert!(
        heard_at < BACKOFF,
        "the failed attempt's end was heard {heard_at:?} in, behind its backoff"
    );

    let response = request.await.expect("the request ran");
    assert_eq!(response.status(), http::StatusCode::OK);
    assert!(started.elapsed() >= BACKOFF, "the retry waited its backoff");
    assert_eq!(said(&heard)[3..], lease("completed Ok 200"));
}

/// A stream the engine ends with its own report is told as the worker's
/// 500, to the collector of an answer read whole and to a streamed answer's
/// reader alike.
#[tokio::test]
async fn the_engine_hears_a_failed_stream_by_the_status_the_breaker_samples() {
    for stream in [false, true] {
        let scheduler = GatedScheduler {
            fail_mid_stream: true,
            ..Default::default()
        };
        let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;

        let response = complete(&pipeline, &components, completion_request(stream), None).await;
        let expected = if stream {
            http::StatusCode::OK
        } else {
            http::StatusCode::INTERNAL_SERVER_ERROR
        };
        assert_eq!(response.status(), expected, "stream: {stream}");
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body is read to its end");

        assert_eq!(
            said_after(&heard, 3).await,
            lease("completed ServerError 500"),
            "stream: {stream}"
        );
    }
}

/// The request's deadline is a timeout wherever it passes. Before the call
/// answered no stream stands, and the execution stage tells it. Behind the
/// call's answer the stream ends with it, and tells it.
#[tokio::test]
async fn the_engine_hears_the_requests_deadline_as_a_timeout() {
    let silence = Duration::from_secs(3600);
    let silent_call = GatedScheduler {
        answer_after: Some(silence),
        ..Default::default()
    };
    let silent_stream = GatedScheduler {
        frame_gap: Some(silence),
        ..Default::default()
    };
    for (scheduler, shape) in [(silent_call, "the call"), (silent_stream, "the stream")] {
        let (pipeline, heard, components, _worker) = hearing(scheduler, REQUEST_TIMEOUT).await;

        let response = complete(&pipeline, &components, completion_request(false), None).await;

        assert_eq!(
            response.status(),
            http::StatusCode::GATEWAY_TIMEOUT,
            "{shape}"
        );
        assert_eq!(said(&heard), lease("completed Timeout 504"), "{shape}");
    }
}

/// A failure the circuit breaker takes no sample of says nothing about the
/// worker, and nobody tells it: the engine could not continue the request
/// under its constraint. The end is heard as it was before, with no status.
#[tokio::test]
async fn the_engine_hears_no_status_of_a_failure_the_breaker_does_not_sample() {
    let scheduler = GatedScheduler {
        fail_constraint: true,
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;

    let response = complete(&pipeline, &components, completion_request(false), None).await;

    assert_eq!(response.status(), http::StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(said(&heard), lease("completed Cancelled"));
}

/// A client that leaves drops the stream before its end, and nobody tells
/// that end: it is heard with no status, as before.
#[tokio::test]
async fn the_engine_hears_a_client_that_left_as_cancelled() {
    let scheduler = GatedScheduler {
        hold_after_first: true,
        ..Default::default()
    };
    let (pipeline, heard, components, worker) = hearing(scheduler, NO_DEADLINE).await;

    let response = complete(&pipeline, &components, completion_request(true), None).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    let client = read_body(response);
    first_response_read(&worker).await;
    client.abort();

    assert_eq!(said_after(&heard, 3).await, lease("completed Cancelled"));
}

/// A batch of prompts is one placement, and the stream of each prompt tells
/// its end through that placement's one handle. The engine hears one end,
/// with the status last told. A prompt that fails ends the batch there, so
/// its status is the last: behind a prompt that ended well, and at a call
/// that opened no stream.
#[tokio::test]
async fn the_engine_hears_a_batch_as_its_one_placement() {
    let batch = |stream: bool| -> Arc<CompletionRequest> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "prompt": ["Hello world", "Hello again"],
                "stream": stream,
            }))
            .expect("a batched completion"),
        )
    };
    for stream in [false, true] {
        let (pipeline, heard, components, _worker) =
            hearing(GatedScheduler::default(), NO_DEADLINE).await;
        let response = complete(&pipeline, &components, batch(stream), None).await;
        assert_eq!(response.status(), http::StatusCode::OK, "stream: {stream}");
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the body is read to its end");
        assert_eq!(
            said_after(&heard, 3).await,
            lease("completed Ok 200"),
            "stream: {stream}"
        );
    }

    let later = GatedScheduler {
        fail_late: Some(1),
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(later, NO_DEADLINE).await;
    let response = complete(&pipeline, &components, batch(false), None).await;
    assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(said(&heard), lease("completed ServerError 500"));

    let scheduler = GatedScheduler {
        fail_first: true,
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;
    let response = complete(&pipeline, &components, batch(false), None).await;
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(said(&heard), lease("completed ServerError 503"));
}

/// A failed call's answer bears the end its dispatch owes Grand Central
/// from the dispatch to the stage's plan, which takes it off and tells it.
/// The answer that leaves the stage bears none: no later reader of the
/// answer can tell that end again.
#[tokio::test]
async fn the_answer_that_leaves_the_stage_bears_no_end_owed() {
    let scheduler = GatedScheduler {
        fail_always: true,
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;

    let response = complete(&pipeline, &components, completion_request(false), None).await;

    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(said(&heard), lease("completed ServerError 503"));
    assert!(!bears_untold_end(&response));
}

/// A request the classic policies placed owes Grand Central no end: its
/// failed call marks its answer with none.
#[tokio::test]
async fn a_classic_requests_failed_answer_bears_no_end_owed() {
    let scheduler = GatedScheduler {
        fail_always: true,
        ..Default::default()
    };
    let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    let (pipeline, components, _worker) = placed_by(policies, scheduler, NO_DEADLINE).await;

    let response = complete(&pipeline, &components, completion_request(false), None).await;

    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert!(!bears_untold_end(&response));
}

/// A batch whose later prompt's call fails is told once, by that failure's
/// status. The call opened no stream, so the stage tells it. The stream of
/// the prompt whose call answered is dropped unread, and tells nothing.
#[tokio::test]
async fn the_engine_hears_a_batch_whose_later_call_failed_once_by_its_status() {
    let scheduler = GatedScheduler {
        fail_call: Some(1),
        ..Default::default()
    };
    let (pipeline, heard, components, _worker) = hearing(scheduler, NO_DEADLINE).await;
    let batch: Arc<CompletionRequest> = Arc::new(
        serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "prompt": ["Hello world", "Hello again"],
        }))
        .expect("a batched completion"),
    );

    let response = complete(&pipeline, &components, batch, None).await;

    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(said(&heard), lease("completed ServerError 503"));
    assert!(!bears_untold_end(&response));
}
