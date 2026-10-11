//! A worker stream that fails after the response started reaches the
//! accounting against the leg it came from: the engine's own failure on the
//! decode leg as an `engine_error`, a prefill leg lost before its completion
//! as a `backend_error` of the prefill worker.
use std::time::Instant;

use openai_protocol::worker::HealthCheckConfig;
use smg_grpc_client::vllm_engine::proto;
use tracing_test::traced_test;

use super::{
    eof_tests::{chat_spec, chunk, complete, scripted_stream, CharacterTokenizer},
    *,
};
use crate::{
    observability::metrics::test_support::render_with_recorder,
    routers::grpc::{
        context::WorkerSelection,
        proto_wrapper::ENGINE_ERROR_MESSAGE,
        utils::{stream_failure::UpstreamLeg, StreamFailure, StreamSource},
    },
    worker::{BasicWorkerBuilder, Worker, WorkerRegistry},
};

const REQUEST_ID: &str = "chatcmpl-failure";
const PREFILL_URL: &str = "grpc://prefill-1:30000";
const DECODE_URL: &str = "grpc://decode-1:30000";

fn processor() -> StreamingProcessor {
    StreamingProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::new(Arc::new(WorkerRegistry::new()), None, None),
        metrics_labels::BACKEND_PD,
    )
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: REQUEST_ID.to_string(),
        model: "failure-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

fn worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// The accounting source of a disaggregated dispatch to the two workers.
fn pd_source() -> StreamSource {
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: worker(PREFILL_URL),
        decode: worker(DECODE_URL),
        runtime_type: Default::default(),
    };
    StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
        .with_backend(metrics_labels::BACKEND_PD)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a current-thread runtime")
}

/// The rendered line of `family` whose labels contain every `label`.
fn series<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
    rendered.lines().find(|line| {
        line.starts_with(&format!("{family}{{")) && labels.iter().all(|label| line.contains(label))
    })
}

/// Stream a disaggregated chat answer whose legs are `prefill` and `decode`
/// and return the failure the streaming layer reports.
async fn stream_pd(
    prefill: Vec<proto::GenerateResponse>,
    prefill_status: &'static str,
    decode: Vec<proto::GenerateResponse>,
) -> StreamFailure {
    let (prefill, prefill_server) = scripted_stream(prefill, prefill_status).await;
    let (decode, decode_server) = scripted_stream(decode, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor()
        .process_prefill_decode_streaming_chunks(
            prefill,
            decode,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            chat_spec(false),
            &tx,
            context::PdTiming {
                prefill_start: Instant::now(),
                decode_sent: Instant::now(),
                runtime: "test",
            },
            vec![None],
            None,
        )
        .await;
    drop(tx);
    drop(rx);
    prefill_server.abort();
    decode_server.abort();
    result.expect_err("the stream failed")
}

fn the_warn(lines: &[&str], needles: &[&str]) -> Result<(), String> {
    let line = lines
        .iter()
        .find(|line| line.contains("Stream ended by the engine before completion"))
        .ok_or_else(|| "no WARN for the failed stream".to_string())?;
    for needle in needles {
        if !line.contains(needle) {
            return Err(format!("{needle:?} missing from the WARN: {line}"));
        }
    }
    Ok(())
}

#[test]
#[traced_test]
fn a_decode_leg_the_engine_fails_is_an_engine_error_of_the_decode_worker() {
    let failure = runtime().block_on(stream_pd(
        vec![complete(0, "stop")],
        "0",
        vec![chunk(0, "a"), complete(0, "error")],
    ));
    assert!(
        matches!(
            &failure,
            StreamFailure::Upstream {
                leg: UpstreamLeg::Answer,
                completion_tokens: 1,
                ..
            }
        ),
        "{failure:?}"
    );
    assert_eq!(
        failure.to_string(),
        format!("Stream error: {ENGINE_ERROR_MESSAGE}"),
        "the client's error event carries the engine's failure, as before"
    );
    let rendered = render_with_recorder(|| pd_source().account(&failure));
    assert!(
        series(
            &rendered,
            "smg_router_request_errors_total",
            &[
                r#"backend_type="pd""#,
                r#"endpoint="chat""#,
                r#"error_type="engine_error""#,
                r#"model="failure-test""#,
            ],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the request's outcome is one engine_error:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="decode""#, r#"error_type="engine_error""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the decode leg carries the failure:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="prefill""#]
        )
        .is_none(),
        "the prefill leg, which completed, is not blamed:\n{rendered}"
    );
    logs_assert(|lines: &[&str]| {
        the_warn(
            lines,
            &[
                "WARN",
                REQUEST_ID,
                DECODE_URL,
                r#"worker_type="decode""#,
                r#"error_type="engine_error""#,
                "completion_tokens=1",
                ENGINE_ERROR_MESSAGE,
            ],
        )
    });
}

#[test]
#[traced_test]
fn a_prefill_leg_lost_before_its_completion_is_a_backend_error_of_the_prefill_worker() {
    // The prefill leg's RPC ends with UNAVAILABLE before its Complete.
    let failure = runtime().block_on(stream_pd(
        vec![],
        "14",
        vec![chunk(0, "a"), complete(0, "stop")],
    ));
    assert!(
        matches!(
            &failure,
            StreamFailure::Upstream {
                leg: UpstreamLeg::Prefill,
                completion_tokens: 0,
                ..
            }
        ),
        "{failure:?}"
    );
    assert!(
        failure.to_string().starts_with("Prefill stream error: "),
        "{failure}"
    );
    let rendered = render_with_recorder(|| pd_source().account(&failure));
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="prefill""#, r#"error_type="backend_error""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the prefill leg carries the failure as a backend error:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="decode""#]
        )
        .is_none(),
        "the decode leg is not blamed:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_router_request_errors_total",
            &[r#"backend_type="pd""#, r#"error_type="backend_error""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the request's outcome is one backend_error:\n{rendered}"
    );
    logs_assert(|lines: &[&str]| {
        the_warn(
            lines,
            &[
                "WARN",
                REQUEST_ID,
                PREFILL_URL,
                r#"worker_type="prefill""#,
                r#"error_type="backend_error""#,
            ],
        )
    });
}
