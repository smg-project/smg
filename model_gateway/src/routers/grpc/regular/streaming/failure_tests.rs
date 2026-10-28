//! A worker stream that fails after the response started reaches the
//! accounting against the side it came from: the engine's own failure on the
//! decode side as an `engine_error`, a prefill side lost before its completion
//! as a `backend_error` of the prefill worker; and a decode side the engine
//! failed before its first token, with no reason of its own, is read against
//! the pair's bootstrap port before the client is told.
use std::{net::TcpListener, time::Instant};

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
        utils::{stream_failure::UpstreamStream, StreamFailure, StreamSource},
    },
    test_support::RefusedAddress,
    worker::{
        test_support::no_health_check, BasicWorkerBuilder, Worker, WorkerRegistry, WorkerType,
    },
};

const REQUEST_ID: &str = "chatcmpl-failure";
const PREFILL_URL: &str = "grpc://prefill-1:30000";
const DECODE_URL: &str = "grpc://decode-1:30000";
/// A prefill side on the loopback host, whose bootstrap port a test can hold
/// or leave closed.
const LOOPBACK_PREFILL_URL: &str = "grpc://127.0.0.1:30000";

fn processor() -> StreamingProcessor {
    StreamingProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::with_flags(Arc::new(WorkerRegistry::new()), None, None),
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
            .health_config(no_health_check())
            .build(),
    )
}

/// The accounting source of a disaggregated dispatch to the two workers.
fn pd_source() -> StreamSource {
    pd_source_with(worker(PREFILL_URL))
}

/// The accounting source of a disaggregated dispatch whose prefill side is
/// `prefill`.
fn pd_source_with(prefill: Arc<dyn Worker>) -> StreamSource {
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode: worker(DECODE_URL),
        runtime_type: Default::default(),
    };
    StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
        .with_backend(metrics_labels::BACKEND_PD)
}

/// The prefill side on the loopback host, registered with `bootstrap_port`.
fn loopback_prefill(bootstrap_port: u16) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(LOOPBACK_PREFILL_URL)
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(bootstrap_port))
            .health_config(no_health_check())
            .build(),
    )
}

/// The decode side of a disaggregated chat answer, ended by the engine's own
/// report before its first token, as the streaming layer reports it.
fn decode_leg_failed_before_its_first_token(rt: &tokio::runtime::Runtime) -> StreamFailure {
    let failure = rt.block_on(stream_pd(
        vec![complete(0, "stop")],
        "0",
        vec![complete(0, "error")],
    ));
    assert!(
        matches!(
            &failure,
            StreamFailure::Upstream {
                stream: UpstreamStream::Answer,
                completion_tokens: 0,
                ..
            }
        ),
        "{failure:?}"
    );
    failure
}

/// The error event the client reads when `failure` from `source` ends the
/// response, as its JSON, and what was counted meanwhile.
fn client_error_event(
    rt: &tokio::runtime::Runtime,
    failure: &StreamFailure,
    source: &StreamSource,
) -> (Value, String) {
    let (tx, mut rx) = sse_channel();
    let rendered =
        render_with_recorder(|| rt.block_on(utils::send_stream_failure(&tx, failure, source)));
    drop(tx);
    let frame = rt
        .block_on(rx.recv())
        .expect("the error event")
        .expect("a frame");
    let frame = String::from_utf8(frame.to_vec()).expect("utf-8");
    let event = serde_json::from_str(frame.trim_start_matches("data: ").trim())
        .unwrap_or_else(|e| panic!("{e}: {frame}"));
    (event, rendered)
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

/// Stream a disaggregated chat answer whose sides are `prefill` and `decode`
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
fn a_decode_side_the_engine_fails_is_an_engine_error_of_the_decode_worker() {
    let failure = runtime().block_on(stream_pd(
        vec![complete(0, "stop")],
        "0",
        vec![chunk(0, "a"), complete(0, "error")],
    ));
    assert!(
        matches!(
            &failure,
            StreamFailure::Upstream {
                stream: UpstreamStream::Answer,
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
                r#"reason="engine_error""#,
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
        "the decode side carries the failure:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="prefill""#]
        )
        .is_none(),
        "the prefill side, which completed, is not blamed:\n{rendered}"
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

/// A decode side the engine fails before its first token, with no reason of
/// its own, behind a prefill side whose bootstrap port nothing listens on: the
/// client's error event keeps its type and names the dead rendezvous after
/// the engine's report, the WARN names the port, the failure stays the
/// decode worker's engine error, and the rendezvous family counts it under
/// the request phase on the prefill side.
#[test]
#[traced_test]
fn a_decode_leg_failed_before_its_first_token_names_the_pairs_dead_bootstrap_port() {
    let refused = RefusedAddress::reserve();
    let port = refused.port();
    let source = pd_source_with(loopback_prefill(port));
    let rt = runtime();
    let failure = decode_leg_failed_before_its_first_token(&rt);
    let (event, rendered) = client_error_event(&rt, &failure, &source);
    assert_eq!(event["error"]["type"], "stream_error", "{event}");
    let message = event["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.starts_with(&format!(
            "Stream error: {ENGINE_ERROR_MESSAGE}; the prefill side's bootstrap port did not \
             accept a connection ("
        )) && message.ends_with("), so the sides could not rendezvous"),
        "{message}"
    );
    assert!(
        !message.contains("127.0.0.1"),
        "the port's address is the log line's, not the client's: {message}"
    );
    assert!(
        series(
            &rendered,
            "smg_pd_rendezvous_failures_total",
            &[r#"leg="prefill""#, r#"phase="request""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the dead port is counted under the request phase:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_router_request_errors_total",
            &[r#"backend_type="pd""#, r#"error_type="engine_error""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the request's outcome stays one engine_error:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="decode""#, r#"error_type="engine_error""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the decode side carries the failure:\n{rendered}"
    );
    let port_field = format!("bootstrap_port={port}");
    logs_assert(|lines: &[&str]| {
        the_warn(
            lines,
            &[
                "WARN",
                REQUEST_ID,
                DECODE_URL,
                LOOPBACK_PREFILL_URL,
                r#"worker_type="decode""#,
                r#"error_type="engine_error""#,
                r#"bootstrap_host="127.0.0.1""#,
                port_field.as_str(),
                r#"bootstrap_port_source="registered""#,
                "probe_error=",
                "could not rendezvous",
            ],
        )
    });
}

/// The same failure behind a bootstrap port that accepts a connection is
/// told as before: the engine's report alone, nothing under the request
/// phase.
#[test]
fn a_decode_leg_failed_behind_a_live_bootstrap_port_keeps_the_engines_report() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("address").port();
    let source = pd_source_with(loopback_prefill(port));
    let rt = runtime();
    let failure = decode_leg_failed_before_its_first_token(&rt);
    let (event, rendered) = client_error_event(&rt, &failure, &source);
    assert_eq!(event["error"]["type"], "stream_error", "{event}");
    assert_eq!(
        event["error"]["message"],
        format!("Stream error: {ENGINE_ERROR_MESSAGE}"),
        "{event}"
    );
    assert!(
        !rendered.contains(r#"phase="request""#),
        "nothing is counted under the request phase:\n{rendered}"
    );
}

#[test]
#[traced_test]
fn a_prefill_side_lost_before_its_completion_is_a_backend_error_of_the_prefill_worker() {
    // The prefill side's RPC ends with UNAVAILABLE before its Complete.
    let failure = runtime().block_on(stream_pd(
        vec![],
        "14",
        vec![chunk(0, "a"), complete(0, "stop")],
    ));
    assert!(
        matches!(
            &failure,
            StreamFailure::Upstream {
                stream: UpstreamStream::Prefill,
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
        "the prefill side carries the failure as a backend error:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="decode""#]
        )
        .is_none(),
        "the decode side is not blamed:\n{rendered}"
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
