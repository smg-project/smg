use metrics_exporter_prometheus::PrometheusBuilder;

use super::*;
use crate::{
    anthropic::worker::probe::{counted_failure, series_line},
    test_support::upstream,
};

/// An upstream error status ends the iteration with the error the client
/// is told and one error counted against the model under `backend_error`
/// and the status's reason, with a duration sample; a success passes
/// uncounted.
#[test]
fn an_upstream_error_status_is_counted_against_the_model() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let refused = upstream_answer(upstream(503, None, "{}"), "loop-refused", Instant::now())
        .expect_err("an error status ends the iteration");
    assert_eq!(
        refused,
        "Worker returned error status: 503 Service Unavailable"
    );
    assert_eq!(
        counted_failure(&handle, "loop-refused"),
        (
            "backend_error".to_string(),
            "engine_unavailable".to_string(),
            1,
            1
        )
    );

    let passed = upstream_answer(upstream(200, None, "{}"), "loop-ok", Instant::now())
        .expect("a success passes");
    assert_eq!(passed.status(), StatusCode::OK);
    assert!(series_line(&handle, "smg_router_request_errors_total", "loop-ok").is_none());
}

/// A stream that broke under the loop is counted by what broke, its
/// error text passing through; a client that went away is passed through
/// uncounted.
#[test]
fn a_broken_stream_is_counted_by_what_broke_and_a_departed_client_is_not() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let dropped = stream_failure(
        sse::StreamFailure::Read("Stream read error: connection reset".to_string()),
        "loop-dropped",
        Instant::now(),
    );
    assert_eq!(dropped, "Stream read error: connection reset");
    assert_eq!(
        counted_failure(&handle, "loop-dropped"),
        (
            "streaming_error".to_string(),
            "transport_disconnect".to_string(),
            1,
            1
        )
    );

    let late = stream_failure(
        sse::StreamFailure::Timeout("Stream read error: operation timed out".to_string()),
        "loop-late",
        Instant::now(),
    );
    assert_eq!(late, "Stream read error: operation timed out");
    assert_eq!(
        counted_failure(&handle, "loop-late"),
        ("timeout".to_string(), "deadline".to_string(), 1, 1)
    );

    let unreadable = stream_failure(
        sse::StreamFailure::BadFrame("Failed to parse SSE data: expected value".to_string()),
        "loop-unreadable",
        Instant::now(),
    );
    assert_eq!(unreadable, "Failed to parse SSE data: expected value");
    assert_eq!(
        counted_failure(&handle, "loop-unreadable"),
        ("parse_error".to_string(), "bad_response".to_string(), 1, 1)
    );

    let gone = stream_failure(sse::StreamFailure::ClientGone, "loop-gone", Instant::now());
    assert_eq!(gone, sse::CLIENT_DISCONNECTED_ERROR);
    assert!(series_line(&handle, "smg_router_request_errors_total", "loop-gone").is_none());
}
