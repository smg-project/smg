use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::messages::MessageStreamEvent;
use serde_json::Value;
use smg_worker::{
    health::circuit_breaker::CircuitBreakerConfig,
    test_support::{no_health_check, RefusedAddress},
};

use super::*;
use crate::{routers::grpc::backend_client::BasicWorkerBuilder, server::http_client_builder};

const CAUSE: &str = "error decoding response body: error reading a body from connection: \
                         connection reset";

/// A worker whose breaker opens on the first failure.
fn worker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("http://engine-under-test:8000")
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            })
            .health_config(no_health_check())
            .build(),
    )
}

fn cut(status: StatusCode) -> UpstreamCut {
    UpstreamCut {
        status,
        cause: CAUSE.to_string(),
        code: None,
    }
}

/// A call the client's own timeout ends: the listener's backlog takes
/// the connect and nobody answers the request.
async fn timed_out_call() -> reqwest::Error {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    http_client_builder()
        .no_proxy()
        .timeout(std::time::Duration::from_millis(50))
        .build()
        .unwrap()
        .get(format!("http://{addr}/v1/chat/completions"))
        .send()
        .await
        .unwrap_err()
}

/// The source of a response to `route`, read from a fresh worker.
fn source(route: &'static str) -> (RelaySource, Arc<dyn Worker>) {
    let worker = worker();
    (
        RelaySource::new("relay-test", route, Arc::clone(&worker)),
        worker,
    )
}

/// `cut` ends a chat stream read from a fresh worker: the worker, and the
/// metrics as rendered.
fn account(cut: &UpstreamCut) -> (Arc<dyn Worker>, String) {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let (source, worker) = source("/v1/chat/completions");
    metrics::with_local_recorder(&recorder, || source.account(cut, 839));
    (worker, handle.render())
}

/// A call to a worker nobody listens at: the connect is refused, and the
/// error carries the worker's URL as reqwest attaches it.
async fn refused_call() -> (std::net::SocketAddr, reqwest::Error) {
    let refused = RefusedAddress::reserve();
    let addr = refused.addr();
    let error = http_client_builder()
        .no_proxy()
        .build()
        .unwrap()
        .get(format!("http://{addr}/v1/chat/completions"))
        .send()
        .await
        .unwrap_err();
    assert_eq!(error.url().map(|url| url.port()), Some(Some(addr.port())));
    (addr, error)
}

/// The cause a client is told is the transport's reason and its chain, and
/// never the worker's address reqwest puts in the outer error: the same
/// words head a side's 502 and the error event that ends a cut stream.
#[tokio::test]
async fn the_cause_of_a_failed_call_names_the_reason_and_not_the_worker() {
    let (addr, error) = refused_call().await;

    let text = cause(&error);
    assert!(text.starts_with("error sending request: "), "{text}");
    assert!(!text.contains(&addr.to_string()), "{text}");
    assert!(!text.contains("for url"), "{text}");
    assert!(text.contains("onnect"), "the reason stays: {text}");

    let (source, _worker) = source("/v1/chat/completions");
    let event = source.error_event(&UpstreamCut::of(&error));
    let event = std::str::from_utf8(&event).unwrap();
    assert!(event.contains(&text), "{event}");
    assert!(!event.contains(&addr.to_string()), "{event}");
}

/// The rendered sample line of `family` with all of `labels`, if any.
fn sample<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
    rendered.lines().find(|line| {
        line.starts_with(&format!("{family}{{")) && labels.iter().all(|label| line.contains(label))
    })
}

#[test]
fn a_cut_after_a_successful_head_is_charged_to_the_worker_as_a_backend_error() {
    let (worker, rendered) = account(&cut(StatusCode::BAD_GATEWAY));

    assert!(
        !worker.circuit_breaker_can_execute(),
        "the worker's threshold-1 breaker samples the cut"
    );
    let worker_errors = sample(
        &rendered,
        "smg_worker_errors_total",
        &[
            "worker_type=\"regular\"",
            "connection_mode=\"http\"",
            "error_type=\"backend_error\"",
        ],
    )
    .unwrap_or_else(|| panic!("a worker error is counted:\n{rendered}"));
    assert!(worker_errors.ends_with(" 1"), "{worker_errors}");
    let router_errors = sample(
        &rendered,
        "smg_router_request_errors_total",
        &[
            "router_type=\"http\"",
            "backend_type=\"regular\"",
            "connection_mode=\"http\"",
            "model=\"relay-test\"",
            "endpoint=\"chat\"",
            "error_type=\"backend_error\"",
            "reason=\"transport_disconnect\"",
        ],
    )
    .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
    assert!(router_errors.ends_with(" 1"), "{router_errors}");
}

#[test]
fn a_timeout_mid_body_counts_as_a_timeout() {
    let (worker, rendered) = account(&cut(StatusCode::GATEWAY_TIMEOUT));

    assert!(!worker.circuit_breaker_can_execute());
    assert!(
        sample(
            &rendered,
            "smg_worker_errors_total",
            &["connection_mode=\"http\"", "error_type=\"timeout\""]
        )
        .is_some(),
        "{rendered}"
    );
    assert!(
        sample(
            &rendered,
            "smg_router_request_errors_total",
            &[
                "router_type=\"http\"",
                "error_type=\"timeout\"",
                "reason=\"deadline\""
            ]
        )
        .is_some(),
        "{rendered}"
    );
}

/// The decode side of a disaggregated pair is charged as the decode
/// worker's failure, under the disaggregated router's labels.
#[test]
fn a_cut_on_the_decode_side_is_counted_under_the_disaggregated_router_s_labels() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let worker = worker();
    let source =
        RelaySource::decode_side("relay-test", "/v1/chat/completions", Arc::clone(&worker));

    metrics::with_local_recorder(&recorder, || {
        source.account(&cut(StatusCode::BAD_GATEWAY), 839);
    });

    let rendered = handle.render();
    assert!(
        !worker.circuit_breaker_can_execute(),
        "the decode worker's threshold-1 breaker samples the cut"
    );
    let worker_errors = sample(
        &rendered,
        "smg_worker_errors_total",
        &[
            "worker_type=\"decode\"",
            "connection_mode=\"http\"",
            "error_type=\"backend_error\"",
        ],
    )
    .unwrap_or_else(|| panic!("a decode worker error is counted:\n{rendered}"));
    assert!(worker_errors.ends_with(" 1"), "{worker_errors}");
    let router_errors = sample(
        &rendered,
        "smg_router_request_errors_total",
        &[
            "router_type=\"http\"",
            "backend_type=\"pd\"",
            "connection_mode=\"http\"",
            "model=\"relay-test\"",
            "endpoint=\"chat\"",
            "error_type=\"backend_error\"",
        ],
    )
    .unwrap_or_else(|| panic!("a disaggregated router error is counted:\n{rendered}"));
    assert!(router_errors.ends_with(" 1"), "{router_errors}");
    assert!(
        sample(
            &rendered,
            "smg_worker_errors_total",
            &["worker_type=\"regular\""]
        )
        .is_none(),
        "nothing is charged to a regular worker:\n{rendered}"
    );
}

#[test]
fn the_event_of_an_openai_style_stream_is_a_stream_error_followed_by_done() {
    let (source, _worker) = source("/v1/chat/completions");

    let event = source.error_event(&cut(StatusCode::BAD_GATEWAY));

    let text = std::str::from_utf8(&event).unwrap();
    let (error, done) = text.split_once("\n\n").unwrap();
    let error: Value = serde_json::from_str(error.strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(error["error"]["type"], STREAM_ERROR_TYPE);
    assert_eq!(error["error"]["message"], format!("Stream error: {CAUSE}"));
    assert_eq!(done, "data: [DONE]\n\n");
}

/// The request's own timeout names itself in the event of an
/// OpenAI-style stream, as its 504 does on a unary answer; a refused
/// connect and a cut of any other kind have no code.
#[tokio::test]
async fn the_event_of_a_stream_the_request_timeout_cut_names_the_timeouts_code() {
    let (source, _worker) = source("/v1/chat/completions");
    let error_of = |cut: &UpstreamCut| -> Value {
        let event = source.error_event(cut);
        let text = std::str::from_utf8(&event).unwrap();
        let (error, done) = text.split_once("\n\n").unwrap();
        assert_eq!(done, "data: [DONE]\n\n");
        serde_json::from_str(error.strip_prefix("data: ").unwrap()).unwrap()
    };

    let timed_out = error_of(&UpstreamCut::of(&timed_out_call().await));
    assert_eq!(timed_out["error"]["type"], STREAM_ERROR_TYPE);
    assert_eq!(timed_out["error"]["code"], "call_upstream_timeout");

    let (_addr, refused) = refused_call().await;
    for other in [UpstreamCut::of(&refused), cut(StatusCode::BAD_GATEWAY)] {
        let error = error_of(&other);
        assert!(error["error"].get("code").is_none(), "{error}");
    }
}

/// The Messages API's error event has no code field: the request
/// timeout ends its stream with the event of any other cut.
#[tokio::test]
async fn the_event_of_a_messages_stream_the_request_timeout_cut_has_no_code() {
    let (source, _worker) = source("/v1/messages");

    let event = source.error_event(&UpstreamCut::of(&timed_out_call().await));

    let text = std::str::from_utf8(&event).unwrap();
    assert!(text.starts_with("event: error\ndata: "), "{text:?}");
    assert!(!text.contains("call_upstream_timeout"), "{text:?}");
}

#[test]
fn the_event_of_a_messages_stream_is_the_api_s_error_event_without_done() {
    let (source, _worker) = source("/v1/messages");

    let event = source.error_event(&cut(StatusCode::BAD_GATEWAY));

    let text = std::str::from_utf8(&event).unwrap();
    let data = text
        .strip_prefix("event: error\ndata: ")
        .and_then(|rest| rest.strip_suffix("\n\n"))
        .unwrap_or_else(|| panic!("one error event: {text:?}"));
    match serde_json::from_str::<MessageStreamEvent>(data).unwrap() {
        MessageStreamEvent::Error { error } => {
            assert_eq!(error.error_type, MESSAGES_API_ERROR_TYPE);
            assert_eq!(error.message, format!("Stream error: {CAUSE}"));
        }
        other => panic!("the protocol's error event, got {other:?}"),
    }
    assert!(!text.contains("[DONE]"), "{text:?}");
}

/// After a partial line the event opens a frame of its own; on a
/// boundary it is written as it is.
#[test]
fn the_event_that_ends_a_cut_opens_a_frame_after_a_partial_line() {
    let (source, _worker) = source("/v1/chat/completions");
    let cut = cut(StatusCode::BAD_GATEWAY);
    let on_boundary = source.error_event_on_its_own_frame(&cut, b"}\n\n");
    assert_eq!(on_boundary, source.error_event(&cut));
    let after_partial = source.error_event_on_its_own_frame(&cut, b"\"con");
    assert!(
        after_partial.starts_with(b"\n\ndata: "),
        "{after_partial:?}"
    );
    assert_eq!(&after_partial[2..], &source.error_event(&cut)[..]);
}

/// The boundary is read from the last bytes sent, across chunks.
#[test]
fn the_frame_boundary_is_read_from_the_last_bytes_sent() {
    let mut tail = SentTail::default();
    assert!(on_frame_boundary(tail.as_slice()), "nothing sent yet");
    note_sent(&mut tail, b"data: {\"choices\":[]}\n\n");
    assert!(on_frame_boundary(tail.as_slice()));
    note_sent(&mut tail, b"data: {\"cho");
    assert!(!on_frame_boundary(tail.as_slice()), "a frame cut mid-line");
    note_sent(&mut tail, b"ices\":[]}\n");
    assert!(
        !on_frame_boundary(tail.as_slice()),
        "one newline is not a blank line"
    );
    note_sent(&mut tail, b"\n");
    assert!(
        on_frame_boundary(tail.as_slice()),
        "the blank line split across two chunks"
    );
    note_sent(&mut tail, b"event: x\r\ndata: 1\r\n\r\n");
    assert!(on_frame_boundary(tail.as_slice()));
    assert_eq!(
        tail.as_slice(),
        b"\r\n\r\n",
        "the last four bytes, oldest first"
    );
    note_sent(&mut tail, b"ab");
    assert_eq!(
        tail.as_slice(),
        b"\r\nab",
        "two new bytes keep two old ones"
    );
    note_sent(&mut tail, b"");
    assert_eq!(tail.as_slice(), b"\r\nab", "an empty chunk changes nothing");
}
