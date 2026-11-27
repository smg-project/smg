use std::net::TcpListener;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use openai_protocol::worker::WorkerType;
use smg_worker::{
    health::circuit_breaker::{CircuitBreakerConfig, CircuitState},
    identity::basic::ConnectionMode,
    test_support::{no_health_check, RefusedAddress},
};

use super::*;
use crate::routers::grpc::{
    backend_client::BasicWorkerBuilder, proto_wrapper::ENGINE_ERROR_MESSAGE,
};

fn dispatch() -> DispatchMetadata {
    DispatchMetadata {
        request_id: "chatcmpl-cut".to_string(),
        model: "stream-failure-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

/// A worker whose breaker opens on the first failure.
fn worker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("grpc://engine-under-test:8000")
            .connection_mode(ConnectionMode::Grpc)
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            })
            .health_config(no_health_check())
            .build(),
    )
}

fn cut() -> StreamFailure {
    StreamFailure::upstream(
        tonic::Status::internal("h2 protocol error: error reading a body from connection"),
        3_191,
    )
}

/// The rendered sample line of `family` with all of `labels`, if any.
fn sample<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
    rendered.lines().find(|line| {
        line.starts_with(&format!("{family}{{")) && labels.iter().all(|label| line.contains(label))
    })
}

/// `status` ends a chat stream read from a fresh worker whose breaker
/// opens on the first failure: the worker, and what was counted.
fn account_on_a_fresh_worker(status: tonic::Status) -> (Arc<dyn Worker>, PrometheusHandle) {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let worker = worker();
    let selection = WorkerSelection::Single {
        worker: Arc::clone(&worker).into(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    metrics::with_local_recorder(&recorder, || {
        source.account(&StreamFailure::upstream(status, 7));
    });
    (worker, handle)
}

/// The `error_type` of the one `smg_worker_errors_total` sample of the
/// regular worker, when the failure was counted against it.
fn worker_error_type(rendered: &str) -> Option<&str> {
    let line = sample(
        rendered,
        "smg_worker_errors_total",
        &["worker_type=\"regular\"", "connection_mode=\"grpc\""],
    )?;
    assert!(line.ends_with(" 1"), "{line}");
    Some(label(line, "error_type"))
}

/// The `error_type` of the one `smg_router_request_errors_total` sample
/// of the chat endpoint.
fn router_error_type(rendered: &str) -> &str {
    let line = sample(
        rendered,
        "smg_router_request_errors_total",
        &[
            "router_type=\"grpc\"",
            "backend_type=\"regular\"",
            "model=\"stream-failure-test\"",
            "endpoint=\"chat\"",
        ],
    )
    .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
    assert!(line.ends_with(" 1"), "{line}");
    label(line, "error_type")
}

/// The `reason` of the one `smg_router_request_errors_total` sample of
/// the chat endpoint.
fn router_error_reason(rendered: &str) -> &str {
    let line = sample(
        rendered,
        "smg_router_request_errors_total",
        &["router_type=\"grpc\"", "endpoint=\"chat\""],
    )
    .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
    label(line, "reason")
}

/// The value of the label `name` on a rendered sample line.
fn label<'a>(line: &'a str, name: &str) -> &'a str {
    let key = format!("{name}=\"");
    let start = line
        .find(&key)
        .unwrap_or_else(|| panic!("{name} on {line}"))
        + key.len();
    let end = start + line[start..].find('"').expect("a closed label");
    &line[start..end]
}

#[test]
fn an_upstream_failure_is_a_stream_error_with_the_message_clients_know() {
    let failure = cut();
    assert_eq!(failure.error_type(), "stream_error");
    assert_eq!(failure.messages_error_type(), "api_error");
    assert_eq!(
        failure.to_string(),
        "Stream error: h2 protocol error: error reading a body from connection"
    );
    let failure = StreamFailure::from("Channel closed".to_string());
    assert_eq!(failure.error_type(), "internal_error");
    assert_eq!(failure.messages_error_type(), "api_error");
    assert_eq!(failure.to_string(), "Channel closed");
}

#[test]
fn a_status_after_the_first_chunk_is_classed_as_a_failed_dispatch_of_it() {
    let class = |status: tonic::Status| UpstreamClass::of(&status);
    for status in [
        tonic::Status::invalid_argument("the grammar does not compile"),
        tonic::Status::failed_precondition("bad"),
        tonic::Status::out_of_range("bad"),
    ] {
        let class = class(status);
        assert_eq!(
            (class.http, class.error_type, class.breaker_sample),
            (StatusCode::BAD_REQUEST, "validation_error", Some(400))
        );
        assert!(!class.is_worker_failure());
        assert_eq!(class.client_type("stream_error"), "invalid_request_error");
    }
    let timeout = class(tonic::Status::deadline_exceeded("too slow"));
    assert_eq!(
        (timeout.http, timeout.error_type, timeout.breaker_sample),
        (StatusCode::GATEWAY_TIMEOUT, "timeout", Some(504))
    );
    assert!(timeout.is_worker_failure());
    for (status, http) in [
        (
            tonic::Status::unavailable("gone"),
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            tonic::Status::internal("broke"),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            tonic::Status::unknown("broke"),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
    ] {
        let class = class(status);
        assert_eq!(
            (class.http, class.error_type, class.breaker_sample),
            (http, "backend_error", Some(http.as_u16()))
        );
        assert!(class.is_worker_failure());
        assert_eq!(class.client_type("stream_error"), "stream_error");
    }
    // Capacity pushback is not the worker's failure (and the breaker
    // takes no sample of a 429 either way); it counts as the rate-limit
    // error the client reads, not as an internal error of the router.
    let capacity = class(tonic::Status::resource_exhausted("full"));
    assert_eq!(
        (capacity.http, capacity.error_type, capacity.breaker_sample),
        (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", Some(429))
    );
    assert!(!capacity.is_worker_failure());
    assert_eq!(capacity.client_type("stream_error"), "rate_limit_error");
}

#[test]
fn a_cancelled_stream_is_a_cut_by_the_engines_server() {
    let status = tonic::Status::cancelled("stream reset");
    let class = UpstreamClass::of(&status);
    assert_eq!(
        (class.http, class.error_type, class.breaker_sample),
        (StatusCode::BAD_GATEWAY, "backend_error", Some(502))
    );
    assert_eq!(
        StreamFailure::upstream(status.clone(), 7).error_type(),
        "stream_error"
    );
    let (worker, handle) = account_on_a_fresh_worker(status);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    let rendered = handle.render();
    assert_eq!(worker_error_type(&rendered), Some("backend_error"));
    assert_eq!(router_error_type(&rendered), "backend_error");
    assert_eq!(router_error_reason(&rendered), "transport_disconnect");
}

/// A stream that ended with an HTTP/2 fault (the connection to the
/// worker went away under it) is the worker's backend error, a stream
/// error to the client and a breaker sample, whatever code tonic picked
/// from the frame's reason: never the rate-limit class a GOAWAY
/// ENHANCE_YOUR_CALM would read as.
#[test]
fn a_stream_cut_by_an_http2_fault_is_the_workers_backend_error() {
    let status = tonic::Status::from_error(Box::new(smg_grpc_client::h2::Error::from(
        smg_grpc_client::h2::Reason::ENHANCE_YOUR_CALM,
    )));
    assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    let class = UpstreamClass::of(&status);
    assert_eq!(
        (class.http, class.error_type, class.breaker_sample),
        (StatusCode::BAD_GATEWAY, "backend_error", Some(502))
    );
    assert_eq!(class.reason, "transport_disconnect");
    assert!(class.is_worker_failure());
    assert_eq!(class.client_type("stream_error"), "stream_error");
}

#[test]
fn a_failure_of_the_requests_media_takes_no_breaker_sample() {
    // A worker that predates the media-fault trailer reports the media
    // host's failure on UNAVAILABLE, in the connector's words. The client
    // is answered for the host (a 502, so it knows to retry), but the
    // router counts it as the request's error (the reference is the
    // request's input), not the worker's: no breaker sample, no
    // worker-failure class, and the router-error family keeps the
    // request's label.
    let status = tonic::Status::unavailable(
        "media fetch of https://media.example/a.png timed out after 10s",
    );
    let class = UpstreamClass::of(&status);
    assert_eq!(
        (class.http, class.error_type, class.breaker_sample),
        (StatusCode::BAD_GATEWAY, "validation_error", None)
    );
    assert!(!class.is_worker_failure());
    let failure = StreamFailure::upstream(status.clone(), 7);
    assert_eq!(failure.error_type(), "server_error");
    let (worker, handle) = account_on_a_fresh_worker(status);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    let rendered = handle.render();
    assert_eq!(worker_error_type(&rendered), None, "{rendered}");
    assert_eq!(router_error_type(&rendered), "validation_error");
    assert_eq!(router_error_reason(&rendered), "media_fault");
}

#[test]
fn a_request_the_engine_rejects_after_its_first_chunk_is_a_validation_error() {
    let status = tonic::Status::invalid_argument("the grammar does not compile");
    let failure = StreamFailure::upstream(status.clone(), 7);
    assert_eq!(failure.error_type(), "invalid_request_error");
    assert_eq!(failure.messages_error_type(), "invalid_request_error");
    assert_eq!(
        failure.to_string(),
        "Stream error: the grammar does not compile"
    );
    let (worker, handle) = account_on_a_fresh_worker(status);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    let rendered = handle.render();
    assert_eq!(worker_error_type(&rendered), None, "{rendered}");
    assert_eq!(router_error_type(&rendered), "validation_error");
    assert_eq!(router_error_reason(&rendered), "invalid_request");
}

/// The gateway's own request timeout names itself in the event as its
/// 504 does; a deadline the engine reports has no code of its own.
#[test]
fn a_stream_cut_by_the_request_timeout_carries_the_timeouts_code() {
    let bound = smg_grpc_client::RequestDeadline::after(std::time::Duration::from_secs(5)).status();
    assert_eq!(
        StreamFailure::upstream(bound, 0).code(),
        Some("call_upstream_timeout")
    );
    let engines_own = tonic::Status::deadline_exceeded("decode step took too long");
    assert_eq!(StreamFailure::upstream(engines_own, 7).code(), None);
}

#[test]
fn a_deadline_the_engine_exceeds_after_its_first_chunk_is_a_timeout() {
    let status = tonic::Status::deadline_exceeded("decode step took too long");
    assert_eq!(
        StreamFailure::upstream(status.clone(), 7).error_type(),
        "stream_error"
    );
    let (worker, handle) = account_on_a_fresh_worker(status);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    let rendered = handle.render();
    assert_eq!(worker_error_type(&rendered), Some("timeout"));
    assert_eq!(router_error_type(&rendered), "timeout");
    assert_eq!(router_error_reason(&rendered), "deadline");
}

#[test]
fn an_unavailable_or_internal_status_after_the_first_chunk_is_a_backend_error() {
    for (status, reason) in [
        (
            tonic::Status::unavailable("connection reset"),
            "engine_unavailable",
        ),
        (
            tonic::Status::internal("h2 protocol error: connection reset"),
            "engine_internal",
        ),
    ] {
        assert_eq!(
            StreamFailure::upstream(status.clone(), 7).error_type(),
            "stream_error"
        );
        let (worker, handle) = account_on_a_fresh_worker(status);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
        let rendered = handle.render();
        assert_eq!(worker_error_type(&rendered), Some("backend_error"));
        // One class; the reason tells the worker's own UNAVAILABLE from
        // its INTERNAL.
        assert_eq!(router_error_type(&rendered), "backend_error");
        assert_eq!(router_error_reason(&rendered), reason);
    }
}

#[test]
fn capacity_pushback_after_the_first_chunk_is_not_the_workers_failure() {
    let status = tonic::Status::resource_exhausted("kv cache full");
    assert_eq!(
        StreamFailure::upstream(status.clone(), 7).error_type(),
        "rate_limit_error"
    );
    let (worker, handle) = account_on_a_fresh_worker(status);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    let rendered = handle.render();
    assert_eq!(worker_error_type(&rendered), None);
    // The router counts the rate-limit error the client read, not an
    // internal error.
    assert_eq!(router_error_type(&rendered), "rate_limit_error");
    assert_eq!(router_error_reason(&rendered), "overloaded");
}

#[test]
fn an_upstream_failure_is_charged_to_the_worker_and_counted() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let worker = worker();
    let selection = WorkerSelection::Single {
        worker: Arc::clone(&worker).into(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);

    metrics::with_local_recorder(&recorder, || source.account(&cut()));

    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    let rendered = handle.render();
    let worker_errors = sample(
        &rendered,
        "smg_worker_errors_total",
        &[
            "worker_type=\"regular\"",
            "connection_mode=\"grpc\"",
            "error_type=\"backend_error\"",
        ],
    )
    .unwrap_or_else(|| panic!("a worker error is counted:\n{rendered}"));
    assert!(worker_errors.ends_with(" 1"), "{worker_errors}");
    let router_errors = sample(
        &rendered,
        "smg_router_request_errors_total",
        &[
            "router_type=\"grpc\"",
            "backend_type=\"regular\"",
            "model=\"stream-failure-test\"",
            "endpoint=\"chat\"",
            "error_type=\"backend_error\"",
        ],
    )
    .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
    assert!(router_errors.ends_with(" 1"), "{router_errors}");
}

#[test]
fn a_disaggregated_stream_is_charged_to_its_decode_side() {
    let prefill = worker();
    let decode = worker();
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: Arc::clone(&prefill),
        decode: Arc::clone(&decode).into(),
        runtime_type: Default::default(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    let recorder = PrometheusBuilder::new().build_recorder();
    metrics::with_local_recorder(&recorder, || source.account(&cut()));
    assert_eq!(decode.circuit_breaker_state(), CircuitState::Open);
    assert_eq!(prefill.circuit_breaker_state(), CircuitState::Closed);
    assert!(sample(
        &recorder.handle().render(),
        "smg_worker_errors_total",
        &["worker_type=\"decode\""]
    )
    .is_some());
}

/// The decode side of a disaggregated pair the engine retired at its
/// handshake is counted under the rendezvous family; the prefill side's
/// stream is counted where it is drained, and the same status from a
/// regular worker is not a rendezvous failure.
#[test]
fn a_decode_side_retired_at_its_handshake_is_counted_as_a_rendezvous_failure() {
    let status = tonic::Status::internal(
        "Decode handshake failed for request rank=0 req.rid='r2' req.bootstrap_room=42 with \
             exception KVTransferError(bootstrap_room=42): Request 42 timed out after 120.0s in \
             KVPoll.Bootstrapping",
    );
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: worker(),
        decode: worker().into(),
        runtime_type: Default::default(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
        .with_backend(metrics_labels::BACKEND_PD);
    let recorder = PrometheusBuilder::new().build_recorder();
    metrics::with_local_recorder(&recorder, || {
        source.account(&StreamFailure::upstream(status.clone(), 0));
    });
    let rendered = recorder.handle().render();
    let line = sample(
        &rendered,
        "smg_pd_rendezvous_failures_total",
        &["leg=\"decode\"", "phase=\"bootstrap\""],
    )
    .unwrap_or_else(|| panic!("the handshake failure is counted:\n{rendered}"));
    assert!(line.ends_with(" 1"), "{line}");

    // The prefill side's stream is counted where it is drained, not here.
    let recorder = PrometheusBuilder::new().build_recorder();
    metrics::with_local_recorder(&recorder, || {
        source.account(&StreamFailure::prefill(status.clone()));
    });
    assert!(
        !recorder
            .handle()
            .render()
            .contains("smg_pd_rendezvous_failures_total{"),
        "{}",
        recorder.handle().render()
    );

    let (_worker, handle) = account_on_a_fresh_worker(status);
    assert!(
        !handle
            .render()
            .contains("smg_pd_rendezvous_failures_total{"),
        "{}",
        handle.render()
    );
}

/// A prefill side on the loopback host whose partners are handed `port`,
/// its breaker opening on the first failure like [`worker`]'s.
fn prefill_on(port: u16) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("grpc://127.0.0.1:30000")
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .bootstrap_port(Some(port))
            .health_config(no_health_check())
            .build(),
    )
}

/// The source of a disaggregated chat answer whose prefill side is
/// `prefill`, with the decode side it is read from.
fn pd_source_with(prefill: Arc<dyn Worker>) -> (StreamSource, Arc<dyn Worker>) {
    let decode = worker();
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode: Arc::clone(&decode).into(),
        runtime_type: Default::default(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
        .with_backend(metrics_labels::BACKEND_PD);
    (source, decode)
}

/// The engine's own report, before the first token.
fn reasonless_before_the_first_token() -> StreamFailure {
    StreamFailure::upstream(tonic::Status::internal(ENGINE_ERROR_MESSAGE), 0)
}

/// A decode side the engine failed before its first token, with no reason
/// of its own, behind a prefill side whose bootstrap port nothing listens
/// on: the failure is explained by the dead port, the client's message
/// names it after the engine's report, the class stays the engine's
/// error (the decode worker's breaker and error family, the router
/// family), and the rendezvous family counts it under the request phase
/// on the prefill side.
#[tokio::test]
async fn a_decode_leg_failed_before_its_first_token_is_explained_by_the_dead_bootstrap_port() {
    let refused = RefusedAddress::reserve();
    let (source, decode) = pd_source_with(prefill_on(refused.port()));
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let explained = {
        let _guard = metrics::set_default_local_recorder(&recorder);
        source.explain(&reasonless_before_the_first_token()).await
    }
    .expect("the dead port explains the failure");
    assert!(
        matches!(
            &explained,
            StreamFailure::Upstream {
                stream: UpstreamStream::Answer,
                completion_tokens: 0,
                rendezvous: Some(_),
                ..
            }
        ),
        "{explained:?}"
    );
    let message = explained.to_string();
    assert!(
        message.starts_with(&format!(
            "Stream error: {ENGINE_ERROR_MESSAGE}; the prefill side's bootstrap port did not \
                 accept a connection ("
        )) && message.ends_with("), so the sides could not rendezvous"),
        "{message}"
    );
    assert_eq!(explained.error_type(), "stream_error");
    assert_eq!(explained.messages_error_type(), "api_error");

    metrics::with_local_recorder(&recorder, || source.account(&explained));
    assert_eq!(decode.circuit_breaker_state(), CircuitState::Open);
    let rendered = handle.render();
    assert!(
        sample(
            &rendered,
            "smg_pd_rendezvous_failures_total",
            &["leg=\"prefill\"", "phase=\"request\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        sample(
            &rendered,
            "smg_worker_errors_total",
            &["worker_type=\"decode\"", "error_type=\"engine_error\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        sample(
            &rendered,
            "smg_router_request_errors_total",
            &["backend_type=\"pd\"", "error_type=\"engine_error\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{rendered}"
    );
}

/// Nothing explains a failure that is not a reasonless report of the
/// decode side before its first token: a port that accepts, a side past
/// its first token, the prefill side's own failure, a status with the
/// engine's words, a single worker. Each keeps its message, and the
/// rendezvous family counts no request phase.
#[tokio::test]
async fn any_other_failure_is_left_as_it_is() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let live = listener.local_addr().expect("address").port();
    let (live_source, _) = pd_source_with(prefill_on(live));
    let refused = RefusedAddress::reserve();
    let (dead_source, _) = pd_source_with(prefill_on(refused.port()));
    let single = WorkerSelection::Single {
        worker: worker().into(),
    };
    let single_source =
        StreamSource::new(Some(&single), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let explained = {
        let _guard = metrics::set_default_local_recorder(&recorder);
        [
            live_source
                .explain(&reasonless_before_the_first_token())
                .await,
            dead_source
                .explain(&StreamFailure::upstream(
                    tonic::Status::internal(ENGINE_ERROR_MESSAGE),
                    3,
                ))
                .await,
            dead_source
                .explain(&StreamFailure::prefill(tonic::Status::internal(
                    ENGINE_ERROR_MESSAGE,
                )))
                .await,
            dead_source
                .explain(&StreamFailure::upstream(
                    tonic::Status::internal(
                        "Decode handshake failed for request rank=0 req.rid='r2' \
                             req.bootstrap_room=42",
                    ),
                    0,
                ))
                .await,
            dead_source.explain(&cut()).await,
            single_source
                .explain(&reasonless_before_the_first_token())
                .await,
        ]
    };
    assert!(explained.iter().all(Option::is_none), "{explained:?}");
    assert!(
        !handle.render().contains("phase=\"request\""),
        "{}",
        handle.render()
    );
}

#[test]
fn the_engines_own_failure_is_an_engine_error_charged_to_the_worker() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let worker = worker();
    let selection = WorkerSelection::Single {
        worker: Arc::clone(&worker).into(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    let failure = StreamFailure::upstream(tonic::Status::internal(ENGINE_ERROR_MESSAGE), 3);
    assert_eq!(failure.error_type(), "stream_error");
    assert_eq!(
        failure.to_string(),
        format!("Stream error: {ENGINE_ERROR_MESSAGE}")
    );
    metrics::with_local_recorder(&recorder, || source.account(&failure));
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    let rendered = recorder.handle().render();
    assert!(
        sample(
            &rendered,
            "smg_worker_errors_total",
            &["worker_type=\"regular\"", "error_type=\"engine_error\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        sample(
            &rendered,
            "smg_router_request_errors_total",
            &["endpoint=\"chat\"", "error_type=\"engine_error\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        !rendered.contains("error_type=\"backend_error\""),
        "the engine's own failure is not a backend error:\n{rendered}"
    );
}

#[test]
fn a_broken_connection_stays_a_backend_error() {
    let error_type = |status: tonic::Status| UpstreamClass::of(&status).error_type;
    assert_eq!(
        error_type(tonic::Status::internal(
            "h2 protocol error: error reading a body from connection"
        )),
        metrics_labels::ERROR_BACKEND
    );
    assert_eq!(
        error_type(tonic::Status::unavailable(ENGINE_ERROR_MESSAGE)),
        metrics_labels::ERROR_BACKEND,
        "only the engine's own Internal report is an engine error"
    );
    assert_eq!(
        error_type(tonic::Status::internal(ENGINE_ERROR_MESSAGE)),
        metrics_labels::ERROR_ENGINE
    );
    assert_eq!(
        error_type(tonic::Status::invalid_argument("bad")),
        metrics_labels::ERROR_VALIDATION,
        "a request's own fault keeps the dispatch path's label"
    );
}

#[test]
fn a_prefill_failure_is_charged_to_the_prefill_side() {
    let prefill = worker();
    let decode = worker();
    let selection = WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill: Arc::clone(&prefill),
        decode: Arc::clone(&decode).into(),
        runtime_type: Default::default(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    let failure = StreamFailure::prefill(tonic::Status::unavailable("side died"));
    assert_eq!(failure.error_type(), "stream_error");
    assert_eq!(failure.to_string(), "Prefill stream error: side died");
    let recorder = PrometheusBuilder::new().build_recorder();
    metrics::with_local_recorder(&recorder, || source.account(&failure));
    assert_eq!(prefill.circuit_breaker_state(), CircuitState::Open);
    assert_eq!(decode.circuit_breaker_state(), CircuitState::Closed);
    let rendered = recorder.handle().render();
    assert!(
        sample(
            &rendered,
            "smg_worker_errors_total",
            &["worker_type=\"prefill\"", "error_type=\"backend_error\""]
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        sample(
            &rendered,
            "smg_worker_errors_total",
            &["worker_type=\"decode\""]
        )
        .is_none(),
        "the decode side is not blamed:\n{rendered}"
    );
}

#[test]
fn a_failure_of_the_gateway_leaves_the_worker_alone() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let worker = worker();
    let selection = WorkerSelection::Single {
        worker: Arc::clone(&worker).into(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    metrics::with_local_recorder(&recorder, || {
        source.account(&StreamFailure::from("Channel closed".to_string()));
    });
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    let rendered = recorder.handle().render();
    assert!(!rendered.contains("smg_worker_errors_total{"), "{rendered}");
    assert!(
        !rendered.contains("smg_router_request_errors_total{"),
        "{rendered}"
    );
}

/// One classifier serves both sides of the first chunk: a cancelled
/// dispatch (the worker's side ending the call, its connection closed
/// under the side's start) is the same cut by the engine's server as a
/// cancelled stream, a bad gateway the worker is charged for, never the
/// request's fault.
#[test]
fn a_dispatch_status_is_classed_as_the_same_status_after_the_first_chunk() {
    let cancelled = UpstreamClass::of(&tonic::Status::cancelled("operation was canceled"));
    assert_eq!(
        (
            cancelled.http,
            cancelled.error_type,
            cancelled.breaker_sample
        ),
        (StatusCode::BAD_GATEWAY, "backend_error", Some(502))
    );
    assert!(cancelled.is_worker_failure());
}

#[test]
fn a_failed_dispatch_counts_against_the_worker_only_when_it_is_the_workers_failure() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        for status in [
            tonic::Status::unavailable("gone"),
            tonic::Status::internal("broke"),
            tonic::Status::deadline_exceeded("too slow"),
            tonic::Status::internal(ENGINE_ERROR_MESSAGE),
            tonic::Status::resource_exhausted("full"),
            tonic::Status::invalid_argument("bad"),
            tonic::Status::not_found("no such model"),
            tonic::Status::cancelled("gone away"),
            // The request's own media failed at the worker (a 400 to the
            // client, nothing for the breaker): not the worker's failure.
            tonic::Status::unavailable("HTTP error while fetching media: 503 from the origin"),
        ] {
            UpstreamClass::account_dispatch_failure(
                metrics_labels::WORKER_DECODE,
                metrics_labels::CONNECTION_GRPC,
                &status,
            );
        }
    });
    let rendered = handle.render();
    let count = |error_type: &str| {
        sample(
            &rendered,
            "smg_worker_errors_total",
            &[
                "worker_type=\"decode\"",
                "connection_mode=\"grpc\"",
                &format!("error_type=\"{error_type}\""),
            ],
        )
        .map(|line| line.rsplit(' ').next().expect("a value").to_owned())
    };
    // The worker's failures, each under its class (the cancelled call
    // among the backend errors)...
    assert_eq!(count("backend_error").as_deref(), Some("3"), "{rendered}");
    assert_eq!(count("timeout").as_deref(), Some("1"), "{rendered}");
    assert_eq!(count("engine_error").as_deref(), Some("1"), "{rendered}");
    // ...and nothing for the request's fault or capacity pushback.
    for error_type in [
        "rate_limit_error",
        "validation_error",
        "no_workers",
        "internal_error",
    ] {
        assert_eq!(count(error_type), None, "{rendered}");
    }
}

#[test]
fn a_collected_stream_failure_counts_against_the_worker_it_came_from() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let worker = worker();
    metrics::with_local_recorder(&recorder, || {
        UpstreamClass::account_collected_failure(
            worker.as_ref(),
            &tonic::Status::internal(ENGINE_ERROR_MESSAGE),
        );
        UpstreamClass::account_collected_failure(
            worker.as_ref(),
            &tonic::Status::resource_exhausted("full"),
        );
    });
    let rendered = handle.render();
    // Under the worker's own labels (a gRPC worker): the engine's report
    // only, capacity pushback is not its failure.
    let line = sample(
        &rendered,
        "smg_worker_errors_total",
        &["worker_type=\"regular\"", "connection_mode=\"grpc\""],
    )
    .unwrap_or_else(|| panic!("counted under the worker's labels:\n{rendered}"));
    assert_eq!(label(line, "error_type"), "engine_error");
    assert!(line.ends_with(" 1"), "{line}");
}

/// A reading that panicked is the gateway's own failure: an
/// `internal_error` event under `reader_panicked`, the worker uncharged,
/// the output counted under the table that read it.
#[test]
fn a_reading_that_panicked_is_an_internal_error_under_its_code_and_charges_no_worker() {
    let panicked: Result<(), _> =
        super::super::isolation::isolated_sync(|| panic!("the reader slipped"));
    let failure = StreamFailure::from(panicked.expect_err("the panic").read_by(Some("qwen3")));
    assert_eq!(failure.error_type(), INTERNAL_ERROR_TYPE);
    assert_eq!(failure.messages_error_type(), MESSAGES_API_ERROR_TYPE);
    assert_eq!(failure.code(), Some(READER_PANICKED_CODE));
    // The client's text names the failure; the message is the log line's.
    assert_eq!(
        failure.to_string(),
        "the gateway's reader of the response panicked"
    );
    assert!(failure.class().is_none());
    // Accounted: the table's outcome, nothing of the worker's record.
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let worker = worker();
    let selection = WorkerSelection::Single {
        worker: Arc::clone(&worker).into(),
    };
    let source = StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
    metrics::with_local_recorder(&recorder, || source.account(&failure));
    let rendered = handle.render();
    let outputs = sample(
        &rendered,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"panicked\""],
    )
    .unwrap_or_else(|| panic!("the table's outcome:\n{rendered}"));
    assert!(outputs.ends_with(" 1"), "{outputs}");
    assert!(worker_error_type(&rendered).is_none(), "{rendered}");
    assert!(
        sample(&rendered, "smg_router_request_errors_total", &[]).is_none(),
        "{rendered}"
    );
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}
