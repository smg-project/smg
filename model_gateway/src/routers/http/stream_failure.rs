//! A relayed streaming response whose upstream body ended with an error
//! before its natural end: what the client is told and what the gateway
//! counts. The HTTP relay's counterpart of the gRPC path's `stream_failure`,
//! shared by the regular relay and the disaggregated router's decode relay.
//!
//! Once the response head is out, a failure cannot change the status code.
//! The relay ends an SSE response the way the gRPC path ends one: a typed
//! error event in the shape of the route's API, then the body's own end, so
//! the transfer closes cleanly and the client reads why the stream stopped
//! instead of a truncated transfer. A streamed body of any other type is cut
//! as before, so a partial document is never taken for a complete one. Either
//! way the failure is counted the way a failed dispatch of the same kind is:
//! the worker's circuit breaker samples the status, the worker-error and
//! router-error families count it under one `error_type`, and one WARN names
//! the worker, the route and the cause.

use std::{error::Error as _, sync::Arc};

use axum::http::StatusCode;
use bytes::Bytes;
use serde_json::json;
use tracing::warn;

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::grpc::utils::{
        error_type_from_status, route_to_endpoint,
        stream_failure::{MESSAGES_API_ERROR_TYPE, STREAM_ERROR_TYPE},
    },
    worker::Worker,
};

/// The route whose streams end with the Messages API's own error event.
const MESSAGES_ROUTE: &str = "/v1/messages";

/// An upstream body error, classed the way a failed dispatch of it is: the
/// client's timeout expiring mid-body is a gateway timeout, as it is when it
/// expires before the head; any other error (the connection reset or closed
/// before the body's end, a transfer the worker left malformed) is a bad
/// gateway. Both are the worker's failure.
pub(crate) struct UpstreamCut {
    status: StatusCode,
    cause: String,
}

impl UpstreamCut {
    /// The class and the cause of `error`.
    pub(crate) fn of(error: &reqwest::Error) -> Self {
        let status = if error.is_timeout() {
            StatusCode::GATEWAY_TIMEOUT
        } else {
            StatusCode::BAD_GATEWAY
        };
        Self {
            status,
            cause: cause(error),
        }
    }
}

/// `error` and its sources, outermost first: a body error reads "error
/// decoding response body" on its own, and keeps the transport's reason (the
/// connection reset, the transfer closed before its end) in the chain.
fn cause(error: &reqwest::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        text.push_str(": ");
        text.push_str(&inner.to_string());
        source = inner.source();
    }
    text
}

/// Where a relayed response comes from, kept by the relay task for the
/// accounting of a failure after the response head went out: the worker,
/// and the `worker_type` and `backend_type` labels its failure is counted
/// under.
pub(crate) struct RelaySource {
    model: String,
    route: &'static str,
    worker: Arc<dyn Worker>,
    worker_type: &'static str,
    backend: &'static str,
}

impl RelaySource {
    /// The response to `route` for `model`, read from `worker` by the regular
    /// relay.
    pub(crate) fn new(model: &str, route: &'static str, worker: Arc<dyn Worker>) -> Self {
        Self::with_labels(
            model,
            route,
            worker,
            metrics_labels::WORKER_REGULAR,
            metrics_labels::BACKEND_REGULAR,
        )
    }

    /// The response to `route` for `model`, read from the decode leg `worker`
    /// of a disaggregated prefill/decode pair: a cut is the decode worker's
    /// failure, counted under the disaggregated router's labels.
    pub(crate) fn decode_leg(model: &str, route: &'static str, worker: Arc<dyn Worker>) -> Self {
        Self::with_labels(
            model,
            route,
            worker,
            metrics_labels::WORKER_DECODE,
            metrics_labels::BACKEND_PD,
        )
    }

    fn with_labels(
        model: &str,
        route: &'static str,
        worker: Arc<dyn Worker>,
        worker_type: &'static str,
        backend: &'static str,
    ) -> Self {
        Self {
            model: model.to_string(),
            route,
            worker,
            worker_type,
            backend,
        }
    }

    /// The worker the response is read from.
    pub(crate) fn worker(&self) -> &dyn Worker {
        self.worker.as_ref()
    }

    /// Log and count `cut`, the upstream body error that ended the response
    /// after `bytes_relayed` bytes of a successful head: the worker's circuit
    /// breaker samples the status of its class, the worker-error and
    /// router-error families count it under the source's labels and the
    /// class's `error_type`, and one WARN names the worker, its type, the
    /// route, how far the response had come and
    /// the cause. The task runs under the request's span, so the line carries
    /// the request's id as every line of the request does.
    pub(crate) fn account(&self, cut: &UpstreamCut, bytes_relayed: u64) {
        let error_type = error_type_from_status(cut.status);
        warn!(
            worker = self.worker.url(),
            worker_type = self.worker_type,
            route = self.route,
            bytes_relayed,
            http_status = cut.status.as_u16(),
            error_type,
            "Stream ended by the worker before completion: {}",
            cut.cause
        );
        self.worker.record_outcome(cut.status.as_u16());
        Metrics::record_worker_error(
            self.worker_type,
            metrics_labels::CONNECTION_HTTP,
            error_type,
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_HTTP,
            self.backend,
            metrics_labels::CONNECTION_HTTP,
            &self.model,
            route_to_endpoint(self.route),
            error_type,
        );
    }

    /// The SSE event that ends a response `cut` by the worker, in the shape
    /// of the route's API. An OpenAI-style stream (Chat Completions,
    /// Completions, generate, Responses) gets the error event the gRPC path
    /// writes for a stream the worker failed, typed `stream_error`, then
    /// `[DONE]`; a Messages stream gets the API's own `error` event, typed
    /// `api_error` as the gRPC path types a worker's failure there, and no
    /// `[DONE]`, which the protocol does not have.
    pub(crate) fn error_event(&self, cut: &UpstreamCut) -> Bytes {
        let message = format!("Stream error: {}", cut.cause);
        if self.route == MESSAGES_ROUTE {
            // The Messages API's error body: `{"type": "error", "error":
            // {"type", "message"}}`, the shape of the protocol's error event.
            let event = json!({
                "type": "error",
                "error": {"type": MESSAGES_API_ERROR_TYPE, "message": message},
            });
            Bytes::from(format!("event: error\ndata: {event}\n\n"))
        } else {
            let event = json!({"error": {"message": message, "type": STREAM_ERROR_TYPE}});
            Bytes::from(format!("data: {event}\n\ndata: [DONE]\n\n"))
        }
    }
}

/// Whether a relayed SSE body stands on a frame boundary after `tail`, the
/// last bytes sent: nothing yet, or a blank line. The error event that ends
/// a cut body is written on its own frame, so a cut that lands inside a
/// frame (a reset mid-write, an intermediary that re-chunks) does not merge
/// the partial line and the event into one line a client cannot parse.
pub(crate) fn on_frame_boundary(tail: &[u8]) -> bool {
    tail.is_empty() || tail.ends_with(b"\n\n") || tail.ends_with(b"\r\n\r\n")
}

/// Keep the last four bytes of what was sent: enough to recognise a blank
/// line whatever chunking the bytes arrived in.
pub(crate) fn note_sent(tail: &mut Vec<u8>, bytes: &[u8]) {
    tail.extend_from_slice(&bytes[bytes.len().saturating_sub(4)..]);
    if tail.len() > 4 {
        let excess = tail.len() - 4;
        tail.drain(..excess);
    }
}

#[cfg(test)]
mod tests {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::{messages::MessageStreamEvent, worker::HealthCheckConfig};
    use serde_json::Value;

    use super::*;
    use crate::worker::{circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder};

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
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn cut(status: StatusCode) -> UpstreamCut {
        UpstreamCut {
            status,
            cause: CAUSE.to_string(),
        }
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

    /// The rendered sample line of `family` with all of `labels`, if any.
    fn sample<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
        rendered.lines().find(|line| {
            line.starts_with(&format!("{family}{{"))
                && labels.iter().all(|label| line.contains(label))
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
                &["router_type=\"http\"", "error_type=\"timeout\""]
            )
            .is_some(),
            "{rendered}"
        );
    }

    /// The decode leg of a disaggregated pair is charged as the decode
    /// worker's failure, under the disaggregated router's labels.
    #[test]
    fn a_cut_on_the_decode_leg_is_counted_under_the_disaggregated_router_s_labels() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let worker = worker();
        let source =
            RelaySource::decode_leg("relay-test", "/v1/chat/completions", Arc::clone(&worker));

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

    /// The boundary is read from the last bytes sent, across chunks.
    #[test]
    fn the_frame_boundary_is_read_from_the_last_bytes_sent() {
        let mut tail = Vec::new();
        assert!(on_frame_boundary(&tail), "nothing sent yet");
        note_sent(&mut tail, b"data: {\"choices\":[]}\n\n");
        assert!(on_frame_boundary(&tail));
        note_sent(&mut tail, b"data: {\"cho");
        assert!(!on_frame_boundary(&tail), "a frame cut mid-line");
        note_sent(&mut tail, b"ices\":[]}\n");
        assert!(!on_frame_boundary(&tail), "one newline is not a blank line");
        note_sent(&mut tail, b"\n");
        assert!(
            on_frame_boundary(&tail),
            "the blank line split across two chunks"
        );
        note_sent(&mut tail, b"event: x\r\ndata: 1\r\n\r\n");
        assert!(on_frame_boundary(&tail));
        assert!(tail.len() <= 4);
    }
}
