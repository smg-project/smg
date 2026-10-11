//! A streaming response that ends before its final chunk: what the client is
//! told and what the gateway counts.
//!
//! Once the response headers are out, a failure cannot change the status
//! code: it reaches the client as an SSE error event followed by `[DONE]`.
//! The event's `type` tells a stream the engine ended (its connection broke,
//! or it answered a status after its first chunk) from a failure of the
//! gateway's own processing, and the former is counted where a failed
//! dispatch is counted: the worker-error and router-error families and the
//! worker's circuit breaker, as the HTTP relay already does for a relayed
//! stream that breaks.

use std::{fmt, sync::Arc};

use axum::http::StatusCode;
use tracing::{error, warn};

use super::chat_utils::{send_error_sse, SseSender};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::grpc::context::{DispatchMetadata, WorkerSelection},
    worker::Worker,
};

/// The SSE error `type` of a stream the engine ended before completion: the
/// type the Responses pipeline gives the same failure.
pub(crate) const STREAM_ERROR_TYPE: &str = "stream_error";
/// The SSE error `type` of a failure in the gateway's own processing.
pub(crate) const INTERNAL_ERROR_TYPE: &str = "internal_error";

/// Why a streaming response ended before its final chunk.
#[derive(Debug)]
pub(crate) enum StreamFailure {
    /// The engine's response stream ended with an error after the response
    /// began: the connection broke (the engine went away mid-stream) or the
    /// engine answered a status after its first chunk. `completion_tokens` is
    /// how far the response had come.
    Upstream {
        status: tonic::Status,
        completion_tokens: u32,
    },
    /// The gateway's own processing failed.
    Internal(String),
}

impl StreamFailure {
    /// The engine's stream ended with `status` after `completion_tokens`.
    pub(crate) fn upstream(status: tonic::Status, completion_tokens: u32) -> Self {
        Self::Upstream {
            status,
            completion_tokens,
        }
    }

    /// The `type` of the SSE error event that ends the response.
    pub(crate) fn error_type(&self) -> &'static str {
        match self {
            Self::Upstream { .. } => STREAM_ERROR_TYPE,
            Self::Internal(_) => INTERNAL_ERROR_TYPE,
        }
    }
}

impl From<String> for StreamFailure {
    fn from(message: String) -> Self {
        Self::Internal(message)
    }
}

impl From<&str> for StreamFailure {
    fn from(message: &str) -> Self {
        Self::Internal(message.to_string())
    }
}

impl fmt::Display for StreamFailure {
    /// The `message` of the SSE error event; an upstream failure keeps the
    /// wording clients have been reading.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upstream { status, .. } => write!(f, "Stream error: {}", status.message()),
            Self::Internal(message) => f.write_str(message),
        }
    }
}

/// Where a streaming response is read from, kept by the task that writes it
/// for the accounting of a failure after the response began.
#[derive(Clone)]
pub(crate) struct StreamSource {
    request_id: String,
    model: String,
    endpoint: &'static str,
    backend_type: &'static str,
    /// The worker whose stream ends the response: the one worker, or the
    /// decode leg of a disaggregated pair.
    worker: Option<Arc<dyn Worker>>,
    worker_type: &'static str,
}

impl StreamSource {
    /// The response to `dispatch` on `endpoint`, read from the request's
    /// selection: its one worker, or its decode leg.
    pub(crate) fn new(
        workers: Option<&WorkerSelection>,
        dispatch: &DispatchMetadata,
        endpoint: &'static str,
    ) -> Self {
        let (worker, worker_type) = match workers {
            Some(selection) => match selection.decode_worker() {
                Some(decode) => (Some(Arc::clone(decode)), metrics_labels::WORKER_DECODE),
                None => (selection.single().cloned(), metrics_labels::WORKER_REGULAR),
            },
            None => (None, metrics_labels::WORKER_REGULAR),
        };
        Self {
            request_id: dispatch.request_id.clone(),
            model: dispatch.model.clone(),
            endpoint,
            backend_type: metrics_labels::BACKEND_REGULAR,
            worker,
            worker_type,
        }
    }

    /// The backend label the response's router error carries.
    pub(crate) fn with_backend(mut self, backend_type: &'static str) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Log and count `failure`. A stream the engine ended is a worker failure:
    /// the breaker outcome of a bad gateway, the worker-error and router-error
    /// counters a failed dispatch increments, and one WARN naming the worker,
    /// the request and how far the response had come. A failure of the
    /// gateway's own processing leaves the worker's record alone.
    pub(crate) fn account(&self, failure: &StreamFailure) {
        let StreamFailure::Upstream {
            status,
            completion_tokens,
        } = failure
        else {
            return;
        };
        warn!(
            request_id = %self.request_id,
            worker = self.worker.as_ref().map_or("none", |worker| worker.url()),
            completion_tokens,
            grpc_code = ?status.code(),
            "Stream ended by the engine before completion: {}",
            status.message()
        );
        if let Some(worker) = &self.worker {
            worker.record_outcome(StatusCode::BAD_GATEWAY.as_u16());
        }
        Metrics::record_worker_error(
            self.worker_type,
            metrics_labels::CONNECTION_GRPC,
            metrics_labels::ERROR_BACKEND,
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &self.model,
            self.endpoint,
            metrics_labels::ERROR_BACKEND,
        );
    }
}

/// End a failed streaming response: the accounting, then the SSE error event
/// typed for the failure (the caller's `[DONE]` follows). A client that has
/// already gone is told nothing, and a failure of the gateway's own processing
/// is logged only when there was a client to fail.
pub(crate) async fn send_stream_failure(
    tx: &SseSender,
    failure: &StreamFailure,
    source: &StreamSource,
) {
    source.account(failure);
    if tx.is_closed() {
        return;
    }
    if let StreamFailure::Internal(message) = failure {
        error!(
            request_id = %source.request_id,
            "Streaming response failed in the gateway: {message}"
        );
    }
    send_error_sse(tx, failure, failure.error_type()).await;
}

#[cfg(test)]
mod tests {
    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder,
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

    fn cut() -> StreamFailure {
        StreamFailure::upstream(
            tonic::Status::internal("h2 protocol error: error reading a body from connection"),
            3_191,
        )
    }

    /// The rendered sample line of `family` with all of `labels`, if any.
    fn sample<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
        rendered.lines().find(|line| {
            line.starts_with(&format!("{family}{{"))
                && labels.iter().all(|label| line.contains(label))
        })
    }

    #[test]
    fn an_upstream_failure_is_a_stream_error_with_the_message_clients_know() {
        let failure = cut();
        assert_eq!(failure.error_type(), "stream_error");
        assert_eq!(
            failure.to_string(),
            "Stream error: h2 protocol error: error reading a body from connection"
        );
        let failure = StreamFailure::from("Channel closed".to_string());
        assert_eq!(failure.error_type(), "internal_error");
        assert_eq!(failure.to_string(), "Channel closed");
    }

    #[test]
    fn an_upstream_failure_is_charged_to_the_worker_and_counted() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let worker = worker();
        let selection = WorkerSelection::Single {
            worker: Arc::clone(&worker),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);

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
    fn a_disaggregated_stream_is_charged_to_its_decode_leg() {
        let prefill = worker();
        let decode = worker();
        let selection = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: Arc::clone(&prefill),
            decode: Arc::clone(&decode),
            runtime_type: Default::default(),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
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

    #[test]
    fn a_failure_of_the_gateway_leaves_the_worker_alone() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let worker = worker();
        let selection = WorkerSelection::Single {
            worker: Arc::clone(&worker),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
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
}
