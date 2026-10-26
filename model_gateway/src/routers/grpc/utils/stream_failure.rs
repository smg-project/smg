//! A streaming response that ends before its final chunk: what the client is
//! told and what the gateway counts.
//!
//! Once the response headers are out, a failure cannot change the status
//! code: it reaches the client as an SSE error event followed by `[DONE]`.
//! A status the engine answered after the response began (its connection
//! broke, its server cancelled the stream, it finished the request with an
//! error of its own, or it rejected the request after its first chunk) is
//! classed the way a failed dispatch of the same status is, by
//! [`UpstreamClass`]: the worker whose stream it was (the one worker, a
//! decode side, or the prefill side drained after the response began; the
//! prefill side the sequential dispatch drains before the response begins
//! is classed as a dispatch, the class its breaker sample comes from) is
//! charged (its
//! circuit breaker, the worker-error family) only for the classes a failed
//! dispatch charges it for, the router-error family counts every one under
//! the same `error_type`, and the event's `type` tells a stream the worker
//! failed (`stream_error`) from a request's own fault (the HTTP error body's
//! type) and from a failure of the gateway's own processing
//! (`internal_error`). A dispatch the engine answered with a status (a
//! refusal before the first token) is classed the same way, so the worker
//! is charged for it, in the worker-error family, exactly when it would be
//! for the same status after the first token (see
//! [`UpstreamClass::account_dispatch_failure`]). A worker's report that the
//! engine could not continue the request under its constraint is the
//! request's outcome, counted under its own `error_type`, charged to no
//! worker, and the event names it by the `code` the HTTP answer carries.
//! A reading that panicked in the gateway, caught at the request's boundary
//! (`utils::isolation`), is the request's own failure: an `internal_error`
//! event under `reader_panicked`, charged to no worker, counted under the
//! table that read the output ([`Panicked::record`]).

use std::{fmt, sync::Arc};

use axum::http::StatusCode;
use smg_grpc_client::is_constraint_failure;
use tracing::{error, warn};

use super::{
    chat_utils::{send_error_sse_with_code, SseSender},
    isolation::{Panicked, READER_PANICKED_CODE},
    metrics::error_type_from_status,
    pd_bootstrap::{account_pd_side_failure, dead_rendezvous_of, DeadRendezvous},
    tonic_ext::{is_the_engines_own_report, TonicStatusExt, CONSTRAINT_FAILED_CODE},
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        error::error_type_for_status,
        grpc::context::{DispatchMetadata, WorkerSelection},
    },
    worker::{worker::WorkerTypeExt, ConnectionModeExt, Worker},
};

/// The SSE error `type` of a stream the worker failed: the engine's
/// connection broke, its server cancelled the stream, or it answered a server
/// error or a timeout after its first chunk. The type the Responses pipeline
/// gives the same failure.
pub(crate) const STREAM_ERROR_TYPE: &str = "stream_error";
/// The SSE error `type` of a failure in the gateway's own processing.
pub(crate) const INTERNAL_ERROR_TYPE: &str = "internal_error";
/// The messages API's error `type` for a failure on the server's side, the
/// one its clients already read for a stream that failed.
pub(crate) const MESSAGES_API_ERROR_TYPE: &str = "api_error";

/// Which of a request's worker streams ended with the error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UpstreamStream {
    /// The stream that answers the request: the one worker's, or the decode
    /// side's of a disaggregated pair.
    Answer,
    /// The prefill side of a disaggregated pair, drained before the answer.
    Prefill,
}

/// How a status the engine answered after the response began is classed:
/// what a failed dispatch of the same status answers and counts, so a
/// request the engine rejects after its first chunk is a validation error
/// and a stream its server cut is a backend error, whichever side of the
/// first chunk the status arrives on. The one report a failed dispatch never
/// sees, the engine finishing the request with an error of its own, is an
/// `engine_error`: the worker's failure, told apart from a failed call to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UpstreamClass {
    /// The HTTP status a failed dispatch answers for the status.
    pub(crate) http: StatusCode,
    /// The `error_type` label the status counts under.
    pub(crate) error_type: &'static str,
    /// The `reason` label the status counts under: the cause behind the
    /// class (see [`TonicStatusExt::error_reason`]).
    pub(crate) reason: &'static str,
    /// What the worker's circuit breaker samples for the status; `None` for
    /// a failure that says nothing about the worker (the request's media).
    pub(crate) breaker_sample: Option<u16>,
}

impl UpstreamClass {
    /// The class of `status`, through the dispatch path's own mapping
    /// ([`TonicStatusExt::http_status`]): the same status is the same failure
    /// on either side of the first chunk. A cancelled RPC is what the
    /// engine's server produces when it resets its streams or its connection
    /// closes under a call (a stop whose grace period ran out, a dead peer:
    /// the transport's codes the liveness monitor reads the same way), a bad
    /// gateway before the first token as after it.
    pub(crate) fn of(status: &tonic::Status) -> Self {
        let http = status.http_status();
        let error_type = if is_constraint_failure(status) {
            metrics_labels::ERROR_CONSTRAINT
        } else if is_the_engines_own_report(status) {
            metrics_labels::ERROR_ENGINE
        } else if status.media_fault().is_some() {
            // The request's media failed, on whichever side of the fetch: the
            // router counts it as the request's error (the reference is the
            // request's input) and never as the worker's, whatever status
            // tells the client to do about it (a 400 to fix the reference, a
            // 502 to try the host again later).
            metrics_labels::ERROR_VALIDATION
        } else {
            error_type_from_status(http)
        };
        Self {
            http,
            error_type,
            reason: status.error_reason(),
            breaker_sample: status.breaker_status(),
        }
    }

    /// Count a dispatch the worker answered `status` to, in the worker-error
    /// family under the worker's labels, when the status is the worker's
    /// failure: a refusal before the first token is the worker's exactly as
    /// often as the same status after it. Once per attempt, next to the
    /// breaker sample the attempt takes through its ledger; the client's
    /// status and the router-error family are the pipeline's. A rejection of
    /// the request (4xx) and capacity pushback (429) are not charged to the
    /// worker, as after the first token.
    pub(crate) fn account_dispatch_failure(
        worker_type: &'static str,
        connection_mode: &'static str,
        status: &tonic::Status,
    ) {
        Self::of(status).account_worker_failure(worker_type, connection_mode);
    }

    /// Count a stream `status` ended with against the worker that served it,
    /// under `worker_type` and `connection_mode`, when it is the worker's
    /// failure: the post-first-chunk class, next to the breaker sample as
    /// everywhere else.
    pub(crate) fn account_stream_failure(
        worker_type: &'static str,
        connection_mode: &'static str,
        status: &tonic::Status,
    ) {
        Self::of(status).account_worker_failure(worker_type, connection_mode);
    }

    /// One sample of the worker-error family for this class, when the status
    /// is the worker's failure AND says something about the worker: a failure
    /// the breaker does not sample (the request's own media) is not the
    /// worker's either, whatever its HTTP class.
    fn account_worker_failure(&self, worker_type: &'static str, connection_mode: &'static str) {
        if self.breaker_sample.is_some() && self.is_worker_failure() {
            Metrics::record_worker_error(worker_type, connection_mode, self.error_type);
        }
    }

    /// Count a stream `status` ended with, read into a non-streamed answer,
    /// against `worker` in the worker-error family when it is the worker's
    /// failure: what [`StreamSource::account`] does for a streamed answer,
    /// without the streamed answer's event. The client's status and the
    /// router-error family are the pipeline's.
    pub(crate) fn account_collected_failure(worker: &dyn Worker, status: &tonic::Status) {
        Self::account_stream_failure(
            worker.worker_type().as_metric_label(),
            worker.connection_mode().as_metric_label(),
            status,
        );
    }

    /// Whether the status is the worker's failure: a server error, a timeout
    /// or the engine's own report, the classes a failed dispatch charges to
    /// the worker. A request's own fault and capacity pushback are not.
    pub(crate) fn is_worker_failure(&self) -> bool {
        matches!(
            self.error_type,
            metrics_labels::ERROR_BACKEND
                | metrics_labels::ERROR_TIMEOUT
                | metrics_labels::ERROR_ENGINE
        )
    }

    /// The error `type` the client reads: `worker_failure` for the worker's
    /// failure, otherwise the type the HTTP error body carries for the status
    /// (`invalid_request_error`, `rate_limit_error`, ...), the one the client
    /// would have read had the status come before the first chunk.
    fn client_type(&self, worker_failure: &'static str) -> &'static str {
        if self.is_worker_failure() {
            worker_failure
        } else {
            error_type_for_status(self.http)
        }
    }
}

/// Why a streaming response ended before its final chunk.
#[derive(Debug)]
pub(crate) enum StreamFailure {
    /// The engine's response stream ended with an error after the response
    /// began: the connection broke (the engine went away mid-stream), the
    /// engine answered a status after its first chunk, or it finished the
    /// request with an error of its own (`finish_reason == "error"`, the KV
    /// handoff of a decode side that failed). `completion_tokens` is how far
    /// the response had come; `stream` says which worker's stream it was;
    /// `rendezvous` is the pair's bootstrap port when the gateway probed it
    /// behind the failure and found it dead (see [`StreamSource::explain`]).
    Upstream {
        status: tonic::Status,
        completion_tokens: u32,
        stream: UpstreamStream,
        rendezvous: Option<DeadRendezvous>,
    },
    /// The gateway's own processing failed.
    Internal(String),
    /// The gateway's reading of the response panicked, caught at the
    /// request's boundary (`utils::isolation`): the request's failure, not
    /// the task's.
    Panicked(Panicked),
}

impl StreamFailure {
    /// The engine's answer stream ended with `status` after `completion_tokens`.
    pub(crate) fn upstream(status: tonic::Status, completion_tokens: u32) -> Self {
        Self::Upstream {
            status,
            completion_tokens,
            stream: UpstreamStream::Answer,
            rendezvous: None,
        }
    }

    /// The prefill side's stream ended with `status` before its completion.
    pub(crate) fn prefill(status: tonic::Status) -> Self {
        Self::Upstream {
            status,
            completion_tokens: 0,
            stream: UpstreamStream::Prefill,
            rendezvous: None,
        }
    }

    /// The class of an upstream failure's status; a failure of the gateway's
    /// own processing has none.
    pub(crate) fn class(&self) -> Option<UpstreamClass> {
        match self {
            Self::Upstream { status, .. } => Some(UpstreamClass::of(status)),
            Self::Internal(_) | Self::Panicked(_) => None,
        }
    }

    /// The `type` of the SSE error event that ends the response:
    /// `stream_error` for a stream the worker failed, the HTTP error body's
    /// type for any other status the engine answered, `internal_error` for a
    /// failure of the gateway's own processing.
    pub(crate) fn error_type(&self) -> &'static str {
        self.class().map_or(INTERNAL_ERROR_TYPE, |class| {
            class.client_type(STREAM_ERROR_TYPE)
        })
    }

    /// The same for the messages API's error event, whose type for a failure
    /// on the server's side is `api_error`.
    pub(crate) fn messages_error_type(&self) -> &'static str {
        self.class().map_or(MESSAGES_API_ERROR_TYPE, |class| {
            class.client_type(MESSAGES_API_ERROR_TYPE)
        })
    }

    /// The `code` of the SSE error event that ends the response, when the
    /// failure has one of its own: `constraint_failed` for a worker's report
    /// that the engine could not continue the request under its constraint,
    /// `reader_panicked` for a reading that panicked; each the code the HTTP
    /// answer to the same failure carries. `None` otherwise.
    pub(crate) fn code(&self) -> Option<&'static str> {
        match self {
            Self::Upstream { status, .. } if is_constraint_failure(status) => {
                Some(CONSTRAINT_FAILED_CODE)
            }
            Self::Panicked(_) => Some(READER_PANICKED_CODE),
            Self::Upstream { .. } | Self::Internal(_) => None,
        }
    }
}

impl From<Panicked> for StreamFailure {
    fn from(panicked: Panicked) -> Self {
        Self::Panicked(panicked)
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
    /// wording clients have been reading, with the dead rendezvous port the
    /// gateway found behind it, when it did, after the engine's report.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upstream {
                status,
                stream: UpstreamStream::Answer,
                rendezvous,
                ..
            } => {
                write!(f, "Stream error: {}", status.message())?;
                match rendezvous {
                    Some(rendezvous) => write!(f, "; {rendezvous}"),
                    None => Ok(()),
                }
            }
            Self::Upstream {
                status,
                stream: UpstreamStream::Prefill,
                ..
            } => write!(f, "Prefill stream error: {}", status.message()),
            Self::Internal(message) => f.write_str(message),
            Self::Panicked(panicked) => write!(f, "{panicked}"),
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
    /// decode side of a disaggregated pair.
    worker: Option<Arc<dyn Worker>>,
    worker_type: &'static str,
    /// The prefill side of a disaggregated pair, whose stream is drained
    /// before the answer's.
    prefill: Option<Arc<dyn Worker>>,
}

impl StreamSource {
    /// The response to `dispatch` on `endpoint`, read from the request's
    /// selection: its one worker, or its decode side.
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
            prefill: workers.and_then(WorkerSelection::prefill_worker).cloned(),
        }
    }

    /// The backend label the response's router error carries.
    pub(crate) fn with_backend(mut self, backend_type: &'static str) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// `failure` explained, when the gateway can explain it: the decode side
    /// of a disaggregated pair the engine failed before its first token, with
    /// no reason of its own, is read against the pair's bootstrap port,
    /// probed now (see [`dead_rendezvous_of`]). The failure with the dead
    /// port on it; `None` for any other failure, or a port that accepted.
    pub(crate) async fn explain(&self, failure: &StreamFailure) -> Option<StreamFailure> {
        let StreamFailure::Upstream {
            status,
            completion_tokens: 0,
            stream: UpstreamStream::Answer,
            rendezvous: None,
        } = failure
        else {
            return None;
        };
        let rendezvous = dead_rendezvous_of(self.prefill.as_ref(), status).await?;
        Some(StreamFailure::Upstream {
            status: status.clone(),
            completion_tokens: 0,
            stream: UpstreamStream::Answer,
            rendezvous: Some(rendezvous),
        })
    }

    /// Log and count `failure`. A status the engine answered after the
    /// response began is accounted, on the side whose stream it was, as a
    /// failed dispatch of the same status is (see [`UpstreamClass`]): the
    /// worker's circuit breaker samples the HTTP status of its class, the
    /// worker-error family counts it only when it is the worker's failure,
    /// the router-error family counts every one, all under one `error_type`;
    /// and one WARN names the worker, its side, the request, the class and how
    /// far the response had come (and the pair's dead bootstrap port, when
    /// the failure was explained by one). A failure of the gateway's own
    /// processing leaves the worker's record alone; a reading that panicked
    /// is recorded as the isolation records it ([`Panicked::record`]): the
    /// table's outcome and one ERROR line naming the request.
    pub(crate) fn account(&self, failure: &StreamFailure) {
        if let StreamFailure::Panicked(panicked) = failure {
            panicked.record(&self.request_id, &self.model);
            return;
        }
        let StreamFailure::Upstream {
            status,
            completion_tokens,
            stream,
            rendezvous,
        } = failure
        else {
            return;
        };
        let (worker, worker_type) = match stream {
            UpstreamStream::Answer => (self.worker.as_ref(), self.worker_type),
            UpstreamStream::Prefill => (self.prefill.as_ref(), metrics_labels::WORKER_PREFILL),
        };
        let class = UpstreamClass::of(status);
        let worker_url = worker.map_or("none", |worker| worker.url());
        match rendezvous {
            Some(DeadRendezvous(port)) => warn!(
                request_id = %self.request_id,
                worker = worker_url,
                worker_type,
                completion_tokens,
                grpc_code = ?status.code(),
                http_status = class.http.as_u16(),
                error_type = class.error_type,
                reason = class.reason,
                prefill = self.prefill.as_ref().map_or("none", |prefill| prefill.url()),
                bootstrap_host = port.host.as_str(),
                bootstrap_port = port.port,
                bootstrap_port_source = port.source.label(),
                probe_error = %port.error,
                "Stream ended by the engine before completion, and the prefill side's bootstrap \
                 port did not accept a connection: the sides could not rendezvous: {}",
                status.message()
            ),
            None => warn!(
                request_id = %self.request_id,
                worker = worker_url,
                worker_type,
                completion_tokens,
                grpc_code = ?status.code(),
                http_status = class.http.as_u16(),
                error_type = class.error_type,
                reason = class.reason,
                "Stream ended by the engine before completion: {}",
                status.message()
            ),
        }
        if let (Some(worker), Some(sample)) = (worker, class.breaker_sample) {
            worker.record_outcome(sample);
        }
        // The decode side of a disaggregated pair the engine retired with its
        // rendezvous verdict counts under the rendezvous family too; the
        // prefill side's failure is counted where its stream is drained.
        if matches!(stream, UpstreamStream::Answer) && worker_type == metrics_labels::WORKER_DECODE
        {
            account_pd_side_failure(worker_type, status);
        }
        // Under the worker's own transport label, as the dispatch sites count;
        // the router family below keeps its gRPC-router label.
        class.account_worker_failure(
            worker_type,
            worker.map_or(metrics_labels::CONNECTION_GRPC, |worker| {
                worker.connection_mode().as_metric_label()
            }),
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &self.model,
            self.endpoint,
            class.error_type,
            class.reason,
        );
    }
}

/// End a failed streaming response: the accounting, then the SSE error event
/// typed for the failure (the caller's `[DONE]` follows). A client that has
/// already gone is told nothing, and a failure of the gateway's own processing
/// is logged only when there was a client to fail. A decode side the engine
/// failed without a reason is first read against the pair's bootstrap port
/// (see [`StreamSource::explain`]), so the line, the counts and the event
/// name the dead rendezvous when there is one.
pub(crate) async fn send_stream_failure(
    tx: &SseSender,
    failure: &StreamFailure,
    source: &StreamSource,
) {
    let explained = source.explain(failure).await;
    let failure = explained.as_ref().unwrap_or(failure);
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
    send_error_sse_with_code(tx, failure, failure.error_type(), failure.code()).await;
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
    use openai_protocol::worker::WorkerType;

    use super::*;
    use crate::{
        routers::grpc::proto_wrapper::ENGINE_ERROR_MESSAGE,
        test_support::RefusedAddress,
        worker::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            test_support::no_health_check,
            BasicWorkerBuilder, ConnectionMode,
        },
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
            line.starts_with(&format!("{family}{{"))
                && labels.iter().all(|label| line.contains(label))
        })
    }

    /// `status` ends a chat stream read from a fresh worker whose breaker
    /// opens on the first failure: the worker, and what was counted.
    fn account_on_a_fresh_worker(status: tonic::Status) -> (Arc<dyn Worker>, PrometheusHandle) {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let worker = worker();
        let selection = WorkerSelection::Single {
            worker: Arc::clone(&worker),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
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
    fn a_disaggregated_stream_is_charged_to_its_decode_side() {
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
            decode: worker(),
            runtime_type: Default::default(),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
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
            decode: Arc::clone(&decode),
            runtime_type: Default::default(),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
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
        let single = WorkerSelection::Single { worker: worker() };
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
            worker: Arc::clone(&worker),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
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
            decode: Arc::clone(&decode),
            runtime_type: Default::default(),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
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
            worker: Arc::clone(&worker),
        };
        let source =
            StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT);
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
}
