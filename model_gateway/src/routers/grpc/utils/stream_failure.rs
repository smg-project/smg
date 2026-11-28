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
use smg_external_router::retry::REQUEST_TIMEOUT_CODE;
use smg_grpc_client::{is_constraint_failure, is_request_timeout};
use smg_worker::identity::{
    basic::{ConnectionModeExt, Worker, WorkerTypeExt},
    endpoint::Endpoint,
};
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
    /// `call_upstream_timeout` for a stream the gateway's request timeout
    /// cut, `reader_panicked` for a reading that panicked; each the code the
    /// HTTP answer to the same failure carries. `None` otherwise.
    pub(crate) fn code(&self) -> Option<&'static str> {
        match self {
            Self::Upstream { status, .. } if is_constraint_failure(status) => {
                Some(CONSTRAINT_FAILED_CODE)
            }
            Self::Upstream { status, .. } if is_request_timeout(status) => {
                Some(REQUEST_TIMEOUT_CODE)
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

/// The per-choice state at the engine's `index` (the unit's own index on the
/// completion stream), or the decoding failure an index past the choices the
/// request expects is: the chat and completion streams report it alike.
pub(crate) fn choice_mut<T>(choices: &mut [T], index: u32) -> Result<&mut T, StreamFailure> {
    let expected = choices.len();
    choices.get_mut(index as usize).ok_or_else(|| {
        StreamFailure::from(format!(
            "engine sent choice index {index}; the request expects {expected} choice(s)"
        ))
    })
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
                worker = %Endpoint::redacted_str(worker_url),
                worker_type,
                completion_tokens,
                grpc_code = ?status.code(),
                http_status = class.http.as_u16(),
                error_type = class.error_type,
                reason = class.reason,
                prefill = self.prefill.as_ref().map_or("none", |prefill| prefill.redacted_url()),
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
                worker = %Endpoint::redacted_str(worker_url),
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
#[path = "../../../../tests/unit/routers/grpc/utils/stream_failure/tests.rs"]
mod tests;
