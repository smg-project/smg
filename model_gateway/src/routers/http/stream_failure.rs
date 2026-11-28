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

use axum::{http::StatusCode, response::Response};
use bytes::Bytes;
use serde_json::json;
use smg_external_router::retry::{is_request_timeout, REQUEST_TIMEOUT_CODE};
use smg_worker::identity::basic::Worker;
use tracing::warn;

use crate::{
    observability::metrics::{error_reason_from_status, metrics_labels, Metrics},
    routers::grpc::utils::{
        error_type_from_status, route_to_endpoint,
        stream_failure::{MESSAGES_API_ERROR_TYPE, STREAM_ERROR_TYPE},
    },
};

/// The route whose streams end with the Messages API's own error event.
const MESSAGES_ROUTE: &str = "/v1/messages";

/// Response extension: the worker's circuit breaker takes its sample of a
/// streamed response with a successful head from the relay (the head's
/// success before the relay task is spawned where the breaker is not closed,
/// the stream's one sample at its end where it is), so the caller does not
/// record the head (a second success sample would count twice toward closing
/// a half-open breaker, and a closed breaker's would erase the failures a
/// cut counts into).
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeadSuccessRecorded;

/// Mark `response` as a streamed response whose successful head its relay
/// samples (see [`HeadSuccessRecorded`]).
pub(crate) fn mark_head_success_recorded(response: &mut Response) {
    response.extensions_mut().insert(HeadSuccessRecorded);
}

/// Whether the successful head of `response` is its relay's to sample, so
/// its caller must not record it.
pub(crate) fn head_success_recorded(response: &Response) -> bool {
    response.extensions().get::<HeadSuccessRecorded>().is_some()
}

/// An upstream body error, classed the way a failed dispatch of it is: the
/// client's timeout expiring mid-body is a gateway timeout, as it is when it
/// expires before the head; any other error (the connection reset or closed
/// before the body's end, a transfer the worker left malformed) is a bad
/// gateway. Both are the worker's failure.
pub(crate) struct UpstreamCut {
    status: StatusCode,
    cause: String,
    /// The code the event that ends an OpenAI-style stream names the cut
    /// with, when the cut has one of its own: the request timeout's, the
    /// code of the 504 the same request gets unary.
    code: Option<&'static str>,
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
            code: is_request_timeout(error).then_some(REQUEST_TIMEOUT_CODE),
        }
    }

    /// The status the cut is accounted and answered under.
    pub(crate) fn status(&self) -> StatusCode {
        self.status
    }
}

/// `error` and its sources, outermost first, without the request's URL: a
/// body error reads "error decoding response body" on its own, and keeps the
/// transport's reason (the connection reset, the transfer closed before its
/// end) in the chain; the URL reqwest names in the outer error ("error sending
/// request for url (...)") is the worker's address, which the gateway logs and
/// a client's answer does not carry. The detail a side's failure below HTTP is
/// answered with (see `pd_router`), and the HTTP router's for a call it made.
pub(crate) fn cause(error: &reqwest::Error) -> String {
    let mut text = error.to_string();
    if let Some(url) = error.url() {
        // reqwest's Display ends with the URL, after the kind's words.
        let suffix = format!(" for url ({url})");
        if text.ends_with(&suffix) {
            text.truncate(text.len() - suffix.len());
        }
    }
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

    /// The response to `route` for `model`, read from the decode side `worker`
    /// of a disaggregated prefill/decode pair: a cut is the decode worker's
    /// failure, counted under the disaggregated router's labels.
    pub(crate) fn decode_side(model: &str, route: &'static str, worker: Arc<dyn Worker>) -> Self {
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
    ///
    /// The breaker sample sits outside the request's attempt ledger on
    /// purpose: the relay task outlives the attempt that produced the head.
    /// The shared relay (`routers::http::stream_relay`) samples a closed
    /// breaker once a stream, so the cut is that stream's one sample; on a
    /// breaker that is not closed it records a successful head before it
    /// spawns its task. Both hold for the regular router and the PD router's
    /// decode side alike, so the cut is never erased by the head's success.
    /// A request retried onto the same worker after a failed head is charged
    /// for that head and for the cut.
    pub(crate) fn account(&self, cut: &UpstreamCut, bytes_relayed: u64) {
        let error_type = error_type_from_status(cut.status);
        let reason = error_reason_from_status(cut.status);
        warn!(
            worker = self.worker.redacted_url(),
            worker_type = self.worker_type,
            route = self.route,
            bytes_relayed,
            http_status = cut.status.as_u16(),
            error_type,
            reason,
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
            reason,
        );
    }

    /// The SSE event that ends a response `cut` by the worker, in the shape
    /// of the route's API. An OpenAI-style stream (Chat Completions,
    /// Completions, generate, Responses) gets the error event the gRPC path
    /// writes for a stream the worker failed, typed `stream_error`, under the
    /// request timeout's code when that is what cut the body, then `[DONE]`;
    /// a Messages stream gets the API's own `error` event, typed
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
            let mut error = json!({"message": message, "type": STREAM_ERROR_TYPE});
            if let Some(code) = cut.code {
                error["code"] = json!(code);
            }
            let event = json!({"error": error});
            Bytes::from(format!("data: {event}\n\ndata: [DONE]\n\n"))
        }
    }

    /// [`Self::error_event`] written on its own frame: when `sent_tail` (the
    /// last bytes relayed, kept by [`note_sent`]) does not stand on a frame
    /// boundary, a blank line opens one first, so a cut inside a frame
    /// leaves the partial line as it is and the event parses on its own.
    /// The regular relay and the disaggregated router's decode relay end a
    /// cut SSE body with it.
    pub(crate) fn error_event_on_its_own_frame(
        &self,
        cut: &UpstreamCut,
        sent_tail: &[u8],
    ) -> Bytes {
        let event = self.error_event(cut);
        if on_frame_boundary(sent_tail) {
            return event;
        }
        let mut framed = Vec::with_capacity(event.len() + 2);
        framed.extend_from_slice(b"\n\n");
        framed.extend_from_slice(&event);
        Bytes::from(framed)
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

/// The last four bytes of what was sent: enough to recognise a blank line
/// whatever chunking the bytes arrived in, kept in place (a `Vec` for them
/// was one allocation per streamed response).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SentTail {
    bytes: [u8; 4],
    len: usize,
}

impl SentTail {
    /// The bytes, oldest first, for [`on_frame_boundary`].
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// Keep the last four bytes of what was sent.
pub(crate) fn note_sent(tail: &mut SentTail, bytes: &[u8]) {
    let fresh = &bytes[bytes.len().saturating_sub(4)..];
    let keep = (4 - fresh.len()).min(tail.len);
    let kept_from = tail.len - keep;
    tail.bytes.copy_within(kept_from..tail.len, 0);
    tail.bytes[keep..keep + fresh.len()].copy_from_slice(fresh);
    tail.len = keep + fresh.len();
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/http/stream_failure/tests.rs"]
mod tests;
