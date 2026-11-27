//! HTTP metrics collection (SMG Layer 1 metrics).
//!
//! `HttpMetricsLayer` wraps the inner service to record per-request
//! duration plus the in-flight connection count via
//! `InFlightRequestTracker`. The path label is the matched axum route
//! template (or `"other"` when unmatched) to bound metric cardinality.
//! The layer is the outermost one with a duty that ends with the response
//! body, and the one that wraps the body: the `EdgeBody` it builds carries
//! this layer's in-flight guard and response count together with what the
//! layers inside hand it (the request span, the admission permit, the SSE
//! keep-alive), so a streamed frame crosses one wrapper.
//!
//! `smg_http_responses_total` counts a request once, when its outcome is
//! known: the response's status once the response has gone out in full, or
//! [`CLIENT_CLOSED_REQUEST_STATUS`] (499, nginx's spelling) when the client
//! closed the connection first, before the response was produced or while
//! its body was still streaming. Counted at the head, a stream the client
//! cut was a 200 and a request the client abandoned before its answer was
//! nothing at all.
//!
//! A body that failed after its head went out (the relay's upstream stream
//! broke mid-body) keeps the head's status, told apart by the `error_code`
//! [`BODY_FAILED_CODE`]: counted as its head alone, a truncated answer was a
//! complete one. So does a body whose producer told the client of a failure
//! in the body itself and then ended it as planned (the error event a
//! stream ends with, then `[DONE]`), whatever the failure's cause: a worker
//! that cut the stream, a constraint the engine could not keep, a fault of
//! the request found after the first frame, a failure in the gateway's own
//! reading. Its producer's word reaches the edge as `BodyFailure`, and the
//! response counts as a body that failed, as the client read it; the event's
//! own type names the cause to the client.
//!
//! A body has gone out in full at its last frame, or once its data frames
//! add up to the length the response declared (`Content-Length`): an
//! HTTP/1.1 server writes that many bytes and asks for no more, so a relayed
//! upstream answer that declared its length is never polled for its trailing
//! end. The 499 also covers the rare request the server itself dropped
//! before it answered: a handler that panicked (the edge stack catches no
//! panic) and a connection cut by a forced shutdown after the drain timeout
//! look the same from here, a request's future dropped without a response.

use std::{
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    body::{Body, Bytes},
    extract::{MatchedPath, Request},
    response::Response,
};
use http_body::Body as _;
use openai_protocol::validated::RejectionCode;
use tokio::task::futures::TaskLocalFuture;
use tower::{Layer, Service};

use super::{
    duty::{Duty, DutyFuture},
    edge_body::{BodyDuties, EdgeBody},
    request_start::{clock_now, RequestStarted},
};
use crate::{
    observability::{
        inflight_tracker::{InFlightGuard, InFlightRequestTracker},
        metrics::{method_to_static_str, Metrics},
    },
    routers::{
        error::{extract_error_code_from_response, HEADER_X_SMG_ERROR_CODE},
        grpc::utils::stream_failure::STREAM_ERROR_TYPE,
    },
};

/// The `error_code` label of a response: the code the gateway's error builder
/// set, else the code of a body the JSON extractor refused (its envelope is
/// built in the protocols crate, outside the builder), else `""` (a success,
/// or an upstream body: an upstream's `X-SMG-Error-Code` header is never read,
/// so forged codes cannot grow the label set).
fn error_code_label(response: &Response) -> &str {
    let code = extract_error_code_from_response(response);
    if !code.is_empty() {
        return code;
    }
    response
        .extensions()
        .get::<RejectionCode>()
        .map_or("", |code| code.0.as_str())
}

/// The `status_code` label of a request whose client closed the connection
/// before the response completed (nginx's 499), and the `error_code` that
/// names the outcome on the same sample.
pub const CLIENT_CLOSED_REQUEST_STATUS: u16 = 499;
pub const CLIENT_CLOSED_REQUEST_CODE: &str = "client_closed_request";

/// The `error_code` of a response whose body failed after its head was out
/// (the relay's upstream stream broke mid-body): the sample keeps the head's
/// status; a gateway error code the head carried stays, else this code names
/// the failure where an empty code would say the response went out in full.
/// The type the SSE error event gives a stream the worker failed, shared with
/// it so the two names cannot drift apart.
pub const BODY_FAILED_CODE: &str = STREAM_ERROR_TYPE;

/// The length a response declares for its body: its `Content-Length`, or
/// else what the body itself knows exactly. An HTTP/1.1 server writes that
/// many bytes and stops asking for frames, so a body that has sent them has
/// gone out in full whether or not its trailing `None` is ever polled.
fn declared_body_length(response: &Response) -> Option<u64> {
    response
        .headers()
        .get(http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok()?.trim().parse().ok())
        .or_else(|| response.body().size_hint().exact())
}

/// Where a response body stands: the length its response declared and the
/// data bytes out so far. The body has gone out in full at its last frame,
/// or once the bytes add up to the declared length: the one rule by which
/// the edge body counts the response and closes its span, so a response
/// cannot count as complete and read as cut short.
pub(crate) struct BodyProgress {
    declared: Option<u64>,
    sent: u64,
}

impl BodyProgress {
    pub(crate) fn new(response: &Response) -> Self {
        Self {
            declared: declared_body_length(response),
            sent: 0,
        }
    }

    /// The length the response declared, when it did.
    pub(super) fn declared(&self) -> Option<u64> {
        self.declared
    }

    /// The data bytes out so far.
    pub(super) fn sent(&self) -> u64 {
        self.sent
    }

    /// A frame went out; `at_end` is the body's own word (`is_end_stream`)
    /// after it. Whether the body has now gone out in full.
    pub(crate) fn frame_out(&mut self, frame: &http_body::Frame<Bytes>, at_end: bool) -> bool {
        if let Some(data) = frame.data_ref() {
            self.sent = self.sent.saturating_add(data.len() as u64);
        }
        self.out_in_full(at_end)
    }

    /// Whether the body has gone out in full; `at_end` is the body's own
    /// word (`is_end_stream`). Asked at the head, before any frame, it says
    /// whether the response has nothing to send.
    pub(crate) fn out_in_full(&self, at_end: bool) -> bool {
        at_end || self.declared.is_some_and(|declared| self.sent >= declared)
    }
}

/// Tower Layer for HTTP metrics collection (SMG Layer 1 metrics)
#[derive(Clone)]
pub struct HttpMetricsLayer {
    tracker: Arc<InFlightRequestTracker>,
}

impl HttpMetricsLayer {
    pub fn new(tracker: Arc<InFlightRequestTracker>) -> Self {
        Self { tracker }
    }
}

impl<S> Layer<S> for HttpMetricsLayer {
    type Service = HttpMetricsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HttpMetricsMiddleware {
            inner,
            in_flight_request_tracker: self.tracker.clone(),
        }
    }
}

/// Tower Service for HTTP metrics collection
#[derive(Clone)]
pub struct HttpMetricsMiddleware<S> {
    inner: S,
    in_flight_request_tracker: Arc<InFlightRequestTracker>,
}

impl<S> Service<Request> for HttpMetricsMiddleware<S>
where
    S: Service<Request, Response = Response>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = HttpMetricsFuture<TaskLocalFuture<RequestStarted, S::Future>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let method = method_to_static_str(req.method().as_str());
        let path = PathLabel::of(req.extensions());
        // The request's entry stamp: read from the clock once, here; the
        // layers inside read it from the request's scope (the access log's
        // latency and the in-flight age count from it; see `request_start`):
        // their `call`s run under the stamp's sync scope below and their
        // futures under its scoped future, so they read it from either.
        let started = RequestStarted::now();
        let start = started.instant();

        let tracker = Arc::clone(&self.in_flight_request_tracker);
        let guard = tracker.track_from(start);
        Metrics::set_http_connections_active(tracker.len());
        // Counted once the outcome is known. Dropped with the future before
        // the handler answered (what the server does with a request whose
        // client closed the connection), it counts the request as closed by
        // the client.
        let outcome = ResponseOutcome::pending(path);
        // The inner service is called here, not from the future: the layer
        // keeps no clone of the stack inside it per request (a clone of a
        // route is a heap copy of the service in it), and its future is the
        // route's own with the edge's duty beside it, not a box around both.
        // The call runs under the stamp's sync scope and the future under
        // its scoped future: the layers inside read the stamp from both.
        DutyFuture::new(
            started.scope(started.sync_scope(|| self.inner.call(req))),
            EdgeDuty {
                method,
                start,
                tracker,
                outcome,
                guard,
            },
        )
    }
}

/// The metrics layer's future: `F`, the route's future under the request's
/// start in its task local, with the edge's bookkeeping as its duty.
pub type HttpMetricsFuture<F> = DutyFuture<F, EdgeDuty>;

/// What the metrics layer holds for a request between its arrival and the
/// route's answer: the edge's bookkeeping of the answer (the duration, the
/// response head counted, the one body wrapper). Dropped with the future
/// before the answer, it counts the request as closed by the client and
/// releases its in-flight guard, with the gauge set from the tracker as at
/// every other release (`unanswered`).
pub struct EdgeDuty {
    method: &'static str,
    start: Instant,
    tracker: Arc<InFlightRequestTracker>,
    outcome: ResponseOutcome,
    guard: InFlightGuard,
}

impl Duty for EdgeDuty {
    /// The route answered: the response head is counted and the body wrapped
    /// once; an error of the inner service releases the guard and counts
    /// nothing.
    fn answered<E>(self, answer: Result<Response, E>) -> Result<Response, E> {
        let Self {
            method,
            start,
            tracker,
            mut outcome,
            guard,
        } = self;
        let mut response = match answer {
            Ok(response) => response,
            Err(e) => {
                // Decrement on error too.
                drop(guard);
                Metrics::set_http_connections_active(tracker.len());
                outcome.forget();
                return Err(e);
            }
        };

        outcome.head(&mut response);

        // Hold the in-flight guard for the response BODY's lifetime, not
        // the handler's: streaming handlers return their response
        // immediately and keep generating from a detached task, and the
        // guard is what `wait_for_drain()` and the age histogram observe.
        // The outcome rides along: the response counts when its body
        // ends, and as a client-closed request when the body is dropped
        // before that. The layers inside handed over what they hold
        // until the body ends; the one wrapper carries it all.
        let progress = BodyProgress::new(&response);
        let duties = BodyDuties::take(&mut response);
        // The head went out: at the access-log layer's stamp of it when
        // that layer is in the stack (it hands the stamp over with the
        // duties), else read here.
        let head_at = duties.head_at.unwrap_or_else(clock_now);
        Metrics::record_http_duration(
            method,
            outcome.path(),
            head_at.saturating_duration_since(start),
        );
        let (parts, body) = response.into_parts();
        let body = Body::new(EdgeBody::new(
            body, guard, tracker, outcome, progress, duties,
        ));
        Ok(Response::from_parts(parts, body))
    }

    /// The route never answered: the request's future was dropped (what the
    /// server does with a request whose client closed the connection). The
    /// in-flight guard goes and the gauge follows it; the outcome, dropped
    /// with the rest, counts the request as closed by the client.
    fn unanswered(self) {
        let Self { tracker, guard, .. } = self;
        // Drop before reading len() so the gauge reflects the decrement.
        drop(guard);
        Metrics::set_http_connections_active(tracker.len());
    }
}

/// What `smg_http_responses_total` counts for one request, recorded exactly
/// once: the response head's status once the response has gone out in full,
/// the head's status with [`BODY_FAILED_CODE`] when its body failed after
/// the head, a client-closed request when the client left first.
pub(super) struct ResponseOutcome {
    path: PathLabel,
    /// The response head's status and gateway error code, once there is one.
    head: Option<(u16, String)>,
    recorded: bool,
}

impl ResponseOutcome {
    pub(super) fn pending(path: PathLabel) -> Self {
        Self {
            path,
            head: None,
            recorded: false,
        }
    }

    /// The path label the response counts under.
    pub(super) fn path(&self) -> &str {
        self.path.as_str()
    }

    /// The response head is out: the status a completed response counts as.
    /// A request the extractors refused (its body, its query or its path)
    /// gets its `X-SMG-Error-Code` header here: their one envelope is built
    /// in the protocols crate, which names the code in the body and in an
    /// extension, and this layer reads that extension for every response.
    pub(super) fn head(&mut self, response: &mut Response) {
        let code = error_code_label(response).to_owned();
        if !code.is_empty() && extract_error_code_from_response(response).is_empty() {
            if let Ok(value) = http::HeaderValue::from_str(&code) {
                response
                    .headers_mut()
                    .insert(HEADER_X_SMG_ERROR_CODE, value);
            }
        }
        self.head = Some((response.status().as_u16(), code));
    }

    /// The response went out in full: count its head's status.
    pub(super) fn completed(&mut self) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        if let Some((status_code, error_code)) = &self.head {
            Metrics::record_http_response(self.path.as_str(), *status_code, error_code);
        }
    }

    /// The body failed after the head went out: count the head's status,
    /// with the failure named unless the head carried a gateway error code.
    pub(super) fn body_failed(&mut self) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        if let Some((status_code, error_code)) = &self.head {
            let error_code = if error_code.is_empty() {
                BODY_FAILED_CODE
            } else {
                error_code.as_str()
            };
            Metrics::record_http_response(self.path.as_str(), *status_code, error_code);
        }
    }

    /// The client closed the connection before the response completed.
    pub(super) fn client_closed(&mut self) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        Metrics::record_http_response(
            self.path.as_str(),
            CLIENT_CLOSED_REQUEST_STATUS,
            CLIENT_CLOSED_REQUEST_CODE,
        );
    }

    /// Nothing to count: the inner service failed instead of answering.
    fn forget(&mut self) {
        self.recorded = true;
    }
}

impl Drop for ResponseOutcome {
    fn drop(&mut self) {
        // Still pending at drop: the request's future went away before the
        // handler answered, which is what the server does with a request
        // whose client closed the connection (and, rarely, with one whose
        // handler panicked or that a forced shutdown cut: the module doc).
        self.client_closed();
    }
}

/// Bounded path label for HTTP metrics: the matched axum route template, or
/// `"other"` when no route matched. Labeling by raw request path would let
/// attacker-controlled URIs create unbounded distinct labels.
pub(super) fn matched_path_label(extensions: &http::Extensions) -> &str {
    extensions
        .get::<MatchedPath>()
        .map_or(UNMATCHED_PATH_LABEL, MatchedPath::as_str)
}

/// The path label of a request that matched no route.
const UNMATCHED_PATH_LABEL: &str = "other";

/// The bounded path label ([`matched_path_label`]) a request's response is
/// counted under once it has gone out: the matched route's own template,
/// shared with the router that holds it (a clone is a reference count, not a
/// copy of the text), or a constant.
#[derive(Clone, Debug)]
pub(super) enum PathLabel {
    Matched(MatchedPath),
    Static(&'static str),
}

impl PathLabel {
    /// The label of a request: its matched route, or `"other"`.
    pub(super) fn of(extensions: &http::Extensions) -> Self {
        extensions
            .get::<MatchedPath>()
            .cloned()
            .map_or(Self::Static(UNMATCHED_PATH_LABEL), Self::Matched)
    }

    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Matched(path) => path.as_str(),
            Self::Static(label) => label,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/metrics/tests.rs"]
mod tests;

/// `smg_http_responses_total` counts a request once, at its outcome: the
/// head's status once the response is out in full, the head's status with
/// `stream_error` when its body failed after the head, 499 when the client
/// closed the connection first.
#[cfg(test)]
#[path = "../../tests/unit/middleware/metrics/client_closed_tests.rs"]
mod client_closed_tests;
