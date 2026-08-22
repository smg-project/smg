//! Tracing/logging integration for the HTTP layer.
//!
//! Wires `tower_http::trace::TraceLayer` with custom span/request/response
//! handlers that propagate W3C trace context, attach the request ID into
//! the span, and record HTTP-level metrics via the observability layer.

use std::time::Duration;

use axum::{extract::Request, response::Response};
use tower_http::{
    classify::ServerErrorsFailureClass,
    trace::{MakeSpan, OnFailure, OnRequest, OnResponse, TraceLayer},
};
use tracing::{debug, error, field::Empty, info, info_span, warn, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::{metrics::matched_path_label, request_id::RequestId};
use crate::observability::{
    metrics::{method_to_static_str, Metrics},
    otel_trace::extract_trace_context_http,
};

/// Response-extension marker for orchestrator probe responses (`/health`,
/// `/readiness`, `/liveness`): every status they return — 503 "not ready"
/// included — is an expected operational state, reported to a machine that
/// polls on a tight interval. The handler that produces the response owns
/// that knowledge and attaches this marker; [`ResponseLogger`] logs marked
/// responses at DEBUG instead of flooding ERROR/INFO once per poll.
#[derive(Clone, Copy, Debug)]
pub struct ProbeResponse;

/// The probe routes on the main listener. Request-start logging for these is
/// demoted alongside [`ProbeResponse`] — the paths are the request-side view
/// of the same contract.
fn is_probe_path(path: &str) -> bool {
    matches!(path, "/health" | "/readiness" | "/liveness")
}

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Extract incoming W3C trace context (traceparent/tracestate) so that
        // server-side spans become children of the caller's distributed trace.
        let parent_cx = extract_trace_context_http(request.headers());

        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span
        let span = info_span!(
            target: "smg::otel-trace",
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = Empty,  // Will be set later
            status_code = Empty,
            latency = Empty,
            error = Empty,
            module = "smg"
        );

        // 0.33 returns a Result; a missing/empty parent context is not actionable here.
        let _ = span.set_parent(parent_cx);
        span
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        let method = method_to_static_str(request.method().as_str());
        let path = matched_path_label(request.extensions());
        Metrics::record_http_request(method, path);

        // Log the request start. Probe polls arrive every couple of seconds
        // forever; keep them out of the INFO access log.
        if is_probe_path(request.uri().path()) {
            debug!(
                target: "smg::request",
                "started processing request"
            );
        } else {
            info!(
                target: "smg::request",
                "started processing request"
            );
        }
    }
}

/// Custom on_response handler
#[derive(Clone, Debug, Default)]
pub struct ResponseLogger;

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status();
        let status_code = status.as_u16();

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status_code);
        // Use microseconds as integer to avoid format! string allocation
        span.record("latency", latency.as_micros() as u64);

        // Log the response completion
        let _enter = span.enter();
        if response.extensions().get::<ProbeResponse>().is_some() {
            // A probe's 503 means "not ready yet" — an expected state a poller
            // reads every couple of seconds, not a server malfunction.
            debug!(
                target: "smg::response",
                "finished probe request"
            );
        } else if status.is_server_error() {
            error!(
                target: "smg::response",
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "smg::response",
                "request failed with client error"
            );
        } else {
            info!(
                target: "smg::response",
                "finished processing request"
            );
        }
    }
}

/// Failure handler that logs only what [`ResponseLogger`] cannot see.
///
/// The default `OnFailure` ERRORs on every 5xx *status*, duplicating the
/// line `ResponseLogger` already emits with more span context — that pair
/// is the double ERROR per failed request in the logs. A status is not a
/// transport failure; the one thing `ResponseLogger` genuinely cannot
/// observe is a body/stream error after the response head went out, so
/// only that arm logs here.
#[derive(Clone, Debug)]
pub struct StreamFailureLogger;

impl OnFailure<ServerErrorsFailureClass> for StreamFailureLogger {
    fn on_failure(&mut self, failure: ServerErrorsFailureClass, _latency: Duration, span: &Span) {
        match failure {
            // Already logged by ResponseLogger with status + latency.
            ServerErrorsFailureClass::StatusCode(_) => {}
            ServerErrorsFailureClass::Error(error) => {
                let _enter = span.enter();
                error!(
                    target: "smg::response",
                    error,
                    "response stream failed after the head was sent"
                );
            }
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
    tower_http::trace::DefaultOnBodyChunk,
    tower_http::trace::DefaultOnEos,
    StreamFailureLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger)
        .on_failure(StreamFailureLogger)
}
