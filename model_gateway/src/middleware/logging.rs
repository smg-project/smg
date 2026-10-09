//! Tracing/logging integration for the HTTP layer.
//!
//! [`RequestTraceLayer`] opens one `http_request` span per request with the
//! `tower_http::trace` handlers below (W3C trace context in, the request ID
//! and trace ID on the span, HTTP-level metrics), runs the handler under it
//! and logs the response. The span is entered for the request, the handler
//! and the response log, and once more if the response stream fails; it is
//! not entered per body frame, so a long SSE stream costs no span bookkeeping
//! per chunk. The span stays open until the body ends, so an exported trace
//! still covers the whole stream.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{body::Body, extract::Request, middleware::Next, response::Response};
use opentelemetry::trace::TraceContextExt;
use tower::{Layer, Service};
use tower_http::{
    classify::ServerErrorsFailureClass,
    trace::{MakeSpan, OnFailure, OnRequest, OnResponse},
};
use tracing::{debug, error, field::Empty, info, info_span, warn, Instrument, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::{metrics::matched_path_label, request_id::RequestId};
use crate::observability::{
    metrics::{method_to_static_str, Metrics},
    otel_trace::{extract_trace_context_http, inject_trace_context_http, is_otel_enabled},
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
            trace_id = Empty,
            status_code = Empty,
            latency = Empty,
            error = Empty,
            otel.status_code = Empty,
            module = "smg"
        );

        // 0.33 returns a Result; a missing/empty parent context is not actionable here.
        let _ = span.set_parent(parent_cx);
        record_trace_id(&span);
        span
    }
}

/// Record the span's W3C trace id as a span field, so every log line written
/// under the request span (`started/finished processing request` included)
/// carries the id the trace backend indexes. The id is the caller's when the
/// request came with a `traceparent`, else the one minted for this request.
/// Nothing is recorded while tracing is off: the span has no trace then.
fn record_trace_id(span: &Span) {
    if !is_otel_enabled() {
        return;
    }
    let context = span.context();
    let span_ref = context.span();
    let span_context = span_ref.span_context();
    if span_context.is_valid() {
        span.record("trace_id", tracing::field::display(span_context.trace_id()));
    }
}

/// Echo the request's trace context on the response (`traceparent`, plus
/// `tracestate` when the caller sent one), so a client holding a reply can
/// look its trace up without an attribute search. Runs under the request
/// span (inside [`create_logging_layer`]'s trace layer); a no-op while
/// tracing is off.
pub async fn trace_context_response(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    inject_trace_context_http(headers);
    // The propagator writes `tracestate` even when there is none to carry.
    if headers
        .get("tracestate")
        .is_some_and(|state| state.is_empty())
    {
        headers.remove("tracestate");
    }
    response
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
        // The span's status, so the trace backend can filter by outcome: a
        // 5xx is the server's failure (as is a body that fails later, see
        // `StreamFailureLogger`); anything else is left unset, as the HTTP
        // server conventions have it.
        if status.is_server_error() {
            span.record("otel.status_code", "ERROR");
        }

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
                span.record("otel.status_code", "ERROR");
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

/// Create the request-span layer for HTTP logging.
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> RequestTraceLayer {
    RequestTraceLayer
}

/// Tower layer that runs every request under its `http_request` span: the
/// [`RequestSpan`] handlers above, applied by [`RequestTraceService`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestTraceLayer;

impl<S> Layer<S> for RequestTraceLayer {
    type Service = RequestTraceService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestTraceService { inner }
    }
}

/// Tower service of [`RequestTraceLayer`].
#[derive(Clone, Debug)]
pub struct RequestTraceService<S> {
    inner: S,
}

impl<S> Service<Request> for RequestTraceService<S>
where
    S: Service<Request, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let span = RequestSpan.make_span(&request);
        let start = Instant::now();
        let mut inner = self.inner.clone();
        Box::pin(async move {
            RequestLogger.on_request(&request, &span);
            // The handler's future is created and polled under the span, as
            // the trace layer did, so everything it logs carries the span.
            let future = {
                let _enter = span.enter();
                inner.call(request)
            };
            let response = future.instrument(span.clone()).await?;
            ResponseLogger.on_response(&response, start.elapsed(), &span);
            Ok(response.map(|body| Body::new(SpannedBody { inner: body, span })))
        })
    }
}

/// A response body that keeps its request span alive until the body ends,
/// without entering it per frame; a failure of the stream is logged under the
/// span once.
struct SpannedBody {
    inner: Body,
    span: Span,
}

impl http_body::Body for SpannedBody {
    type Data = <Body as http_body::Body>::Data;
    type Error = <Body as http_body::Body>::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Err(error))) = &poll {
            StreamFailureLogger.on_failure(
                ServerErrorsFailureClass::Error(error.to_string()),
                Duration::ZERO,
                &this.span,
            );
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        http_body::Body::is_end_stream(&self.inner)
    }

    fn size_hint(&self) -> http_body::SizeHint {
        http_body::Body::size_hint(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::{body::Bytes, http::Request, routing::get, Router};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use tracing_subscriber::{layer::SubscriberExt, Layer, Registry};

    use super::*;

    /// Counts how often any span is entered on this thread's subscriber.
    struct EnterCounter(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> Layer<S> for EnterCounter {
        fn on_enter(
            &self,
            _id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[tokio::test]
    async fn a_streamed_body_does_not_enter_the_request_span_per_frame() {
        let enters = Arc::new(AtomicUsize::new(0));
        let _guard = tracing::subscriber::set_default(
            Registry::default().with(EnterCounter(enters.clone())),
        );
        let frames = 64usize;
        let app = Router::new()
            .route(
                "/stream",
                get(move || async move {
                    Body::from_stream(futures::stream::iter(
                        (0..frames)
                            .map(|_| Ok::<_, std::io::Error>(Bytes::from_static(b"data: x\n\n"))),
                    ))
                }),
            )
            .layer(create_logging_layer());
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/stream")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let before_body = enters.load(Ordering::Relaxed);
        assert!(before_body >= 1, "the request itself runs under the span");
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        assert_eq!(body.len(), frames * b"data: x\n\n".len());
        let during_body = enters.load(Ordering::Relaxed) - before_body;
        assert!(
            during_body < frames,
            "the request span was entered {during_body} times while {frames} frames streamed"
        );
    }
}
