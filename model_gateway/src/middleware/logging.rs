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

use super::{
    metrics::{matched_path_label, BodyProgress, CLIENT_CLOSED_REQUEST_STATUS},
    request_id::RequestId,
};
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

/// Response-extension marker for a refusal whose refusing router wrote the
/// log line for it already (one WARN per refusal, rate-limited under a
/// sustained condition, with the request id, the model and the pool's
/// state). [`ResponseLogger`] logs a marked response at DEBUG instead of a
/// second line at ERROR: a client retrying at once against a model whose
/// workers are all down turned every refusal into three ERROR lines, and
/// the log into a storm.
#[derive(Clone, Copy, Debug)]
pub struct ReportedRefusal;

/// The probe routes on the main listener. Request-start logging for these is
/// demoted alongside [`ProbeResponse`] — the paths are the request-side view
/// of the same contract.
fn is_probe_path(path: &str) -> bool {
    matches!(path, "/health" | "/readiness" | "/liveness")
}

/// Opens the `http_request` span: method, path, HTTP version, and the
/// request id, trace id, status and latency recorded as they become known.
///
/// The span carries the request's path, never its query string. The query
/// is where a client mistakenly puts a credential (`?api_key=...`): the
/// request is refused, but a logged URI would keep the key in every access
/// log record of the request. Routing rarely needs the query (`/v1/realtime`'s
/// `?model=` is the exception, and that handler reads it itself; the list
/// routes page on it the same way), so the access log gives it up: a
/// trade-off, not permission to route on query parameters, and not a
/// field to widen back to the URI.
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
            path = %request.uri().path(),
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
        } else if response.extensions().get::<ReportedRefusal>().is_some() {
            // The refusing router wrote this refusal's line; a second one per
            // request here, at ERROR or WARN, is what made a retry loop a log
            // storm.
            debug!(
                target: "smg::response",
                "request refused; reported by the refusing router"
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
            let mut close_log = ClientCloseLog::new(span.clone(), start);
            // The handler's future is created and polled under the span, as
            // the trace layer did, so everything it logs carries the span.
            let future = {
                let _enter = span.enter();
                inner.call(request)
            };
            let response = match future.instrument(span.clone()).await {
                Ok(response) => response,
                Err(error) => {
                    close_log.done();
                    return Err(error);
                }
            };
            ResponseLogger.on_response(&response, start.elapsed(), &span);
            close_log.head_sent();
            let progress = BodyProgress::new(&response);
            Ok(response.map(|body| {
                Body::new(SpannedBody {
                    inner: body,
                    span,
                    close_log,
                    polled: false,
                    progress,
                })
            }))
        })
    }
}

/// The one line a request whose client closed the connection first gets, at
/// INFO: a client that gives up (an SDK's timeout, a user who navigated away,
/// a proxy cutting an idle stream) is a normal outcome, which the metrics
/// layer counts as status 499; nothing on the gateway's side failed. Logged
/// when the request's future is dropped before the handler answered, or when
/// the response body is dropped before its end (its last frame, or the length
/// the response declared). The rare request the server dropped itself (a
/// handler panic, a forced shutdown) gets the same line: from here the two
/// are one event.
struct ClientCloseLog {
    span: Span,
    start: Instant,
    stage: Stage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The handler has not answered yet.
    BeforeResponse,
    /// The response head is out; its body may still be streaming.
    Streaming,
    /// The body ended (a failure of it is `StreamFailureLogger`'s), or the
    /// inner service failed: nothing to log at drop.
    Done,
}

impl ClientCloseLog {
    fn new(span: Span, start: Instant) -> Self {
        Self {
            span,
            start,
            stage: Stage::BeforeResponse,
        }
    }

    fn head_sent(&mut self) {
        self.stage = Stage::Streaming;
    }

    fn done(&mut self) {
        self.stage = Stage::Done;
    }

    /// The client closed the connection: one line for the stage the request
    /// was in, once.
    fn client_closed(&mut self) {
        match std::mem::replace(&mut self.stage, Stage::Done) {
            Stage::BeforeResponse => {
                // The access-log fields the response line would have carried.
                self.span
                    .record("status_code", CLIENT_CLOSED_REQUEST_STATUS);
                self.span
                    .record("latency", self.start.elapsed().as_micros() as u64);
                let _enter = self.span.enter();
                info!(
                    target: "smg::response",
                    "client closed the connection before the response"
                );
            }
            Stage::Streaming => {
                let _enter = self.span.enter();
                info!(
                    target: "smg::response",
                    "client closed the connection before the response stream ended"
                );
            }
            Stage::Done => {}
        }
    }
}

impl Drop for ClientCloseLog {
    fn drop(&mut self) {
        self.client_closed();
    }
}

/// A response body that keeps its request span alive until the body ends,
/// without entering it per frame; a failure of the stream is logged under the
/// span once, and so is the client closing the connection before the end.
struct SpannedBody {
    inner: Body,
    span: Span,
    close_log: ClientCloseLog,
    /// Whether a frame was ever asked for: a body dropped before any poll is
    /// one the server never meant to send (the answer to a HEAD request), not
    /// one the client cut short.
    polled: bool,
    /// The declared length and the bytes out: the end-of-body rule shared
    /// with the metrics wrapper.
    progress: BodyProgress,
}

impl http_body::Body for SpannedBody {
    type Data = <Body as http_body::Body>::Data;
    type Error = <Body as http_body::Body>::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        this.polled = true;
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        match &poll {
            Poll::Ready(Some(Err(error))) => {
                StreamFailureLogger.on_failure(
                    ServerErrorsFailureClass::Error(error.to_string()),
                    Duration::ZERO,
                    &this.span,
                );
                this.close_log.done();
            }
            Poll::Ready(None) => this.close_log.done(),
            Poll::Ready(Some(Ok(frame))) => {
                // At its end: the last frame, or the declared length out.
                if this
                    .progress
                    .frame_out(frame, http_body::Body::is_end_stream(&this.inner))
                {
                    this.close_log.done();
                }
            }
            Poll::Pending => {}
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

impl Drop for SpannedBody {
    fn drop(&mut self) {
        // Dropped before its end: the client closed the connection. A body
        // that was never polled was not cut short.
        if self.polled {
            self.close_log.client_closed();
        } else {
            self.close_log.done();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    use axum::{
        body::Bytes,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
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

    /// Counts the events of one level on this thread's subscriber.
    struct LevelCounter(tracing::Level, Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> Layer<S> for LevelCounter {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == self.0 {
                self.1.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A 503 the refusing stage reported itself is not logged again at ERROR
    /// by the access log; an unmarked 503 still is.
    #[tokio::test]
    async fn a_reported_refusal_is_not_logged_again_at_error() {
        let errors = Arc::new(AtomicUsize::new(0));
        let _guard = tracing::subscriber::set_default(
            Registry::default().with(LevelCounter(tracing::Level::ERROR, errors.clone())),
        );
        let refused = || async {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
            response.extensions_mut().insert(ReportedRefusal);
            response
        };
        let failed = || async {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
            response
        };
        let app = Router::new()
            .route("/refused", get(refused))
            .route("/failed", get(failed))
            .layer(create_logging_layer());
        for (path, expected_errors) in [("/refused", 0), ("/failed", 1)] {
            errors.store(0, Ordering::Relaxed);
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                errors.load(Ordering::Relaxed),
                expected_errors,
                "ERROR lines written for {path}"
            );
        }
    }

    /// Every field a span is opened with, recorded later, or carried by an
    /// event on this thread's subscriber, as `name=value`.
    #[derive(Clone, Default)]
    struct Fields(Arc<Mutex<Vec<String>>>);

    impl Fields {
        fn visit(&self, record: &dyn Fn(&mut dyn tracing::field::Visit)) {
            struct Collect<'a>(&'a mut Vec<String>);
            impl tracing::field::Visit for Collect<'_> {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    self.0.push(format!("{}={value:?}", field.name()));
                }
            }
            record(&mut Collect(&mut self.0.lock().expect("fields")));
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for Fields {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.visit(&|visitor| attrs.record(visitor));
        }

        fn on_record(
            &self,
            _id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.visit(&|visitor| values.record(visitor));
        }

        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            self.visit(&|visitor| event.record(visitor));
        }
    }

    /// The access log keeps the request's path and never its query string: a
    /// credential a client mistakenly puts in the URL stays out of the logs,
    /// even though the request itself is refused.
    #[tokio::test]
    async fn the_access_log_carries_the_path_without_the_query_string() {
        let fields = Fields::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(fields.clone()));
        let app = Router::new()
            .route("/v1/models", get(|| async { "ok" }))
            .layer(create_logging_layer());
        let key = "sk-test-7f3a9c0e11";
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/models?api_key={key}"))
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let recorded = fields.0.lock().expect("fields").clone();
        assert!(
            !recorded.iter().any(|field| field.contains(key)),
            "the credential reached the log fields: {recorded:?}"
        );
        assert!(
            recorded.iter().any(|field| field == "path=/v1/models"),
            "the request path is missing from the log fields: {recorded:?}"
        );
    }
}

/// The access log's one line for a client that closed the connection before
/// the response completed, at INFO.
#[cfg(test)]
mod client_closed_tests {
    use std::{
        convert::Infallible,
        sync::{Arc, Mutex},
        task::Poll,
    };

    use axum::{
        body::{Body, Bytes},
        http::Request,
        response::Response,
        routing::get,
        Router,
    };
    use http::header::CONTENT_LENGTH;
    use http_body_util::BodyExt;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;
    use tracing::Level;
    use tracing_subscriber::{layer::SubscriberExt, Layer, Registry};

    use super::*;

    /// Every event of the `smg::response` target, as (level, message).
    #[derive(Clone, Default)]
    struct ResponseLines(Arc<Mutex<Vec<(Level, String)>>>);

    impl ResponseLines {
        fn lines(&self) -> Vec<(Level, String)> {
            self.0.lock().expect("lines").clone()
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for ResponseLines {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if event.metadata().target() != "smg::response" {
                return;
            }
            struct Message(String);
            impl tracing::field::Visit for Message {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }
            }
            let mut message = Message(String::new());
            event.record(&mut message);
            self.0
                .lock()
                .expect("lines")
                .push((*event.metadata().level(), message.0));
        }
    }

    fn line(level: Level, message: &str) -> (Level, String) {
        (level, message.to_owned())
    }

    fn request(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .body(Body::empty())
            .expect("request")
    }

    /// `/stream` hands out `body` once; `/slow` never answers.
    fn app(body: Body) -> Router {
        let slot = Arc::new(Mutex::new(Some(body)));
        Router::new()
            .route(
                "/stream",
                get(move || {
                    let body = slot
                        .lock()
                        .expect("slot")
                        .take()
                        .expect("handler called once");
                    async move { Response::new(body) }
                }),
            )
            .route(
                "/slow",
                get(|| async {
                    std::future::pending::<()>().await;
                    "never"
                }),
            )
            .layer(create_logging_layer())
    }

    /// `/relay` answers with `body` under a `Content-Length` of nine bytes,
    /// as the HTTP relay answers an upstream that declared one.
    fn relay_app(body: Body) -> Router {
        let slot = Arc::new(Mutex::new(Some(body)));
        Router::new()
            .route(
                "/relay",
                get(move || {
                    let body = slot
                        .lock()
                        .expect("slot")
                        .take()
                        .expect("handler called once");
                    async move {
                        Response::builder()
                            .header(CONTENT_LENGTH, 9)
                            .body(body)
                            .expect("response")
                    }
                }),
            )
            .layer(create_logging_layer())
    }

    #[tokio::test]
    async fn a_body_under_a_content_length_dropped_after_its_declared_bytes_logs_no_client_close() {
        // On HTTP/1.1 the server stops asking for frames once the declared
        // bytes are out and drops the body: sent in full, not a client that
        // left.
        let sink = ResponseLines::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(sink.clone()));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = relay_app(Body::from_stream(ReceiverStream::new(rx)))
            .oneshot(request("/relay"))
            .await
            .expect("response");
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(9));
        drop(body);
        drop(tx);
        assert_eq!(
            sink.lines(),
            vec![line(Level::INFO, "finished processing request")]
        );
    }

    #[tokio::test]
    async fn a_body_under_a_content_length_dropped_before_its_declared_bytes_logs_a_client_close() {
        // The other side of the declared-length rule: the client left with
        // the body short of its declared bytes.
        let sink = ResponseLines::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(sink.clone()));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = relay_app(Body::from_stream(ReceiverStream::new(rx)))
            .oneshot(request("/relay"))
            .await
            .expect("response");
        tx.send(Ok(Bytes::from_static(b"data:")))
            .await
            .expect("send");
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(5));
        drop(body);
        drop(tx);
        assert_eq!(
            sink.lines(),
            vec![
                line(Level::INFO, "finished processing request"),
                line(
                    Level::INFO,
                    "client closed the connection before the response stream ended"
                ),
            ]
        );
    }

    #[tokio::test]
    async fn a_request_the_client_dropped_before_the_response_logs_one_info_line() {
        let sink = ResponseLines::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(sink.clone()));
        let mut pending = Box::pin(app(Body::empty()).oneshot(request("/slow")));
        assert!(matches!(futures::poll!(pending.as_mut()), Poll::Pending));
        drop(pending);
        assert_eq!(
            sink.lines(),
            vec![line(
                Level::INFO,
                "client closed the connection before the response"
            )]
        );
    }

    #[tokio::test]
    async fn a_stream_the_client_dropped_logs_one_info_line_after_the_access_log() {
        let sink = ResponseLines::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(sink.clone()));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = app(Body::from_stream(ReceiverStream::new(rx)))
            .oneshot(request("/stream"))
            .await
            .expect("response");
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert!(frame.is_data());
        drop(body);
        drop(tx);
        assert_eq!(
            sink.lines(),
            vec![
                line(Level::INFO, "finished processing request"),
                line(
                    Level::INFO,
                    "client closed the connection before the response stream ended"
                ),
            ]
        );
    }

    #[tokio::test]
    async fn a_stream_that_ends_logs_no_client_close() {
        let sink = ResponseLines::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(sink.clone()));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = app(Body::from_stream(ReceiverStream::new(rx)))
            .oneshot(request("/stream"))
            .await
            .expect("response");
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        drop(tx);
        let _ = response.into_body().collect().await.expect("body");
        assert_eq!(
            sink.lines(),
            vec![line(Level::INFO, "finished processing request")]
        );
    }
}
