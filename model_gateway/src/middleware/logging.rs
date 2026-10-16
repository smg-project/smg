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
    borrow::Cow,
    fmt::Write as _,
    future::Future,
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{connect_info::ConnectInfo, Request},
    http::{header::USER_AGENT, HeaderMap, HeaderName},
    middleware::Next,
    response::Response,
};
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
use crate::{
    config::AccessLogField,
    observability::{
        metrics::{method_to_static_str, Metrics},
        otel_trace::{extract_trace_context_http, inject_trace_context_http, is_otel_enabled},
    },
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

/// The optional fields of the access-log record, as `access_log_fields`
/// selected them: none by default. Every request span declares the fields
/// (a field never recorded is never written, so a line keeps the fields it
/// has always had) and the layer records a field only when it is selected,
/// so an unselected field costs a request nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccessLogFields(u16);

impl AccessLogFields {
    /// No optional field: the record as it has always been.
    pub const NONE: Self = Self(0);

    /// The fields `fields` lists.
    #[must_use]
    pub fn from_fields(fields: &[AccessLogField]) -> Self {
        fields
            .iter()
            .fold(Self::NONE, |set, field| Self(set.0 | Self::bit(*field)))
    }

    /// Every optional field.
    #[must_use]
    pub fn all() -> Self {
        Self::from_fields(&AccessLogField::ALL)
    }

    /// Whether `field` is selected.
    #[must_use]
    pub fn contains(self, field: AccessLogField) -> bool {
        self.0 & Self::bit(field) != 0
    }

    fn bit(field: AccessLogField) -> u16 {
        1 << (field as u16)
    }
}

/// The access-log record of one request, handed down the request's
/// extensions while a field only an inner layer learns is selected: the
/// serving path authenticates the caller's key and resolves its tenant in
/// route layers, after the request's first line, and those layers record
/// what they learn here. Absent while neither field is selected, so a
/// resolving layer pays one extension lookup and nothing else.
#[derive(Clone, Debug)]
pub struct AccessLogRecord {
    span: Span,
    fields: AccessLogFields,
}

impl AccessLogRecord {
    /// The caller's tenant, as the serving path keyed it.
    pub fn tenant(&self, tenant: &str) {
        if self.fields.contains(AccessLogField::Tenant) {
            self.span.record("tenant", tenant);
        }
    }

    /// The key the request authenticated with, by its fingerprint: the first
    /// 16 hex digits of the key's SHA-256. The key itself never reaches the
    /// log.
    pub fn key_id(&self, key_hash: &[u8; 32]) {
        if self.fields.contains(AccessLogField::KeyId) {
            let mut id = String::with_capacity(16);
            for byte in &key_hash[..8] {
                // Writing into a String cannot fail.
                let _ = write!(id, "{byte:02x}");
            }
            self.span.record("key_id", id.as_str());
        }
    }
}

/// The longest `User-Agent` or `X-Forwarded-For` value the record carries,
/// in bytes; a longer one is cut at a character boundary.
const HEADER_FIELD_MAX_BYTES: usize = 256;

static X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");
/// The header the routers set on a response with the worker that served it.
static ROUTED_WORKER_ID: HeaderName = HeaderName::from_static("x-smg-routed-worker-id");

/// A header's value as text for the record: decoded lossily (a user agent is
/// not always visible ASCII) and bounded by [`HEADER_FIELD_MAX_BYTES`].
fn header_field<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<Cow<'a, str>> {
    Some(bounded(String::from_utf8_lossy(
        headers.get(name)?.as_bytes(),
    )))
}

/// A header's every field line as one text for the record, the lines joined
/// with `", "` (the combined value RFC 9110 section 5.3 makes of them): a
/// proxy may add a line of its own rather than append to the client's, and
/// the record keeps every hop, in order, for the reader to weigh. Decoded
/// lossily and bounded as [`header_field`] is.
fn header_lines_field<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<Cow<'a, str>> {
    let mut lines = headers.get_all(name).iter();
    let mut text = String::from_utf8_lossy(lines.next()?.as_bytes());
    for line in lines {
        let joined = text.to_mut();
        joined.push_str(", ");
        joined.push_str(&String::from_utf8_lossy(line.as_bytes()));
    }
    Some(bounded(text))
}

/// `text` within [`HEADER_FIELD_MAX_BYTES`], cut at a character boundary.
fn bounded(text: Cow<'_, str>) -> Cow<'_, str> {
    if text.len() <= HEADER_FIELD_MAX_BYTES {
        return text;
    }
    let mut end = HEADER_FIELD_MAX_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    Cow::Owned(text[..end].to_owned())
}

/// The request-side fields of the record, on the span before the request's
/// first line: the peer's address, the forwarded-for header as it came, the
/// user agent.
fn record_request_fields(span: &Span, request: &Request, fields: AccessLogFields) {
    if fields == AccessLogFields::NONE {
        return;
    }
    if fields.contains(AccessLogField::ClientAddr) {
        if let Some(ConnectInfo(peer)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
            let mut address = String::with_capacity(48);
            // Writing into a String cannot fail.
            let _ = write!(address, "{peer}");
            span.record("client_addr", address.as_str());
        }
    }
    if fields.contains(AccessLogField::ForwardedFor) {
        if let Some(value) = header_lines_field(request.headers(), &X_FORWARDED_FOR) {
            span.record("forwarded_for", value.as_ref());
        }
    }
    if fields.contains(AccessLogField::UserAgent) {
        if let Some(value) = header_field(request.headers(), &USER_AGENT) {
            span.record("user_agent", value.as_ref());
        }
    }
}

/// The response-side fields of the record, on the span before the finish
/// line: the worker that answered, the request body's bytes, and the
/// response body's length when the response declared one (a streamed body's
/// bytes are counted as they go out and recorded at its end).
fn record_response_fields(
    span: &Span,
    response: &Response,
    fields: AccessLogFields,
    request_bytes: Option<&AtomicU64>,
    declared: Option<u64>,
) {
    if fields == AccessLogFields::NONE {
        return;
    }
    if fields.contains(AccessLogField::Worker) {
        if let Some(worker) = response
            .headers()
            .get(&ROUTED_WORKER_ID)
            .and_then(|value| value.to_str().ok())
        {
            span.record("worker", worker);
        }
    }
    if let Some(count) = request_bytes {
        span.record("request_bytes", count.load(Ordering::Relaxed));
    }
    if let Some(declared) = declared.filter(|_| fields.contains(AccessLogField::ResponseBytes)) {
        span.record("response_bytes", declared);
    }
}

/// A request body whose data bytes are counted as the handler reads them,
/// for the record's `request_bytes`.
struct CountedBody {
    inner: Body,
    count: Arc<AtomicU64>,
}

impl http_body::Body for CountedBody {
    type Data = <Body as http_body::Body>::Data;
    type Error = <Body as http_body::Body>::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &poll {
            if let Some(data) = frame.data_ref() {
                this.count.fetch_add(data.len() as u64, Ordering::Relaxed);
            }
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
            // The optional fields of the record (`access_log_fields`),
            // recorded when selected, never written otherwise.
            client_addr = Empty,
            forwarded_for = Empty,
            user_agent = Empty,
            tenant = Empty,
            key_id = Empty,
            request_bytes = Empty,
            response_bytes = Empty,
            response_bytes_sent = Empty,
            worker = Empty,
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

/// Create the request-span layer for HTTP logging, with the record's
/// optional `fields`.
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer(fields: AccessLogFields) -> RequestTraceLayer {
    RequestTraceLayer { fields }
}

/// Tower layer that runs every request under its `http_request` span: the
/// [`RequestSpan`] handlers above, applied by [`RequestTraceService`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestTraceLayer {
    fields: AccessLogFields,
}

impl<S> Layer<S> for RequestTraceLayer {
    type Service = RequestTraceService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestTraceService {
            inner,
            fields: self.fields,
        }
    }
}

/// Tower service of [`RequestTraceLayer`].
#[derive(Clone, Debug)]
pub struct RequestTraceService<S> {
    inner: S,
    fields: AccessLogFields,
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
        let fields = self.fields;
        let mut inner = self.inner.clone();
        Box::pin(async move {
            record_request_fields(&span, &request, fields);
            RequestLogger.on_request(&request, &span);
            let mut request = request;
            if fields.contains(AccessLogField::Tenant) || fields.contains(AccessLogField::KeyId) {
                request.extensions_mut().insert(AccessLogRecord {
                    span: span.clone(),
                    fields,
                });
            }
            let mut request_bytes = None;
            if fields.contains(AccessLogField::RequestBytes) {
                let count = Arc::new(AtomicU64::new(0));
                let counted = Arc::clone(&count);
                request = request.map(|body| {
                    Body::new(CountedBody {
                        inner: body,
                        count: counted,
                    })
                });
                request_bytes = Some(count);
            }
            let mut close_log = ClientCloseLog::new(span.clone(), start, request_bytes.clone());
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
            let progress = BodyProgress::new(&response);
            record_response_fields(
                &span,
                &response,
                fields,
                request_bytes.as_deref(),
                progress.declared(),
            );
            ResponseLogger.on_response(&response, start.elapsed(), &span);
            close_log.head_sent();
            let count_bytes = fields.contains(AccessLogField::ResponseBytes);
            // A body without a declared length is a stream: its bytes are
            // known at its end, which then gets a line of its own.
            let end_line = count_bytes && progress.declared().is_none();
            Ok(response.map(|body| {
                Body::new(SpannedBody {
                    inner: body,
                    span,
                    close_log,
                    polled: false,
                    progress,
                    count_bytes,
                    end_line,
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
    /// The request body's bytes read so far, when the record carries them.
    request_bytes: Option<Arc<AtomicU64>>,
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
    fn new(span: Span, start: Instant, request_bytes: Option<Arc<AtomicU64>>) -> Self {
        Self {
            span,
            start,
            stage: Stage::BeforeResponse,
            request_bytes,
        }
    }

    fn head_sent(&mut self) {
        self.stage = Stage::Streaming;
    }

    fn done(&mut self) {
        self.stage = Stage::Done;
    }

    /// The client closed the connection: one line for the stage the request
    /// was in, once. `bytes_sent` are the bytes out so far, when the record
    /// carries them: `response_bytes_sent`, a key of its own, since
    /// `response_bytes` may already hold the declared length from the finish
    /// line and the text format appends a second value rather than replacing
    /// the first.
    fn client_closed(&mut self, bytes_sent: Option<u64>) {
        match std::mem::replace(&mut self.stage, Stage::Done) {
            Stage::BeforeResponse => {
                // The access-log fields the response line would have carried.
                self.span
                    .record("status_code", CLIENT_CLOSED_REQUEST_STATUS);
                self.span
                    .record("latency", self.start.elapsed().as_micros() as u64);
                if let Some(count) = &self.request_bytes {
                    self.span
                        .record("request_bytes", count.load(Ordering::Relaxed));
                }
                let _enter = self.span.enter();
                info!(
                    target: "smg::response",
                    "client closed the connection before the response"
                );
            }
            Stage::Streaming => {
                if let Some(bytes) = bytes_sent {
                    self.span.record("response_bytes_sent", bytes);
                }
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
        self.client_closed(None);
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
    /// Whether the record carries the response body's bytes.
    count_bytes: bool,
    /// Whether the body's end gets a line of its own (a streamed body, whose
    /// bytes are known only then); cleared once the line is written.
    end_line: bool,
}

impl SpannedBody {
    /// The body went out in full.
    fn ended(&mut self) {
        self.close_log.done();
        if std::mem::replace(&mut self.end_line, false) {
            self.span.record("response_bytes", self.progress.sent());
            let _enter = self.span.enter();
            info!(
                target: "smg::response",
                "finished streaming the response"
            );
        }
    }

    /// The bytes out so far, when the record carries them.
    fn bytes_out(&self) -> Option<u64> {
        self.count_bytes.then(|| self.progress.sent())
    }
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
                // The bytes out before the failure, under the early-end key
                // (`response_bytes` may hold the declared length already).
                if let Some(bytes) = this.bytes_out() {
                    this.span.record("response_bytes_sent", bytes);
                }
                StreamFailureLogger.on_failure(
                    ServerErrorsFailureClass::Error(error.to_string()),
                    Duration::ZERO,
                    &this.span,
                );
                this.close_log.done();
            }
            Poll::Ready(None) => this.ended(),
            Poll::Ready(Some(Ok(frame))) => {
                // At its end: the last frame, or the declared length out.
                if this
                    .progress
                    .frame_out(frame, http_body::Body::is_end_stream(&this.inner))
                {
                    this.ended();
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
            let bytes_out = self.bytes_out();
            self.close_log.client_closed(bytes_out);
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
            .layer(create_logging_layer(AccessLogFields::NONE));
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
            .layer(create_logging_layer(AccessLogFields::NONE));
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
            .layer(create_logging_layer(AccessLogFields::NONE));
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
            .layer(create_logging_layer(AccessLogFields::NONE))
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
            .layer(create_logging_layer(AccessLogFields::NONE))
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

/// The record's optional fields: none by default, each written once
/// selected, under the JSON line format and the text format.
#[cfg(test)]
mod field_tests {
    use std::{
        convert::Infallible,
        io,
        net::SocketAddr,
        sync::{Arc, Mutex},
    };

    use axum::{
        body::{Body, Bytes},
        extract::connect_info::ConnectInfo,
        http::{header, HeaderValue, Request, StatusCode},
        middleware::from_fn_with_state,
        response::{IntoResponse, Response},
        routing::post,
        Router,
    };
    use http_body_util::BodyExt;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;
    use tracing_subscriber::{fmt::MakeWriter, layer::SubscriberExt, Registry};

    use super::*;
    use crate::{
        config::{PolicyConfig, RouterConfig, RoutingMode},
        middleware::{
            auth_middleware, route_request_meta_middleware, AuthConfig, TenantResolutionState,
        },
        observability::log_format::{JsonLine, SpanFields},
    };

    const KEY: &str = "sk-test-5e1f0c9a7b3d2e4f";
    const PAYLOAD: &[u8] = br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
    const REPLY: &str = r#"{"id":"chatcmpl-1","choices":[]}"#;
    const WORKER: &str = "http://127.0.0.1:8000";
    const PEER: &str = "198.51.100.7:51514";
    const FORWARDED: &str = "203.0.113.9, 198.51.100.2";
    const AGENT: &str = "test-client/1.0";
    const FRAME: &[u8] = b"data: x\n\n";

    /// The lines a subscriber wrote.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().expect("lines").clone()).expect("utf-8 lines")
        }
    }

    impl io::Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("lines").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Capture {
        type Writer = Capture;

        fn make_writer(&'a self) -> Capture {
            self.clone()
        }
    }

    /// The gateway's JSON line format, into `capture`.
    fn json_subscriber(capture: &Capture) -> impl tracing::Subscriber + Send + Sync {
        Registry::default().with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(capture.clone())
                .fmt_fields(SpanFields)
                .event_format(JsonLine::new()),
        )
    }

    /// The gateway's text format, into `capture`.
    fn text_subscriber(capture: &Capture) -> impl tracing::Subscriber + Send + Sync {
        Registry::default().with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(capture.clone()),
        )
    }

    fn routed(mut response: Response) -> Response {
        response
            .headers_mut()
            .insert("x-smg-routed-worker-id", HeaderValue::from_static(WORKER));
        response
    }

    /// `/v1/chat/completions` answers [`REPLY`]; `/v1/stream` answers
    /// `stream` as an SSE body; both read the request body and name the
    /// worker. The serving path's auth and tenant layers sit inside the
    /// access log, as they do in the gateway.
    fn app(fields: AccessLogFields, stream: Option<Body>) -> Router {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![WORKER.to_string()],
            },
            PolicyConfig::Random,
        );
        let tenants = TenantResolutionState::new(&config).expect("a tenant header name");
        let auth = AuthConfig::new(Some(KEY.to_string()));
        let slot = Arc::new(Mutex::new(stream));
        Router::new()
            .route(
                "/v1/chat/completions",
                post(|body: Bytes| async move {
                    assert_eq!(body.len(), PAYLOAD.len());
                    routed((StatusCode::OK, REPLY).into_response())
                }),
            )
            .route(
                "/v1/stream",
                post({
                    let slot = Arc::clone(&slot);
                    move |body: Bytes| {
                        let stream = slot
                            .lock()
                            .expect("slot")
                            .take()
                            .expect("the stream handler is called once");
                        async move {
                            assert_eq!(body.len(), PAYLOAD.len());
                            routed(
                                ([(header::CONTENT_TYPE, "text/event-stream")], stream)
                                    .into_response(),
                            )
                        }
                    }
                }),
            )
            .route(
                "/v1/sized",
                post(move |body: Bytes| {
                    let stream = slot
                        .lock()
                        .expect("slot")
                        .take()
                        .expect("the sized handler is called once");
                    async move {
                        assert_eq!(body.len(), PAYLOAD.len());
                        routed(
                            Response::builder()
                                .header(header::CONTENT_LENGTH, 2 * FRAME.len())
                                .body(stream)
                                .expect("response"),
                        )
                    }
                }),
            )
            .route_layer(from_fn_with_state(tenants, route_request_meta_middleware))
            .route_layer(from_fn_with_state(auth, auth_middleware))
            .layer(create_logging_layer(fields))
    }

    fn request(path: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header(header::AUTHORIZATION, format!("Bearer {KEY}"))
            .header(USER_AGENT, AGENT)
            .header("x-forwarded-for", FORWARDED)
            .extension(ConnectInfo(PEER.parse::<SocketAddr>().expect("a peer")))
            .body(Body::from(PAYLOAD))
            .expect("request")
    }

    /// The access log's JSON lines (the `smg::request` and `smg::response`
    /// targets), as (message, span object).
    fn access_lines(capture: &Capture) -> Vec<(String, serde_json::Map<String, Value>)> {
        capture
            .text()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("a JSON line"))
            .filter(|line| {
                line["target"]
                    .as_str()
                    .is_some_and(|target| target.starts_with("smg::"))
            })
            .map(|line| {
                (
                    line["message"].as_str().expect("a message").to_owned(),
                    line["span"].as_object().expect("the span").clone(),
                )
            })
            .collect()
    }

    /// The span's keys, sorted (the line writes `name` last).
    fn keys(span: &serde_json::Map<String, Value>) -> Vec<&str> {
        let mut keys: Vec<&str> = span.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// The finish line's span fields as they have always been (no request
    /// id layer in these apps).
    const DEFAULT_FINISH_KEYS: [&str; 7] = [
        "latency",
        "method",
        "module",
        "name",
        "path",
        "status_code",
        "version",
    ];

    /// Without a selection, the record is byte for byte what it was: the
    /// same two lines, the same span keys.
    #[tokio::test]
    async fn the_default_record_keeps_its_fields_under_the_json_format() {
        let capture = Capture::default();
        let _guard = tracing::subscriber::set_default(json_subscriber(&capture));
        let response = app(AccessLogFields::NONE, None)
            .oneshot(request("/v1/chat/completions"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        assert_eq!(body.len(), REPLY.len());

        let lines = access_lines(&capture);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0].0, "started processing request");
        assert_eq!(lines[1].0, "finished processing request");
        assert_eq!(keys(&lines[1].1), DEFAULT_FINISH_KEYS);
    }

    /// Every selected field is on the finish line with its value; the
    /// request-side ones are on the first line too; the key itself is
    /// nowhere.
    #[tokio::test]
    async fn every_selected_field_is_on_the_finish_line_under_the_json_format() {
        let capture = Capture::default();
        let _guard = tracing::subscriber::set_default(json_subscriber(&capture));
        let response = app(AccessLogFields::all(), None)
            .oneshot(request("/v1/chat/completions"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let _ = response.into_body().collect().await.expect("body");

        let lines = access_lines(&capture);
        assert_eq!(lines.len(), 2, "{lines:?}");
        let (message, span) = &lines[1];
        assert_eq!(message, "finished processing request");
        let hex: String = Sha256::digest(KEY.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(span["client_addr"], PEER);
        assert_eq!(span["forwarded_for"], FORWARDED);
        assert_eq!(span["user_agent"], AGENT);
        assert_eq!(span["tenant"], format!("auth:{hex}"));
        assert_eq!(span["key_id"], &hex[..16]);
        assert_eq!(span["request_bytes"], PAYLOAD.len());
        assert_eq!(span["response_bytes"], REPLY.len());
        assert_eq!(span["worker"], WORKER);
        let mut expected: Vec<&str> = DEFAULT_FINISH_KEYS.to_vec();
        expected.extend(AccessLogField::ALL.iter().map(|field| field.as_str()));
        expected.sort_unstable();
        assert_eq!(keys(span), expected);

        let (message, span) = &lines[0];
        assert_eq!(message, "started processing request");
        assert_eq!(span["client_addr"], PEER);
        assert_eq!(span["forwarded_for"], FORWARDED);
        assert_eq!(span["user_agent"], AGENT);
        assert!(
            !capture.text().contains(KEY),
            "the key reached the log: {}",
            capture.text()
        );
    }

    /// A streamed body's bytes are known at its end: the finish line carries
    /// no count, a line at the end does; without the field, no third line.
    #[tokio::test]
    async fn a_streamed_response_gets_its_bytes_on_an_end_line() {
        for (fields, expected_lines) in [(AccessLogFields::all(), 3), (AccessLogFields::NONE, 2)] {
            let capture = Capture::default();
            let _guard = tracing::subscriber::set_default(json_subscriber(&capture));
            let frames = futures::stream::iter(
                (0..3).map(|_| Ok::<_, Infallible>(Bytes::from_static(FRAME))),
            );
            let response = app(fields, Some(Body::from_stream(frames)))
                .oneshot(request("/v1/stream"))
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::OK);
            let body = response
                .into_body()
                .collect()
                .await
                .expect("body")
                .to_bytes();
            assert_eq!(body.len(), 3 * FRAME.len());

            let lines = access_lines(&capture);
            assert_eq!(lines.len(), expected_lines, "{fields:?}: {lines:?}");
            assert_eq!(lines[1].0, "finished processing request");
            assert!(
                !lines[1].1.contains_key("response_bytes"),
                "a stream's bytes are not known at its head: {:?}",
                lines[1]
            );
            if expected_lines == 3 {
                let (message, span) = &lines[2];
                assert_eq!(message, "finished streaming the response");
                assert_eq!(span["response_bytes"], 3 * FRAME.len());
                assert_eq!(span["worker"], WORKER);
                assert_eq!(span["status_code"], 200);
            }
        }
    }

    /// A client that leaves mid-stream gets the bytes that went out on its
    /// line.
    #[tokio::test]
    async fn a_client_that_leaves_mid_stream_gets_the_bytes_out_so_far() {
        let capture = Capture::default();
        let _guard = tracing::subscriber::set_default(json_subscriber(&capture));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = app(
            AccessLogFields::all(),
            Some(Body::from_stream(ReceiverStream::new(rx))),
        )
        .oneshot(request("/v1/stream"))
        .await
        .expect("response");
        tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(FRAME.len()));
        drop(body);
        drop(tx);

        let lines = access_lines(&capture);
        let (message, span) = lines.last().expect("a line");
        assert_eq!(
            message,
            "client closed the connection before the response stream ended"
        );
        assert_eq!(span["response_bytes_sent"], FRAME.len());
        assert!(
            !span.contains_key("response_bytes"),
            "a stream cut short has no length: {span:?}"
        );
        assert_eq!(span["request_bytes"], PAYLOAD.len());
    }

    /// A body under a declared length that the client leaves short of it:
    /// the finish line said the length once, and the client-closed line says
    /// the bytes that went out under a key of its own, so the text format
    /// (which appends a re-recorded field rather than replacing it) carries
    /// each value once.
    #[tokio::test]
    async fn a_declared_body_cut_short_keeps_the_length_once_and_adds_the_bytes_sent() {
        let capture = Capture::default();
        let _guard = tracing::subscriber::set_default(text_subscriber(&capture));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = app(
            AccessLogFields::all(),
            Some(Body::from_stream(ReceiverStream::new(rx))),
        )
        .oneshot(request("/v1/sized"))
        .await
        .expect("response");
        tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(FRAME.len()));
        drop(body);
        drop(tx);

        let text = capture.text();
        let finish = text
            .lines()
            .find(|line| line.contains("finished processing request"))
            .expect("the finish line");
        let declared = format!("response_bytes={}", 2 * FRAME.len());
        assert_eq!(finish.matches(&declared).count(), 1, "{finish}");
        assert!(!finish.contains("response_bytes_sent="), "{finish}");
        let closed = text
            .lines()
            .find(|line| {
                line.contains("client closed the connection before the response stream ended")
            })
            .expect("the client-closed line");
        assert_eq!(closed.matches("response_bytes=").count(), 1, "{closed}");
        assert!(closed.contains(&declared), "{closed}");
        assert_eq!(
            closed.matches("response_bytes_sent=").count(),
            1,
            "{closed}"
        );
        assert!(
            closed.contains(&format!("response_bytes_sent={}", FRAME.len())),
            "{closed}"
        );
    }

    /// A stream that fails after the head went out: the failure line carries
    /// the bytes out so far as `response_bytes_sent`, once, and a declared
    /// length stays on its own key, once, under the text format.
    #[tokio::test]
    async fn a_stream_that_fails_keeps_the_length_once_and_adds_the_bytes_sent() {
        let capture = Capture::default();
        let _guard = tracing::subscriber::set_default(text_subscriber(&capture));
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(4);
        let response = app(
            AccessLogFields::all(),
            Some(Body::from_stream(ReceiverStream::new(rx))),
        )
        .oneshot(request("/v1/sized"))
        .await
        .expect("response");
        tx.send(Ok(Bytes::from_static(FRAME))).await.expect("send");
        tx.send(Err(io::Error::other("upstream cut")))
            .await
            .expect("send the failure");
        drop(tx);
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(FRAME.len()));
        let failure = body.frame().await.expect("the failure");
        assert!(failure.is_err(), "the body ends in the failure");
        drop(body);

        let text = capture.text();
        let failed = text
            .lines()
            .find(|line| line.contains("response stream failed after the head was sent"))
            .expect("the failure line");
        assert_eq!(
            failed.matches("response_bytes_sent=").count(),
            1,
            "{failed}"
        );
        assert!(
            failed.contains(&format!("response_bytes_sent={}", FRAME.len())),
            "{failed}"
        );
        assert!(failed.matches("response_bytes=").count() <= 1, "{failed}");
        assert!(
            failed.contains(&format!("response_bytes={}", 2 * FRAME.len())),
            "{failed}"
        );
    }

    /// The text format carries the selected fields on the finish line, and
    /// none of them without a selection.
    #[tokio::test]
    async fn the_text_format_carries_the_selected_fields_only() {
        for (fields, selected) in [
            (AccessLogFields::NONE, false),
            (AccessLogFields::all(), true),
        ] {
            let capture = Capture::default();
            let _guard = tracing::subscriber::set_default(text_subscriber(&capture));
            let response = app(fields, None)
                .oneshot(request("/v1/chat/completions"))
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::OK);
            let _ = response.into_body().collect().await.expect("body");

            let text = capture.text();
            let finish = text
                .lines()
                .find(|line| line.contains("finished processing request"))
                .expect("the finish line");
            assert!(finish.contains("status_code=200"), "{finish}");
            for needle in [
                format!("client_addr=\"{PEER}\""),
                format!("forwarded_for=\"{FORWARDED}\""),
                format!("user_agent=\"{AGENT}\""),
                "tenant=\"auth:".to_owned(),
                "key_id=\"".to_owned(),
                format!("request_bytes={}", PAYLOAD.len()),
                format!("response_bytes={}", REPLY.len()),
                format!("worker=\"{WORKER}\""),
            ] {
                assert_eq!(
                    finish.contains(&needle),
                    selected,
                    "{needle} selected={selected}: {finish}"
                );
            }
        }
    }

    /// The record handle rides the request only while a field an inner
    /// layer learns (the tenant, the key) is selected.
    #[tokio::test]
    async fn the_record_handle_rides_the_request_only_for_inner_fields() {
        for (fields, present) in [
            (
                AccessLogFields::from_fields(&[AccessLogField::ClientAddr, AccessLogField::Worker]),
                false,
            ),
            (
                AccessLogFields::from_fields(&[AccessLogField::Tenant]),
                true,
            ),
            (AccessLogFields::from_fields(&[AccessLogField::KeyId]), true),
        ] {
            let app = Router::new()
                .route(
                    "/",
                    post(move |request: Request<Body>| async move {
                        if request.extensions().get::<AccessLogRecord>().is_some() == present {
                            StatusCode::OK
                        } else {
                            StatusCode::INTERNAL_SERVER_ERROR
                        }
                    }),
                )
                .layer(create_logging_layer(fields));
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "handle present = {present} for {fields:?}"
            );
        }
    }

    /// A header value is carried as text, lossily decoded and cut at the
    /// bound on a character boundary.
    #[test]
    fn header_fields_are_bounded_at_a_character_boundary() {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static(AGENT));
        assert_eq!(header_field(&headers, &USER_AGENT).as_deref(), Some(AGENT));

        // 255 bytes, then a two-byte character straddling the bound.
        let long = format!("{}é", "a".repeat(HEADER_FIELD_MAX_BYTES - 1));
        headers.insert(
            USER_AGENT,
            HeaderValue::from_bytes(long.as_bytes()).expect("header"),
        );
        let field = header_field(&headers, &USER_AGENT).expect("a value");
        assert_eq!(field.len(), HEADER_FIELD_MAX_BYTES - 1);
        assert!(field.bytes().all(|byte| byte == b'a'));

        headers.insert(
            USER_AGENT,
            HeaderValue::from_bytes(b"\xff\xfeagent").expect("header"),
        );
        assert_eq!(
            header_field(&headers, &USER_AGENT).as_deref(),
            Some("\u{fffd}\u{fffd}agent")
        );
        assert_eq!(header_field(&headers, &X_FORWARDED_FOR), None);
        assert_eq!(header_lines_field(&headers, &X_FORWARDED_FOR), None);
    }

    /// Every `X-Forwarded-For` field line is in the record, in order, joined
    /// as the combined value: a proxy that adds a line of its own is not
    /// hidden behind the line the client sent.
    #[test]
    fn every_forwarded_for_line_is_in_the_record_in_order() {
        let mut headers = HeaderMap::new();
        headers.append(
            X_FORWARDED_FOR.clone(),
            HeaderValue::from_static("203.0.113.9"),
        );
        headers.append(
            X_FORWARDED_FOR.clone(),
            HeaderValue::from_static("198.51.100.2, 192.0.2.1"),
        );
        assert_eq!(
            header_lines_field(&headers, &X_FORWARDED_FOR).as_deref(),
            Some("203.0.113.9, 198.51.100.2, 192.0.2.1")
        );
        assert_eq!(
            header_field(&headers, &X_FORWARDED_FOR).as_deref(),
            Some("203.0.113.9"),
            "the single-line reader sees the first line alone"
        );

        // The joined text is bounded like a single value, at a character
        // boundary.
        let mut headers = HeaderMap::new();
        headers.append(
            X_FORWARDED_FOR.clone(),
            HeaderValue::from_bytes("a".repeat(HEADER_FIELD_MAX_BYTES - 3).as_bytes())
                .expect("header"),
        );
        headers.append(
            X_FORWARDED_FOR.clone(),
            HeaderValue::from_bytes("é".as_bytes()).expect("header"),
        );
        let field = header_lines_field(&headers, &X_FORWARDED_FOR).expect("a value");
        assert_eq!(field.len(), HEADER_FIELD_MAX_BYTES - 1);
        assert!(field.ends_with(", "), "{field}");
    }

    #[test]
    fn the_field_set_holds_what_was_selected() {
        assert_eq!(AccessLogFields::from_fields(&[]), AccessLogFields::NONE);
        let set = AccessLogFields::from_fields(&[AccessLogField::Worker, AccessLogField::Tenant]);
        for field in AccessLogField::ALL {
            assert_eq!(
                set.contains(field),
                matches!(field, AccessLogField::Worker | AccessLogField::Tenant),
                "{field}"
            );
            assert!(AccessLogFields::all().contains(field), "{field}");
            assert!(!AccessLogFields::NONE.contains(field), "{field}");
        }
    }
}
