//! Tracing/logging integration for the HTTP layer.
//!
//! [`RequestTraceLayer`] opens one `http_request` span per request with the
//! `tower_http::trace` handlers below (W3C trace context in, the request ID
//! and trace ID on the span, HTTP-level metrics), runs the handler under it
//! and logs the response. The span is entered for the request, the handler
//! and the response log, and once more if the response stream fails; it is
//! not entered per body frame, so a long SSE stream costs no span bookkeeping
//! per chunk. The span stays open until the body ends, so an exported trace
//! still covers the whole stream: it rides on the edge's one body wrapper
//! (the metrics layer's `EdgeBody`), handed
//! over through the response's extensions with the client-close line the
//! access log owes if the body is dropped first.

use std::{
    borrow::Cow,
    fmt::Write as _,
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
    response::Response,
};
use opentelemetry::trace::TraceContextExt;
use tower::{Layer, Service};
use tower_http::{
    classify::ServerErrorsFailureClass,
    trace::{MakeSpan, OnFailure, OnRequest, OnResponse},
};
use tracing::{
    debug, error, field::Empty, info, info_span, instrument::Instrumented, warn, Instrument, Span,
};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::{
    auth::fingerprint_of,
    duty::{Duty, DutyFuture},
    edge_body::BodyDuties,
    metrics::{matched_path_label, BodyProgress, CLIENT_CLOSED_REQUEST_STATUS},
    request_id::RequestId,
    request_start::{clock_now, RequestStarted},
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
/// log line for it already (one WARN per refusal: a selection's rate-limited
/// under a sustained condition, with the request id, the model and the
/// pool's state; a preparation's with the cause of a client's mistake).
/// [`ResponseLogger`] logs a marked response at DEBUG instead of a
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
            self.span
                .record("key_id", fingerprint_of(key_hash).as_str());
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
            // What the Symphony reader leaves of an output it did not read
            // clean (`regular::symphony::telemetry::Mark`), recorded when
            // the output ends, never for a clean one.
            symphony.table = Empty,
            symphony.outcome = Empty,
            symphony.choice = Empty,
            symphony.reason = Empty,
            symphony.bytes = Empty,
            symphony.malformed_bytes = Empty,
            symphony.finish = Empty,
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
/// span (inside [`create_logging_layer`]'s trace layer) as the response map of
/// `tower::util::MapResponseLayer` (the response's own future carries it,
/// nothing is cloned or boxed per request); a no-op while tracing is off.
pub fn trace_context_on_response(mut response: Response) -> Response {
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
    S: Service<Request, Response = Response>,
{
    type Response = Response;
    type Error = S::Error;
    type Future = RequestTraceFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request) -> Self::Future {
        let span = RequestSpan.make_span(&request);
        // The latency's origin: the edge's entry stamp when the metrics layer
        // runs this request inside its scope (see `request_start`), a read
        // of this layer's own otherwise.
        let start = RequestStarted::current().unwrap_or_else(clock_now);
        let fields = self.fields;
        record_request_fields(&span, &request, fields);
        RequestLogger.on_request(&request, &span);
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
        let close_log = ClientCloseLog::new(span.clone(), start, request_bytes.clone());
        // The handler's future is created here and polled under the span, as
        // the trace layer did, so everything it logs carries the span. The
        // layer keeps no clone of the stack inside it per request (a clone of
        // a route is a heap copy of the service in it), and its future is the
        // route's own under the span with the record's duty beside it, not a
        // box around both.
        let inner = {
            let _enter = span.enter();
            self.inner.call(request)
        };
        DutyFuture::new(
            inner.instrument(span.clone()),
            RecordDuty {
                span,
                start,
                fields,
                request_bytes,
                close_log,
            },
        )
    }
}

/// The request-span layer's future: the route's future under the request's
/// span, with the span's record as its duty.
pub type RequestTraceFuture<F> = DutyFuture<Instrumented<F>, RecordDuty>;

/// What the request-span layer holds for a request between its first line
/// and the route's answer: the response line of the access log and the
/// span's hand-over to the edge's body wrapper. Dropped with the future
/// before the answer, its close log writes the client-closed line.
pub struct RecordDuty {
    span: Span,
    start: Instant,
    fields: AccessLogFields,
    request_bytes: Option<Arc<AtomicU64>>,
    close_log: ClientCloseLog,
}

impl Duty for RecordDuty {
    /// The route answered: the response line is written and the span handed
    /// to the edge's body wrapper; an error of the inner service ends the
    /// record without a line of its own.
    fn answered<E>(self, answer: Result<Response, E>) -> Result<Response, E> {
        let Self {
            span,
            start,
            fields,
            request_bytes,
            mut close_log,
        } = self;
        let mut response = match answer {
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
        let head_at = clock_now();
        ResponseLogger.on_response(&response, head_at.saturating_duration_since(start), &span);
        close_log.head_sent();
        let count_bytes = fields.contains(AccessLogField::ResponseBytes);
        // A body without a declared length is a stream: its bytes are
        // known at its end, which then gets a line of its own.
        let end_line = count_bytes && progress.declared().is_none();
        // The span stays open until the body ends: handed to the edge's one
        // body wrapper with the close line it owes and the bytes it counts,
        // which the wrapper reports at the body's end, failure or drop. The
        // head's stamp rides along for the metrics layer's duration.
        BodyDuties::hand(&mut response, |duties| {
            duties.head_at = Some(head_at);
            duties.span = Some(SpanDuty {
                span,
                close_log,
                count_bytes,
                end_line,
            });
        });
        Ok(response)
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

/// The request span, kept open until the response body ends without being
/// entered per frame, and the access log's client-close line: carried by the
/// edge's one body wrapper, which reports the body's end, failure or drop with
/// the bytes that went out, so the record's byte fields are written here, at
/// the body's end, by that one wrapper.
pub(crate) struct SpanDuty {
    span: Span,
    close_log: ClientCloseLog,
    /// Whether the record carries the response body's bytes.
    count_bytes: bool,
    /// Whether the body's end gets a line of its own (a streamed body, whose
    /// bytes are known only then); cleared once the line is written.
    end_line: bool,
}

impl SpanDuty {
    /// The body ended (its last frame, or the declared length out) with
    /// `bytes_sent` bytes out: a streamed body's end gets its line, with the
    /// bytes, when the record carries them. The line carries the span's
    /// fields as they stand at the body's end, so what the request learned
    /// while it streamed (the Symphony reader's `symphony.*` mark, recorded
    /// when the output ended) is read here, where the finish line went out
    /// before it existed.
    pub(crate) fn ended(&mut self, bytes_sent: u64) {
        self.close_log.done();
        if std::mem::replace(&mut self.end_line, false) {
            self.span.record("response_bytes", bytes_sent);
            let _enter = self.span.enter();
            info!(
                target: "smg::response",
                "finished streaming the response"
            );
        }
    }

    /// The response head went out but its body was never polled (the answer
    /// to a HEAD request, a body complete before anyone read it): the span
    /// closes with it, without an end line or a client-close line.
    pub(crate) fn unpolled(&mut self) {
        self.close_log.done();
    }

    /// The body failed after its head: logged under the span, once, with the
    /// bytes out before the failure under the early-end key (`response_bytes`
    /// may hold the declared length already).
    pub(crate) fn failed(&mut self, error: &axum::Error, bytes_sent: u64) {
        if self.count_bytes {
            self.span.record("response_bytes_sent", bytes_sent);
        }
        StreamFailureLogger.on_failure(
            ServerErrorsFailureClass::Error(error.to_string()),
            Duration::ZERO,
            &self.span,
        );
        self.close_log.done();
    }

    /// The body was dropped before its end: the client closed the connection,
    /// with the `bytes_sent` out so far on its line when the record carries
    /// them.
    pub(crate) fn client_closed(&mut self, bytes_sent: u64) {
        let bytes = self.count_bytes.then_some(bytes_sent);
        self.close_log.client_closed(bytes);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/logging/tests.rs"]
mod tests;

/// The access log's one line for a client that closed the connection before
/// the response completed, at INFO.
#[cfg(test)]
#[path = "../../tests/unit/middleware/logging/client_closed_tests.rs"]
mod client_closed_tests;

/// The record's optional fields: none by default, each written once
/// selected, under the JSON line format and the text format.
#[cfg(test)]
#[path = "../../tests/unit/middleware/logging/field_tests.rs"]
mod field_tests;
