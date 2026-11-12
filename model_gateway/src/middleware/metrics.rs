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
        error::extract_error_code_from_response, grpc::utils::stream_failure::STREAM_ERROR_TYPE,
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
pub(super) struct BodyProgress {
    declared: Option<u64>,
    sent: u64,
}

impl BodyProgress {
    pub(super) fn new(response: &Response) -> Self {
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
    pub(super) fn frame_out(&mut self, frame: &http_body::Frame<Bytes>, at_end: bool) -> bool {
        if let Some(data) = frame.data_ref() {
            self.sent = self.sent.saturating_add(data.len() as u64);
        }
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

        outcome.head(&response);

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
    pub(super) fn head(&mut self, response: &Response) {
        self.head = Some((
            response.status().as_u16(),
            error_code_label(response).to_owned(),
        ));
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
mod tests {
    use std::{
        convert::Infallible,
        sync::{Arc, Mutex},
    };

    use axum::{body::Body, http::Request, routing::get, Router};
    use tokio::sync::mpsc;
    use tower::{ServiceBuilder, ServiceExt};

    use super::*;
    use crate::observability::metrics::is_interned;

    #[test]
    fn matched_path_label_defaults_to_other_when_absent() {
        // No routing has run, so there is no MatchedPath extension.
        let extensions = http::Extensions::new();
        assert_eq!(matched_path_label(&extensions), "other");
    }

    /// Drive `request_uri` through a router that has one dynamic route and a
    /// fallback, returning the label `matched_path_label` observes at a
    /// `Router::layer`-applied middleware. The layer is the outermost wrap (it
    /// also covers the fallback) so both the matched and unmatched branches are
    /// exercised at the same layer the production metrics middleware uses.
    async fn label_at_layer(request_uri: &str) -> String {
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let sink = captured.clone();

        let app = Router::new()
            .route("/v1/responses/{response_id}", get(|| async { "ok" }))
            .fallback(|| async { "fallback" })
            .layer(
                ServiceBuilder::new().map_request(move |req: Request<Body>| {
                    *sink.lock().unwrap() = Some(matched_path_label(req.extensions()).to_owned());
                    req
                }),
            );

        let response = app
            .oneshot(
                Request::builder()
                    .uri(request_uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(response.status().is_success());

        let label = captured.lock().unwrap().clone();
        label.expect("label-capturing layer ran")
    }

    #[tokio::test]
    async fn matched_route_uses_template_label() {
        // A matched dynamic route is labeled by its template, never the raw id.
        assert_eq!(
            label_at_layer("/v1/responses/resp_abc123").await,
            "/v1/responses/{response_id}"
        );
    }

    #[tokio::test]
    async fn unmatched_path_collapses_to_other() {
        // An unmatched path must collapse to "other", not echo the raw URI.
        assert_eq!(label_at_layer("/totally/unregistered/aaaa").await, "other");
    }

    /// A body the JSON extractor refused is labelled with the code its envelope
    /// shows (the extractor builds the envelope outside the gateway's error
    /// builder); the builder's own code keeps precedence; a success has none.
    #[tokio::test]
    async fn validation_rejections_carry_their_code_in_the_metric_label() {
        use axum::{http::header::CONTENT_TYPE, response::IntoResponse, routing::post};
        use openai_protocol::{chat::ChatCompletionRequest, validated::ValidatedJson};

        use crate::routers::error::create_error;

        let app = Router::new().route(
            "/v1/chat/completions",
            post(|ValidatedJson(_): ValidatedJson<ChatCompletionRequest>| async { "ok" }),
        );
        for (body, code) in [
            (
                r#"{"model":"m","messages":[{"role":"user","con"#,
                "invalid_json",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
                "integer_below_min_value",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), http::StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(error_code_label(&response), code, "{body}");
        }
        let built = create_error(http::StatusCode::NOT_FOUND, "model_not_found", "no");
        assert_eq!(error_code_label(&built), "model_not_found");
        let ok: Response = "ok".into_response();
        assert_eq!(error_code_label(&ok), "");
    }

    /// A code an upstream forges in `X-SMG-Error-Code` never becomes an
    /// `error_code` label; the code the gateway's own error builder sets
    /// does. Asked of each value by name, so what neighbouring tests intern
    /// meanwhile cannot move the verdict.
    #[tokio::test]
    async fn an_upstream_forged_error_code_never_becomes_a_label() {
        use axum::{
            extract::Path,
            http::HeaderValue,
            response::{IntoResponse, Response},
        };

        use crate::{
            observability::inflight_tracker::InFlightRequestTracker,
            routers::error::{create_error, HEADER_X_SMG_ERROR_CODE},
        };

        // A handler that returns a fresh X-SMG-Error-Code per response, as an
        // upstream would, and one that refuses through the gateway's builder.
        const GATEWAY_CODE: &str = "forged_code_test_gateway_refusal";
        let app = Router::new()
            .route(
                "/echo/{id}",
                get(|Path(id): Path<String>| async move {
                    let mut response: Response = "ok".into_response();
                    response.headers_mut().insert(
                        HEADER_X_SMG_ERROR_CODE,
                        HeaderValue::from_str(&format!("evil-{id}")).unwrap(),
                    );
                    response
                }),
            )
            .route(
                "/refused",
                get(|| async { create_error(http::StatusCode::NOT_FOUND, GATEWAY_CODE, "no") }),
            )
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));

        let send = |uri: String, expected: http::StatusCode| {
            let app = app.clone();
            async move {
                let response = app
                    .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert_eq!(response.status(), expected);
            }
        };

        const ITERS: usize = 1000;
        for i in 0..ITERS {
            send(format!("/echo/{i}"), http::StatusCode::OK).await;
        }
        send("/refused".to_owned(), http::StatusCode::NOT_FOUND).await;

        for i in 0..ITERS {
            assert!(
                !is_interned(&format!("evil-{i}")),
                "the upstream's forged code evil-{i} became a label"
            );
        }
        assert!(
            is_interned(GATEWAY_CODE),
            "the gateway's own code is the error_code label"
        );
    }

    /// A request id in the path of a matched route never becomes a `path`
    /// label; the route's template does. Asked of each value by name, so
    /// what neighbouring tests intern meanwhile cannot move the verdict.
    #[tokio::test]
    async fn a_request_id_on_a_matched_route_never_becomes_a_label() {
        // Drive the real `HttpMetricsLayer`. Every request matches the dynamic
        // route `/v1/responses/{response_id}`, so each distinct id must record
        // the bounded template label and nothing of its own.
        const TEMPLATE: &str = "/v1/responses/{response_id}";
        let app = Router::new()
            .route(TEMPLATE, get(|| async { "ok" }))
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));

        let send = |uri: String| {
            let app = app.clone();
            async move {
                let response = app
                    .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert!(response.status().is_success());
            }
        };

        const ITERS: usize = 1000;
        for i in 0..ITERS {
            send(format!("/v1/responses/resp_{i}")).await;
        }

        for i in 0..ITERS {
            assert!(
                !is_interned(&format!("/v1/responses/resp_{i}")),
                "the request path /v1/responses/resp_{i} became a label"
            );
        }
        assert!(
            is_interned(TEMPLATE),
            "the route's template is the path label"
        );
    }

    /// Build a router whose `/stream` handler hands out the given body once,
    /// wrapped by the real metrics layer over `tracker`.
    fn stream_app(tracker: Arc<InFlightRequestTracker>, body: Body) -> Router {
        let slot = Arc::new(Mutex::new(Some(body)));
        Router::new()
            .route(
                "/stream",
                get(move || {
                    let body = slot.lock().unwrap().take().expect("handler called once");
                    async move { Response::new(body) }
                }),
            )
            .layer(HttpMetricsLayer::new(tracker))
    }

    #[tokio::test]
    async fn streaming_body_holds_inflight_until_completion() {
        use http_body_util::BodyExt;
        use tokio_stream::wrappers::ReceiverStream;

        let tracker = InFlightRequestTracker::new();
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let app = stream_app(tracker.clone(), Body::from_stream(ReceiverStream::new(rx)));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // The handler has returned but the body is still open: the request
        // must still count as in flight (this is exactly what a detached SSE
        // producer looks like to the middleware).
        assert_eq!(tracker.len(), 1);

        tx.send(Ok(Bytes::from_static(b"chunk"))).await.unwrap();
        drop(tx); // end of stream

        let collected = response.into_body().collect().await.unwrap();
        assert_eq!(collected.to_bytes(), Bytes::from_static(b"chunk"));
        assert_eq!(tracker.len(), 0);
        assert!(
            tracker
                .wait_for_drain(std::time::Duration::from_secs(1))
                .await
        );
    }

    #[tokio::test]
    async fn dropped_streaming_body_releases_inflight() {
        use tokio_stream::wrappers::ReceiverStream;

        let tracker = InFlightRequestTracker::new();
        // Keep the sender alive: the stream never ends on its own.
        let (_tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
        let app = stream_app(tracker.clone(), Body::from_stream(ReceiverStream::new(rx)));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tracker.len(), 1);

        // Client disconnect: the response (and its body) is dropped while the
        // stream is still open — the guard must release immediately.
        drop(response);
        assert_eq!(tracker.len(), 0);
    }

    #[tokio::test]
    async fn already_complete_body_releases_at_wrap() {
        use crate::observability::inflight_tracker::InFlightRequestTracker;

        let tracker = InFlightRequestTracker::new();
        let app = Router::new()
            .route("/empty", get(|| async { Response::new(Body::empty()) }))
            .layer(HttpMetricsLayer::new(tracker.clone()));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/empty")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // An empty body is end-of-stream at wrap time; a consumer may never
        // poll it, so the guard must already be gone.
        assert_eq!(tracker.len(), 0);
        drop(response);
    }

    #[tokio::test]
    async fn final_frame_releases_without_trailing_poll() {
        use http_body_util::BodyExt;

        use crate::observability::inflight_tracker::InFlightRequestTracker;

        let tracker = InFlightRequestTracker::new();
        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .layer(HttpMetricsLayer::new(tracker.clone()));

        let response = app
            .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let mut body = response.into_body();

        // A sized body is not end-of-stream before its data frame...
        assert_eq!(tracker.len(), 1);

        // ...but after the final successful frame the http_body contract lets
        // the consumer stop polling: release must not wait for the None.
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert!(frame.is_data());
        assert_eq!(tracker.len(), 0);
        drop(body);
    }

    #[tokio::test]
    async fn unary_response_releases_inflight_after_body() {
        use http_body_util::BodyExt;

        let tracker = InFlightRequestTracker::new();
        let app = Router::new()
            .route("/ok", get(|| async { "ok" }))
            .layer(HttpMetricsLayer::new(tracker.clone()));

        let response = app
            .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let _ = response.into_body().collect().await.unwrap();
        assert_eq!(tracker.len(), 0);
        assert!(tracker.is_empty());
    }
}

/// `smg_http_responses_total` counts a request once, at its outcome: the
/// head's status once the response is out in full, the head's status with
/// `stream_error` when its body failed after the head, 499 when the client
/// closed the connection first.
#[cfg(test)]
mod client_closed_tests {
    use std::{
        convert::Infallible,
        future::Future,
        pin::pin,
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
    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        middleware::edge_body::BodyFailure,
        observability::inflight_tracker::InFlightRequestTracker, routers::error::create_error,
    };

    /// Run `scenario` under a fresh recorder, on a current-thread runtime so
    /// everything it records lands there: the rendered metrics afterwards.
    fn render<F, Fut>(scenario: F) -> String
    where
        F: FnOnce(PrometheusHandle) -> Fut,
        Fut: Future<Output = ()>,
    {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(scenario(handle.clone()));
        });
        handle.render()
    }

    /// [`render`] on a paused clock: the scenario advances time itself.
    fn render_paused<F, Fut>(scenario: F) -> String
    where
        F: FnOnce(PrometheusHandle) -> Fut,
        Fut: Future<Output = ()>,
    {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        ::metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .start_paused(true)
                .build()
                .expect("runtime");
            rt.block_on(scenario(handle.clone()));
        });
        handle.render()
    }

    /// The `smg_http_request_duration_seconds` sum of `path`.
    fn duration_sum(rendered: &str, path: &str) -> f64 {
        rendered
            .lines()
            .find(|line| {
                line.starts_with("smg_http_request_duration_seconds_sum{")
                    && line.contains(&format!("path=\"{path}\""))
            })
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("a duration sum for {path} in {rendered}"))
    }

    /// The `smg_http_connections_active` sample: the requests in flight at
    /// the edge, as the gauge reports them.
    fn connections_active(rendered: &str) -> f64 {
        rendered
            .lines()
            .find(|line| line.starts_with("smg_http_connections_active "))
            .and_then(|line| line.rsplit(' ').next())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("the in-flight gauge in {rendered}"))
    }

    /// The value of the label `name` on a rendered sample line.
    fn label(line: &str, name: &str) -> String {
        let key = format!("{name}=\"");
        let start = line
            .find(&key)
            .unwrap_or_else(|| panic!("{name} on {line}"))
            + key.len();
        let end = start + line[start..].find('"').expect("a closed label");
        line[start..end].to_owned()
    }

    /// The `smg_http_responses_total` samples of `path`, as
    /// `(status_code, error_code, count)`.
    fn responses(rendered: &str, path: &str) -> Vec<(String, String, u64)> {
        rendered
            .lines()
            .filter(|line| {
                line.starts_with("smg_http_responses_total{")
                    && line.contains(&format!("path=\"{path}\""))
            })
            .map(|line| {
                let count = line
                    .rsplit(' ')
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| panic!("a count on {line}"));
                (label(line, "status_code"), label(line, "error_code"), count)
            })
            .collect()
    }

    fn sample(status_code: &str, error_code: &str) -> Vec<(String, String, u64)> {
        vec![(status_code.to_owned(), error_code.to_owned(), 1)]
    }

    fn request(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .body(Body::empty())
            .expect("request")
    }

    /// A router whose `/stream` handler hands out `body` once, under the
    /// metrics layer.
    fn stream_app(body: Body) -> Router {
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
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
    }

    fn unary_app() -> Router {
        Router::new()
            .route("/ok", get(|| async { "ok" }))
            .route(
                "/refused",
                get(|| async {
                    create_error(
                        http::StatusCode::SERVICE_UNAVAILABLE,
                        "no_available_workers",
                        "no worker".to_owned(),
                    )
                }),
            )
            .route(
                "/slow",
                get(|| async {
                    std::future::pending::<()>().await;
                    "never"
                }),
            )
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
    }

    /// A router whose `/relay` handler answers `status` with `body` under a
    /// `Content-Length` of `length`, as the HTTP relay answers an upstream
    /// that declared one.
    fn relay_app(status: u16, length: u64, body: Body) -> Router {
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
                            .status(status)
                            .header(CONTENT_LENGTH, length)
                            .body(body)
                            .expect("response")
                    }
                }),
            )
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
    }

    /// A router whose `/refused` handler answers the gateway's own 503 error
    /// (its gateway error code on the head) with `body` in place of the
    /// error's JSON.
    fn refused_stream_app(body: Body) -> Router {
        let slot = Arc::new(Mutex::new(Some(body)));
        Router::new()
            .route(
                "/refused",
                get(move || {
                    let body = slot
                        .lock()
                        .expect("slot")
                        .take()
                        .expect("handler called once");
                    async move {
                        let (parts, _) = create_error(
                            http::StatusCode::SERVICE_UNAVAILABLE,
                            "no_available_workers",
                            "no worker".to_owned(),
                        )
                        .into_parts();
                        Response::from_parts(parts, body)
                    }
                }),
            )
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
    }

    /// The error a relayed body ends with when its upstream stream breaks.
    fn broken() -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::ConnectionReset, "upstream reset")
    }

    #[test]
    fn a_stream_whose_body_fails_after_its_head_counts_the_head_status_with_the_failure_named() {
        // The HTTP relay's body ends with an error when the upstream stream
        // breaks after the head: the client received a truncated answer.
        let rendered = render(|handle| async move {
            let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/stream")).await.expect("response");
            assert_eq!(response.status(), http::StatusCode::OK);
            let mut body = response.into_body();
            tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
                .await
                .expect("send");
            let frame = body.frame().await.expect("one frame").expect("frame ok");
            assert!(frame.is_data());
            assert!(responses(&handle.render(), "/stream").is_empty());
            tx.send(Err(broken())).await.expect("send");
            assert!(
                body.frame().await.expect("a frame").is_err(),
                "the body ends with the error"
            );
            // Counted at the failure, once: the head's status, the failure
            // named; the drop that follows adds nothing.
            assert_eq!(
                responses(&handle.render(), "/stream"),
                sample("200", "stream_error")
            );
            drop(body);
            drop(tx);
        });
        assert_eq!(
            responses(&rendered, "/stream"),
            sample("200", "stream_error"),
            "{rendered}"
        );
    }

    /// A router whose `/stream` handler hands out `body` once with the
    /// producer's word `failure` on it, under the metrics layer.
    fn told_stream_app(body: Body, failure: BodyFailure) -> Router {
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
                    let failure = failure.clone();
                    async move {
                        let mut response = Response::new(body);
                        failure.carry(&mut response);
                        response
                    }
                }),
            )
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
    }

    /// A streamed body whose producer told the client of a failure in the
    /// body and then ended it as planned (the error event, then `[DONE]`)
    /// counts as a body that failed, what the client read; one whose
    /// producer told of nothing counts as it went out.
    #[test]
    fn a_body_whose_producer_told_of_a_failure_counts_as_a_failed_body() {
        for (told, error_code) in [(true, "stream_error"), (false, "")] {
            let rendered = render(|handle| async move {
                let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
                let failure = BodyFailure::default();
                let app =
                    told_stream_app(Body::from_stream(ReceiverStream::new(rx)), failure.clone());
                let response = app.oneshot(request("/stream")).await.expect("response");
                let mut body = response.into_body();
                tx.send(Ok(Bytes::from_static(b"data: {}\n\n")))
                    .await
                    .expect("send");
                assert!(body.frame().await.expect("a frame").is_ok());
                if told {
                    failure.tell();
                }
                tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")))
                    .await
                    .expect("send");
                drop(tx);
                assert!(body.frame().await.expect("a frame").is_ok());
                assert!(body.frame().await.is_none(), "the body ends as planned");
                // Counted at the end, once: the head's status, the failure
                // named when the producer told of one.
                assert_eq!(
                    responses(&handle.render(), "/stream"),
                    sample("200", error_code),
                    "told: {told}"
                );
                drop(body);
            });
            assert_eq!(
                responses(&rendered, "/stream"),
                sample("200", error_code),
                "told: {told}: {rendered}"
            );
        }
    }

    #[test]
    fn a_body_that_fails_under_an_error_head_keeps_a_gateway_error_code_the_head_carried() {
        // A relayed error head carries no gateway code: the failure names it.
        let rendered = render(|_| async {
            let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            let app = relay_app(503, 12, Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/relay")).await.expect("response");
            let mut body = response.into_body();
            tx.send(Err(broken())).await.expect("send");
            assert!(body.frame().await.expect("a frame").is_err());
            drop(body);
        });
        assert_eq!(
            responses(&rendered, "/relay"),
            sample("503", "stream_error"),
            "{rendered}"
        );
        // The gateway's own refusal already names its outcome on the head.
        let rendered = render(|_| async {
            let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            let app = refused_stream_app(Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/refused")).await.expect("response");
            let mut body = response.into_body();
            tx.send(Err(broken())).await.expect("send");
            assert!(body.frame().await.expect("a frame").is_err());
            drop(body);
        });
        assert_eq!(
            responses(&rendered, "/refused"),
            sample("503", "no_available_workers"),
            "{rendered}"
        );
    }

    #[test]
    fn a_body_under_a_content_length_counts_as_the_response_once_its_declared_bytes_are_out() {
        // A relayed upstream answer that declared its length: on HTTP/1.1
        // the server writes the declared bytes and stops asking for frames,
        // dropping the body before its trailing None. Sent in full, it counts
        // as its head's status, once the last declared byte is out.
        let rendered = render(|handle| async move {
            let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
            let app = relay_app(503, 12, Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/relay")).await.expect("response");
            assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
            let mut body = response.into_body();
            tx.send(Ok(Bytes::from_static(b"{\"error\":")))
                .await
                .expect("send");
            let frame = body.frame().await.expect("a frame").expect("frame ok");
            assert_eq!(frame.data_ref().map(Bytes::len), Some(9));
            // Nine of the twelve bytes out: the response is still being sent.
            assert!(responses(&handle.render(), "/relay").is_empty());
            tx.send(Ok(Bytes::from_static(b"{}}"))).await.expect("send");
            let frame = body.frame().await.expect("a frame").expect("frame ok");
            assert_eq!(frame.data_ref().map(Bytes::len), Some(3));
            // The declared bytes are out: the server drops the body here,
            // without asking for the stream's end.
            assert_eq!(responses(&handle.render(), "/relay"), sample("503", ""));
            drop(body);
            drop(tx);
        });
        assert_eq!(
            responses(&rendered, "/relay"),
            sample("503", ""),
            "{rendered}"
        );
    }

    #[test]
    fn a_body_under_a_content_length_dropped_before_its_declared_bytes_counts_as_a_client_closed_request(
    ) {
        // The other side of the declared-length rule: nine of twelve bytes
        // out when the client leaves is a body cut short.
        let rendered = render(|_| async {
            let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
            let app = relay_app(503, 12, Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/relay")).await.expect("response");
            let mut body = response.into_body();
            tx.send(Ok(Bytes::from_static(b"{\"error\":")))
                .await
                .expect("send");
            let frame = body.frame().await.expect("a frame").expect("frame ok");
            assert_eq!(frame.data_ref().map(Bytes::len), Some(9));
            drop(body);
            drop(tx);
        });
        assert_eq!(
            responses(&rendered, "/relay"),
            sample("499", "client_closed_request"),
            "{rendered}"
        );
    }

    #[test]
    fn a_stream_the_client_dropped_counts_as_a_client_closed_request() {
        let rendered = render(|handle| async move {
            let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
            let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/stream")).await.expect("response");
            assert_eq!(response.status(), http::StatusCode::OK);
            tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
                .await
                .expect("send");
            let mut body = response.into_body();
            let frame = body.frame().await.expect("one frame").expect("frame ok");
            assert!(frame.is_data());
            // Nothing is counted while the stream is open.
            assert!(responses(&handle.render(), "/stream").is_empty());
            // The client goes away mid-stream: the server drops the body it
            // was writing.
            drop(body);
            drop(tx);
        });
        assert_eq!(
            responses(&rendered, "/stream"),
            sample("499", "client_closed_request"),
            "{rendered}"
        );
    }

    #[test]
    fn a_stream_the_client_dropped_before_its_first_frame_counts_as_a_client_closed_request() {
        let rendered = render(|_| async {
            // The sender stays open: the stream never produces a frame.
            let (_tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
            let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/stream")).await.expect("response");
            let mut body = response.into_body();
            {
                let mut first = pin!(body.frame());
                // The server asked for the first frame (none yet) when the
                // client closed the connection.
                assert!(matches!(futures::poll!(first.as_mut()), Poll::Pending));
            }
            drop(body);
        });
        assert_eq!(
            responses(&rendered, "/stream"),
            sample("499", "client_closed_request"),
            "{rendered}"
        );
    }

    #[test]
    fn a_stream_counts_its_status_once_when_it_ends() {
        let rendered = render(|handle| async move {
            let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
            let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
            let response = app.oneshot(request("/stream")).await.expect("response");
            tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
                .await
                .expect("send");
            drop(tx); // end of stream
            let collected = response.into_body().collect().await.expect("body");
            assert_eq!(collected.to_bytes(), Bytes::from_static(b"data: x\n\n"));
            assert_eq!(responses(&handle.render(), "/stream"), sample("200", ""));
        });
        assert_eq!(
            responses(&rendered, "/stream"),
            sample("200", ""),
            "{rendered}"
        );
    }

    #[test]
    fn a_request_the_client_dropped_before_the_response_counts_as_a_client_closed_request() {
        let rendered = render(|_| async {
            let mut pending = Box::pin(unary_app().oneshot(request("/slow")));
            // The handler is at work (polled once, not done) when the client
            // closes the connection: the server drops the request's future.
            assert!(matches!(futures::poll!(pending.as_mut()), Poll::Pending));
            drop(pending);
        });
        assert_eq!(
            responses(&rendered, "/slow"),
            sample("499", "client_closed_request"),
            "{rendered}"
        );
    }

    /// The gauge is set from the tracker's count at the request's arrival
    /// and at every release of its guard. The future dropped before the
    /// route answered is one of the releases: left out, an idle server
    /// reported one phantom request in flight until its next request.
    #[test]
    fn a_request_the_client_dropped_before_the_response_leaves_the_in_flight_gauge_at_the_tracker_s_count(
    ) {
        let tracker = InFlightRequestTracker::new();
        let rendered = render(|handle| {
            let tracker = Arc::clone(&tracker);
            async move {
                let app = Router::new()
                    .route(
                        "/slow",
                        get(|| async {
                            std::future::pending::<()>().await;
                            "never"
                        }),
                    )
                    .layer(HttpMetricsLayer::new(Arc::clone(&tracker)));
                let mut pending = Box::pin(app.oneshot(request("/slow")));
                assert!(matches!(futures::poll!(pending.as_mut()), Poll::Pending));
                assert_eq!(tracker.len(), 1);
                assert_eq!(connections_active(&handle.render()), 1.0);
                // The client closes the connection: the server drops the
                // request's future, and the guard with it.
                drop(pending);
                assert_eq!(tracker.len(), 0, "the guard went with the future");
            }
        });
        assert_eq!(
            connections_active(&rendered),
            0.0,
            "the gauge reads the tracker's count once the future is gone: {rendered}"
        );
    }

    /// The layers inside read the stamp from their `call`s, before their
    /// futures run (the access log does): this layer calls the inner service
    /// under the stamp's scope, not only the future it returns.
    #[tokio::test]
    async fn the_inner_service_s_call_runs_inside_the_entry_stamp_s_scope() {
        let seen = Arc::new(Mutex::new(None));
        let inner = tower::service_fn({
            let seen = Arc::clone(&seen);
            move |_request: Request<Body>| {
                *seen.lock().expect("the slot") = Some(RequestStarted::current().is_some());
                async { Ok::<_, Infallible>(Response::new(Body::empty())) }
            }
        });
        let mut layered = HttpMetricsLayer::new(InFlightRequestTracker::new()).layer(inner);
        let response = layered
            .ready()
            .await
            .expect("ready")
            .call(request("/scoped"))
            .await
            .expect("response");
        let _ = response.into_body().collect().await.expect("body");
        assert_eq!(
            *seen.lock().expect("the slot"),
            Some(true),
            "the inner service's call ran inside the entry stamp's scope"
        );
    }

    /// The layer reads the clock once at the request's entry and shares the
    /// stamp: the handler finds it in the request's scope, and the
    /// request's duration runs from it to the head, on the runtime's clock.
    #[test]
    fn the_entry_stamp_scopes_the_request_and_the_duration_runs_from_it() {
        let rendered = render_paused(|_| async {
            let app = Router::new()
                .route(
                    "/stamped",
                    get(|| async move {
                        let started =
                            RequestStarted::current().expect("the entry stamp scopes the request");
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        assert_eq!(
                            clock_now().saturating_duration_since(started),
                            std::time::Duration::from_secs(3),
                            "the stamp is on the runtime's clock"
                        );
                        "ok"
                    }),
                )
                .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
            let response = app.oneshot(request("/stamped")).await.expect("response");
            let _ = response.into_body().collect().await.expect("body");
        });
        let seconds = duration_sum(&rendered, "/stamped");
        assert!(
            (seconds - 3.0).abs() < 0.01,
            "duration {seconds} s, expected the 3 s from the entry stamp to the head: {rendered}"
        );
    }

    #[test]
    fn a_unary_response_counts_its_status_once_its_body_is_out() {
        let rendered = render(|_| async {
            let response = unary_app().oneshot(request("/ok")).await.expect("response");
            let _ = response.into_body().collect().await.expect("body");
        });
        assert_eq!(responses(&rendered, "/ok"), sample("200", ""), "{rendered}");
    }

    #[test]
    fn a_refusal_keeps_its_status_and_gateway_error_code() {
        let rendered = render(|_| async {
            let response = unary_app()
                .oneshot(request("/refused"))
                .await
                .expect("response");
            let _ = response.into_body().collect().await.expect("body");
        });
        assert_eq!(
            responses(&rendered, "/refused"),
            sample("503", "no_available_workers"),
            "{rendered}"
        );
    }

    #[test]
    fn a_body_the_server_never_polled_counts_as_the_response_it_was() {
        // The answer to a HEAD request: the server drops the body without
        // asking for a frame. Not a client that left.
        let rendered = render(|_| async {
            let response = unary_app().oneshot(request("/ok")).await.expect("response");
            drop(response);
        });
        assert_eq!(responses(&rendered, "/ok"), sample("200", ""), "{rendered}");
    }
}
