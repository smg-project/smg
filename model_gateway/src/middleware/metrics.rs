//! HTTP metrics collection (SMG Layer 1 metrics).
//!
//! `HttpMetricsLayer` wraps the inner service to record per-request
//! duration plus the in-flight connection count via
//! `InFlightRequestTracker`. The path label is the matched axum route
//! template (or `"other"` when unmatched) to bound metric cardinality.
//!
//! `smg_http_responses_total` counts a request once, when its outcome is
//! known: the response's status once the response has gone out in full, or
//! [`CLIENT_CLOSED_REQUEST_STATUS`] (499, nginx's spelling) when the client
//! closed the connection first, before the response was produced or while
//! its body was still streaming. Counted at the head, a stream the client
//! cut was a 200 and a request the client abandoned before its answer was
//! nothing at all.
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
    pin::Pin,
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
use tower::{Layer, Service};

use crate::{
    observability::{
        inflight_tracker::{InFlightGuard, InFlightRequestTracker},
        metrics::{method_to_static_str, Metrics},
    },
    routers::error::extract_error_code_from_response,
};

/// The `error_code` label of a response: the code the gateway's error builder
/// set, else the code of a body the JSON extractor refused (its envelope is
/// built in the protocols crate, outside the builder), else `""` (a success,
/// or an upstream body: an upstream's `X-SMG-Error-Code` header is never read,
/// so forged codes cannot grow the label set).
pub(crate) fn error_code_label(response: &Response) -> &str {
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
/// or once the bytes add up to the declared length. One rule for the
/// metrics wrapper and the logging wrapper, so a response cannot count as
/// complete in one and as cut short in the other.
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
    S: Service<Request, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        let method = method_to_static_str(req.method().as_str());
        let path = matched_path_label(req.extensions()).to_owned();
        let start = Instant::now();

        let mut inner = self.inner.clone();
        let in_flight_request_tracker = self.in_flight_request_tracker.clone();

        Box::pin(async move {
            let guard = in_flight_request_tracker.track();
            Metrics::set_http_connections_active(in_flight_request_tracker.len());
            // Counted once the outcome is known. Dropped with this future
            // before the handler answered (what the server does with a
            // request whose client closed the connection), it counts the
            // request as closed by the client.
            let mut outcome = ResponseOutcome::pending(path.clone());

            let response = match inner.call(req).await {
                Ok(response) => response,
                Err(e) => {
                    // Decrement on error too.
                    drop(guard);
                    Metrics::set_http_connections_active(in_flight_request_tracker.len());
                    outcome.forget();
                    return Err(e);
                }
            };

            outcome.head(&response);
            Metrics::record_http_duration(method, &path, start.elapsed());

            // Hold the in-flight guard for the response BODY's lifetime, not
            // the handler's: streaming handlers return their response
            // immediately and keep generating from a detached task, and the
            // guard is what `wait_for_drain()` and the age histogram observe.
            // The outcome rides along: the response counts when its body
            // ends, and as a client-closed request when the body is dropped
            // before that.
            let progress = BodyProgress::new(&response);
            let (parts, body) = response.into_parts();
            let body = Body::new(TrackedBody::new(
                body,
                guard,
                in_flight_request_tracker,
                outcome,
                progress,
            ));
            Ok(Response::from_parts(parts, body))
        })
    }
}

/// What `smg_http_responses_total` counts for one request, recorded exactly
/// once: the response head's status once the response has gone out in full,
/// a client-closed request when the client left first.
struct ResponseOutcome {
    path: String,
    /// The response head's status and gateway error code, once there is one.
    head: Option<(u16, String)>,
    recorded: bool,
}

impl ResponseOutcome {
    fn pending(path: String) -> Self {
        Self {
            path,
            head: None,
            recorded: false,
        }
    }

    /// The response head is out: the status a completed response counts as.
    fn head(&mut self, response: &Response) {
        self.head = Some((
            response.status().as_u16(),
            error_code_label(response).to_owned(),
        ));
    }

    /// The response went out in full: count its head's status.
    fn completed(&mut self) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        if let Some((status_code, error_code)) = &self.head {
            Metrics::record_http_response(&self.path, *status_code, error_code);
        }
    }

    /// The client closed the connection before the response completed.
    fn client_closed(&mut self) {
        if std::mem::replace(&mut self.recorded, true) {
            return;
        }
        Metrics::record_http_response(
            &self.path,
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

/// Response-body wrapper that carries the in-flight guard, and the response's
/// count, until the body completes or is dropped.
///
/// Unary bodies release at write-out (same point a handler-scoped guard would
/// observe, modulo the final write); streaming bodies release at stream
/// completion or client disconnect — which is what makes SSE generations
/// visible to graceful drain and stuck-request detection, and a stream the
/// client cut countable as such.
struct TrackedBody {
    inner: Body,
    guard: Option<InFlightGuard>,
    tracker: Arc<InFlightRequestTracker>,
    outcome: ResponseOutcome,
    /// Whether a frame was ever asked for. A body dropped before any poll is
    /// one the server never meant to send (the answer to a HEAD request), not
    /// one the client cut short.
    polled: bool,
    /// The declared length and the bytes out: the end-of-body rule shared
    /// with the logging wrapper.
    progress: BodyProgress,
}

impl TrackedBody {
    fn new(
        inner: Body,
        guard: InFlightGuard,
        tracker: Arc<InFlightRequestTracker>,
        outcome: ResponseOutcome,
        progress: BodyProgress,
    ) -> Self {
        let mut body = Self {
            inner,
            guard: Some(guard),
            tracker,
            outcome,
            polled: false,
            progress,
        };
        // An already-complete body (e.g. an empty unary response) may never be
        // polled at all; do not retain the guard past construction for it,
        // and count the response now.
        if body.inner.is_end_stream() {
            body.ended();
        }
        body
    }

    fn release(&mut self) {
        if let Some(guard) = self.guard.take() {
            // Drop before reading len() so the gauge reflects the decrement.
            drop(guard);
            Metrics::set_http_connections_active(self.tracker.len());
        }
    }

    /// The body reached its end (a failure of the body ends it too): the
    /// response is out and counts as its head's status.
    fn ended(&mut self) {
        self.release();
        self.outcome.completed();
    }
}

impl http_body::Body for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        this.polled = true;
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        // Release as soon as the end of the stream is knowable: at the
        // trailing None, on error, right after a successful final frame
        // when the body reports is_end_stream(), or once the data frames add
        // up to the length the response declared. Under the http_body
        // contract, consumers may stop polling at those points and never ask
        // for the None (an HTTP/1.1 server stops at the declared length).
        // Drop still covers client disconnect.
        match &poll {
            Poll::Ready(None) | Poll::Ready(Some(Err(_))) => this.ended(),
            Poll::Ready(Some(Ok(frame))) => {
                if this.progress.frame_out(frame, this.inner.is_end_stream()) {
                    this.ended();
                }
            }
            Poll::Pending => {}
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for TrackedBody {
    fn drop(&mut self) {
        self.release();
        // Dropped before its end: the server stopped writing the body because
        // the client closed the connection. A body that was never polled was
        // not cut short; it counts as the response it was.
        if self.polled {
            self.outcome.client_closed();
        } else {
            self.outcome.completed();
        }
    }
}

/// Bounded path label for HTTP metrics: the matched axum route template, or
/// `"other"` when no route matched. Labeling by raw request path would let
/// attacker-controlled URIs create unbounded distinct labels.
pub(super) fn matched_path_label(extensions: &http::Extensions) -> &str {
    extensions
        .get::<MatchedPath>()
        .map_or("other", MatchedPath::as_str)
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
    use crate::observability::metrics::interner_size;

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

    #[tokio::test]
    async fn upstream_forged_error_codes_do_not_grow_interner() {
        use axum::{
            extract::Path,
            http::HeaderValue,
            response::{IntoResponse, Response},
        };

        use crate::{
            observability::inflight_tracker::InFlightRequestTracker,
            routers::error::HEADER_X_SMG_ERROR_CODE,
        };

        // Simulates a handler returning a fresh X-SMG-Error-Code per response.
        // Only gateway-set codes may become metric labels, so the interner
        // must stay flat.
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

        send("/echo/warmup".to_owned()).await;
        let size_before = interner_size();

        const ITERS: usize = 1000;
        for i in 0..ITERS {
            send(format!("/echo/{i}")).await;
        }

        let growth = interner_size().saturating_sub(size_before);
        assert!(
            growth < 100,
            "interner grew by {growth} for {ITERS} distinct upstream error codes"
        );
    }

    #[tokio::test]
    async fn distinct_ids_on_matched_route_do_not_grow_interner() {
        // Drive the real `HttpMetricsLayer`. Every request matches the dynamic
        // route `/v1/responses/{response_id}`, so each distinct id must record
        // the bounded template label and leave the never-evicted interner flat.
        let app = Router::new()
            .route("/v1/responses/{response_id}", get(|| async { "ok" }))
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

        // Warm up so the template label and the empty error_code are interned.
        send("/v1/responses/resp_warmup".to_owned()).await;
        let size_before = interner_size();

        const ITERS: usize = 1000;
        for i in 0..ITERS {
            send(format!("/v1/responses/resp_{i}")).await;
        }

        // Slack tolerates strings unrelated parallel tests may intern; an
        // unbounded label would instead grow the interner by ~ITERS.
        let growth = interner_size().saturating_sub(size_before);
        assert!(
            growth < 100,
            "interner grew by {growth} for {ITERS} distinct request ids"
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
/// head's status once the response is out in full, 499 when the client closed
/// the connection first.
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
