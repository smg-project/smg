//! HTTP metrics collection (SMG Layer 1 metrics).
//!
//! `HttpMetricsLayer` wraps the inner service to record per-request
//! duration plus the in-flight connection count via
//! `InFlightRequestTracker`. The path label is the matched axum route
//! template (or `"other"` when unmatched) to bound metric cardinality.

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
use tower::{Layer, Service};

use crate::{
    observability::{
        inflight_tracker::{InFlightGuard, InFlightRequestTracker},
        metrics::{method_to_static_str, Metrics},
    },
    routers::error::extract_error_code_from_response,
};

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

            let response = match inner.call(req).await {
                Ok(response) => response,
                Err(e) => {
                    // Decrement on error too.
                    drop(guard);
                    Metrics::set_http_connections_active(in_flight_request_tracker.len());
                    return Err(e);
                }
            };

            let duration = start.elapsed();
            Metrics::record_http_response(
                &path,
                response.status().as_u16(),
                extract_error_code_from_response(&response),
            );
            Metrics::record_http_duration(method, &path, duration);

            // Hold the in-flight guard for the response BODY's lifetime, not
            // the handler's: streaming handlers return their response
            // immediately and keep generating from a detached task, and the
            // guard is what `wait_for_drain()` and the age histogram observe.
            let (parts, body) = response.into_parts();
            let body = Body::new(TrackedBody {
                inner: body,
                guard: Some(guard),
                tracker: in_flight_request_tracker,
            });
            Ok(Response::from_parts(parts, body))
        })
    }
}

/// Response-body wrapper that carries the in-flight guard until the body
/// completes or is dropped.
///
/// Unary bodies release at write-out (same point a handler-scoped guard would
/// observe, modulo the final write); streaming bodies release at stream
/// completion or client disconnect — which is what makes SSE generations
/// visible to graceful drain and stuck-request detection.
struct TrackedBody {
    inner: Body,
    guard: Option<InFlightGuard>,
    tracker: Arc<InFlightRequestTracker>,
}

impl TrackedBody {
    fn release(&mut self) {
        if self.guard.take().is_some() {
            Metrics::set_http_connections_active(self.tracker.len());
        }
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
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        // Release eagerly at end-of-stream/error rather than waiting for the
        // wrapper's own drop, so a response object kept alive after its body
        // finished does not count as in flight.
        if matches!(&poll, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            this.release();
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
