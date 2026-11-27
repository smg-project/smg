use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use axum::{
    body::{Body, Bytes},
    http::{Request, StatusCode},
    routing::get,
    Router,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use tracing_subscriber::Layer;

use super::*;
use crate::{
    middleware::HttpMetricsLayer,
    observability::{
        inflight_tracker::InFlightRequestTracker, logging::test_support::test_subscriber,
    },
};

/// Counts how often any span is entered on this thread's subscriber.
struct EnterCounter(Arc<AtomicUsize>);

impl<S: tracing::Subscriber> Layer<S> for EnterCounter {
    fn on_enter(&self, _id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn a_streamed_body_does_not_enter_the_request_span_per_frame() {
    let enters = Arc::new(AtomicUsize::new(0));
    let _subscriber = test_subscriber(EnterCounter(enters.clone()));
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
        .layer(create_logging_layer(AccessLogFields::NONE))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
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
    let _subscriber = test_subscriber(LevelCounter(tracing::Level::ERROR, errors.clone()));
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
        .layer(create_logging_layer(AccessLogFields::NONE))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
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
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
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
    let _subscriber = test_subscriber(fields.clone());
    let app = Router::new()
        .route("/v1/models", get(|| async { "ok" }))
        .layer(create_logging_layer(AccessLogFields::NONE))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
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
